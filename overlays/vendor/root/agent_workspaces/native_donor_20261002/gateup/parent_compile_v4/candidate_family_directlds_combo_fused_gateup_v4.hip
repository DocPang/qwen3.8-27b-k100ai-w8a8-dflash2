#include <torch/extension.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>
#include <string>

// H-SMALLM-W-DIRECTLDS-XW-OVERLAP-V1 family extension.
// Exact retained M4 shapes only:
//   gateup: K5120 -> N17408, K_CHUNK1024, N_TILE4, RW4
//   gdn:    K5120 -> N8192,  K_CHUNK512,  N_TILE2, halfwave M4
//   out:    K3072 -> N5120,  K_CHUNK1024, N_TILE2, RW4
//
// GDN/out are byte-for-byte the DirectLDS parent. Gate/up adds one narrow
// scheduling change: issue the retained X 16B global load first, keep it live
// in VGPR, then issue the W raw_buffer_load_lds, and only then commit X to LDS.
// This creates X/W outstanding overlap without changing addresses, barriers,
// arithmetic, reduction, scaling, BF16 stores, or the packed-W contract.
namespace {

using i32x4 = int32_t __attribute__((ext_vector_type(4)));
union BufferResource {
  i32x4 v;
  struct { uint64_t address; uint32_t range; uint32_t config; } f;
};

__device__ __forceinline__ i32x4 make_rsrc(const void* p, uint32_t bytes) {
  BufferResource r{};
  const uint64_t a = reinterpret_cast<uint64_t>(p);
  r.v[0] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(a));
  r.v[1] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(a >> 32));
  r.v[2] = __builtin_amdgcn_readfirstlane(bytes);
  r.v[3] = 0x00020000;
  return r.v;
}

__device__ __forceinline__ int dot_all(i32x4 x, i32x4 w, int acc) {
  acc = __builtin_amdgcn_sdot4(static_cast<int>(x[0]), static_cast<int>(w[0]), acc, false);
  acc = __builtin_amdgcn_sdot4(static_cast<int>(x[1]), static_cast<int>(w[1]), acc, false);
  acc = __builtin_amdgcn_sdot4(static_cast<int>(x[2]), static_cast<int>(w[2]), acc, false);
  acc = __builtin_amdgcn_sdot4(static_cast<int>(x[3]), static_cast<int>(w[3]), acc, false);
  return acc;
}

// ---------------------------------------------------------------- gate/up
constexpr int GU_M=4, GU_K=5120, GU_N=17408, GU_KC=1024, GU_KCS=5;
constexpr int GU_NT=4, GU_RW=4, GU_BLOCK=256, GU_GRID=GU_N/GU_NT;
constexpr int GU_W_TILE=GU_NT*GU_KC;
constexpr int64_t GU_PACKED_BYTES=static_cast<int64_t>(GU_N)*GU_K;

__global__ __launch_bounds__(GU_BLOCK) void gateup_directlds(
    const int8_t* __restrict__ x, const int8_t* __restrict__ packed_w,
    const float* __restrict__ xs, const float* __restrict__ ws,
    hip_bfloat16* __restrict__ out) {
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* sx=smem; unsigned char* sw=smem+GU_M*GU_KC;
  const int tid=threadIdx.x, lane=tid&63, wave=tid>>6;
  const int n_base=blockIdx.x*GU_NT;
  int acc[GU_NT] = {};
  const i32x4 rsrc=make_rsrc(packed_w, static_cast<uint32_t>(GU_PACKED_BYTES));

#pragma unroll
  for (int kc=0; kc<GU_KCS; ++kc) {
    // Exact retained X staging: 4*64 vectors, one per thread.
    const int mr=tid/64, kv=tid-(mr*64);
    const auto* src=reinterpret_cast<const i32x4*>(
        x+static_cast<int64_t>(mr)*GU_K+kc*GU_KC+kv*16);
    const i32x4 x_stage=*src;
#if defined(__HIP_DEVICE_COMPILE__)
    const uint32_t tile_base=static_cast<uint32_t>(
        (static_cast<int64_t>(blockIdx.x)*GU_KCS+kc)*GU_W_TILE);
    if (wave==0) {
      __builtin_amdgcn_raw_buffer_load_lds(rsrc,reinterpret_cast<int32_t*>(sw),16,
          tile_base+lane*16,0,0,0);
    } else if (wave==1) {
      __builtin_amdgcn_raw_buffer_load_lds(rsrc,reinterpret_cast<int32_t*>(sw+GU_KC),16,
          tile_base+GU_KC+lane*16,0,0,0);
    } else if (wave==2) {
      __builtin_amdgcn_raw_buffer_load_lds(rsrc,reinterpret_cast<int32_t*>(sw+2*GU_KC),16,
          tile_base+2*GU_KC+lane*16,0,0,0);
    } else {
      __builtin_amdgcn_raw_buffer_load_lds(rsrc,reinterpret_cast<int32_t*>(sw+3*GU_KC),16,
          tile_base+3*GU_KC+lane*16,0,0,0);
    }
#endif
    // Keep x_stage live across the DirectLDS issue so the X global load and W
    // direct-to-LDS load can overlap.  The logical LDS contents are unchanged.
    *reinterpret_cast<i32x4*>(sx+mr*GU_KC+kv*16)=x_stage;
    __syncthreads();
    const i32x4 xv=*reinterpret_cast<const i32x4*>(sx+wave*GU_KC+lane*16);
#pragma unroll
    for (int nt=0; nt<GU_NT; ++nt) {
      const i32x4 wv=*reinterpret_cast<const i32x4*>(sw+nt*GU_KC+lane*16);
      acc[nt]=dot_all(xv,wv,acc[nt]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int nt=0; nt<GU_NT; ++nt) {
    int v=acc[nt];
#pragma unroll
    for (int sh=32; sh>0; sh>>=1) v+=__shfl_down(v,sh,64);
    const int n=n_base+nt;
    if (lane==0) out[static_cast<int64_t>(wave)*GU_N+n]=
        hip_bfloat16(static_cast<float>(v)*xs[wave]*ws[n]);
  }
}

// --------------------------------------------- fused paired-N gate/up (v3)
// This kernel belongs to the SAME family as gateup_directlds above and keeps
// its X DirectLDS staging and its contiguous packed-W read untouched. The only
// differences are (a) the packed W / ws row order and (b) the epilogue.
//
// Packing contract (see pack_gateup_fused.py): CTA b owns packed slots
//     nt=0 -> logical column g[2b]        (gate)
//     nt=1 -> logical column g[2b+1]      (gate)
//     nt=2 -> logical column u[2b]        (up)
//     nt=3 -> logical column u[2b+1]      (up)
// i.e. packed_slot_row(b,nt) == guf_col(2*b, nt), and ws is permuted with the
// identical row permutation so ws[b*4+nt] is that column's w-scale.
//
// Epilogue: each of the 4 accumulator columns is reduced with the retained
// 64-lane shfl_reduce, scaled by (acc * xs[row]) * ws[col] and rounded to
// bf16 exactly as the retained store does, then SiLU(gate)*up is formed in
// fp32, SiLU rounded to bf16, multiplied by up, rounded to bf16 -> [4, 8704].
constexpr int GUF_HALF = 8704;

__device__ __forceinline__ int guf_col(int q2, int nt) {
  return (nt < 2) ? (q2 + nt) : (GUF_HALF + q2 + (nt - 2));
}

__global__ __launch_bounds__(GU_BLOCK) void gateup_fused_directlds(
    const int8_t* __restrict__ x, const int8_t* __restrict__ packed_w,
    const float* __restrict__ xs, const float* __restrict__ ws,
    hip_bfloat16* __restrict__ out) {
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* sx=smem; unsigned char* sw=smem+GU_M*GU_KC;
  const int tid=threadIdx.x, lane=tid&63, wave=tid>>6;
  const int q2=blockIdx.x*2;
  int acc[GU_NT] = {};
  const i32x4 rsrc=make_rsrc(packed_w, static_cast<uint32_t>(GU_PACKED_BYTES));

#pragma unroll
  for (int kc=0; kc<GU_KCS; ++kc) {
    // Exact DirectLDS X staging: 4*64 vectors, one per thread.
    const int mr=tid/64, kv=tid-(mr*64);
    const auto* src=reinterpret_cast<const i32x4*>(
        x+static_cast<int64_t>(mr)*GU_K+kc*GU_KC+kv*16);
    const i32x4 x_stage=*src;
#if defined(__HIP_DEVICE_COMPILE__)
    const uint32_t tile_base=static_cast<uint32_t>(
        (static_cast<int64_t>(blockIdx.x)*GU_KCS+kc)*GU_W_TILE);
    if (wave==0) {
      __builtin_amdgcn_raw_buffer_load_lds(rsrc,reinterpret_cast<int32_t*>(sw),16,
          tile_base+lane*16,0,0,0);
    } else if (wave==1) {
      __builtin_amdgcn_raw_buffer_load_lds(rsrc,reinterpret_cast<int32_t*>(sw+GU_KC),16,
          tile_base+GU_KC+lane*16,0,0,0);
    } else if (wave==2) {
      __builtin_amdgcn_raw_buffer_load_lds(rsrc,reinterpret_cast<int32_t*>(sw+2*GU_KC),16,
          tile_base+2*GU_KC+lane*16,0,0,0);
    } else {
      __builtin_amdgcn_raw_buffer_load_lds(rsrc,reinterpret_cast<int32_t*>(sw+3*GU_KC),16,
          tile_base+3*GU_KC+lane*16,0,0,0);
    }
#endif
    *reinterpret_cast<i32x4*>(sx+mr*GU_KC+kv*16)=x_stage;
    __syncthreads();
    const i32x4 xv=*reinterpret_cast<const i32x4*>(sx+wave*GU_KC+lane*16);
#pragma unroll
    for (int nt=0; nt<GU_NT; ++nt) {
      const i32x4 wv=*reinterpret_cast<const i32x4*>(sw+nt*GU_KC+lane*16);
      acc[nt]=dot_all(xv,wv,acc[nt]);
    }
    __syncthreads();
  }

  // Retained reduction contract: one full 64-lane shfl_reduce per column,
  // performed for all four columns BEFORE any activation is formed.
  int red[GU_NT];
#pragma unroll
  for (int nt=0; nt<GU_NT; ++nt) {
    int v=acc[nt];
#pragma unroll
    for (int sh=32; sh>0; sh>>=1) v+=__shfl_down(v,sh,64);
    red[nt]=v;
  }
  if (lane==0) {
    const int b4=blockIdx.x*GU_NT;
    const float xsv=xs[wave];
    // Same scale/rounding order as the retained store: ((float)acc * xs) * ws.
    const hip_bfloat16 g0(static_cast<float>(red[0])*xsv*ws[b4+0]);
    const hip_bfloat16 g1(static_cast<float>(red[1])*xsv*ws[b4+1]);
    const hip_bfloat16 u0(static_cast<float>(red[2])*xsv*ws[b4+2]);
    const hip_bfloat16 u1(static_cast<float>(red[3])*xsv*ws[b4+3]);
    const float g0f=static_cast<float>(g0), g1f=static_cast<float>(g1);
    const float u0f=static_cast<float>(u0), u1f=static_cast<float>(u1);
    const hip_bfloat16 s0(g0f/(1.0f+__expf(-g0f)));
    const float a0=static_cast<float>(s0)*u0f;
    const hip_bfloat16 s1(g1f/(1.0f+__expf(-g1f)));
    const float a1=static_cast<float>(s1)*u1f;
    out[static_cast<int64_t>(wave)*GUF_HALF+q2+0]=hip_bfloat16(a0);
    out[static_cast<int64_t>(wave)*GUF_HALF+q2+1]=hip_bfloat16(a1);
  }
}

// ---------------------------------------------------------------- GDN qkvz
constexpr int GD_M=4, GD_K=5120, GD_N=8192, GD_KC=512, GD_KCS=10;
constexpr int GD_NT=2, GD_BLOCK=128, GD_GRID=GD_N/GD_NT;
constexpr int GD_W_TILE=GD_NT*GD_KC;
constexpr int64_t GD_PACKED_BYTES=static_cast<int64_t>(GD_N)*GD_K;

__global__ __launch_bounds__(GD_BLOCK) void gdn_directlds(
    const int8_t* __restrict__ x, const int8_t* __restrict__ packed_w,
    const float* __restrict__ xs, const float* __restrict__ ws,
    hip_bfloat16* __restrict__ out) {
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* sx=smem; unsigned char* sw=smem+GD_M*GD_KC;
  const int tid=threadIdx.x, lane64=tid&63, wave=tid>>6;
  const int subgroup=lane64>>5, lane32=lane64&31;
  const int mr=wave*2+subgroup, n_base=blockIdx.x*GD_NT;
  int acc[GD_NT] = {};
  const i32x4 rsrc=make_rsrc(packed_w, static_cast<uint32_t>(GD_PACKED_BYTES));

#pragma unroll
  for (int kc=0; kc<GD_KCS; ++kc) {
    // Exact retained X staging: 4*32 vectors, one per thread.
    const int xrow=tid/32, kv=tid-(xrow*32);
    const auto* src=reinterpret_cast<const i32x4*>(
        x+static_cast<int64_t>(xrow)*GD_K+kc*GD_KC+kv*16);
    *reinterpret_cast<i32x4*>(sx+xrow*GD_KC+kv*16)=*src;
#if defined(__HIP_DEVICE_COMPILE__)
    // Retained W stage is owned by tid 0..63 on the second copy-loop trip.
    if (wave==0) {
      const uint32_t go=static_cast<uint32_t>(
          (static_cast<int64_t>(blockIdx.x)*GD_KCS+kc)*GD_W_TILE+lane64*16);
      __builtin_amdgcn_raw_buffer_load_lds(rsrc,reinterpret_cast<int32_t*>(sw),16,go,0,0,0);
    }
#endif
    __syncthreads();
    const i32x4 xv=*reinterpret_cast<const i32x4*>(sx+mr*GD_KC+lane32*16);
#pragma unroll
    for (int nt=0; nt<GD_NT; ++nt) {
      const i32x4 wv=*reinterpret_cast<const i32x4*>(sw+nt*GD_KC+lane32*16);
      acc[nt]=dot_all(xv,wv,acc[nt]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int nt=0; nt<GD_NT; ++nt) {
    int v=acc[nt];
#pragma unroll
    for (int sh=16; sh>0; sh>>=1) v+=__shfl_down(v,sh,32);
    const int n=n_base+nt;
    if (lane32==0) out[static_cast<int64_t>(mr)*GD_N+n]=
        hip_bfloat16(static_cast<float>(v)*xs[mr]*ws[n]);
  }
}

// ---------------------------------------------------------------- LA out_proj
constexpr int OT_M=4, OT_K=3072, OT_N=5120, OT_KC=1024, OT_KCS=3;
constexpr int OT_NT=2, OT_RW=4, OT_BLOCK=256, OT_GRID=OT_N/OT_NT;
constexpr int OT_W_TILE=OT_NT*OT_KC;
constexpr int64_t OT_PACKED_BYTES=static_cast<int64_t>(OT_N)*OT_K;

__global__ __launch_bounds__(OT_BLOCK) void out_directlds(
    const int8_t* __restrict__ x, const int8_t* __restrict__ packed_w,
    const float* __restrict__ xs, const float* __restrict__ ws,
    hip_bfloat16* __restrict__ out) {
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* sx=smem; unsigned char* sw=smem+OT_M*OT_KC;
  const int tid=threadIdx.x, lane=tid&63, wave=tid>>6;
  const int n_base=blockIdx.x*OT_NT;
  int acc[OT_NT] = {};
  const i32x4 rsrc=make_rsrc(packed_w, static_cast<uint32_t>(OT_PACKED_BYTES));

#pragma unroll
  for (int kc=0; kc<OT_KCS; ++kc) {
    // Exact retained X staging: 4*64 vectors, one per thread.
    const int mr=tid/64, kv=tid-(mr*64);
    const auto* src=reinterpret_cast<const i32x4*>(
        x+static_cast<int64_t>(mr)*OT_K+kc*OT_KC+kv*16);
    *reinterpret_cast<i32x4*>(sx+mr*OT_KC+kv*16)=*src;
#if defined(__HIP_DEVICE_COMPILE__)
    const uint32_t tile_base=static_cast<uint32_t>(
        (static_cast<int64_t>(blockIdx.x)*OT_KCS+kc)*OT_W_TILE);
    if (wave==0) {
      __builtin_amdgcn_raw_buffer_load_lds(rsrc,reinterpret_cast<int32_t*>(sw),16,
          tile_base+lane*16,0,0,0);
    } else if (wave==1) {
      __builtin_amdgcn_raw_buffer_load_lds(rsrc,reinterpret_cast<int32_t*>(sw+OT_KC),16,
          tile_base+OT_KC+lane*16,0,0,0);
    }
#endif
    __syncthreads();
    const i32x4 xv=*reinterpret_cast<const i32x4*>(sx+wave*OT_KC+lane*16);
#pragma unroll
    for (int nt=0; nt<OT_NT; ++nt) {
      const i32x4 wv=*reinterpret_cast<const i32x4*>(sw+nt*OT_KC+lane*16);
      acc[nt]=dot_all(xv,wv,acc[nt]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int nt=0; nt<OT_NT; ++nt) {
    int v=acc[nt];
#pragma unroll
    for (int sh=32; sh>0; sh>>=1) v+=__shfl_down(v,sh,64);
    const int n=n_base+nt;
    if (lane==0) out[static_cast<int64_t>(wave)*OT_N+n]=
        hip_bfloat16(static_cast<float>(v)*xs[wave]*ws[n]);
  }
}

void check_common(const torch::Tensor& x,const torch::Tensor& pw,
                  const torch::Tensor& xs,const torch::Tensor& ws,
                  int K,int N,int64_t packed_bytes) {
  TORCH_CHECK(x.is_cuda()&&pw.is_cuda()&&xs.is_cuda()&&ws.is_cuda(),"all tensors must be HCU");
  TORCH_CHECK(x.scalar_type()==torch::kInt8&&pw.scalar_type()==torch::kInt8,"x/pw must int8");
  TORCH_CHECK(x.is_contiguous()&&x.dim()==2&&x.size(0)==4&&x.size(1)==K,"x shape drift");
  TORCH_CHECK(pw.is_contiguous()&&pw.numel()==packed_bytes,"packed weight shape drift");
  TORCH_CHECK(xs.scalar_type()==torch::kFloat32&&ws.scalar_type()==torch::kFloat32,"scale dtype drift");
  TORCH_CHECK(xs.numel()>=4&&ws.numel()==N,"scale shape drift");
}

} // namespace

torch::Tensor gateup(torch::Tensor x,torch::Tensor pw,torch::Tensor xs,torch::Tensor ws) {
  check_common(x,pw,xs,ws,GU_K,GU_N,GU_PACKED_BYTES);
  auto out=torch::empty({GU_M,GU_N},x.options().dtype(torch::kBFloat16));
  auto st=c10::hip::getCurrentHIPStream();
  hipLaunchKernelGGL(gateup_directlds,dim3(GU_GRID),dim3(GU_BLOCK),(GU_M+GU_NT)*GU_KC,st,
      x.data_ptr<int8_t>(),pw.data_ptr<int8_t>(),xs.data_ptr<float>(),ws.data_ptr<float>(),
      reinterpret_cast<hip_bfloat16*>(out.data_ptr<at::BFloat16>()));
  return out;
}

torch::Tensor gdn(torch::Tensor x,torch::Tensor pw,torch::Tensor xs,torch::Tensor ws) {
  check_common(x,pw,xs,ws,GD_K,GD_N,GD_PACKED_BYTES);
  auto out=torch::empty({GD_M,GD_N},x.options().dtype(torch::kBFloat16));
  auto st=c10::hip::getCurrentHIPStream();
  hipLaunchKernelGGL(gdn_directlds,dim3(GD_GRID),dim3(GD_BLOCK),(GD_M+GD_NT)*GD_KC,st,
      x.data_ptr<int8_t>(),pw.data_ptr<int8_t>(),xs.data_ptr<float>(),ws.data_ptr<float>(),
      reinterpret_cast<hip_bfloat16*>(out.data_ptr<at::BFloat16>()));
  return out;
}

torch::Tensor outproj(torch::Tensor x,torch::Tensor pw,torch::Tensor xs,torch::Tensor ws) {
  check_common(x,pw,xs,ws,OT_K,OT_N,OT_PACKED_BYTES);
  auto out=torch::empty({OT_M,OT_N},x.options().dtype(torch::kBFloat16));
  auto st=c10::hip::getCurrentHIPStream();
  hipLaunchKernelGGL(out_directlds,dim3(OT_GRID),dim3(OT_BLOCK),(OT_M+OT_NT)*OT_KC,st,
      x.data_ptr<int8_t>(),pw.data_ptr<int8_t>(),xs.data_ptr<float>(),ws.data_ptr<float>(),
      reinterpret_cast<hip_bfloat16*>(out.data_ptr<at::BFloat16>()));
  return out;
}

void check_fused_gateup(const torch::Tensor& x,const torch::Tensor& pw,
                        const torch::Tensor& xs,const torch::Tensor& ws) {
  TORCH_CHECK(x.is_cuda()&&pw.is_cuda()&&xs.is_cuda()&&ws.is_cuda(),
              "fused_gateup: all tensors must be HCU");
  TORCH_CHECK(x.scalar_type()==torch::kInt8&&pw.scalar_type()==torch::kInt8,
              "fused_gateup: x/packed_w must be int8");
  TORCH_CHECK(x.dim()==2&&x.is_contiguous(),"fused_gateup: x must be contiguous 2-D");
  TORCH_CHECK(x.size(0)==GU_M&&x.size(1)==GU_K,
              "fused_gateup: requires x exactly [4,5120] (got [",x.size(0),",",x.size(1),"])");
  TORCH_CHECK(pw.dim()==1&&pw.is_contiguous()&&pw.numel()==GU_PACKED_BYTES,
              "fused_gateup: requires contiguous packed_w of exactly 17408*5120 int8");
  TORCH_CHECK(xs.scalar_type()==torch::kFloat32&&ws.scalar_type()==torch::kFloat32,
              "fused_gateup: xs/ws must be fp32");
  TORCH_CHECK(xs.is_contiguous()&&ws.is_contiguous(),"fused_gateup: xs/ws must be contiguous");
  TORCH_CHECK(xs.numel()==GU_M,"fused_gateup: requires exactly 4 x-scales");
  TORCH_CHECK(ws.numel()==GU_N,"fused_gateup: requires exactly 17408 permuted w-scales");
  TORCH_CHECK(pw.device()==x.device()&&xs.device()==x.device()&&ws.device()==x.device(),
              "fused_gateup: x/packed_w/xs/ws must share one device");
}

// Independent entry: DirectLDS gate/up GEMM with fused paired-N SiLU epilogue.
// Input packed_w/ws MUST come from pack_gateup_fused.pack_gateup_fused.
torch::Tensor fused_gateup(torch::Tensor x,torch::Tensor pw,torch::Tensor xs,torch::Tensor ws) {
  check_fused_gateup(x,pw,xs,ws);
  const c10::DeviceGuard guard(x.device());
  auto out=torch::empty({GU_M,GUF_HALF},x.options().dtype(torch::kBFloat16));
  auto st=c10::hip::getCurrentHIPStream();
  hipLaunchKernelGGL(gateup_fused_directlds,dim3(GU_GRID),dim3(GU_BLOCK),(GU_M+GU_NT)*GU_KC,st,
      x.data_ptr<int8_t>(),pw.data_ptr<int8_t>(),xs.data_ptr<float>(),ws.data_ptr<float>(),
      reinterpret_cast<hip_bfloat16*>(out.data_ptr<at::BFloat16>()));
  return out;
}

std::string identity_fused_gateup(){
  return "H-SMALLM-W-DIRECTLDS-XW-OVERLAP-V1:FAMILY:GATEUP+GDN+OUT:FUSED_GATEUP_V4";
}

std::string identity(){return "H-SMALLM-W-DIRECTLDS-XW-OVERLAP-V1:FAMILY:GATEUP+GDN+OUT";}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  m.def("gateup",&gateup); m.def("gdn",&gdn); m.def("outproj",&outproj); m.def("identity",&identity);
  m.def("fused_gateup",&fused_gateup,"DirectLDS K5120 small-M INT8 gate/up GEMM with fused paired-N SiLU epilogue, M4/K5120/N17408/NTILE4/RW4 only; requires pack_gateup_fused packing",py::arg("x"),py::arg("packed_w"),py::arg("xs"),py::arg("ws"));
  m.def("identity_fused_gateup",&identity_fused_gateup);
}

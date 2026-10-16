#include <torch/extension.h>

#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

// Experimental capacity extension of v29's fixed-width-8 assignment kernel.
// Preserve the exact row/column mapping and int64-offset ABI, but allow up to
// 16 requests by launching enough 64-thread waves to cover batch_size*8 slots.
__global__ __launch_bounds__(64) void assign_fixed8_i64_kernel(
    const int64_t* __restrict__ req_pool_indices,
    int32_t* __restrict__ req_to_token,
    const int64_t* __restrict__ start_offsets,
    const int64_t* __restrict__ out_cache_loc,
    int64_t pool_len,
    int64_t batch_size) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
      static_cast<int64_t>(threadIdx.x);
  const int64_t total = batch_size * 8;
  if (linear >= total) {
    return;
  }

  const int64_t row = linear >> 3;
  const int64_t column = linear & 7;
  const int64_t request = req_pool_indices[row];
  const int64_t destination =
      request * pool_len + start_offsets[row] + column;
  req_to_token[destination] = static_cast<int32_t>(out_cache_loc[linear]);
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a HIP tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void assign_fixed8_i64(
    torch::Tensor req_pool_indices,
    torch::Tensor req_to_token,
    torch::Tensor start_offsets,
    torch::Tensor out_cache_loc,
    int64_t pool_len,
    int64_t batch_size) {
  check_cuda_contiguous(req_pool_indices, "req_pool_indices");
  check_cuda_contiguous(req_to_token, "req_to_token");
  check_cuda_contiguous(start_offsets, "start_offsets");
  check_cuda_contiguous(out_cache_loc, "out_cache_loc");

  TORCH_CHECK(
      req_pool_indices.scalar_type() == torch::kInt64,
      "req_pool_indices must be int64");
  TORCH_CHECK(
      req_to_token.scalar_type() == torch::kInt32,
      "req_to_token must be int32");
  TORCH_CHECK(
      start_offsets.scalar_type() == torch::kInt64,
      "start_offsets must be int64");
  TORCH_CHECK(
      out_cache_loc.scalar_type() == torch::kInt64,
      "out_cache_loc must be int64");
  TORCH_CHECK(req_to_token.dim() == 2, "req_to_token must be 2-D");
  TORCH_CHECK(pool_len == req_to_token.size(1), "pool_len mismatch");
  TORCH_CHECK(batch_size > 0 && batch_size <= 16, "batch_size must be 1..16");
  TORCH_CHECK(
      req_pool_indices.numel() >= batch_size,
      "req_pool_indices shorter than batch_size");
  TORCH_CHECK(
      start_offsets.numel() >= batch_size,
      "start_offsets shorter than batch_size");
  TORCH_CHECK(
      out_cache_loc.numel() == batch_size * 8,
      "out_cache_loc must contain exactly batch_size*8 elements");
  TORCH_CHECK(
      req_pool_indices.device() == req_to_token.device() &&
          start_offsets.device() == req_to_token.device() &&
          out_cache_loc.device() == req_to_token.device(),
      "all tensors must be on the same HIP device");

  const int64_t total = batch_size * 8;
  const int64_t blocks = (total + 63) / 64;
  hipStream_t stream = c10::hip::getCurrentHIPStream();
  hipLaunchKernelGGL(
      assign_fixed8_i64_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(64),
      0,
      stream,
      req_pool_indices.data_ptr<int64_t>(),
      req_to_token.data_ptr<int32_t>(),
      start_offsets.data_ptr<int64_t>(),
      out_cache_loc.data_ptr<int64_t>(),
      pool_len,
      batch_size);
  const hipError_t error = hipGetLastError();
  TORCH_CHECK(
      error == hipSuccess,
      "assign_fixed8_i64 c16exp kernel launch failed: ",
      hipGetErrorString(error));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "assign_fixed8_i64",
      &assign_fixed8_i64,
      "K100AI DFlash fixed-width-8 req-to-token assignment with int64 offsets, experimental batch<=16");
}

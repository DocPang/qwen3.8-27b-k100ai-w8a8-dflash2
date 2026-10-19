"""Fused GDN in-proj split for the CONTIGUOUS-BLOCK checkpoint layout (this model).

Why this exists.  qwen3_5.py:462 gates the vendor fused split on
    num_v_heads // num_k_heads in [1, 2, 4]
and our ratio is 48/16 = 3, so every GDN layer takes the slow else-branch:
    query,key,value,z,b,a = fix_query_key_value_ordering(...)   # views, cheap
    b = b.contiguous(); a = a.contiguous()                      # 2 launches/layer
    mixed_qkv = torch.cat((query,key,value), -1)                # 1 launch/layer
i.e. 3 launches per layer (~144 per step; ledger: direct_copy 228 calls/1.425 ms,
cat_copy 48.5 calls/0.345 ms).

The vendor kernel cannot be reused at ratio 3: its own docstring says it reads the
per-head-group INTERLEAVED layout [g0_q,g0_k,g0_v,g0_z, g1_...], while this
checkpoint stores contiguous blocks [q|k|v|z] -- that is exactly why the [1,2,4]
gate exists (a correctness guard, not an oversight).

This kernel performs the same work in ONE launch per layer and is pure data
movement, so it must be bit-exact against the reference else-branch.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_split_fused_kernel(
    qkvz_ptr, ba_ptr, mq_ptr, z_ptr, b_ptr, a_ptr,
    stride_qkvz, stride_ba, stride_mq, stride_z,
    QKV_DIM: tl.constexpr,   # 2*K_TP + V_TP = 5120
    Z_DIM: tl.constexpr,     # V_TP = 3072
    BA_HALF: tl.constexpr,   # NV_TP = 24
    BLK: tl.constexpr,       # >= QKV_DIM + Z_DIM, power of two
    BLK_BA: tl.constexpr,    # >= 2*BA_HALF, power of two
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLK)
    qkvz_row = qkvz_ptr + row * stride_qkvz
    x = tl.load(qkvz_row + offs, mask=offs < (QKV_DIM + Z_DIM))
    tl.store(mq_ptr + row * stride_mq + offs, x, mask=offs < QKV_DIM)
    tl.store(z_ptr + row * stride_z + (offs - QKV_DIM), x,
             mask=(offs >= QKV_DIM) & (offs < QKV_DIM + Z_DIM))
    boffs = tl.arange(0, BLK_BA)
    bx = tl.load(ba_ptr + row * stride_ba + boffs, mask=boffs < 2 * BA_HALF)
    tl.store(b_ptr + row * BA_HALF + boffs, bx, mask=boffs < BA_HALF)
    tl.store(a_ptr + row * BA_HALF + (boffs - BA_HALF), bx,
             mask=(boffs >= BA_HALF) & (boffs < 2 * BA_HALF))


def gdn_split_fused(qkvz, ba, k_tp, v_tp, nv_tp):
    S = qkvz.shape[0]
    qkv_dim = 2 * k_tp + v_tp
    blk = 1 << (qkv_dim + v_tp - 1).bit_length()
    blk_ba = 1 << (2 * nv_tp - 1).bit_length()
    dev = qkvz.device
    mixed_qkv = torch.empty((S, qkv_dim), dtype=qkvz.dtype, device=dev)
    z = torch.empty((S, v_tp), dtype=qkvz.dtype, device=dev)
    b = torch.empty((S, nv_tp), dtype=ba.dtype, device=dev)
    a = torch.empty((S, nv_tp), dtype=ba.dtype, device=dev)
    _gdn_split_fused_kernel[(S,)](
        qkvz, ba, mixed_qkv, z, b, a,
        qkvz.stride(0), ba.stride(0), mixed_qkv.stride(0), z.stride(0),
        qkv_dim, v_tp, nv_tp, blk, blk_ba,
        num_warps=8,
    )
    return mixed_qkv, z, b, a


def reference(qkvz, ba, k_tp, v_tp, nv_tp, head_v):
    query, key, value, z = qkvz.split([k_tp, k_tp, v_tp, v_tp], dim=-1)
    b, a = ba.split([nv_tp, nv_tp], dim=-1)
    value = value.reshape(value.size(0), -1, head_v)
    z = z.reshape(z.size(0), -1, head_v)
    b = b.contiguous()
    a = a.contiguous()
    query, key, value = (x.reshape(x.shape[0], -1) for x in (query, key, value))
    mixed_qkv = torch.cat((query, key, value), dim=-1)
    return mixed_qkv, z.reshape(z.size(0), -1), b, a


if __name__ == "__main__":
    torch.manual_seed(0)
    k_tp, v_tp, nv_tp, head_v = 1024, 3072, 24, 128
    allok = True
    for S in (1, 8, 16, 32):
        qkvz = torch.randn(S, 2 * k_tp + 2 * v_tp, dtype=torch.bfloat16, device="cuda")
        ba = torch.randn(S, 2 * nv_tp, dtype=torch.bfloat16, device="cuda")
        rm, rz, rb, ra = reference(qkvz, ba, k_tp, v_tp, nv_tp, head_v)
        fm, fz, fb, fa = gdn_split_fused(qkvz, ba, k_tp, v_tp, nv_tp)
        ok = all(bool(torch.equal(x, y)) for x, y in ((rm, fm), (rz, fz), (rb, fb), (ra, fa)))
        allok = allok and ok
        print("S=%-3d bit_exact=%-5s shapes mq=%s z=%s b=%s a=%s" % (
            S, ok, tuple(fm.shape), tuple(fz.shape), tuple(fb.shape), tuple(fa.shape)))
    print("ALL_BIT_EXACT", allok)

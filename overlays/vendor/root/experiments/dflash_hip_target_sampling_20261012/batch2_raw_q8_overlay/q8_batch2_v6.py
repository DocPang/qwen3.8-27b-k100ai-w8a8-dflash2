"""Modern DFlash2 verify q8 exact-shape functional backport on 2026-10 combined_v9.

Run combined_v9 unchanged first. Then replace only the batch1 causal
DFlash verifier shape q=8, QH12/KVH2/D256/BF16/page64/fa2 with one raw
flash_attn_2_cuda.paged_attention call.  Isolated 64K/128K/257.9K gates show
bitwise equality versus the existing 2xq4 verifier and ~2.0-2.26x attention
speedup.  Every non-audited shape falls back to the combined_v9 parent.
"""
from __future__ import annotations

import os
import runpy
import torch
import flash_attn_2_cuda as _fa2

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_ROOT}/experiments/tp2_dflash2_retest_20261008/pagefix_q8_overlay/pagefix_parent.py"
runpy.run_path(_PARENT, run_name="__q38_modern_tp2_dflash2_q8_v3_parent__")

from sglang.srt.layers.attention import flashattention_backend as _fab
from sglang.srt.layers.attention import flashattention_interface as _fai

_parent = _fab.vllm_flash_attn_varlen_func
_seen = False


def _raw_q8_else_parent(
    q,
    k,
    v,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    fa_version,
    q_descale,
    k_descale,
    v_descale,
):
    global _seen
    qlen = int(max_seqlen_q)
    batch = int(cu_seqlens_q.numel() - 1)
    eligible = (
        qlen == 8
        and tuple(q.shape) == (batch * 8, 12, 256)
        and batch in (1, 2)
        and int(seqused_k.numel()) == batch
        and bool(causal)
        and window_size == (-1, -1)
        and block_table is not None
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and int(k.ndim) == 4
        and int(v.ndim) == 4
        and int(k.shape[1]) == 2
        and int(v.shape[1]) == 2
        and int(k.shape[2]) == 64
        and int(k.shape[3]) == 256
        and int(v.shape[2]) == 256
        and int(v.shape[3]) == 64
        and int(fa_version) == 2
    )
    if not eligible:
        return _parent(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            block_table=block_table,
            fa_version=fa_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    if not _seen:
        _seen = True
        print(
            "[K100 DFlash2 TP2 raw-q8 v6] ACTIVE batch1+2 q=8 verifier -> single raw paged_attention "
            "QH12/KVH2/D256/BF16/page64",
            flush=True,
        )

    out = torch.empty_like(q)
    _fa2.paged_attention(
        out,
        q.reshape(batch, 8, 12, 256),
        k,
        v,
        softmax_scale,
        block_table,
        seqused_k,
        None,
        "auto",
        q_descale,
        k_descale,
        v_descale,
        max_seqlen_k,
        None,
        0,  # modern Hygon FA2 paged_attention ABI: extra int, frozen legacy TP4 raw-q8 precedent
    )
    return out


if os.environ.get("Q38_DISABLE_RAW_Q8") == "1":
    print("[K100 DFlash2 TP2 raw-q8 v6] DISABLED by Q38_DISABLE_RAW_Q8=1 -> parent varlen path", flush=True)
else:
    _fai.vllm_flash_attn_varlen_func = _raw_q8_else_parent
    _fab.vllm_flash_attn_varlen_func = _raw_q8_else_parent

print(
    "[K100 DFlash2 TP2 longkv BM128 v5] installed: v4 prefill/qtail preserved; "
    "exact batch1+2 causal q8 verifier uses raw paged_attention; all else parent fallback",
    flush=True,
)

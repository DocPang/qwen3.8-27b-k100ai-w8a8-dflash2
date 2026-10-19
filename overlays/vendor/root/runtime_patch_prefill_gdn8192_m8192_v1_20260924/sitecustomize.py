"""Exact large-M8192 GDN QKVZ Triton-config overlay for retained TP2.

Parent:
  runtime_patch_spec_gain_m4_gateup_down8704_gdn8192_laout3072_v1_20260923

Runtime delta:
  Only when the existing compressed-tensors W8A8 path calls triton_scaled_mm with
  best_config=None and exact tensors
      A: [8192, 5120] int8
      B: [5120, 8192] int8
      output: bf16
      bias: None
  inject the frozen vendor-derived config:
      BM256 BN128 BK64 GROUP4 SPLIT_K1 stages1 warps4

All other calls fall through to the retained parent unchanged.
"""

from __future__ import annotations

import hashlib
import os
import runpy

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = os.getenv(
    "Q38_PREFILL_GDN8192_PARENT",
    f"{_ROOT}/runtime_patch_spec_gain_m4_gateup_down8704_gdn8192_laout3072_v1_20260923/sitecustomize.py",
)
_PARENT_SHA256 = "054263dfbb44d4d9e96f10cb916bdfed8462ba6785cfb1a421b9b811e2c0abb5"

_CFG = {
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 4,
    "SPLIT_K": 1,
    "num_stages": 1,
    "num_warps": 4,
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


_actual = _sha256(_PARENT)
if _actual != _PARENT_SHA256:
    raise RuntimeError(f"prefill GDN8192 parent drift: {_actual}")

runpy.run_path(_PARENT, run_name="__q38_prefill_gdn8192_parent__")

try:
    _cmd = (
        open("/proc/self/cmdline", "rb")
        .read()
        .replace(b"\0", b" ")
        .decode("utf-8", "replace")
    )
except Exception:
    _cmd = ""

_HELPERS = (
    "tvm_ffi/utils/_build_optional_torch_c_dlpack.py",
    "_build_optional_torch_c_dlpack.py",
    "multiprocessing.resource_tracker",
    "/usr/local/bin/ninja",
    " ninja --version",
)

if not any(marker in _cmd for marker in _HELPERS):
    import torch
    from sglang.srt.layers.quantization.compressed_tensors import quant_ops as _ops

    _parent_scaled_mm = _ops.triton_scaled_mm
    _hits = 0

    def _prefill_gdn8192_scaled_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias=None,
        best_config=None,
    ) -> torch.Tensor:
        global _hits

        exact = (
            best_config is None
            and isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and isinstance(scale_a, torch.Tensor)
            and isinstance(scale_b, torch.Tensor)
            and a.ndim == 2
            and b.ndim == 2
            and tuple(a.shape) == (8192, 5120)
            and tuple(b.shape) == (5120, 8192)
            and a.dtype is torch.int8
            and b.dtype is torch.int8
            and out_dtype is torch.bfloat16
            and bias is None
            and a.is_contiguous()
            and scale_a.dtype is torch.float32
            and scale_b.dtype is torch.float32
            and scale_a.numel() >= 8192
            and scale_b.numel() == 8192
        )

        if exact:
            _hits += 1
            if _hits <= 24:
                print(
                    f"[Q38 PREFILL GDN8192 M8192 V1] hit #{_hits}: "
                    "M8192 K5120->N8192 BM256 BN128 BK64 G4 S1 W4",
                    flush=True,
                )
            return _parent_scaled_mm(
                a, b, scale_a, scale_b, out_dtype, bias, _CFG
            )

        return _parent_scaled_mm(
            a, b, scale_a, scale_b, out_dtype, bias, best_config
        )

    _ops.triton_scaled_mm = _prefill_gdn8192_scaled_mm
    print(
        "[Q38 PREFILL GDN8192 M8192 V1] installed; exact missing-config "
        "M8192 K5120->N8192 only; all unmatched paths preserve retained parent",
        flush=True,
    )

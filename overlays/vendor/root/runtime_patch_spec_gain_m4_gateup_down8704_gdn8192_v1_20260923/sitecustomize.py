"""Exact-M4 GDN QKVZ overlay on top of gate_up + down8704 candidate.

Adds only the bitwise-validated speculative-target shape:

    M=4, K=5120, N=8192, int8 x int8 -> bf16, no bias.

Current-runtime paired re-screen on 2026-09-23:
  graph official median ~123.37 us
  native median ~88.98 us
  median paired gain ~27.86%, 7/7 positive
  eager and CUDA-graph outputs bitwise exact.

All unmatched calls fall through to the already validated gate_up + down8704
parent unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import runpy


_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = os.getenv(
    "Q38_SPECGAIN_GDN8192_PARENT",
    f"{_ROOT}/runtime_patch_spec_gain_m4_gateup_v3b_down8704_v1_20260923/sitecustomize.py",
)
_PARENT_SHA256 = "0031051f1c8a04516880d65c1a4436d3f658b3e8a0132e98659204c4e9abaaf7"
_GDN_SO = os.getenv(
    "Q38_SPECGAIN_GDN8192_SO",
    f"{_ROOT}/native_ext/k100_int8_smallm_k5120_n8192_v1_hybrid.so",
)
_GDN_SO_SHA256 = "c158c6b3402d1a2d43b154d6960734ce4f17f448b5288ba361d531e8a21621f2"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if _sha256(_PARENT) != _PARENT_SHA256:
    raise RuntimeError(f"spec-gain GDN8192 parent drift: {_sha256(_PARENT)}")
if _sha256(_GDN_SO) != _GDN_SO_SHA256:
    raise RuntimeError(f"spec-gain GDN8192 SO drift: {_sha256(_GDN_SO)}")

runpy.run_path(_PARENT, run_name="__q38_specgain_gdn8192_parent__")

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

    _spec = importlib.util.spec_from_file_location(
        "k100_int8_smallm_k5120_n8192_v1_hybrid", _GDN_SO
    )
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"cannot load GDN8192 extension: {_GDN_SO}")
    _gdn = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_gdn)
    if not callable(getattr(_gdn, "matmul_half", None)):
        raise RuntimeError(f"GDN8192 extension has no matmul_half(): {_GDN_SO}")

    _parent_scaled_mm = _ops.triton_scaled_mm
    _hits = 0

    def _m4_gdn8192_scaled_mm(
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
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and isinstance(scale_a, torch.Tensor)
            and isinstance(scale_b, torch.Tensor)
            and a.ndim == 2
            and b.ndim == 2
            and tuple(a.shape) == (4, 5120)
            and tuple(b.shape) == (5120, 8192)
            and a.dtype is torch.int8
            and b.dtype is torch.int8
            and out_dtype is torch.bfloat16
            and bias is None
            and a.is_contiguous()
            and b.t().is_contiguous()
            and scale_a.dtype is torch.float32
            and scale_b.dtype is torch.float32
            and scale_a.numel() >= 4
            and scale_b.numel() == 8192
        )

        if exact:
            out = _gdn.matmul_half(
                a,
                b.t(),
                scale_a.reshape(-1),
                scale_b.reshape(-1),
                2,
            )
            _hits += 1
            if _hits <= 24:
                print(
                    f"[Q38 SPECGAIN M4 GDN8192 V1] hit #{_hits}: "
                    "M4 K5120->N8192 halfwave ntile2",
                    flush=True,
                )
            return out

        return _parent_scaled_mm(
            a, b, scale_a, scale_b, out_dtype, bias, best_config
        )

    _ops.triton_scaled_mm = _m4_gdn8192_scaled_mm
    print(
        "[Q38 SPECGAIN M4 GDN8192 V1] installed; exact int8 M4 "
        "K5120->N8192 only; all unmatched paths preserve down8704 parent",
        flush=True,
    )

"""Exact M4 linear-attention out_proj native tiled overlay.

Parent is the retained gate/up + down8704 + GDN8192 speculative candidate.
This child replaces only the exact W8A8 target shape:

    M=4, K=3072, N=5120, int8 x int8 -> bf16, no bias.

The native kernel is the frozen true-M4 tiled winner validated independently
in eager and CUDA Graph modes with bitwise-exact output.  All unmatched calls
fall through to the parent implementation unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import runpy


_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = os.getenv(
    "Q38_SPECGAIN_LAOUT3072_PARENT",
    f"{_ROOT}/runtime_patch_spec_gain_m4_gateup_down8704_gdn8192_v1_20260923/sitecustomize.py",
)
_PARENT_SHA256 = "e7ca2a7ddbd05bf638ec919a2379644073757a6f80a200400076ee3175084dca"
_LAOUT_SO = os.getenv(
    "Q38_SPECGAIN_LAOUT3072_SO",
    f"{_ROOT}/native_ext/k100_int8_smallm_k3072_n5120_m4_v1_tiled.so",
)
_LAOUT_SO_SHA256 = "ece271b41dd37c3a44cb608b57f5b8aeba182cbe58ec0035b72e98b62303d7b0"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if _sha256(_PARENT) != _PARENT_SHA256:
    raise RuntimeError(f"spec-gain laout3072 parent drift: {_sha256(_PARENT)}")
if _sha256(_LAOUT_SO) != _LAOUT_SO_SHA256:
    raise RuntimeError(f"spec-gain laout3072 SO drift: {_sha256(_LAOUT_SO)}")

runpy.run_path(_PARENT, run_name="__q38_specgain_laout3072_parent__")

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
        "k100_int8_smallm_k3072_n5120_m4_v1_tiled", _LAOUT_SO
    )
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"cannot load LA out_proj extension: {_LAOUT_SO}")
    _laout = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_laout)
    if not callable(getattr(_laout, "matmul", None)):
        raise RuntimeError(f"LA out_proj extension has no matmul(): {_LAOUT_SO}")

    _parent_scaled_mm = _ops.triton_scaled_mm
    _hits = 0

    def _m4_laout3072_scaled_mm(
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
            and tuple(a.shape) == (4, 3072)
            and tuple(b.shape) == (3072, 5120)
            and a.dtype is torch.int8
            and b.dtype is torch.int8
            and out_dtype is torch.bfloat16
            and bias is None
            and a.is_contiguous()
            and b.t().is_contiguous()
            and scale_a.dtype is torch.float32
            and scale_b.dtype is torch.float32
            and scale_a.numel() >= 4
            and scale_b.numel() == 5120
        )

        if exact:
            out = _laout.matmul(
                a,
                b.t(),
                scale_a.reshape(-1),
                scale_b.reshape(-1),
                1024,
                2,
                4,
            )
            _hits += 1
            if _hits <= 24:
                print(
                    f"[Q38 SPECGAIN M4 LAOUT3072 V1] hit #{_hits}: "
                    "M4 K3072->N5120 true-M4 tiled kchunk1024 ntile2 rowwaves4",
                    flush=True,
                )
            return out

        return _parent_scaled_mm(
            a, b, scale_a, scale_b, out_dtype, bias, best_config
        )

    _ops.triton_scaled_mm = _m4_laout3072_scaled_mm
    print(
        "[Q38 SPECGAIN M4 LAOUT3072 V1] installed; exact int8 M4 "
        "K3072->N5120 only; all unmatched paths preserve GDN8192 parent",
        flush=True,
    )

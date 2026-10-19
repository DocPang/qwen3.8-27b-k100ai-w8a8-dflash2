"""Exact-M4 W8A8 down-projection overlay on top of gate_up v3b.

The parent overlay already replaces only the exact speculative-target gate/up
shape M=4,K=5120,N=17408.  This child adds one more bitwise-validated shape:

    M=4, K=8704, N=5120, int8 x int8 -> bf16, no bias.

The native K8704 v1 tiled primitive was paired against the current parent on
2026-09-23.  ntile=2,row_waves=4 was bitwise exact (max_abs=0) and measured
~109.68 us median versus ~261.36 us for the current parent path.  Every other
shape falls through to the parent unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import runpy


_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = os.getenv(
    "Q38_SPECGAIN_DOWN8704_PARENT",
    f"{_ROOT}/runtime_patch_spec_gain_m4_gateup_v3b_global_v3_20260923/sitecustomize.py",
)
_PARENT_SHA256 = "89a8861be7fec2856a03558edfa14717da7244bad9b9ed29ba3859ff24c845d1"
_SMALLM_SO = os.getenv(
    "Q38_SPECGAIN_SMALLM_K8704_V1_SO",
    f"{_ROOT}/native_ext/k100_int8_smallm_k8704_v1_tiled.so",
)
_SMALLM_SHA256 = "3b417c3303d6a4b99d8b26f47b369d237c2a7be7cc9a1d626115392041f27d4c"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if _sha256(_PARENT) != _PARENT_SHA256:
    raise RuntimeError(f"spec-gain down8704 parent drift: {_sha256(_PARENT)}")
if _sha256(_SMALLM_SO) != _SMALLM_SHA256:
    raise RuntimeError(f"spec-gain K8704 SO drift: {_sha256(_SMALLM_SO)}")

runpy.run_path(_PARENT, run_name="__q38_specgain_down8704_parent__")

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
        "k100_int8_smallm_k8704_v1_tiled", _SMALLM_SO
    )
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"cannot load K8704 small-M extension: {_SMALLM_SO}")
    _smallm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_smallm)
    if not callable(getattr(_smallm, "matmul", None)):
        raise RuntimeError(f"K8704 small-M extension has no matmul(): {_SMALLM_SO}")

    _parent_scaled_mm = _ops.triton_scaled_mm
    _hits = 0

    def _m4_down8704_scaled_mm(
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
            and tuple(a.shape) == (4, 8704)
            and tuple(b.shape) == (8704, 5120)
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
            out = _smallm.matmul(
                a,
                b.t(),
                scale_a.reshape(-1),
                scale_b.reshape(-1),
                2,
                4,
            )
            _hits += 1
            if _hits <= 24:
                print(
                    f"[Q38 SPECGAIN M4 DOWN8704 V1] hit #{_hits}: "
                    "M4 K8704->N5120 ntile2 rowwaves4",
                    flush=True,
                )
            return out

        return _parent_scaled_mm(
            a, b, scale_a, scale_b, out_dtype, bias, best_config
        )

    _ops.triton_scaled_mm = _m4_down8704_scaled_mm
    print(
        "[Q38 SPECGAIN M4 DOWN8704 V1] installed; exact int8 M4 "
        "K8704->N5120 only; all unmatched paths preserve gate_up-v3b parent",
        flush=True,
    )

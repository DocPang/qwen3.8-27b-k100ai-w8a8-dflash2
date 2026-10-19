"""Global exact-M4 W8A8 gate_up overlay for speculative target graph capture.

The previous verify-context experiment proved that EAGLE target kernels are
selected while the target CUDA graph is captured, before the Python
``EAGLEWorkerV2.verify`` scope is active.  This overlay therefore dispatches
globally, but only for the exact bitwise-validated W8A8 shape:

    M=4, K=5120, N=17408, int8 x int8 -> bf16, no bias.

The native v3b primitive was re-screened on the current 2026-09-23 runtime and
is bitwise exact with 156.86 us median versus 294.22 us for the parent path.
All unmatched calls fall through to the current speculative parent unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import runpy


_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = os.getenv(
    "Q38_SPECGAIN_GATEUP_V3_PARENT",
    f"{_ROOT}/runtime_patch_spec_gain_mtp_gateup10009_fullhead11055_m4_v1_20260922/sitecustomize.py",
)
_PARENT_SHA256 = "c4fb75694686bc060875c154c4e42b31b94c8bf6885f56fe5b022384cadf63b9"
_SMALLM_SO = os.getenv(
    "Q38_SPECGAIN_SMALLM_K5120_V3B_SO",
    f"{_ROOT}/native_ext/k100_int8_smallm_k5120_v3b_tiled.so",
)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if _sha256(_PARENT) != _PARENT_SHA256:
    raise RuntimeError(f"spec-gain parent drift: {_sha256(_PARENT)}")

runpy.run_path(_PARENT, run_name="__q38_specgain_gateup_v3_parent__")

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
        "k100_int8_smallm_k5120_v3b_tiled", _SMALLM_SO
    )
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"cannot load small-M v3b extension: {_SMALLM_SO}")
    _smallm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_smallm)
    if not callable(getattr(_smallm, "matmul", None)):
        raise RuntimeError(f"small-M extension has no matmul(): {_SMALLM_SO}")

    _parent_scaled_mm = _ops.triton_scaled_mm
    _hits = 0

    def _m4_gateup_v3b_scaled_mm(
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
            and tuple(b.shape) == (5120, 17408)
            and a.dtype is torch.int8
            and b.dtype is torch.int8
            and out_dtype is torch.bfloat16
            and bias is None
            and a.is_contiguous()
            and b.t().is_contiguous()
            and scale_a.dtype is torch.float32
            and scale_b.dtype is torch.float32
            and scale_a.numel() >= 4
            and scale_b.numel() == 17408
        )

        if exact:
            out = _smallm.matmul(
                a,
                b.t(),
                scale_a.reshape(-1),
                scale_b.reshape(-1),
                4,
                4,
            )
            _hits += 1
            if _hits <= 24:
                print(
                    f"[Q38 SPECGAIN M4 GATEUP V3B GLOBAL] hit #{_hits}: "
                    "M4 K5120->N17408 ntile4 rw4",
                    flush=True,
                )
            return out

        return _parent_scaled_mm(
            a, b, scale_a, scale_b, out_dtype, bias, best_config
        )

    _ops.triton_scaled_mm = _m4_gateup_v3b_scaled_mm
    print(
        "[Q38 SPECGAIN M4 GATEUP V3B GLOBAL] installed; exact int8 M4 "
        "K5120->N17408 only; all unmatched paths preserve parent",
        flush=True,
    )


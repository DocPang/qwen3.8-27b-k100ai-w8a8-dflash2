"""Official TP2 no-spec exact-M1 native GEMV adapter, direct-fallback v3.

Scope is intentionally narrow:
  * preserve the frozen C0 compressed-tensors correctness compatibility;
  * intercept only W8A8 BF16-output M=1 TP2 rank-local projection shapes;
  * use the already validated K100AI LDS-x donor kernels;
  * return M>1/prefill and unmatched calls directly to the exact LightOP function
    used by the frozen official wrapper, removing one extra Python frame;
  * leave unmatched M1 shapes, bias, other dtypes, and scheduler behavior on
    the official SourceFind/Hygon path.

This is a no-spec v3 experiment. It does not import the historical v1.3.3
runtime stack or speculative TARGET_VERIFY adapters.
"""
from __future__ import annotations

import importlib.util
import os
import runpy


_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_C0 = f"{_ROOT}/runtime_patch_newbase_c0_helper_correctness_20260913/sitecustomize.py"

# Keep the exact correctness shim used by the official I arm.
runpy.run_path(_C0, run_name="__q38_official_nonspec_m1_native_c0__")

try:
    _cmd = open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
except Exception:
    _cmd = ""

_HELPERS = (
    "tvm_ffi/utils/_build_optional_torch_c_dlpack.py",
    "_build_optional_torch_c_dlpack.py",
    "multiprocessing.resource_tracker",
    "/usr/local/bin/ninja",
    " ninja --version",
)

if not any(x in _cmd for x in _HELPERS):
    import torch
    from sglang.srt.layers.quantization.compressed_tensors import quant_ops as _ops

    _K5120_SO = os.getenv(
        "SGLANG_Q38_OFFICIAL_M1_K5120_SO",
        f"{_ROOT}/native_ext/k100_int8_gemv_k5120_ldsx_v1_sglang.so",
    )
    _ROW_SO = os.getenv(
        "SGLANG_Q38_OFFICIAL_M1_ROW_SO",
        f"{_ROOT}/native_ext/k100_int8_gemv_tp2_row_ldsx_v1_sglang.so",
    )

    def _load(name: str, path: str):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load native extension {name}: {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not callable(getattr(mod, "gemv", None)):
            raise RuntimeError(f"native extension {name} does not expose gemv(): {path}")
        return mod

    _k5120 = _load("k100_int8_gemv_k5120_ldsx_v1_sglang", _K5120_SO)
    _row = _load("k100_int8_gemv_tp2_row_ldsx_v1_sglang", _ROW_SO)
    _official_wrapper = _ops.triton_scaled_mm
    _lightop_scaled_mm = _ops.quant_ops.triton_scaled_mm
    if not callable(_lightop_scaled_mm):
        raise RuntimeError("official LightOP triton_scaled_mm is not callable")

    # All five shapes are TP2 rank-local W8A8 projections observed in the
    # matched no-spec traces. Both donor extensions require [N,K] contiguous
    # weights; the official strategy-1 path stores the logical operand as
    # [K,N], whose transpose is contiguous.
    _K5120 = {
        (5120, 17408): (2, "gate_up_tp2"),
        (5120, 8192): (2, "gdn_qkvz_tp2"),
        (5120, 7168): (2, "full_qkv_tp2"),
    }
    _ROW = {
        (8704, 5120): (2, "down_tp2"),
        (3072, 5120): (2, "out_tp2"),
    }
    _seen: set[tuple[int, int]] = set()

    def _official_tp2_m1_native_scaled_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias=None,
        best_config=None,
    ) -> torch.Tensor:
        # v2 still paid an extra Python frame on every M>1 call:
        # adapter -> official wrapper -> LightOP.  The frozen official wrapper
        # is a one-line passthrough, so v3 returns directly to that exact
        # LightOP callable for the dominant fallback path.
        if not isinstance(a, torch.Tensor) or a.ndim != 2 or int(a.shape[0]) != 1:
            return _lightop_scaled_mm(
                a, b, scale_a, scale_b, out_dtype, bias, best_config
            )
        if (
            bias is None
            and out_dtype is torch.bfloat16
            and isinstance(b, torch.Tensor)
            and isinstance(scale_a, torch.Tensor)
            and isinstance(scale_b, torch.Tensor)
            and a.dtype is torch.int8
            and b.dtype is torch.int8
            and b.ndim == 2
            and a.is_contiguous()
            and b.t().is_contiguous()
            and scale_a.dtype is torch.float32
            and scale_b.dtype is torch.float32
            and scale_a.numel() >= 1
            and int(b.shape[0]) == int(a.shape[1])
        ):
            k = int(a.shape[1])
            n = int(b.shape[1])
            cfg = _K5120.get((k, n))
            mod = _k5120
            if cfg is None:
                cfg = _ROW.get((k, n))
                mod = _row
            if cfg is not None and scale_b.numel() == n:
                waves, label = cfg
                key = (k, n)
                if key not in _seen:
                    _seen.add(key)
                    print(
                        f"[Q38 OFFICIAL TP2 NONSPEC M1 NATIVE] ACTIVE {label} "
                        f"M1 K{k}->N{n}; waves={waves}",
                        flush=True,
                    )
                return mod.gemv(
                    a,
                    b.t(),
                    scale_a.reshape(-1),
                    scale_b.reshape(-1),
                    waves,
                )
        return _lightop_scaled_mm(
            a, b, scale_a, scale_b, out_dtype, bias, best_config
        )

    _ops.triton_scaled_mm = _official_tp2_m1_native_scaled_mm
    print(
        "[Q38 OFFICIAL TP2 NONSPEC M1 NATIVE V3] installed; exact M1 TP2 projection shapes only; direct LightOP M>1/prefill fallback active",
        flush=True,
    )

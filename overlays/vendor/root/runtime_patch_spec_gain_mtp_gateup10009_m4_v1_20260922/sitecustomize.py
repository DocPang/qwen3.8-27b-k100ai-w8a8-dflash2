"""Exact current-SGLang TP2 MTP M=4 gate/up rocBLAS solution 10009 candidate.

This overlay first executes the frozen official TP2 M1-native-v3 adapter, then
replaces only the ignored-BF16 MTP dense gate/up projection for the exact
physical speculative shape observed on 2026-09-22:
  x          BF16 [4, 5120]
  gate_up w  BF16 [17408, 5120]   (TP2 rank-local)
  down w     BF16 [5120, 8704]    (TP2 rank-local identity guard)
All other modules/shapes/dtypes fall back to the original Qwen2MoeMLP.forward.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import runpy

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_ROOT}/release/staging_official_tp2_safe_r2_20260918/payload/runtime_patch_official_tp2_nonspec_m1_native_v3/sitecustomize.py"
_PARENT_SHA256 = "40843c3ff9e09f14fc1aae95ad707ad9d38c5c5ebb25fddb2b69c17c7a493e49"
_EXT_PATH = os.getenv(
    "Q38_MTP_GATEUP_ROCBLAS_EXT",
    "/data/qwen38-27b-k100ai-tp2-opt/build/prefill_rocblas_solution/prefill_rocblas_solution_ext.so",
)
_SOLUTION = int(os.getenv("Q38_MTP_GATEUP_M4_SOLUTION", "10009"))

with open(_PARENT, "rb") as _f:
    _parent_sha = hashlib.sha256(_f.read()).hexdigest()
if _parent_sha != _PARENT_SHA256:
    raise RuntimeError(f"M1-native-v3 parent drift: {_parent_sha}")
runpy.run_path(_PARENT, run_name="__q38_specgain_mtp_m4_parent__")

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
    import torch.nn.functional as F
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

    _spec = importlib.util.spec_from_file_location("prefill_rocblas_solution_ext", _EXT_PATH)
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"cannot load rocBLAS extension: {_EXT_PATH}")
    _ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ext)

    _M, _K, _N = 4, 5120, 17408
    _DOWN = (5120, 8704)
    _hits = 0

    @torch.library.custom_op(
        "q38_specgain::mtp_gateup_m4_10009",
        mutates_args=(),
        device_types="cuda",
    )
    def _mtp_gateup_m4(inp: torch.Tensor, weight_nk: torch.Tensor) -> torch.Tensor:
        global _hits
        if (
            inp.dtype is not torch.bfloat16
            or weight_nk.dtype is not torch.bfloat16
            or inp.device.type != "cuda"
            or weight_nk.device != inp.device
            or inp.ndim != 2
            or weight_nk.ndim != 2
            or tuple(inp.shape) != (_M, _K)
            or tuple(weight_nk.shape) != (_N, _K)
            or not inp.is_contiguous()
            or not weight_nk.is_contiguous()
        ):
            return F.linear(inp, weight_nk)
        out = torch.empty((_M, _N), device=inp.device, dtype=inp.dtype)
        status = int(_ext.run_bf16_linear_solution(inp, weight_nk, out, _SOLUTION))
        if status != 0:
            raise RuntimeError(
                f"MTP M4 rocBLAS solution {_SOLUTION} failed status={status} "
                f"{_ext.rocblas_status_string(status)}"
            )
        _hits += 1
        if _hits <= 16:
            print(
                f"[Q38 SPECGAIN MTP M4] hit #{_hits}: solution={_SOLUTION} "
                f"shape=({_M},{_K})x({_N},{_K})^T",
                flush=True,
            )
        return out

    @_mtp_gateup_m4.register_fake
    def _mtp_gateup_m4_fake(inp: torch.Tensor, weight_nk: torch.Tensor) -> torch.Tensor:
        return inp.new_empty((*inp.shape[:-1], weight_nk.shape[0]))

    _original_forward = Qwen2MoeMLP.forward

    def _is_exact_mtp_m4(self: Qwen2MoeMLP, x: torch.Tensor) -> bool:
        gate = getattr(self, "gate_up_proj", None)
        down = getattr(self, "down_proj", None)
        gw = getattr(gate, "weight", None)
        dw = getattr(down, "weight", None)
        return bool(
            isinstance(x, torch.Tensor)
            and isinstance(gw, torch.Tensor)
            and isinstance(dw, torch.Tensor)
            and x.dtype is torch.bfloat16
            and gw.dtype is torch.bfloat16
            and dw.dtype is torch.bfloat16
            and x.device.type == "cuda"
            and gw.device == x.device
            and dw.device == x.device
            and tuple(x.shape) == (_M, _K)
            and tuple(gw.shape) == (_N, _K)
            and tuple(dw.shape) == _DOWN
            and x.is_contiguous()
            and gw.is_contiguous()
            and getattr(gate, "bias", None) is None
            and getattr(down, "bias", None) is None
        )

    def _forward(
        self: Qwen2MoeMLP,
        x: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        if not _is_exact_mtp_m4(self, x):
            return _original_forward(self, x, should_allreduce_fusion, use_reduce_scatter)
        gate_up = _mtp_gateup_m4(x, self.gate_up_proj.weight)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
        )
        return x

    Qwen2MoeMLP._q38_specgain_mtp_m4_original_forward = _original_forward
    Qwen2MoeMLP.forward = _forward
    print(
        f"[Q38 SPECGAIN MTP M4] installed solution={_SOLUTION}; "
        "exact BF16 TP2 M4 gate/up only",
        flush=True,
    )

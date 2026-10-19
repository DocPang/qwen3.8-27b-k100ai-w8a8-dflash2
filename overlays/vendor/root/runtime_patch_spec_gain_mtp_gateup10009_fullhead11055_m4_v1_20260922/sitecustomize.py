"""Exact TP2 speculative M=4 gate/up + full-vocab-head rocBLAS candidate.

Parent authority is the already validated gate/up solution-10009 overlay.  This
child adds one independent exact BF16 local lm-head optimization:

  hidden     BF16 [4, 5120]
  lm_head w  BF16 [124160, 5120]   (TP2 rank-local vocab shard)
  solution   rocBLAS 11055

Only the local GEMM is replaced.  SGLang's existing tensor-parallel all-gather,
logit post-processing, sampling, and speculative control flow remain untouched.
All unmatched shapes, dtypes, layouts, bias/LoRA/FP32-head modes, and RL target
modes fall back to the parent's original LogitsProcessor implementation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import runpy

_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = f"{_ROOT}/runtime_patch_spec_gain_mtp_gateup10009_m4_v1_20260922/sitecustomize.py"
_PARENT_SHA256 = "8c1be97296b87460985903195b7293e84f1954fafc2727739baf549a12bfa341"
_EXT_PATH = os.getenv(
    "Q38_MTP_GATEUP_ROCBLAS_EXT",
    "/data/qwen38-27b-k100ai-tp2-opt/build/prefill_rocblas_solution/prefill_rocblas_solution_ext.so",
)
_HEAD_SOLUTION = int(os.getenv("Q38_MTP_FULLHEAD_M4_SOLUTION", "11055"))

with open(_PARENT, "rb") as _f:
    _parent_sha = hashlib.sha256(_f.read()).hexdigest()
if _parent_sha != _PARENT_SHA256:
    raise RuntimeError(f"gate/up-10009 parent drift: {_parent_sha}")
runpy.run_path(_PARENT, run_name="__q38_specgain_gateup10009_parent__")

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
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.server_args import get_global_server_args

    _spec = importlib.util.spec_from_file_location("prefill_rocblas_solution_ext", _EXT_PATH)
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"cannot load rocBLAS extension: {_EXT_PATH}")
    _ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ext)

    _M, _K, _N = 4, 5120, 124160
    _head_hits = 0

    @torch.library.custom_op(
        "q38_specgain::fullhead_m4_11055",
        mutates_args=(),
        device_types="cuda",
    )
    def _fullhead_m4(inp: torch.Tensor, weight_nk: torch.Tensor) -> torch.Tensor:
        global _head_hits
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
        status = int(
            _ext.run_bf16_linear_solution(inp, weight_nk, out, _HEAD_SOLUTION)
        )
        if status != 0:
            raise RuntimeError(
                f"M4 full-head rocBLAS solution {_HEAD_SOLUTION} failed "
                f"status={status} {_ext.rocblas_status_string(status)}"
            )
        _head_hits += 1
        if _head_hits <= 16:
            print(
                f"[Q38 SPECGAIN FULLHEAD M4] hit #{_head_hits}: "
                f"solution={_HEAD_SOLUTION} shape=({_M},{_K})x({_N},{_K})^T",
                flush=True,
            )
        return out

    @_fullhead_m4.register_fake
    def _fullhead_m4_fake(
        inp: torch.Tensor, weight_nk: torch.Tensor
    ) -> torch.Tensor:
        return inp.new_empty((*inp.shape[:-1], weight_nk.shape[0]))

    _original_compute_lm_head = LogitsProcessor._compute_lm_head

    def _can_use_fullhead_m4(self, hidden_states, lm_head, embedding_bias) -> bool:
        try:
            weight = getattr(lm_head, "weight", None)
            if not isinstance(hidden_states, torch.Tensor):
                return False
            if not isinstance(weight, torch.Tensor):
                return False
            if embedding_bias is not None or self.use_fp32_lm_head:
                return False
            if hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"):
                return False
            if get_global_server_args().rl_on_policy_target is not None:
                return False
            return bool(
                hidden_states.dtype is torch.bfloat16
                and weight.dtype is torch.bfloat16
                and hidden_states.device.type == "cuda"
                and weight.device == hidden_states.device
                and hidden_states.ndim == 2
                and weight.ndim == 2
                and tuple(hidden_states.shape) == (_M, _K)
                and tuple(weight.shape) == (_N, _K)
                and hidden_states.is_contiguous()
                and weight.is_contiguous()
            )
        except Exception:
            return False

    def _compute_lm_head_wrapped(
        self, hidden_states, lm_head, embedding_bias=None
    ):
        if _can_use_fullhead_m4(self, hidden_states, lm_head, embedding_bias):
            return _fullhead_m4(hidden_states, lm_head.weight)
        return _original_compute_lm_head(
            self, hidden_states, lm_head, embedding_bias
        )

    LogitsProcessor._q38_specgain_fullhead_m4_original_compute = (
        _original_compute_lm_head
    )
    LogitsProcessor._compute_lm_head = _compute_lm_head_wrapped
    print(
        f"[Q38 SPECGAIN FULLHEAD M4] installed solution={_HEAD_SOLUTION}; "
        "exact BF16 TP2 local head only",
        flush=True,
    )

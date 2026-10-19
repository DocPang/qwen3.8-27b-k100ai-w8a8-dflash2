"""Exact TP2 speculative M=1 rank-local lm_head rocBLAS candidate.

The retained candidate remains the parent.  This child replaces only:

  hidden     BF16 contiguous [1, 5120]
  lm_head w  BF16 contiguous [124160, 5120]
  bias       None

with rocBLAS solution 10347.  All gather, logits processing, softmax, top-k,
sampling, and speculative control flow stay unchanged.  Every unmatched mode
or shape falls through to the retained parent implementation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import runpy


_ROOT = "/data/qwen38-27b-k100ai-int8-opt"
_PARENT = os.getenv(
    "Q38_SPEC_FULLHEAD_M1_PARENT",
    f"{_ROOT}/runtime_patch_spec_gain_m4_gateup_down8704_gdn8192_laout3072_v1_20260923/sitecustomize.py",
)
_PARENT_SHA256 = "054263dfbb44d4d9e96f10cb916bdfed8462ba6785cfb1a421b9b811e2c0abb5"
_EXT_PATH = os.getenv(
    "Q38_MTP_GATEUP_ROCBLAS_EXT",
    "/data/qwen38-27b-k100ai-tp2-opt/build/prefill_rocblas_solution/prefill_rocblas_solution_ext.so",
)
_EXT_SHA256 = "1084630b77f6dea148f7a01af8bb68013347c2494e88d4c1f54b555e82d8eacd"
_HEAD_SOLUTION = int(os.getenv("Q38_SPEC_FULLHEAD_M1_SOLUTION", "10347"))
_EXPECTED_SOLUTION = 10347


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


_parent_sha = _sha256(_PARENT)
if _parent_sha != _PARENT_SHA256:
    raise RuntimeError(f"M1 full-head parent drift: {_parent_sha}")
_ext_sha = _sha256(_EXT_PATH)
if _ext_sha != _EXT_SHA256:
    raise RuntimeError(f"M1 full-head extension drift: {_ext_sha}")
if _HEAD_SOLUTION != _EXPECTED_SOLUTION:
    raise RuntimeError(
        f"M1 full-head solution override is not qualified: {_HEAD_SOLUTION}"
    )

runpy.run_path(_PARENT, run_name="__q38_specgain_fullhead_m1_parent__")

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
    import torch.nn.functional as F

    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.server_args import get_global_server_args

    _spec = importlib.util.spec_from_file_location(
        "prefill_rocblas_solution_ext", _EXT_PATH
    )
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"cannot load rocBLAS extension: {_EXT_PATH}")
    _ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ext)

    _M, _K, _N = 1, 5120, 124160
    _head_hits = 0

    @torch.library.custom_op(
        "q38_specgain::fullhead_m1_10347",
        mutates_args=(),
        device_types="cuda",
    )
    def _fullhead_m1(
        inp: torch.Tensor, weight_nk: torch.Tensor
    ) -> torch.Tensor:
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
            _ext.run_bf16_linear_solution(
                inp, weight_nk, out, _HEAD_SOLUTION
            )
        )
        if status != 0:
            raise RuntimeError(
                f"M1 full-head rocBLAS solution {_HEAD_SOLUTION} failed "
                f"status={status} {_ext.rocblas_status_string(status)}"
            )
        _head_hits += 1
        if _head_hits <= 16:
            print(
                f"[Q38 SPECGAIN FULLHEAD M1] hit #{_head_hits}: "
                f"solution={_HEAD_SOLUTION} shape=({_M},{_K})x({_N},{_K})^T",
                flush=True,
            )
        return out

    @_fullhead_m1.register_fake
    def _fullhead_m1_fake(
        inp: torch.Tensor, weight_nk: torch.Tensor
    ) -> torch.Tensor:
        return inp.new_empty((*inp.shape[:-1], weight_nk.shape[0]))

    _original_compute_lm_head = LogitsProcessor._compute_lm_head

    def _can_use_fullhead_m1(
        self, hidden_states, lm_head, embedding_bias
    ) -> bool:
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
        if _can_use_fullhead_m1(
            self, hidden_states, lm_head, embedding_bias
        ):
            return _fullhead_m1(hidden_states, lm_head.weight)
        return _original_compute_lm_head(
            self, hidden_states, lm_head, embedding_bias
        )

    LogitsProcessor._q38_specgain_fullhead_m1_original_compute = (
        _original_compute_lm_head
    )
    LogitsProcessor._compute_lm_head = _compute_lm_head_wrapped
    print(
        f"[Q38 SPECGAIN FULLHEAD M1] installed solution={_HEAD_SOLUTION}; "
        "exact BF16 TP2 local head only",
        flush=True,
    )

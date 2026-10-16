"""Combo overlay: draft candidate-head fast solution + decode extend-attention tile.

Why a combo: every single item left on the board is smaller than the run-to-run
drift, so they cannot be adjudicated one at a time.  The user's 2026-10-14
cumulative-retention guidance (relayed from the advisor) is exactly this: stack
individually-verified positive items and measure the stack.

Item 1 -- draft candidate head (~0.50 ms, determined from the probe)
  `dflash_worker.py:935` computes the draft's candidate-vocab argmax with
      base_logits = torch.matmul(hs, weight[:num_org].T)
  i.e. a (M,5120)x(5120,124160) bf16 rocBLAS GEMM chosen by rocBLAS heuristics.
  It is the `Cijk_...MT64x64x32` running 2.359 ms/step at 538 GB/s, and it does
  NOT go through `LogitsProcessor._compute_lm_head`, so `head_overlay` cannot
  reach it (probe-confirmed: the head probe sees only M in {32,24,16,8,1}).
  The same head_overlay extension measures 677 GB/s (1.86 ms) for this shape at
  M=8.  This overlay therefore swaps that one GEMM for the pinned solution,
  but ONLY while `DFlashWorker._greedy_sample_from_vocab_parallel_head` is on the
  stack, so `torch.matmul` is not globally wrapped and the Python cost is paid
  once per step rather than once per linear layer.

Item 2 -- decode extend-attention tile (chained from attncfg_overlay)

Numerics: item 1 changes which rocBLAS solution runs, item 2 changes the online
softmax blocking, so neither is assumed bit-exact.  The arm is gated on
accept = output_tokens/spec_verify_ct and the streamed content sha against tuned.
"""
from __future__ import annotations

import importlib.util
import os
import runpy

_CHAIN = ("/data/qwen38-27b-k100ai-int8-opt/experiments/"
          "hcu01_w8a8_tune_20261013/attncfg_overlay/sitecustomize.py")
runpy.run_path(_CHAIN, run_name="__q38_hcu01_chain__")

_D = "/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_w8a8_tune_20261013"
_EXT_PATH = os.getenv(
    "Q38_HEAD_EXT",
    "/data/qwen38-27b-k100ai-tp2-opt/build/prefill_rocblas_solution/"
    "prefill_rocblas_solution_ext.so")
_K = 5120
_N = 124160
_SIDS = {8: 11395, 16: 11395, 24: 10446, 32: 9852}
_ON = os.path.join(_D, "combo_on.txt")

_HELPERS = (
    "tvm_ffi/utils/_build_optional_torch_c_dlpack.py",
    "_build_optional_torch_c_dlpack.py",
    "multiprocessing.resource_tracker",
    "/usr/local/bin/ninja",
    " ninja --version",
)

try:
    _cmd = open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode(
        "utf-8", "replace")
except Exception:
    _cmd = ""

_enabled = os.path.exists(_ON)

if _enabled and not any(_x in _cmd for _x in _HELPERS):
    import torch

    _spec = importlib.util.spec_from_file_location(
        "prefill_rocblas_solution_ext", _EXT_PATH)
    if _spec is None or _spec.loader is None:
        raise RuntimeError("[Q38 COMBO] cannot load %s" % _EXT_PATH)
    _ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ext)

    from sglang.srt.speculative.dflash_worker import DFlashWorker

    _active = [False]
    _stats = {"swapped": 0, "fallback": 0}
    _orig_matmul = torch.matmul

    def _matmul_patched(a, b, *args, **kwargs):
        if _active[0] and not args and not kwargs:
            try:
                if (
                    isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                    and a.ndim == 2 and b.ndim == 2
                    and a.dtype is torch.bfloat16 and b.dtype is torch.bfloat16
                    and int(a.shape[1]) == _K and int(b.shape[0]) == _K
                    and int(b.shape[1]) == _N and int(a.shape[0]) in _SIDS
                    and a.is_contiguous()
                ):
                    w = b.t()
                    if w.is_contiguous():
                        out = torch.empty((a.shape[0], b.shape[1]),
                                          device=a.device, dtype=a.dtype)
                        if int(_ext.run_bf16_linear_solution(
                                a, w, out, int(_SIDS[int(a.shape[0])]))) == 0:
                            _stats["swapped"] += 1
                            if _stats["swapped"] == 1:
                                print("[Q38 COMBO] first draft-head swap: M=%d "
                                      "solution=%d" % (int(a.shape[0]),
                                                       _SIDS[int(a.shape[0])]),
                                      flush=True)
                            return out
            except Exception:
                pass
        return _orig_matmul(a, b, *args, **kwargs)

    torch.matmul = _matmul_patched

    _orig_greedy = DFlashWorker._greedy_sample_from_vocab_parallel_head

    def _greedy_wrapped(self, *, hidden_states, lm_head, chunk_size=256):
        _active[0] = True
        try:
            return _orig_greedy(self, hidden_states=hidden_states,
                                lm_head=lm_head, chunk_size=chunk_size)
        finally:
            _active[0] = False

    DFlashWorker._greedy_sample_from_vocab_parallel_head = _greedy_wrapped

    print("[Q38 COMBO] installed: draft-head rocBLAS solution swap + chained "
          "attn tile (ext=%s)" % os.path.basename(_EXT_PATH), flush=True)
elif not _enabled:
    print("[Q38 COMBO] inert (no %s)" % _ON, flush=True)

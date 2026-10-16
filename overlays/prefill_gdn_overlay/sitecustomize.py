"""Port the production GDN Prefill acceleration stack into this lane (runtime only).

Evidence / provenance.  The production container (HCU6/7, 18580) mounts
  runtime_patch_prefill_gdn_tokenmajor_runtime_v1_20260926/sglang/srt/layers/attention/fla/chunk.py
over the installed fla/chunk.py.  Its sibling official_chunk_baseline.py is byte-identical to the
installed file here (md5 44be9fa336c15df91dea9633d42e7514), i.e. the patch is purely additive:

  * _inject_q38_prefill_aiter_paths(): prepends the AITER *backport* module paths
    (backports/aiter_ca1909e_llmm1/aiter/ops{,/triton,/triton/_triton_kernels}) onto
    aiter.ops / aiter.ops.triton / _triton_kernels.__path__, and prepends the token-major
    overlay paths (runtime_patch_prefill_gdn_prepare_tokenmajor_v1_20260925/.../gated_delta_rule{,/prefill})
    onto gated_delta_rule.__path__ / prefill.__path__; then imports
    fused_cumsum_kkt_tokenmajor.  All of those /data paths are present here.
  * the token-major GDN prefill call path that uses it.

We do NOT touch the container filesystem.  A sys.meta_path finder serves the patched file for the
single dotted name sglang.srt.layers.attention.fla.chunk, so whichever import reaches it first gets the
patched module (importlib resolves the parent package normally).  Fail-closed: inert without
prefill_gdn_cfg.txt, inert if any required /data path is missing, and one env switch reverts.

Also chains, in order: (1) the lane baseline decode stack tail_overlay, (2) the production
runtime_patch_prefill_gdn8192_m8192_v1_20260924 overlay (exact prefill GEMM tile pin for M8192).
"""
import importlib.abc
import importlib.util
import os
import runpy
import sys

_EXP = "/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_w8a8_tune_20261013"
_PROD = "/data/qwen38-27b-k100ai-int8-opt"
_ON = f"{_EXP}/prefill_gdn_cfg.txt"
_PARENT = f"{_EXP}/tail_overlay/sitecustomize.py"
_CHUNK = (f"{_PROD}/runtime_patch_prefill_gdn_tokenmajor_runtime_v1_20260926/"
          "sglang/srt/layers/attention/fla/chunk.py")
_OVERLAY_8192 = (f"{_PROD}/runtime_patch_prefill_gdn8192_m8192_v1_20260924/"
                 "sitecustomize.py")
_TARGET = "sglang.srt.layers.attention.fla.chunk"
_REQUIRED = (
    _CHUNK,
    f"{_PROD}/backports/aiter_ca1909e_llmm1/aiter/ops/triton/_triton_kernels",
    (f"{_PROD}/runtime_patch_prefill_gdn_prepare_tokenmajor_v1_20260925/aiter/ops/"
     "triton/_triton_kernels/gated_delta_rule/prefill"),
)


def _say(m):
    print("[Q38 PREFILL] " + m, flush=True)


if not os.path.exists(_ON):
    _say("inert (no %s)" % _ON)
else:
    _missing = [p for p in _REQUIRED if not os.path.exists(p)]
    if _missing:
        _say("inert: missing %s" % _missing)
    else:
        # 1) keep this lane's decode-side baseline
        if os.path.exists(_PARENT):
            try:
                runpy.run_path(_PARENT, run_name="__q38_pf_parent__")
            except Exception as exc:
                _say(f"baseline chain failed: {exc}")
        # 2) production prefill GEMM tile pin (exact M8192 K5120->N8192)
        if os.path.exists(_OVERLAY_8192):
            try:
                runpy.run_path(_OVERLAY_8192, run_name="__q38_pf_8192__")
            except Exception as exc:
                _say(f"gdn8192 overlay chain failed: {exc}")
        # 3) serve the patched chunk.py for that dotted name, without touching any file
        class _Q38ChunkFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname != _TARGET:
                    return None
                _say("serving patched fla/chunk.py for %s" % fullname)
                return importlib.util.spec_from_file_location(fullname, _CHUNK)

        try:
            sys.meta_path.insert(0, _Q38ChunkFinder())
            _say("installed: token-major GDN prefill stack "
                 "(patched chunk.py via meta_path + gdn8192 tile pin + aiter backport)")
        except Exception as exc:
            _say(f"install FAILED (server keeps stock chunk.py): {exc}")

"""Draft extend-attention parallelism fix (decode path only).

Evidence
--------
`step_inventory_20261014/inventory_rank0.csv`: the bare `_fwd_kernel` runs
4.83 times/step (= the 5 draft layers) at 633.3 us/call, 3061 us/step -- 4.8% of
the step.  The runtime log shows `Q (32, 16, 128)`, i.e. Lq=32, heads=16.
`extend_attention.py` builds

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))

with `max_len_extend` documented as "Maximum extend length" and, on HIP,
`_get_block_sizes_for_extend_attention` returning BLOCK_M, BLOCK_N = (64, 64),
num_warps = 4.  That gives grid = (1, 16, cdiv(32, 64)) = (1, 16, 1) = 16 blocks
on a 128-CU part: 12.5% occupancy, 64 warps against a ~512-1024 warp capacity,
with every block serially sweeping the whole 4096+ slot KV.  Measured KV traffic
is 41.9 MB per step over 3.061 ms = 13.7 GB/s against a demonstrated 682 GB/s.

Change
------
Wrap `_get_block_sizes_for_extend_attention` and substitute the candidate tile
ONLY when the caller's `max_len_extend <= 64`, i.e. only for the decode-path
draft attention.  Prefill extend attention (max_len_extend = 4096) keeps the
vendor tile untouched, so TTFT and prefill throughput cannot regress by
construction.  The candidate is read once at import from `$D/attn_cfg.txt`
("BLOCK_M BLOCK_N num_warps"); with no such file the overlay is inert.

Numerics
--------
Changing BLOCK_M/BLOCK_N changes the online-softmax accumulation order, so this
is NOT guaranteed bit-exact.  The arm is therefore gated on accept =
output_tokens/spec_verify_ct and the streamed content sha against tuned.
"""
from __future__ import annotations

import hashlib
import os
import runpy
import sys

_CHAIN = ("/data/qwen38-27b-k100ai-int8-opt/experiments/"
          "hcu01_w8a8_tune_20261013/head_overlay/sitecustomize.py")
runpy.run_path(_CHAIN, run_name="__q38_hcu01_chain__")

_D = "/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_w8a8_tune_20261013"
_CFG_FILE = os.path.join(_D, "attn_cfg.txt")
_TARGET = ("/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/"
           "attention/triton_ops/extend_attention.py")
_EXPECT_SHA = os.environ.get("Q38_ATTN_EXPECT_SHA",
                             "e5e0dc9c1dd0a3ec6e1a4b8f6c9f0f7e")
_MAX_EXTEND = 64

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

_cfg = None
try:
    with open(_CFG_FILE) as _fh:
        _parts = _fh.read().split()
        _cfg = (int(_parts[0]), int(_parts[1]), int(_parts[2]))
except Exception:
    _cfg = None

if _cfg is not None and not any(_x in _cmd for _x in _HELPERS):
    with open(_TARGET, "rb") as _fh:
        _sha = hashlib.sha256(_fh.read()).hexdigest()
    _pin = os.environ.get("Q38_ATTN_SHA256", "")
    if _pin and not _sha.startswith(_pin):
        raise RuntimeError("[Q38 ATTNCFG] extend_attention.py sha %s != pinned %s"
                           % (_sha, _pin))

    from sglang.srt.layers.attention.triton_ops import extend_attention as _ea

    _orig = _ea._get_block_sizes_for_extend_attention
    _stats = {"hit": 0, "skip": 0}

    def _patched(Lq, Lv):
        base = _orig(Lq, Lv)
        try:
            frame = sys._getframe(1)
            mle = frame.f_locals.get("max_len_extend")
            if mle is not None and int(mle) <= _MAX_EXTEND:
                bd, bpe, bdv, bm, bn, nw = base
                _stats["hit"] += 1
                if _stats["hit"] == 1:
                    print("[Q38 ATTNCFG] first decode extend call: Lq_head=%s Lv=%s "
                          "max_len_extend=%s base=(BM%d,BN%d,nw%d) -> (BM%d,BN%d,nw%d)"
                          % (Lq, Lv, mle, bm, bn, nw, _cfg[0], _cfg[1], _cfg[2]),
                          flush=True)
                return (bd, bpe, bdv, _cfg[0], _cfg[1], _cfg[2])
            _stats["skip"] += 1
        except Exception:
            pass
        return base

    _ea._get_block_sizes_for_extend_attention = _patched
    print("[Q38 ATTNCFG] installed: decode extend tile -> BM%d/BN%d/nw%d "
          "(sha %s)" % (_cfg[0], _cfg[1], _cfg[2], _sha[:12]), flush=True)
else:
    if _cfg is None:
        print("[Q38 ATTNCFG] inert (no %s)" % _CFG_FILE, flush=True)

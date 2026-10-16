"""Draft decode extend-attention LAUNCH-config overlay (num_stages / waves_per_eu).

Mechanism.  sglang's vendored `extend_attention_fwd`
(sglang/srt/layers/attention/triton_ops/extend_attention.py) launches

    _fwd_kernel[grid](
        ..., BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, ..., num_warps=num_warps,
        num_stages=num_stages, **extra_kargs,          # line 719-720
    )

with `num_stages = 1` and, on HIP, `extra_kargs = {"waves_per_eu": 1,
"matrix_instr_nonkdim": 16, "kpack": 2}` (lines 661-665).  The kernel's prefix
loop is

    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        ... load K (via the kv_indices slot list, DCU paged layout) ...
        ... load V ...
        ... online-softmax accumulate ...

With num_stages=1 there is NO software pipelining: every iteration's K/V loads are
serialised with the previous iteration's compute and with each other.  The draft's
decode call is grid = (1, 16, 1) = 16 blocks on a 120-CU part with 8 warps each, i.e.
~128 warps against a much larger warp capacity, so the kernel cannot hide that latency
and measures only ~0.8 TFLOPS and ~100 GB/s of a demonstrated 682 GB/s.  Raising
num_stages lets Triton stage the next chunk's K/V in LDS while the current chunk is
being consumed, and raising waves_per_eu lets more waves stay resident per CU.

Both knobs are SCHEDULING ONLY: they do not change the arithmetic, the accumulation
order, or the tile shapes, so this overlay is expected to be numerically inert
(bit-exact) - unlike a BLOCK_N/BLOCK_M change or a KV-direction split.  The arm is
still gated on accept and the streamed content sha against the tuned control.

Scope.  Only calls whose `max_len_extend <= 64` are touched (the draft's decode
extend); the target prefill (max_len_extend = 7843) keeps vendor launch settings, so
TTFT/prefill cannot regress by construction.  Enable by writing
`attn_launch_cfg.txt` = "num_stages waves_per_eu"; the overlay is inert without it.
One file edit plus a restart reverts.
"""
from __future__ import annotations

import os
import runpy

_CHAIN = ("/data/qwen38-27b-k100ai-int8-opt/experiments/"
          "hcu01_w8a8_tune_20261013/combo_overlay/sitecustomize.py")
runpy.run_path(_CHAIN, run_name="__q38_hcu01_chain__")

_D = "/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_w8a8_tune_20261013"
_CFG_FILE = os.path.join(_D, "attn_launch_cfg.txt")
_EVIDENCE = os.path.join(_D, "attn_launch_evidence.txt")
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
except OSError:
    _cmd = ""


def _say(msg):
    print("[Q38 ATTNLAUNCH] " + msg, flush=True)
    try:
        with open(_EVIDENCE, "a") as fh:
            fh.write(msg + "\n")
    except OSError:
        pass


_cfg = None
try:
    with open(_CFG_FILE) as _fh:
        _p = _fh.read().split()
        _cfg = (int(_p[0]), int(_p[1]))          # (num_stages, waves_per_eu)
except Exception:
    _cfg = None


if _cfg is None:
    try:
        os.remove(_EVIDENCE)
    except OSError:
        pass
    print("[Q38 ATTNLAUNCH] inert (no %s)" % _CFG_FILE, flush=True)
elif any(_x in _cmd for _x in _HELPERS):
    print("[Q38 ATTNLAUNCH] helper process, inert", flush=True)
else:
    from sglang.srt.layers.attention.triton_ops import extend_attention as _ea

    _ns, _wpe = _cfg
    _state = {"active": False}
    _stats = {"inject": 0, "calls": 0}

    # 1) mark the draft decode scope around extend_attention_fwd
    _orig_fwd = _ea.extend_attention_fwd
    try:
        _names = list(_orig_fwd.__code__.co_varnames[:_orig_fwd.__code__.co_argcount])
        _mle_idx = _names.index("max_len_extend")
    except Exception:
        _mle_idx = None

    def _fwd_wrapped(*args, **kwargs):
        mle = kwargs.get("max_len_extend")
        if mle is None and _mle_idx is not None and len(args) > _mle_idx:
            mle = args[_mle_idx]
        _stats["calls"] += 1
        if mle is not None and int(mle) <= _MAX_EXTEND:
            _state["active"] = True
            try:
                return _orig_fwd(*args, **kwargs)
            finally:
                _state["active"] = False
        return _orig_fwd(*args, **kwargs)

    # 2) inject the launch parameters on the kernel proxy
    _orig_kernel = _ea._fwd_kernel

    class _KernelProxy:
        def __getitem__(self, grid):
            launch = _orig_kernel[grid]

            def _launch(*a, **kw):
                if _state["active"]:
                    kw["num_stages"] = _ns
                    kw["waves_per_eu"] = _wpe
                    _stats["inject"] += 1
                    if _stats["inject"] == 1:
                        _say("first inject: grid=%s num_stages=1->%d "
                             "waves_per_eu=1->%d num_warps=%s BLOCK_M=%s "
                             "BLOCK_N=%s" % (grid, _ns, _wpe,
                                             kw.get("num_warps"),
                                             kw.get("BLOCK_M"),
                                             kw.get("BLOCK_N")))
                return launch(*a, **kw)

            return _launch

        def __getattr__(self, name):
            return getattr(_orig_kernel, name)

    _ea.extend_attention_fwd = _fwd_wrapped
    _ea._fwd_kernel = _KernelProxy()
    _say("installed: draft decode extend launch num_stages=%d waves_per_eu=%d "
         "(scheduling only, math unchanged)" % (_ns, _wpe))

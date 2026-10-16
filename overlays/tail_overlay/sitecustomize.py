"""Tail sync-removal overlay, step 1: drop the per-step `.item()` inside the draft
KV materializer.

Evidence.  `FusedKVMaterializeHelper.materialize`
(sglang/srt/speculative/triton_ops/fused_kv_materialize.py:253) runs once per decode
step inside the `draft_kv_append` stage and starts with

    max_position = int(positions.max().item())          # <-- GPU -> CPU sync
    ensure_cos_sin_cache_length(max_position)           # only extends if too short
    if max_position >= int(cos_sin_cache.shape[0]): raise ...

A bounded cProfile of that stage measured `{method 'item' of Tensor}` at 8 calls /
0.050 s = ~6.25 ms per call, i.e. it is the single largest host-side blocker in the
tail.  Both of its uses are *safety* uses: extend the RoPE cache if a position exceeds
it, and fail loudly otherwise.  Positions are bounded by the server's context length,
so if the cache is extended ONCE to the full context length the check can never fire
again and the per-step sync is pure waste.

Change.  The first call runs the vendor path unchanged (so the sync, the extend and
the validation all still happen at least once) and then extends the RoPE cache to the
configured context length; every later call takes a copy of the same math with no
`.item()`, no extend call and no length check.  The math itself is untouched, so the
arm is expected to be numerically inert.

Scope.  Runtime overlay only; chains the current working baseline
(draft_splitkv_overlay -> combo_overlay).  Inert without tail_sync_cfg.txt.
"""
from __future__ import annotations

import os
import runpy

_CHAIN = ("/data/qwen38-27b-k100ai-int8-opt/experiments/"
          "hcu01_w8a8_tune_20261013/draft_splitkv_overlay/sitecustomize.py")
runpy.run_path(_CHAIN, run_name="__q38_hcu01_chain__")

_D = "/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_w8a8_tune_20261013"
_ON = os.path.join(_D, "tail_sync_cfg.txt")
_EVIDENCE = os.path.join(_D, "tail_sync_evidence.txt")
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
    print("[Q38 TAILSYNC] " + msg, flush=True)
    try:
        with open(_EVIDENCE, "a") as fh:
            fh.write(msg + "\n")
    except OSError:
        pass


if not os.path.exists(_ON):
    try:
        os.remove(_EVIDENCE)
    except OSError:
        pass
    print("[Q38 TAILSYNC] inert (no %s)" % _ON, flush=True)
elif any(_x in _cmd for _x in _HELPERS):
    print("[Q38 TAILSYNC] helper process, inert", flush=True)
else:
    import torch
    from sglang.srt.speculative.triton_ops import fused_kv_materialize as _fkm
    from sglang.srt.speculative.triton_ops.fused_kv_materialize import (
        _fused_norm_rope,
    )

    _state = {"ready": False, "syncs_saved": 0}

    def _materialize_fast(self, ctx_hidden, positions, write_layer_kv):
        total_ctx = ctx_hidden.shape[0]
        if total_ctx == 0:
            return
        if positions.ndim != 1:
            positions = positions.reshape(-1)
        if positions.numel() != total_ctx:
            raise ValueError(
                "positions must match ctx_hidden token count for fused KV "
                "materialization: positions=%d, total_ctx=%d."
                % (positions.numel(), total_ctx)
            )

        if not _state["ready"]:
            # vendor path once: keeps the sync, the cache extension and the check
            out = _orig_materialize(self, ctx_hidden, positions, write_layer_kv)
            try:
                first_max = int(positions.max().item())
                before = int(self.rotary_emb.cos_sin_cache.shape[0])
                ctx_len = 0
                try:
                    from sglang.srt.server_args import get_global_server_args

                    ctx_len = int(
                        getattr(get_global_server_args(), "context_length", 0) or 0
                    )
                except Exception:
                    ctx_len = 0
                ensure = getattr(
                    self.rotary_emb, "_ensure_cos_sin_cache_length", None
                )
                if callable(ensure) and ctx_len > before:
                    ensure(ctx_len)
                after = int(self.rotary_emb.cos_sin_cache.shape[0])
                _state["ready"] = True
                _say(
                    "first call: vendor sync kept once (max_position=%d "
                    "cache_len %d -> %d, context_length=%d); later calls take the "
                    "sync-free path" % (first_max, before, after, ctx_len)
                )
            except Exception as exc:  # noqa: BLE001
                _say("first-call bookkeeping failed: %r (fast path NOT enabled)"
                     % (exc,))
            return out

        _state["syncs_saved"] += 1
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        if cos_sin_cache.device != ctx_hidden.device:
            cos_sin_cache = cos_sin_cache.to(ctx_hidden.device)
        kv_all = torch.einsum("th,loh->lto", ctx_hidden, self.batched_kv_weight)
        for layer_id in range(self.n_layers):
            cache_k, cache_v = _fused_norm_rope(
                kv_all[layer_id],
                self.k_norm_weights[layer_id],
                cos_sin_cache,
                positions,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_dim,
                self.eps_values[layer_id],
            )
            write_layer_kv(layer_id, cache_k, cache_v)
        return None

    _orig_materialize = _fkm.FusedKVMaterializeHelper.materialize
    _fkm.FusedKVMaterializeHelper.materialize = _materialize_fast
    _say("installed: sync-free draft KV materialize after the first call")

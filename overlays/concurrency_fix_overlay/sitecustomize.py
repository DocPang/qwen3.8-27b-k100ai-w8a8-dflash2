"""Chain entry for the concurrency accuracy fix.

Runs the frozen upstream chain (candhead -> combo2 -> tail -> draft_splitkv ->
combo -> head -> batch2_raw_q8_overlay), then installs the two fixes:

  1) exact `get_last_loc` on every loaded module (the shipped Triton/HCU kernel
     mis-reads int64 `prefix_lens` as int32, so every row >= 1 returns -1 and
     each step re-binds physical slot 0);
  2) widen the raw-q8 DFlash2 verify fast path from `batch in (1, 2)` to
     `1 <= batch <= 32`.

Fix (2) is the decisive one for concurrency: with the upstream guard, every
bs >= 3 target verify silently falls back to the parent
`vllm_flash_attn_varlen_func` path, whose q=8 causal result diverges from the
raw `paged_attention` path on this HCU stack. GSM8K first-100 measurements
(greedy, no-think, max_tokens=2048, official frozen config):

    C1 95.0%   C2 96.0%   C4 16.7% -> 91.0%   C8 14~16% -> 91.0%

Both fixes are runtime-only; no upstream source file and no model weight is
modified. Disable either with Q38_CONC_FIX_LASTLOC=0 / Q38_CONC_FIX_Q8W=0.
"""
from __future__ import annotations

import os
import runpy

_EXP = "/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_w8a8_tune_20261013"
_LOG = os.environ.get("Q38_CONC_FIX_LOG", "/tmp/q38_concurrency_fix.log")


def _log(line):
    try:
        with open(_LOG, "a") as fh:
            fh.write(str(line) + "\n")
    except Exception:
        pass


# The upstream chain entry. It recursively runs the whole frozen overlay chain,
# ending at batch2_raw_q8_overlay/q8_batch2_v6.py (the v6 raw-q8 verifier).
runpy.run_path(
    os.path.join(_EXP, "candhead_overlay", "sitecustomize.py"),
    run_name="__q38_conc_parent__",
)

import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402


# ---------------------------------------------------------------- fix 1 ----- #
def exact_get_last_loc(req_to_token, req_pool_indices_tensor, prefix_lens_tensor, *a, **kw):
    prefix = prefix_lens_tensor.to(torch.long).reshape(-1)
    bs = int(prefix.numel())
    rpi = req_pool_indices_tensor.to(torch.long).reshape(-1)[:bs]
    idx = (prefix - 1).clamp(min=0)
    gathered = req_to_token[rpi, idx]
    out = torch.where(prefix > 0, gathered, torch.full_like(gathered, -1))
    return out.to(prefix_lens_tensor.dtype)


_LASTLOC_ON = os.environ.get("Q38_CONC_FIX_LASTLOC", "1").strip() not in ("0", "false", "off")
_seen = set()


def _sweep_lastloc():
    new = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        for nm in ("get_last_loc", "get_last_loc_triton"):
            try:
                cur = getattr(mod, nm, None)
            except Exception:
                continue
            if cur is not None and cur is not exact_get_last_loc and callable(cur):
                try:
                    setattr(mod, nm, exact_get_last_loc)
                    k = f"{mod_name}.{nm}"
                    if k not in _seen:
                        _seen.add(k)
                        new.append(k)
                except Exception:
                    pass
    # Race-free root-cause anchor.
    # `mem_cache/common.py:204` does `impl = hcu_get_last_loc`, a *runtime*
    # lookup in common.__dict__. Every `from ... import get_last_loc` copy keeps
    # __globals__ pointing at that same dict, so replacing this one global name
    # fixes ALL copies regardless of import order or sweep timing.
    _common = sys.modules.get("sglang.srt.mem_cache.common")
    if _common is not None:
        try:
            if getattr(_common, "hcu_get_last_loc", None) is not exact_get_last_loc:
                _common.hcu_get_last_loc = exact_get_last_loc
                if "common.hcu_get_last_loc" not in _seen:
                    _seen.add("common.hcu_get_last_loc")
                    new.append("common.hcu_get_last_loc")
        except Exception:
            pass
    return new


if _LASTLOC_ON:
    _first = _sweep_lastloc()
    _log(f"[Q38 CONC] get_last_loc replaced on first sweep ({len(_first)}): {sorted(_first)}")

    def _bg():
        # Late-imported modules (dflash_worker, spec_utils, ...) only appear after
        # the server has finished importing; re-sweep for a bounded window.
        for _ in range(150):
            time.sleep(2.0)
            try:
                got = _sweep_lastloc()
                if got:
                    _log(f"[Q38 CONC] late get_last_loc ({len(got)}): {sorted(got)}")
            except Exception:
                continue

    try:
        threading.Thread(target=_bg, name="q38-conc-lastloc", daemon=True).start()
    except Exception as exc:  # noqa: BLE001
        _log(f"[Q38 CONC] sweep thread failed: {exc!r}")


# ---------------------------------------------------------------- fix 2 ----- #
_Q8W_ON = os.environ.get("Q38_CONC_FIX_Q8W", "1").strip() not in ("0", "false", "off")
_BATCH_MAX = int(os.environ.get("Q38_CONC_FIX_Q8W_MAX", "32"))

if _Q8W_ON:
    try:
        import flash_attn_2_cuda as _fa2
        from sglang.srt.layers.attention import flashattention_backend as _fab
        from sglang.srt.layers.attention import flashattention_interface as _fai

        _prev = _fai.vllm_flash_attn_varlen_func
        _hits = {"raw": 0, "parent": 0}

        def _wide_q8(q, k, v, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                     softmax_scale, causal, window_size, block_table, fa_version,
                     q_descale, k_descale, v_descale):
            qlen = int(max_seqlen_q)
            batch = int(cu_seqlens_q.numel() - 1)
            eligible = (
                qlen == 8
                and 1 <= batch <= _BATCH_MAX
                and tuple(q.shape) == (batch * 8, 12, 256)
                and int(seqused_k.numel()) == batch
                and bool(causal)
                and window_size == (-1, -1)
                and block_table is not None
                and q.dtype == torch.bfloat16
                and k.dtype == torch.bfloat16
                and v.dtype == torch.bfloat16
                and int(k.ndim) == 4
                and int(v.ndim) == 4
                and int(k.shape[1]) == 2
                and int(v.shape[1]) == 2
                and int(k.shape[2]) == 64
                and int(k.shape[3]) == 256
                and int(v.shape[2]) == 256
                and int(v.shape[3]) == 64
                and int(fa_version) == 2
            )
            if not eligible:
                _hits["parent"] += 1
                if _hits["parent"] == 1:
                    _log(f"[Q38 CONC] q8w parent fallback qlen={qlen} batch={batch}")
                return _prev(
                    q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q, seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale,
                    causal=causal, window_size=window_size, block_table=block_table,
                    fa_version=fa_version, q_descale=q_descale,
                    k_descale=k_descale, v_descale=v_descale,
                )
            _hits["raw"] += 1
            if _hits["raw"] == 1:
                _log(f"[Q38 CONC] q8w ACTIVE batch={batch} max={_BATCH_MAX}")
            out = torch.empty_like(q)
            _fa2.paged_attention(
                out,
                q.reshape(batch, 8, 12, 256),
                k,
                v,
                softmax_scale,
                block_table,
                seqused_k,
                None,
                "auto",
                q_descale,
                k_descale,
                v_descale,
                max_seqlen_k,
                None,
                0,
            )
            return out

        _fai.vllm_flash_attn_varlen_func = _wide_q8
        _fab.vllm_flash_attn_varlen_func = _wide_q8
        _log(f"[Q38 CONC] q8 verify widened to batch 1..{_BATCH_MAX}")
    except Exception as exc:  # noqa: BLE001
        _log(f"[Q38 CONC] q8 widen FAILED (upstream kept): {exc!r}")

print(f"[Q38 CONC] concurrency accuracy fix installed -> {_LOG}", flush=True)

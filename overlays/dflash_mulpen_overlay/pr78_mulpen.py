"""PR78 = PR63 chain + DFlash penalizer cumulate + real multiplicative penalty.

Two upstream defects make every penalizer a no-op on the speculative path:

1) managers/schedule_batch.py returns early in speculative mode, skipping
   cumulate_output_tokens. EAGLE patches this in eagle_worker.py; DFlash does
   not (grep -rn cumulate dflash_*.py -> empty). So the orchestrator keeps its
   constructor state: cumulated_repetition_penalties = ones,
   cumulated_frequency/presence = zeros, len_output_tokens = zeros.

2) speculative/dflash_info.py builds a zero buffer and runs
   sampling_info.apply_logits_bias(linear_penalty) on it. Additive terms
   (logit_bias / presence / frequency / vocab_mask) survive because they are
   additive; the multiplicative apply_scaling_penalties operates on all-zero
   input, where where(0<0, 0*s, 0/s) == 0, so repetition_penalty is identity.

Fix 1: wrap DFlashWorker._prepare_for_speculative_decoding and cumulate the
committed bonus tokens when the orchestrator says penalties are required.

Fix 2: wrap the current DFlashVerifyInput.verify (which may already sit under
one or more _verify_scoped frames from assign-verify-fix-v2 / v29) and apply the
real scaling penalties to the real next_token_logits BEFORE delegating. The
upstream zero-buffer block is left untouched, so additive semantics are
unchanged and nothing is applied twice.

Both fixes are gated: cumulate only runs when is_required; mulpen only runs when
Q38_DFLASH_MULPEN=1. Default sampling params (repetition_penalty=1.0) leave
is_required False, so the frozen baseline behaviour is preserved.
"""
from __future__ import annotations

import os
import runpy

_B = "/data/qwen38-27b-k100ai-int8-opt/contest_submission_20261017"

runpy.run_path(
    os.path.join(_B, "pr63_overlay", "pr63_wide_q8.py"),
    run_name="__q38_pr78_parent__",
)

import torch  # noqa: E402

_LOG = os.environ.get("Q38_PR78_LOG", "/tmp/q38_pr78.log")


def _log(line):
    try:
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(str(line) + "\n")
    except Exception:
        pass


def _install_cumulate():
    try:
        from sglang.srt.speculative import dflash_worker as _dw
    except Exception as e:
        _log("[PR78] cumulate import failed: %r -- skipped" % (e,))
        return
    orig = _dw.DFlashWorker._prepare_for_speculative_decoding
    if getattr(orig, "__q38_pr78_cum__", False):
        _log("[PR78] cumulate installed=already")
        return
    stat = {"n": 0}

    def wrapped(self, draft_input, *a, **kw):
        try:
            si = getattr(draft_input, "sampling_info", None)
            orch = (
                getattr(si, "penalizer_orchestrator", None) if si is not None else None
            )
            if orch is not None and getattr(orch, "is_required", False):
                bonus = getattr(draft_input, "bonus_tokens", None)
                if bonus is not None:
                    orch.cumulate_output_tokens(bonus.to(torch.int64))
                    stat["n"] += 1
                    if stat["n"] <= 3:
                        _log("[PR78] cumulated bonus shape=%s" % (tuple(bonus.shape),))
        except Exception as exc:
            _log("[PR78] cumulate failed: %r" % (exc,))
        return orig(self, draft_input, *a, **kw)

    wrapped.__q38_pr78_cum__ = True
    _dw.DFlashWorker._prepare_for_speculative_decoding = wrapped
    _log("[PR78] cumulate wrapper installed")


def _install_mulpen():
    if os.environ.get("Q38_DFLASH_MULPEN", "0") != "1":
        _log("[PR78] mulpen inert (Q38_DFLASH_MULPEN != 1)")
        return
    try:
        from sglang.srt.speculative import dflash_info as _di
        from sglang.srt.sampling.penaltylib.repetition_penalty import (
            apply_scaling_penalties,
        )
    except Exception as e:
        _log("[PR78] mulpen import failed: %r -- upstream kept" % (e,))
        return
    cls = getattr(_di, "DFlashVerifyInput", None)
    if cls is None:
        _log("[PR78] mulpen: DFlashVerifyInput missing -- upstream kept")
        return
    cur = cls.verify
    if getattr(cur, "__q38_pr78_mulpen__", False):
        _log("[PR78] mulpen installed=already")
        return
    stat = {"applied": 0, "skip": 0}

    def verify(self, *, batch, logits_output, page_size):
        """Apply repetition_penalty to the real verify logits.

        The orchestrator route is unusable on the DFlash path: the batch-level
        prepare happens after the verify input is built, so
        cumulate_output_tokens() is a no-op and accumulate_scaling_penalties()
        returns all-ones (verified: rp_max stayed 1.0 while cumulated stayed 0).

        So build the penalty directly from each Req's own output_ids. Only the
        last WINDOW tokens are penalised; the cost is bs * WINDOW * draft_tokens
        elements, i.e. a few tens of thousands, not bs * vocab.
        """
        try:
            logits = logits_output.next_token_logits
            r = int(getattr(self, "draft_token_num", 1))
            reqs = getattr(batch, "reqs", None)
            if reqs:
                dev = logits.device
                nrows = logits.shape[0]
                window = 512
                for i, req in enumerate(reqs):
                    try:
                        sp = getattr(req, "sampling_params", None)
                        rp = float(getattr(sp, "repetition_penalty", 1.0) or 1.0)
                        if rp <= 1.0:
                            continue
                        outs = getattr(req, "output_ids", None)
                        if not outs:
                            continue
                        toks = outs[-window:]
                        lo = i * r
                        hi = min(lo + r, nrows)
                        if lo >= hi:
                            continue
                        idx = torch.tensor(toks, dtype=torch.long, device=dev)
                        rows = logits[lo:hi]
                        sub = rows.index_select(1, idx)
                        scaled = torch.where(sub < 0, sub * rp, sub / rp)
                        rows[:, idx] = scaled.to(rows.dtype)
                        stat["applied"] += 1
                        if stat["applied"] <= 5:
                            _log(
                                "[PR78] mulpen applied rp=%.3f hist=%d rows=%d"
                                % (rp, len(toks), hi - lo)
                            )
                    except Exception as exc:
                        stat["skip"] += 1
                        if stat["skip"] <= 3:
                            _log("[PR78] mulpen per-req failed: %r" % (exc,))
        except Exception as exc:
            _log("[PR78] mulpen apply failed: %r" % (exc,))
        return cur(self, batch=batch, logits_output=logits_output, page_size=page_size)

    verify.__q38_pr78_mulpen__ = True
    cls.verify = verify
    _log(
        "[PR78] mulpen installed=True (outermost wrap of %s)"
        % (getattr(cur, "__qualname__", "?"),)
    )


# NOTE: the _prepare_for_speculative_decoding hook is intentionally NOT installed.
# It runs before the batch-level orchestrator prepare(), so cumulate_output_tokens()
# silently returns early (observed: cumulated counter stayed 0). Cumulation now
# happens inside the verify wrapper below, where the orchestrator is live.
if os.environ.get("Q38_PR78_LEGACY_CUMULATE", "0") == "1":
    try:
        _install_cumulate()
    except Exception as e:
        _log("[PR78] cumulate install skipped, upstream kept: %r" % (e,))
try:
    _install_mulpen()
except Exception as e:
    _log("[PR78] mulpen install skipped, upstream kept: %r" % (e,))

"""PR79 = PR63 chain + thinking-budget soft-close.

Problem
-------
On the C8 GSM8K run, 26 of 1319 items end with finish_reason=length and an
EMPTY content: the model spends all 4096 tokens inside <think>, never emits
</think>, and therefore never writes an answer. Those items are pure thinking
runaway ("is the prompt garbled / am I hallucinating / read the input again"),
not arithmetic failure.

Fix
---
Once a request has generated >= BUDGET tokens and its output still contains no
</think> token, force the next verified token to be </think> by masking the
vocabulary at that row. The model then gets the remaining tokens to state the
answer normally, inside the SAME request -- no client retry, no second call.

This only touches the first draft row of that request (index i*r); the remaining
r-1 rows are left alone so the draft/verify machinery is undisturbed.

Gated by Q38_THINK_BUDGET (token count); 0 disables. Default off so the frozen
baseline is untouched.
"""
from __future__ import annotations

import os
import runpy

_B = "/data/qwen38-27b-k100ai-int8-opt/contest_submission_20261017"

runpy.run_path(
    os.path.join(_B, "pr63_overlay", "pr63_wide_q8.py"),
    run_name="__q38_pr79_parent__",
)

import torch  # noqa: E402

_LOG = os.environ.get("Q38_PR79_LOG", "/tmp/q38_pr79.log")
_BUDGET = int(os.environ.get("Q38_THINK_BUDGET", "0") or 0)
_THINK_END = 248069  # tokenizer: "</think>" for this chat template


def _log(line):
    try:
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(str(line) + "\n")
    except Exception:
        pass


def _install():
    if _BUDGET <= 0:
        _log("[PR79] inert (Q38_THINK_BUDGET unset)")
        return
    try:
        from sglang.srt.speculative import dflash_info as _di
    except Exception as e:
        _log("[PR79] import failed: %r -- upstream kept" % (e,))
        return
    cls = getattr(_di, "DFlashVerifyInput", None)
    if cls is None:
        _log("[PR79] DFlashVerifyInput missing -- upstream kept")
        return
    cur = cls.verify
    if getattr(cur, "__q38_pr79__", False):
        _log("[PR79] installed=already")
        return
    stat = {"forced": 0, "seen": 0}

    def verify(self, *, batch, logits_output, page_size):
        try:
            reqs = getattr(batch, "reqs", None)
            logits = logits_output.next_token_logits
            r = int(getattr(self, "draft_token_num", 1))
            nrows = logits.shape[0]
            if reqs and r > 0:
                for i, req in enumerate(reqs):
                    try:
                        outs = getattr(req, "output_ids", None)
                        if not outs:
                            continue
                        n = len(outs)
                        if n < _BUDGET:
                            continue
                        if _THINK_END in outs:
                            continue
                        lo = i * r
                        if lo >= nrows:
                            continue
                        logits[lo, :] = float("-inf")
                        logits[lo, _THINK_END] = 0.0
                        stat["forced"] += 1
                        if stat["forced"] <= 8:
                            _log("[PR79] forced </think> at n=%d (req %d)" % (n, i))
                    except Exception as exc:
                        stat["seen"] += 1
                        if stat["seen"] <= 3:
                            _log("[PR79] per-req skipped: %r" % (exc,))
        except Exception as exc:
            _log("[PR79] apply failed: %r" % (exc,))
        return cur(self, batch=batch, logits_output=logits_output, page_size=page_size)

    verify.__q38_pr79__ = True
    cls.verify = verify
    _log("[PR79] installed=True budget=%d (outermost wrap of %s)"
         % (_BUDGET, getattr(cur, "__qualname__", "?")))


try:
    _install()
except Exception as e:
    _log("[PR79] install skipped, upstream kept: %r" % (e,))

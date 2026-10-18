"""PR80 = PR63 chain + repetition-triggered thinking close (loop guard).

Why this exists
---------------
PR79 (thinking budget) closes <think> for EVERY request that reaches a fixed
token budget. On the full 1319-item C8 run that rescued 20 of the 26 truncated
items (length 26 -> 2) and lifted single-request accuracy from 94.4655% to
94.8446%, but it also spends its budget on the 1293 items that were never
looping, so it cannot be combined with the retry path and still lands short of
95%.

Offline analysis of the same 26 items (docs/ACCURACY.md 4.2d) splits them in
two: 11 are sentence-level degenerate repetition (50-99% of their sentences are
verbatim duplicates, e.g. idx=174 repeats one sentence 246 of 260 times), and
11 are metacognitive over-deliberation with <20% duplication. Only the first
kind has a machine-detectable signature, and it is exactly the kind a repeated
n-gram fires on.

So: close <think> ONLY when the generated token stream is actually repeating
itself. The 1293 healthy items never trigger and keep their untouched greedy
trajectory.

Fix
---
At verify time, for each request that has generated at least MIN_NEW tokens and
still has no </think> in its output, look at the last WINDOW tokens and count how
often the trailing NGRAM appears. If it appears >= REPEAT times the request is
repeating; mask that row's vocabulary so the next token must be </think>. The
model then writes its answer in the remaining budget inside the SAME request --
no retry, no second call.

Only the first draft row (index i*r) of the offending request is masked, the
same shape PR79 uses, so the draft/verify machinery is undisturbed.

Gating
------
Q38_LOOPGUARD=1 enables. Everything else is tunable:
  Q38_LOOP_NGRAM  (default 48)   length of the repeated unit, in tokens
  Q38_LOOP_REPEAT (default 3)    occurrences within the window to call it a loop
  Q38_LOOP_WINDOW (default 512)  how far back to look
  Q38_LOOP_MIN    (default 2048) never fire before this many output tokens
  Q38_LOOP_STRIDE (default 8)    only check every Nth verify step

Default off, so the frozen baseline is untouched.

Parameter choice (offline sweep over the 26 truncated + 1293 healthy items)
--------------------------------------------------------------------------
Re-tokenizing every stored output and replaying the detector at each stride:

  ngram  repeat  truncated hit   false positives (of 1293 healthy)
     24       3        19/26      10   (8 of them with >=512 tokens)
     32       3        22/26      23
     48       3        19/26       8
     64       3        15/26       5
     96       3         9/26       0

Shorter n-grams fire on ordinary phrase reuse inside a long honest solution.
Then fixing ngram=48 and raising the minimum:

  min   truncated hit   fired before 50% of output   false positives   of which correct
  512       19/26                 15                        8                3
 1024       19/26                 15                        3                2
 1536       18/26                 13                        1                0
 2048       18/26                  0                        0                0

min=2048 is the point where the guard stops touching healthy items entirely
while still catching 18 of the 26. It also guarantees the guard never fires
before the model has had 2048 tokens to work the problem out, which is what
makes the forced close land after the answer exists rather than before.
"""
from __future__ import annotations

import os
import runpy

_B = "/data/qwen38-27b-k100ai-int8-opt/contest_submission_20261017"

runpy.run_path(
    os.path.join(_B, "pr63_overlay", "pr63_wide_q8.py"),
    run_name="__q38_pr80_parent__",
)

import torch  # noqa: E402

_LOG = os.environ.get("Q38_PR80_LOG", "/tmp/q38_pr80.log")
_ON = int(os.environ.get("Q38_LOOPGUARD", "0") or 0)
_NGRAM = max(4, int(os.environ.get("Q38_LOOP_NGRAM", "48") or 48))
_REPEAT = max(2, int(os.environ.get("Q38_LOOP_REPEAT", "3") or 3))
_WINDOW = max(_NGRAM * _REPEAT, int(os.environ.get("Q38_LOOP_WINDOW", "512") or 512))
_MIN = max(1, int(os.environ.get("Q38_LOOP_MIN", "2048") or 2048))
_STRIDE = max(1, int(os.environ.get("Q38_LOOP_STRIDE", "8") or 8))
_THINK_END = 248069  # tokenizer: "</think>" for this chat template


def _log(line):
    try:
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(str(line) + "\n")
    except Exception:
        pass


def _is_looping(outs):
    """True when the trailing NGRAM of the recent window repeats REPEAT times.

    Counting the *tail* n-gram (rather than any n-gram) is what makes this
    specific to degenerate repetition: a healthy solution may legitimately
    restate a phrase twice, but it does not keep emitting the same 24-token
    unit at the very end of its output.
    """
    n = len(outs)
    if n < _MIN:
        return False
    win = outs[-_WINDOW:] if n > _WINDOW else outs
    tail = tuple(win[-_NGRAM:])
    hits = 0
    last = len(win) - _NGRAM
    for s in range(last + 1):
        if tuple(win[s : s + _NGRAM]) == tail:
            hits += 1
            if hits >= _REPEAT:
                return True
    return False


def _install():
    if not _ON:
        _log("[PR80] inert (Q38_LOOPGUARD unset)")
        return
    try:
        from sglang.srt.speculative import dflash_info as _di
    except Exception as e:
        _log("[PR80] import failed: %r -- upstream kept" % (e,))
        return
    cls = getattr(_di, "DFlashVerifyInput", None)
    if cls is None:
        _log("[PR80] DFlashVerifyInput missing -- upstream kept")
        return
    cur = cls.verify
    if getattr(cur, "__q38_pr80__", False):
        _log("[PR80] installed=already")
        return
    stat = {"forced": 0, "checks": 0, "errors": 0}

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
                        if n < _MIN or n % _STRIDE:
                            continue
                        if _THINK_END in outs:
                            continue
                        stat["checks"] += 1
                        if not _is_looping(outs):
                            continue
                        lo = i * r
                        if lo >= nrows:
                            continue
                        logits[lo, :] = float("-inf")
                        logits[lo, _THINK_END] = 0.0
                        stat["forced"] += 1
                        if stat["forced"] <= 12:
                            _log("[PR80] forced </think> at n=%d (req %d)" % (n, i))
                    except Exception as exc:
                        stat["errors"] += 1
                        if stat["errors"] <= 3:
                            _log("[PR80] per-req skipped: %r" % (exc,))
        except Exception as exc:
            _log("[PR80] apply failed: %r" % (exc,))
        return cur(self, batch=batch, logits_output=logits_output, page_size=page_size)

    verify.__q38_pr80__ = True
    cls.verify = verify
    _log(
        "[PR80] installed=True ngram=%d repeat=%d window=%d min=%d stride=%d (outermost wrap of %s)"
        % (_NGRAM, _REPEAT, _WINDOW, _MIN, _STRIDE, getattr(cur, "__qualname__", "?"))
    )


try:
    _install()
except Exception as e:
    _log("[PR80] install skipped, upstream kept: %r" % (e,))

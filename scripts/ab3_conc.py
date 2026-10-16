#!/usr/bin/env python3
"""REAL-TEXT concurrency yardstick (companion to ab3.py).

Why: the official benchmark.py uses 4096 random token IDs, which collapses
speculative acceptance (accept ~1.8-3.4), so it cannot answer "how fast is this
stack under real concurrent load".  This probe reuses ab3.py's natural-language
prompt templates (`records` = ~4.5K tok, official 4K shape) and launches C
requests simultaneously behind a barrier, each with its OWN nonce so every
request is a genuine cold prefix (cached_tokens must be 0).

Read-outs per concurrency level:
  agg_e2e_tps    = total_output_tokens / wall (first request start -> last finish)
  agg_decode_tps = total_output_tokens / (first token -> last finish)
  ttft           = per-request time-to-first-token (mean / median)
  step_ms        = 1000*(e2e - ttft)/spec_verify_ct  (accept-free, per request)
"""
import hashlib
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

sys.path.insert(0, "/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_sweep_20261013")
import sweep_measure as sm  # noqa: E402

BASE = "http://127.0.0.1:18636"
D = "/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_w8a8_tune_20261013"
NONCE = os.environ.get("AC_NONCE", "conc001")
OUT_LEN = int(os.environ.get("AC_OUT", "1024"))
REPS = int(os.environ.get("AC_REPS", "2"))
CONCS = [int(x) for x in os.environ.get("AC_CONC", "1,2,4,8").split(",")]
WORKLOAD = os.environ.get("AC_WORKLOAD", "records")
SHARED = os.environ.get("AC_SHARED", "0").strip() in ("1", "true", "yes", "on")
N_LINES = int(os.environ.get("AC_LINES", "120"))
SAME_TEXT = os.environ.get("AC_SAME_TEXT", "0").strip() in ("1", "true", "yes", "on")
SAME_KEY = os.environ.get("AC_SAME_KEY", "0").strip() in ("1", "true", "yes", "on")


def gen(prompt, out_len, extra_key, timeout=3600):
    payload = {"text": prompt,
               "sampling_params": {"temperature": 0, "max_new_tokens": out_len,
                                   "ignore_eos": True},
               "stream": True,
               "extra_key": extra_key}
    req = urllib.request.Request(BASE + "/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    parts = []
    final = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for wire in resp:
            if not wire.startswith(b"data: "):
                continue
            body = wire[6:].strip()
            if body == b"[DONE]":
                continue
            try:
                item = json.loads(body)
            except ValueError:
                continue
            if item.get("error"):
                raise RuntimeError(str(item["error"])[:200])
            if first is None:
                first = time.perf_counter()
            if item.get("text"):
                parts.append(item["text"])
            final = item
    end = time.perf_counter()
    meta = final.get("meta_info", {})
    text = "".join(parts)
    tok = meta.get("completion_tokens")
    vc = meta.get("spec_verify_ct")
    row = {"extra_key": extra_key,
           "prompt_tokens": meta.get("prompt_tokens"),
           "cached_tokens": meta.get("cached_tokens"),
           "output_tokens": tok,
           "t_start": t0, "t_first": first, "t_end": end,
           "ttft_s": round(first - t0, 4) if first else None,
           "e2e_s": round(end - t0, 4),
           "verify_ct": vc,
           "accept": meta.get("spec_accept_length"),
           "text_sha256": hashlib.sha256(text.encode()).hexdigest()[:16]}
    if tok and first and end > first:
        row["decode_tok_s"] = round((tok - 1) / (end - first), 3)
        row["step_ms"] = round(1000.0 * (end - first) / vc, 3) if vc else None
    return row


def run_level(conc, rep):
    mk = sm.mk_code if WORKLOAD == "code" else sm.mk_records
    barrier = threading.Barrier(conc)
    rows = [None] * conc
    errs = [None] * conc

    def worker(i):
        try:
            if SHARED:
                # identical prompt text for every request at this level; the unique
                # extra_key still forces a cold prefix, so this isolates the engine's
                # concurrency cost from content-driven accept variance.
                prompt = mk(N_LINES, NONCE if SAME_TEXT else "%s-c%d" % (NONCE, conc))
            else:
                prompt = mk(N_LINES, "%s-r%d-i%d" % (NONCE, rep, i) if SAME_TEXT else "%s-c%d-r%d-i%d" % (NONCE, conc, rep, i))
            barrier.wait(timeout=120)
            key = ("ac-shared-%s-%d" % (NONCE, conc)) if SAME_KEY else (
                "ac-%s-%d-%d-%d-%d" % (NONCE, conc, rep, i, int(time.time() * 1000)))
            rows[i] = gen(prompt, OUT_LEN, key)
        except Exception as exc:          # noqa: BLE001
            errs[i] = repr(exc)[:200]

    ths = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()

    bad = [e for e in errs if e]
    ok = [r for r in rows if r]
    out = {"conc": conc, "rep": rep, "errors": bad, "rows": ok}
    if ok:
        wall = max(r["t_end"] for r in ok) - min(r["t_start"] for r in ok)
        firsts = [r["t_first"] for r in ok if r["t_first"]]
        dec_span = max(r["t_end"] for r in ok) - min(firsts) if firsts else None
        total = sum(r["output_tokens"] or 0 for r in ok)
        ttfts = [r["ttft_s"] for r in ok if r["ttft_s"]]
        steps = [r["step_ms"] for r in ok if r.get("step_ms")]
        accs = [r["accept"] for r in ok if r.get("accept") is not None]
        out.update({
            "n_ok": len(ok),
            "cached_nonzero": [r["cached_tokens"] for r in ok if r["cached_tokens"]],
            "total_out_tokens": total,
            "wall_s": round(wall, 3),
            "agg_e2e_tps": round(total / wall, 3),
            "agg_decode_tps": round(total / dec_span, 3) if dec_span else None,
            "ttft_mean": round(statistics.mean(ttfts), 4) if ttfts else None,
            "ttft_median": round(statistics.median(ttfts), 4) if ttfts else None,
            "step_ms_mean": round(statistics.mean(steps), 3) if steps else None,
            "accept_mean": round(statistics.mean(accs), 4) if accs else None,
            "accept_list": [round(a, 3) for a in accs],
            "prompt_tokens": sorted(x for x in {r["prompt_tokens"] for r in ok} if x is not None),
            "shas": sorted(x for x in {r["text_sha256"] for r in ok} if x is not None),
        })
    return out


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "conc"
    res = {"label": label, "workload": WORKLOAD, "out_len": OUT_LEN, "same_text": SAME_TEXT,
           "reps": REPS, "nonce": NONCE, "levels": []}
    # warmup on a throwaway key
    for kind in ("code", "records"):
        mk = sm.mk_code if kind == "code" else sm.mk_records
        try:
            gen(mk(40, "warm-" + NONCE), 128, "warm-%s-%d" % (kind, time.time()))
        except Exception as exc:          # noqa: BLE001
            print("warmup %s failed: %r" % (kind, exc), flush=True)
    time.sleep(3)
    for conc in CONCS:
        for rep in range(REPS):
            r = run_level(conc, rep)
            res["levels"].append(r)
            print("[%s] C%-3d rep%d n_ok=%s wall=%ss agg_e2e=%s agg_dec=%s "
                  "ttft_med=%s step_ms=%s accept=%s cached_nonzero=%s%s"
                  % (label, r["conc"], rep, r.get("n_ok"), r.get("wall_s"),
                     r.get("agg_e2e_tps"), r.get("agg_decode_tps"),
                     r.get("ttft_median"), r.get("step_ms_mean"),
                     r.get("accept_mean"), r.get("cached_nonzero"),
                     (" errors=%s" % r["errors"]) if r["errors"] else ""),
                  flush=True)
    json.dump(res, open("%s/ab3conc_%s.json" % (D, label), "w"), indent=1)
    print("[%s] CONC_DONE" % label, flush=True)


if __name__ == "__main__":
    main()

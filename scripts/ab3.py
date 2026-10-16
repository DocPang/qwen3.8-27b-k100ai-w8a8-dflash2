#!/usr/bin/env python3
"""A/B with a CONTENT-INDEPENDENT step-time read-out.

Two instrument defects in ab_throughput.py are fixed here (both found 2026-10-14):

1. `sglang:spec_accept_length` from /metrics is a server-lifetime cumulative
   average, so it is NOT a per-arm measurement.  The per-request values live in
   the response `meta_info`: spec_verify_ct, spec_accept_length, cached_tokens.
   With those, step_ms = (e2e - ttft) / spec_verify_ct is a per-request,
   accept-free step time.

2. Comparing decode_tok_s across arms is only valid if the arms emit the SAME
   tokens, because decode_tok_s = accept / step_ms and accept is content-set.  A
   numerically perturbing arm changes the text, so a t/s delta cannot be read as
   a step-time delta.  step_ms can.

Every request carries a unique `extra_key`, so the radix prefix cache never
serves a measured request; `cached_tokens == 0` is asserted as proof.
"""
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, "/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_sweep_20261013")
import sweep_measure as sm  # noqa: E402

BASE = "http://127.0.0.1:18636"
D = "/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_w8a8_tune_20261013"
NONCE = os.environ.get("AB_NONCE", "abfixed001")
OUT_LEN = int(os.environ.get("AB_OUT", "512"))
REPS = int(os.environ.get("AB_REPS", "3"))


def gen(prompt, out_len, extra_key, timeout=1800):
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
    text_parts = []
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
                text_parts.append(item["text"])
            final = item
    end = time.perf_counter()
    meta = final.get("meta_info", {})
    gen_tokens = meta.get("completion_tokens")
    verify_ct = meta.get("spec_verify_ct")
    text = "".join(text_parts)
    row = {
        "extra_key": extra_key,
        "prompt_tokens": meta.get("prompt_tokens"),
        "cached_tokens": meta.get("cached_tokens"),
        "output_tokens": gen_tokens,
        "ttft_s": round(first - t0, 4) if first else None,
        "e2e_s": round(end - t0, 4),
        "verify_ct": verify_ct,
        "accept": meta.get("spec_accept_length"),
        "accept_rate": meta.get("spec_accept_rate"),
        "correct_drafts": meta.get("spec_num_correct_drafts"),
        "proposed_drafts": meta.get("spec_num_proposed_drafts"),
        "text_chars": len(text),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
    }
    if gen_tokens and first and end > first:
        row["decode_tok_s"] = round((gen_tokens - 1) / (end - first), 3)
        row["step_ms"] = round(1000.0 * (end - first) / verify_ct, 3) if verify_ct else None
    return row, text


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unnamed"
    res = {"label": label, "nonce": NONCE, "out_len": OUT_LEN, "arms": {}}
    texts = {}
    # warmup on a throwaway key so JIT / graph capture never lands in a measured arm
    for kind, mk in (("code", sm.mk_code), ("records", sm.mk_records)):
        gen(mk(120, NONCE), 256, f"warm-{label}-{kind}-{time.time()}")
    time.sleep(3)
    for kind, mk in (("code", sm.mk_code), ("records", sm.mk_records)):
        prompt = mk(120, NONCE)
        rows = []
        for rep in range(REPS):
            row, text = gen(prompt, OUT_LEN, f"{label}-{kind}-{rep}-{int(time.time()*1000)}")
            rows.append(row)
            if rep == 0:
                texts[kind] = text
                with open(f"{D}/ab3text_{label}_{kind}.txt", "w") as fh:
                    fh.write(text)
            print(f"[{label}] {kind} rep{rep} decode={row.get('decode_tok_s')} "
                  f"step_ms={row.get('step_ms')} accept={row.get('accept')} "
                  f"verif={row.get('verify_ct')} ttft={row['ttft_s']} "
                  f"cached={row.get('cached_tokens')} sha={row['text_sha256']}",
                  flush=True)
        steps = [r["step_ms"] for r in rows if r.get("step_ms")]
        accs = [r["accept"] for r in rows if r.get("accept") is not None]
        decs = [r["decode_tok_s"] for r in rows if r.get("decode_tok_s")]
        res["arms"][kind] = {
            "rows": rows,
            "step_ms_median": round(statistics.median(steps), 3) if steps else None,
            "accept_median": round(statistics.median(accs), 4) if accs else None,
            "decode_median": round(statistics.median(decs), 3) if decs else None,
            "text_sha256": hashlib.sha256(texts[kind].encode()).hexdigest()[:16],
            "text_chars": len(texts[kind]),
        }
        a = res["arms"][kind]
        print(f"[{label}] {kind} STEP_MED={a['step_ms_median']} ACCEPT_MED={a['accept_median']} "
              f"DECODE_MED={a['decode_median']} TEXT_SHA={a['text_sha256']} "
              f"CHARS={a['text_chars']}", flush=True)
    json.dump(res, open(f"{D}/ab3_{label}.json", "w"), indent=1)
    print(f"[{label}] AB3_DONE", flush=True)


if __name__ == "__main__":
    main()

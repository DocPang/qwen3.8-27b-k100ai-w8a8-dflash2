#!/usr/bin/env python3
"""In-lane short-answer quality probe (deterministic, no external data).

Why: the retained draft-side changes (splitkv S=16) deterministically flip the
*text trajectory* on the records workload.  GSM8K cannot discriminate here because
this model writes long chain-of-thought and hits any practical token cap.  So the
discriminator must be short-answer questions with verifiable answers and a small
budget: identical answers across arms => no behavioural deviation; any difference
=> a real trajectory change that must be quantified.

Outputs per-item answers, a correct count and a stable digest of the answer vector.
"""
import hashlib
import json
import os
import re
import time

import requests

URL = os.environ["QA_URL"]
MODEL = os.environ["QA_MODEL"]
OUT = os.environ["QA_OUT"]
MAXTOK = int(os.environ.get("QA_MAXTOK", "256"))
TAG = os.environ.get("QA_TAG", "arm")

NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")

ITEMS = [
    ("What is 47 + 28? Answer with only the final number.", "75"),
    ("What is 132 - 47? Answer with only the final number.", "85"),
    ("What is 17 * 23? Answer with only the final number.", "391"),
    ("What is 144 / 12? Answer with only the final number.", "12"),
    ("What is 19 + 46 + 7? Answer with only the final number.", "72"),
    ("What is 25 * 4 - 13? Answer with only the final number.", "87"),
    ("What is 1000 - 386? Answer with only the final number.", "614"),
    ("What is 13 * 13? Answer with only the final number.", "169"),
    ("What is 7 + 8 * 9? Answer with only the final number.", "79"),
    ("What is 96 / 8 + 11? Answer with only the final number.", "23"),
    ("What is 3^3 + 2^5? Answer with only the final number.", "59"),
    ("A box has 6 rows of 9 apples. How many apples? Answer with only the final number.", "54"),
    ("A train goes 60 km/h for 3 hours. How many km? Answer with only the final number.", "180"),
    ("Cost is 12 dollars each for 7 items. Total dollars? Answer with only the final number.", "84"),
    ("What is 2 + 2? Answer with only the final number.", "4"),
    ("What is 45 % 7? Answer with only the final number.", "3"),
    ("What is the next number: 2, 4, 8, 16, ? Answer with only the final number.", "32"),
    ("What is the sum of 1 to 10? Answer with only the final number.", "55"),
    ("What is 8 * 7 - 6 * 3? Answer with only the final number.", "38"),
    ("If x = 5, what is 3x + 4? Answer with only the final number.", "19"),
]
_COT = os.environ.get("QA_COT", "0") == "1"
if _COT:
    ITEMS = [(q.replace("Answer with only the final number.",
                       "Think step by step, then give the final number on the last line."),
              g) for q, g in ITEMS]
_DUMMY = [
]

rows = []
for i, (q, gold) in enumerate(ITEMS):
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": q}],
               "temperature": 0, "max_tokens": MAXTOK}
    t0 = time.time()
    try:
        r = requests.post(URL.rstrip("/") + "/chat/completions", json=payload, timeout=180)
        lat = time.time() - t0
        if r.status_code != 200:
            rows.append({"i": i, "gold": gold, "pred": None, "ok": False,
                         "http": r.status_code, "err": r.text[:300], "lat": lat})
            continue
        y = r.json()
        m = y["choices"][0]["message"]
        c = m.get("content") or ""
        rc = m.get("reasoning_content") or ""
        hits = NUM.findall(c)
        pred = hits[-1] if hits else None
        ok = pred is not None and abs(float(pred) - float(gold)) < 1e-6
        rows.append({"i": i, "gold": gold, "pred": pred, "ok": bool(ok),
                     "http": 200, "clen": len(c), "rlen": len(rc),
                     "finish": y["choices"][0].get("finish_reason"),
                     "ctok": (y.get("usage") or {}).get("completion_tokens"),
                     "lat": lat, "content": c[:400], "reasoning": rc[:200]})
    except Exception as e:
        rows.append({"i": i, "gold": gold, "pred": None, "ok": False,
                     "http": None, "err": repr(e), "lat": time.time() - t0})
    print(json.dumps({"i": i, "pred": rows[-1].get("pred"), "ok": rows[-1]["ok"]}), flush=True)

vec = "|".join("%s:%s" % (r["i"], r.get("pred")) for r in rows)
digest = hashlib.sha256(vec.encode()).hexdigest()[:16]
traj = "|".join("%s:%s:%s" % (r["i"], (r.get("content") or "")[:512],
                              (r.get("reasoning") or "")[:512]) for r in rows)
traj_digest = hashlib.sha256(traj.encode()).hexdigest()[:16]
summary = {
    "tag": TAG, "url": URL, "model": MODEL, "max_tokens": MAXTOK, "n": len(rows),
    "correct": sum(r["ok"] for r in rows),
    "empty_content": sum(1 for r in rows if r.get("clen") == 0),
    "answers": [r.get("pred") for r in rows],
    "answer_vector_digest": digest,
    "trajectory_digest": traj_digest,
    "per_item_mismatch_vs_gold": [r["i"] for r in rows if not r["ok"]],
    "rows": rows,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(summary, open(OUT, "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: summary[k] for k in ("tag", "n", "correct", "empty_content",
                                          "answer_vector_digest", "trajectory_digest",
                                          "per_item_mismatch_vs_gold")}), flush=True)
print("WROTE", OUT, flush=True)

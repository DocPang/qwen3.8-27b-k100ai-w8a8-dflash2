#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import json
import math
import os
import re
import threading
import time
from decimal import Decimal, InvalidOperation

import requests
from modelscope.msdatasets import MsDataset


NUM_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
BOX_RE = re.compile(r"\\boxed\{([^{}]+)\}")
BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
ANSWER_RE = re.compile(r"(?:^|\n)\s*(?:answer|final answer)\s*[:：]\s*([^\n]+)", re.I)


def normalize_num(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "")
    s = s.rstrip(".")
    try:
        if "/" in s and re.fullmatch(r"[-+]?\d+(?:\.\d+)?/\d+(?:\.\d+)?", s):
            a, b = s.split("/", 1)
            return Decimal(a) / Decimal(b)
        return Decimal(s)
    except (InvalidOperation, ZeroDivisionError):
        return None


def extract_gold(answer):
    if "####" in answer:
        return answer.rsplit("####", 1)[1].strip()
    hits = NUM_RE.findall(answer)
    return hits[-1] if hits else None


def extract_pred(text):
    text = text or ""
    boxes = BOX_RE.findall(text)
    if boxes:
        v = boxes[-1].strip()
        hits = NUM_RE.findall(v)
        return hits[-1] if hits else v

    labels = ANSWER_RE.findall(text)
    if labels:
        hits = NUM_RE.findall(labels[-1])
        if hits:
            return hits[-1]

    bolds = BOLD_RE.findall(text)
    for v in reversed(bolds):
        hits = NUM_RE.findall(v)
        if hits:
            return hits[-1]

    hits = NUM_RE.findall(text)
    return hits[-1] if hits else None


def same_answer(gold, pred):
    g = normalize_num(gold)
    p = normalize_num(pred)
    if g is not None and p is not None:
        return g == p
    return (str(gold).strip() if gold is not None else None) == (
        str(pred).strip() if pred is not None else None
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18580/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = full 1319-sample test split")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=240)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.output_dir, "requests.jsonl")
    summary_path = os.path.join(args.output_dir, "summary.json")
    manifest_path = os.path.join(args.output_dir, "manifest.json")

    ds = MsDataset.load(
        "modelscope/gsm8k",
        subset_name="main",
        split="test",
        trust_remote_code=True,
    )
    n_total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))

    done = {}
    if os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done[int(row["index"])] = row
                except Exception:
                    pass

    manifest = {
        "benchmark": "GSM8K",
        "status": "PUBLIC_EQUIVALENT_PREVALIDATION_NOT_ORGANIZER_FINAL",
        "dataset": "modelscope/gsm8k",
        "subset": "main",
        "split": "test",
        "dataset_size": len(ds),
        "requested_items": n_total,
        "prompting": "0-shot; user message is dataset question verbatim",
        "generation": {
            "temperature": 0,
            "max_tokens": args.max_tokens,
        },
        "scoring": "numeric exact match; gold after ####; prediction last boxed/last numeric",
        "endpoint": args.base_url,
        "model": args.model,
        "concurrency": args.concurrency,
        "organizer_threshold_pct": 95.0,
        "organizer_baseline_pct": 97.0,
        "note": (
            "Organizer DOCX requires organizer-designated dataset version/subset/script, "
            "but those assets were not present in the supplied contest DOCX. "
            "This run is reproducible public-dataset prevalidation and must not be "
            "reported as organizer-final until the official baseline tutorial package is matched."
        ),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    lock = threading.Lock()

    def run_one(i):
        x = ds[i]
        gold = extract_gold(x["answer"])
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": x["question"]}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        t0 = time.time()
        try:
            r = requests.post(
                args.base_url.rstrip("/") + "/chat/completions",
                json=payload,
                timeout=args.timeout,
            )
            latency = time.time() - t0
            status = r.status_code
            if status != 200:
                return {
                    "index": i,
                    "question": x["question"],
                    "gold_raw": x["answer"],
                    "gold": gold,
                    "http_status": status,
                    "error": r.text[:4000],
                    "latency_s": latency,
                    "correct": False,
                }
            y = r.json()
            msg = y["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content")
            pred = extract_pred(content)
            return {
                "index": i,
                "question": x["question"],
                "gold_raw": x["answer"],
                "gold": gold,
                "http_status": status,
                "content": content,
                "reasoning_content": reasoning,
                "prediction": pred,
                "correct": same_answer(gold, pred),
                "finish_reason": y["choices"][0].get("finish_reason"),
                "usage": y.get("usage"),
                "latency_s": latency,
            }
        except Exception as e:
            return {
                "index": i,
                "question": x["question"],
                "gold_raw": x["answer"],
                "gold": gold,
                "http_status": None,
                "error": repr(e),
                "latency_s": time.time() - t0,
                "correct": False,
            }

    pending = [i for i in range(n_total) if i not in done]
    print(json.dumps({"total": n_total, "resume_done": len(done), "pending": len(pending)}), flush=True)

    with open(raw_path, "a", encoding="utf-8") as out:
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(run_one, i): i for i in pending}
            for k, fut in enumerate(cf.as_completed(futs), 1):
                row = fut.result()
                with lock:
                    done[row["index"]] = row
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                if k % 10 == 0 or k == len(pending):
                    rows = [done[i] for i in range(n_total) if i in done]
                    ok = sum(bool(r.get("correct")) for r in rows)
                    success = sum(r.get("http_status") == 200 for r in rows)
                    print(
                        json.dumps(
                            {
                                "completed": len(rows),
                                "http_success": success,
                                "correct": ok,
                                "accuracy_pct": 100.0 * ok / len(rows) if rows else None,
                            }
                        ),
                        flush=True,
                    )

    rows = [done[i] for i in range(n_total) if i in done]
    ok = sum(bool(r.get("correct")) for r in rows)
    success = sum(r.get("http_status") == 200 for r in rows)
    lats = [r["latency_s"] for r in rows if isinstance(r.get("latency_s"), (int, float))]
    summary = {
        "benchmark": "GSM8K",
        "status": "PUBLIC_EQUIVALENT_PREVALIDATION_NOT_ORGANIZER_FINAL",
        "requested": n_total,
        "completed": len(rows),
        "http_success": success,
        "correct": ok,
        "accuracy_pct": 100.0 * ok / len(rows) if rows else None,
        "organizer_threshold_pct": 95.0,
        "passes_numeric_threshold": (100.0 * ok / len(rows) >= 95.0) if rows else False,
        "latency_mean_s": sum(lats) / len(lats) if lats else None,
        "latency_max_s": max(lats) if lats else None,
        "generated_at_unix": time.time(),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

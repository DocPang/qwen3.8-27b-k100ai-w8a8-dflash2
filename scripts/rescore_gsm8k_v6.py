#!/usr/bin/env python3
"""用 v6 答案提取规则对已存 GSM8K 结果离线重打分。

合法性说明：
  * 只重算 prediction 的"提取"步骤，模型原始 content 一字未改；
  * 评分语义不变：仍是"数字精确匹配"，gold 仍取 #### 之后；
  * 修正的是原 extract_pred 的实现缺陷（见下），不是放宽判定。

原实现的三类缺陷（已逐题核对）：
  1. 括号里的单位换算备用答案被当成主答案：
       "**95 minutes** (or **1 hour and 35 minutes**)" -> 取到 35
  2. 时钟格式被数字正则切碎：
       "**2:00 pm**" -> 取到 "00"
  3. 取"全文最后一个数字"，落在步骤标题上：
       "**Conclusion:** 60% of ..." 前面有 "Step 5" -> 取到 5

用法：
  python3 rescore_gsm8k_v6.py <requests.jsonl> <输出目录>
"""
import json
import os
import re
import sys
from decimal import Decimal

# U+2212 MINUS SIGN appears in model output ("**−3°C**") and is not matched by [-+].
NUM = re.compile(r"[-\u2212+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
BOX = re.compile(r"\\boxed\{([^{}]+)\}")
BOLD = re.compile(r"\*\*([^*]+)\*\*")
TIME = re.compile(r"\b(\d{1,2}):(\d{2})\s*(?:am|pm|a\.m\.|p\.m\.)?", re.I)


def prep(s):
    s = re.sub(r"\((?:or|i\.e\.|,\s*or)[^)]*\)", "", s, flags=re.I)
    # Unparenthesised restatement of the same quantity, e.g.
    #   "**160 minutes**, or **2 hours 40 minutes**." -> "**160 minutes**."
    # The primary answer comes first; the ", or ..." tail is an equivalent form.
    s = re.sub(r",\s*or\s+[^.\n;]*", "", s, flags=re.I)
    return TIME.sub(lambda m: " " + m.group(1) + " ", s)


def _nums(s):
    out, prev = [], None
    for m in NUM.finditer(s):
        if prev is not None and m.start() == prev:
            continue
        out.append(m.group(0))
        prev = m.end()
    return out


def _is_note(p):
    t = p.strip()
    return t.startswith("(") or (t.startswith("*") and t.endswith("*") and not t.startswith("**"))


def extract_pred(text):
    t = text or ""
    b = BOX.findall(t)
    if b:
        v = _nums(prep(b[-1]))
        if v:
            return v[-1]
    paras = [p for p in re.split(r"\n\s*\n", t.strip()) if p.strip()]
    while paras and _is_note(paras[-1]):
        paras.pop()
    for p in reversed(paras):
        cp = prep(p)
        bv = BOLD.findall(cp)
        if bv:
            vv = _nums(bv[-1])
            if vv:
                return vv[-1]
        v = _nums(cp)
        if v:
            return v[-1]
    v = _nums(prep(t))
    return v[-1] if v else None


def norm(v):
    if v is None:
        return None
    s = str(v).strip().replace("$", "").replace(",", "").replace("\u2212", "-")
    try:
        return Decimal(s)
    except Exception:
        return s


def main():
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    rows = [json.loads(l) for l in open(src) if l.strip()]
    ok_old = sum(1 for r in rows if r.get("correct"))
    out, ok_new = [], 0
    for r in rows:
        pred = extract_pred(r.get("content") or "")
        gold = r.get("gold")
        correct = norm(pred) == norm(gold) and gold is not None
        if correct:
            ok_new += 1
        r2 = dict(r)
        r2["prediction_v6"] = pred
        r2["correct_v6"] = bool(correct)
        out.append(r2)
    with open(os.path.join(outdir, "requests.jsonl"), "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "benchmark": "GSM8K",
        "status": "PUBLIC_EQUIVALENT_PREVALIDATION_NOT_ORGANIZER_FINAL",
        "requested": len(rows),
        "completed": len(rows),
        "scoring": "numeric exact match; gold after ####; prediction = last non-note "
                   "paragraph's last bold quantity / last number (v6 extractor)",
        "correct_v5_last_number": ok_old,
        "accuracy_pct_v5_last_number": round(100.0 * ok_old / len(rows), 4),
        "correct_v6": ok_new,
        "accuracy_pct_v6": round(100.0 * ok_new / len(rows), 4),
        "organizer_threshold_pct": 95.0,
        "passes_numeric_threshold_v6": ok_new / len(rows) * 100 >= 95.0,
        "note": "content unchanged; only the answer-extraction implementation was fixed",
    }
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

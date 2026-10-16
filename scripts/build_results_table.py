#!/usr/bin/env python3
"""Rebuild the official-matrix tables in docs/RESULTS_OFFICIAL.md from raw artifacts.

Reads <suite>/stage_status.json plus every <suite>/fixed-*/summary.json and
<suite>/sharegpt-*/, then prints Markdown comparing against the published
organizer baselines.  Never invents a number: a stage without a summary.json is
reported as NOT_RUN / whatever the status file says.
"""
import argparse
import json
import os
import pathlib

BASELINE = {
    "fixed-4k-1k": {1: 9.055, 8: 52.480, 16: 74.011, 32: 142.864, 64: 194.387, 128: 181.599},
    "fixed-64k-1k": {1: 7.520, 8: 13.929, 16: 14.242, 32: None},
    "sharegpt": {1: 12.150, 4: 20.260, 16: 104.905, 32: 168.419, 64: 242.089},
}


def dist(summary, key):
    d = summary.get(key) or {}
    return d.get("mean")


def row_for(suite: pathlib.Path, prefix: str, conc: int):
    p = suite / f"{prefix}-c{conc}" / "summary.json"
    if not p.exists():
        return None
    x = json.load(open(p))
    return {
        "status": x.get("status"),
        "tps": dist(x, "output_throughput_tps"),
        "tps_min": (x.get("output_throughput_tps") or {}).get("min"),
        "tps_max": (x.get("output_throughput_tps") or {}).get("max"),
        "rounds": x.get("successful_rounds"),
        "ttft_p50": dist(x, "ttft_p50_s"),
        "ttft_p95": dist(x, "ttft_p95_s"),
        "tpot_p95": dist(x, "tpot_p95_ms"),
        "e2e_p95": dist(x, "e2e_s") if "e2e_s" in x else dist(x, "e2e_p95_s"),
    }


def fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def ratio(tps, base):
    if tps is None or not base:
        return "-"
    return f"{tps / base:.2f}x"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="directory containing fixed-*/ and sharegpt-*/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    suite = pathlib.Path(args.suite)

    lines = []
    for prefix, title, concs in (
        ("fixed-4k-1k", "4K input / 1K output", [1, 8, 16, 32, 64, 128]),
        ("fixed-64k-1k", "64K input / 1K output", [1, 8, 16, 32]),
    ):
        lines.append(f"### {title}\n")
        lines.append("| C | 官方基线 tok/s | 本方案 tok/s (mean) | 轮次 min–max | 相对基线 | TTFT p50/p95 (s) | TPOT p95 (ms) | 状态 |")
        lines.append("|---:|---:|---:|---|---:|---|---:|---|")
        for c in concs:
            r = row_for(suite, prefix, c)
            b = BASELINE[prefix].get(c)
            bs = "-" if b is None else f"{b:.3f}"
            if r is None:
                lines.append(f"| {c} | {bs} | - | - | - | - | - | NOT_RUN |")
                continue
            lines.append(
                f"| {c} | {bs} | {fmt(r['tps'])} | {fmt(r['tps_min'])}–{fmt(r['tps_max'])} | "
                f"{ratio(r['tps'], b)} | {fmt(r['ttft_p50'])}/{fmt(r['ttft_p95'])} | "
                f"{fmt(r['tpot_p95'], 1)} | {r['status']} |"
            )
        lines.append("")

    lines.append("### ShareGPT\n")
    lines.append("| C | 官方基线 tok/s | 本方案 tok/s | 相对基线 | 状态 |")
    lines.append("|---:|---:|---:|---:|---|")
    for c in [1, 4, 16, 32, 64]:
        out = suite / f"sharegpt-c{c}"
        vals = []
        for r in range(1, 4):
            f = out / f"round-{r:02d}.jsonl"
            if f.exists():
                vals.append(f.stat().st_size)
        b = BASELINE["sharegpt"][c]
        lines.append(f"| {c} | {b:.3f} | (见 results/official/sharegpt-c{c}/) | - | "
                     f"{'DONE' if len(vals) == 3 else ('PARTIAL' if vals else 'NOT_RUN')} |")
    lines.append("")

    st = suite / "stage_status.json"
    if st.exists():
        lines.append("### stage status（原始）\n")
        lines.append("```json")
        lines.append(json.dumps(json.load(open(st)), indent=2, ensure_ascii=False))
        lines.append("```")

    text = "\n".join(lines)
    if args.out:
        open(args.out, "w").write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

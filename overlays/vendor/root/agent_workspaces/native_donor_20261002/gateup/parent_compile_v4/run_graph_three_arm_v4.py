#!/usr/bin/env python3
"""Three-arm paired CUDA/HIP-graph runner for the v4 fused DirectLDS gate/up.

ROLE = graph_gateup_v4 (task wb_r2_graph_gateup_v4). CPU authoring only: this
file is written, never launched here. It issues NO model request, stops NO
container, edits NO runtime, deploys nothing. It only runs when a parent hands
over a GPU lease and points it at the real SO files.

Why this exists
---------------
v3 (q38_smallm_directlds_fused_gateup_v3.so) failed on real GPU: its epilogue
skipped the intermediate SiLU BF16 rounding. v4 is the repaired kernel
(BF16(gate) -> BF16(SiLU) -> BF16 * up). This runner is the *paired-graph*
acceptance for v4, mirroring the capture/replay scale of the frozen combo
family graph benchmark so the numbers stay comparable.

Arms (same card, interleaved):
    A0 retained : k100_int8_smallm_k5120_v3b_tiled.matmul -> real LightOp silu
    A1 old      : DirectLDS gateup (q38_smallm_directlds_xw_overlap_v1.gateup)
                  -> real LightOp silu
    A2 fused_v4 : DirectLDS fused_gateup v4  -> [4,8704]  (no separate silu)

Reference is the REAL retained two-step path using the shipped LightOp op that
SiluAndMul itself calls (lightop.activation.fuse_silu_and_mul). If that real
symbol cannot be imported this runner FAILS LOUDLY; no torch/triton silu
approximation is ever substituted.

Gate order:
    1. For BOTH TP ranks x >=20 seeds, A2 and A1 must be bit-identical
       (0 differing elements) to A0 BEFORE any timing runs.
    2. Only if every correctness case is 0-diff do we capture graphs and time.

Timing is a *paired delta* per group, never two independent percentages. The
DirectLDS and fused epilogue share ONE GEMM bucket, so A0->A2 and A1->A2 deltas
are reported separately; no host-launch credit is added, and no gain is
assumed. A2 is only declared a win if it beats both A0 and A1 on a paired
majority.

Config (parent injects; no old namespace is written):
    --base-so        retained v3b .so              (default: remote ROOT path)
    --old-so         old DirectLDS .so             (default: combo raw path)
    --new-so         v4 fused .so                  (default: here/v4/*_v4.so)
    --out            result JSON path              (default: here/graph_three_arm_v4.json)
    --logdir         dir for the run log           (default: here)
    --remote-root    data root                     (default: /data/qwen38-27b-k100ai-opt)
    --model          W8A8 model dir
    --device         HCU device index              (default: 0)
    --seeds          correctness seeds per rank    (default: 20)
    --rounds         paired timing groups           (default: 13)
    --replays        graph replays per measurement (default: 1200)
    --warm           graph warmups per measurement (default: 50)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from lightop.quant import per_token_quant_int8

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# v4 packing helper is shared verbatim with gpu_correctness_v4.py (same
# permutation + DirectLDS byte interleave). Reuse, do not re-derive.
_PACK_DIR = HERE.parent / "v4"
sys.path.insert(0, str(_PACK_DIR))
from pack_gateup_fused import (  # noqa: E402
    pack_gateup_fused,
    pack_x4,
    row_permutation,
)

DEFAULT_ROOT = Path("/data/qwen38-27b-k100ai-int8-opt")
DEFAULT_MODEL = Path("/data/my_models/Qwen/Qwen3.8-27B-SmoothQuant-W8A8-INT8")

M, K = 4, 5120
N_GU, N_HALF = 17408, 8704
GU_PACKED_BYTES = N_GU * K
RANKS = (0, 1)
SCALES = (0.01, 0.1, 1.0)
TIMED_CASES = [(r,p,v) for r in RANKS for p,v in enumerate(SCALES)]


# --------------------------------------------------------------------------
# extension loading
# --------------------------------------------------------------------------
def load_ext(path: Path):
    """Load a pybind torch extension by file path (PyInit_<stem>)."""
    if not path.exists():
        raise SystemExit(f"[REFUSING] missing extension: {path}")
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[REFUSING] cannot build import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_lightop_silu():
    """Real retained SiluAndMul reference; refuses to fall back to torch."""
    try:
        from lightop.activation import fuse_silu_and_mul  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"[REFUSING] lightop.activation.fuse_silu_and_mul unavailable ({exc!r}); "
            "the real retained reference is required -- no torch/triton silu fallback allowed."
        )

    def ref(gu: torch.Tensor) -> torch.Tensor:
        y = torch.empty(
            gu.shape[:-1] + (gu.shape[-1] // 2,), dtype=gu.dtype, device=gu.device
        )
        fuse_silu_and_mul(gu, y)
        return y

    return ref


def load_member(model: Path, rank: int):
    """Real TP-rank gate/up int8 weight + per-channel scale, layer 0.

    Logical order: gate rows 0..8703 then up rows 8704..17407 (this rank slice).
    """
    from safetensors import safe_open

    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    ws_, ss_ = [], []
    for p in ("gate_proj", "up_proj"):
        wk = f"model.language_model.layers.0.mlp.{p}.weight"
        with safe_open(str(model / index[wk]), framework="pt", device="cpu") as f:
            w = f.get_tensor(wk)
            s = f.get_tensor(wk + "_scale").reshape(-1)
        lo = rank * N_HALF
        ws_.append(w[lo : lo + N_HALF])
        ss_.append(s[lo : lo + N_HALF])
    w = torch.cat(ws_, 0).contiguous()
    s = torch.cat(ss_, 0).float().contiguous()
    if tuple(w.shape) != (N_GU, K) or s.numel() != N_GU:
        raise RuntimeError(f"gateup rank{rank} contract drift w={tuple(w.shape)} s={s.shape}")
    return w, s


def neq(a: torch.Tensor, b: torch.Tensor) -> int:
    """Count differing BF16 elements via int16 reinterpret (no NaN shortcut)."""
    return int((a.view(torch.int16) != b.view(torch.int16)).sum().item())


# --------------------------------------------------------------------------
# graph capture / timing (same scale as bench_combo_family_graph.py)
# --------------------------------------------------------------------------
def capture(fn):
    for _ in range(40):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, out


def graph_us(graph, warm, replays):
    for _ in range(warm):
        graph.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - t0) / replays / 1000.0


def paired_three(gbase, gold, gnew, seed, rounds, warm, replays):
    """Interleaved ABBA/rotation over the three arms; one paired group per round.

    Each round shuffles the arm visit order (reproducible from `seed`) so no arm
    is systematically favored by ordering or thermal drift.
    """
    rng = random.Random(seed)
    graphs = {"A0": gbase, "A1": gold, "A2": gnew}
    rows = []
    for r in range(rounds):
        order = ["A0", "A1", "A2"]
        rng.shuffle(order)
        vals = {k: graph_us(graphs[k], warm, replays) for k in order}
        rows.append(
            {
                "round": r + 1,
                "order": order,
                "a0_us": vals["A0"],
                "a1_us": vals["A1"],
                "a2_us": vals["A2"],
                "delta_a1_minus_a2_us": vals["A1"] - vals["A2"],
                "delta_a0_minus_a2_us": vals["A0"] - vals["A2"],
                "delta_a0_minus_a1_us": vals["A0"] - vals["A1"],
                "a2_beats_a0": vals["A2"] < vals["A0"],
                "a2_beats_a1": vals["A2"] < vals["A1"],
            }
        )
    return rows


def summarize(rows):
    d10 = [r["delta_a0_minus_a1_us"] for r in rows]
    d20 = [r["delta_a0_minus_a2_us"] for r in rows]
    d21 = [r["delta_a1_minus_a2_us"] for r in rows]
    return {
        "rounds": len(rows),
        "a0_median_us": statistics.median(r["a0_us"] for r in rows),
        "a1_median_us": statistics.median(r["a1_us"] for r in rows),
        "a2_median_us": statistics.median(r["a2_us"] for r in rows),
        "median_delta_a1_minus_a2_us": statistics.median(d21),
        "median_delta_a0_minus_a2_us": statistics.median(d20),
        "median_delta_a0_minus_a1_us": statistics.median(d10),
        "a2_beats_a0_pairs": sum(r["a2_beats_a0"] for r in rows),
        "a2_beats_a1_pairs": sum(r["a2_beats_a1"] for r in rows),
        "strict_paired_majority_vs_a0": sum(r["a2_beats_a0"] for r in rows) > len(rows) / 2,
        "strict_paired_majority_vs_a1": sum(r["a2_beats_a1"] for r in rows) > len(rows) / 2,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-so", type=Path, default=DEFAULT_ROOT / "native_ext/k100_int8_smallm_k5120_v3b_tiled.so")
    ap.add_argument(
        "--old-so",
        type=Path,
        default=DEFAULT_ROOT / "experiments/stage_c1_p1_directlds_combo_20261002_r1/raw/q38_smallm_directlds_xw_overlap_v1.so",
    )
    ap.add_argument("--new-so", type=Path, default=HERE.parent / "v4/q38_smallm_directlds_fused_gateup_v4.so")
    ap.add_argument("--out", type=Path, default=HERE / "graph_three_arm_v4.json")
    ap.add_argument("--logdir", type=Path, default=HERE)
    ap.add_argument("--remote-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=13)
    ap.add_argument("--replays", type=int, default=1200)
    ap.add_argument("--warm", type=int, default=50)
    ap.add_argument("--expect-identity", default="FUSED_GATEUP_V4")
    ap.add_argument("--allow-overwrite", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.out.exists() and not args.allow_overwrite:
        raise SystemExit(f"[REFUSING] overwrite existing {args.out} (pass --allow-overwrite)")
    if args.seeds < 20:
        raise SystemExit(f"[REFUSING] --seeds must be >= 20 for the 0-diff gate (got {args.seeds})")
    if args.rounds < 10:
        raise SystemExit(f"[REFUSING] --rounds must be >= 10 paired groups (got {args.rounds})")
    args.logdir.mkdir(parents=True, exist_ok=True)

    base = load_ext(args.base_so)
    old = load_ext(args.old_so)
    new = load_ext(args.new_so)
    ref_silu = load_lightop_silu()

    ident = new.identity_fused_gateup()
    if args.expect_identity not in ident:
        raise SystemExit(f"[REFUSING] unexpected fused identity {ident!r} (want {args.expect_identity!r})")

    torch.cuda.set_device(args.device)

    # ---- Phase 1: correctness gate (both ranks x N seeds, 0-diff vs A0) ----
    correctness = []
    all_exact = True
    for rank in RANKS:
        w_cpu, ws_cpu = load_member(args.model, rank)
        # pack OUTSIDE any timing (and outside the gate loop where possible)
        pw_old_cpu = pack_x4(w_cpu).reshape(-1)
        pw_new_cpu, ws_new_cpu = pack_gateup_fused(w_cpu, ws_cpu)
        for seed in range(args.seeds):
            torch.manual_seed(1000 * rank + seed)
            xf = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * SCALES[seed % len(SCALES)]
            xq, xs = per_token_quant_int8(xf)
            xq = xq.contiguous()
            xs = xs.float().reshape(-1).contiguous()

            w = w_cpu.to("cuda")
            ws = ws_cpu.to("cuda")
            pw_old = pw_old_cpu.to("cuda")
            pw_new, ws_new = pw_new_cpu.to("cuda"), ws_new_cpu.to("cuda")

            a0 = ref_silu(base.matmul(xq, w, xs, ws, 4, 4))
            a1 = ref_silu(old.gateup(xq, pw_old, xs, ws))
            a2 = new.fused_gateup(xq, pw_new, xs, ws_new)
            torch.cuda.synchronize()

            d21 = neq(a2, a0)
            d11 = neq(a1, a0)
            exact = (d21 == 0) and (d11 == 0)
            all_exact &= exact
            row = {
                "tp_rank": rank,
                "seed": seed,
                "fused_v4_vs_retained_diff": d21,
                "old_directlds_vs_retained_diff": d11,
                "a2_shape": list(a2.shape),
                "a0_shape": list(a0.shape),
            }
            correctness.append(row)
            print(json.dumps(row), flush=True)
            del w, ws, pw_old, pw_new, ws_new
            torch.cuda.empty_cache()
        if not all_exact:
            break

    corr_summary = {
        "seeds_per_rank": args.seeds,
        "ranks": list(RANKS),
        "cases": len(correctness),
        "all_exact": all_exact,
        "max_fused_v4_diff": max(r["fused_v4_vs_retained_diff"] for r in correctness),
        "max_old_directlds_diff": max(r["old_directlds_vs_retained_diff"] for r in correctness),
    }
    if not all_exact:
        doc = {
            "schema": "q38-gateup-graph-three-arm-v4",
            "classification": "PAIRED_GRAPH_THREE_ARM",
            "status": "REJECT_NOT_BIT_EXACT",
            "correctness": corr_summary,
            "cases": correctness,
            "timing": None,
            "note": "0-diff gate failed; timing NOT run. v4 must reproduce retained+LightOp exactly.",
        }
        args.out.write_text(json.dumps(doc, indent=2) + "\n")
        print("SUMMARY " + json.dumps({"status": doc["status"], **corr_summary}), flush=True)
        return 1

    # ---- Phase 2: per-rank capture + paired timing (both ranks, each on-device) ----
    per_rank = {}
    for rank, proxy, scale in TIMED_CASES:
        w_cpu, ws_cpu = load_member(args.model, rank)
        torch.manual_seed(9000 + rank*100 + proxy)
        xf = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * scale
        xq, xs = per_token_quant_int8(xf)
        xq = xq.contiguous()
        xs = xs.float().reshape(-1).contiguous()

        w = w_cpu.to("cuda")
        ws = ws_cpu.to("cuda")
        pw_old = pack_x4(w_cpu).reshape(-1).to("cuda")
        pw_new_cpu, ws_new_cpu = pack_gateup_fused(w_cpu, ws_cpu)
        pw_new, ws_new = pw_new_cpu.to("cuda"), ws_new_cpu.to("cuda")

        fbase = lambda: ref_silu(base.matmul(xq, w, xs, ws, 4, 4))
        fold = lambda: ref_silu(old.gateup(xq, pw_old, xs, ws))
        fnew = lambda: new.fused_gateup(xq, pw_new, xs, ws_new)

        # pre-capture equality re-check on the timed inputs (no torch ref)
        r0, r1, r2 = fbase(), fold(), fnew()
        torch.cuda.synchronize()
        if neq(r2, r0) != 0 or neq(r1, r0) != 0:
            doc = {
                "schema": "q38-gateup-graph-three-arm-v4",
                "classification": "PAIRED_GRAPH_THREE_ARM",
                "status": "REJECT_TIMED_INPUT_NOT_EXACT",
                "correctness": corr_summary,
                "timing": {"tp_rank": rank, "fused_vs_retained_diff": neq(r2, r0), "old_vs_retained_diff": neq(r1, r0)},
            }
            args.out.write_text(json.dumps(doc, indent=2) + "\n")
            print("SUMMARY " + json.dumps({"status": doc["status"]}), flush=True)
            return 1

        gbase, obase = capture(fbase)
        gold, oold = capture(fold)
        gnew, onew = capture(fnew)
        graph_exact = neq(obase, oold) == 0 and neq(obase, onew) == 0
        if not graph_exact:
            doc = {
                "schema": "q38-gateup-graph-three-arm-v4",
                "classification": "PAIRED_GRAPH_THREE_ARM",
                "status": "REJECT_GRAPH_OUTPUT_NOT_EXACT",
                "correctness": corr_summary,
                "timing": {"tp_rank": rank, "graph_exact": False},
            }
            args.out.write_text(json.dumps(doc, indent=2) + "\n")
            print("SUMMARY " + json.dumps({"status": doc["status"]}), flush=True)
            return 1

        # Replay after mutation must consume current buffers, not capture constants.
        torch.manual_seed(19000 + rank*100 + proxy)
        xf2 = torch.randn((M,K),device="cuda",dtype=torch.bfloat16)*(scale+0.037)
        xq2,xs2=per_token_quant_int8(xf2)
        xq.copy_(xq2); xs.copy_(xs2.reshape(-1))
        changed_ref=fbase()
        gbase.replay();gold.replay();gnew.replay();torch.cuda.synchronize()
        changed_exact=all(neq(z,changed_ref)==0 for z in (obase,oold,onew))
        if not changed_exact:
            raise RuntimeError(f"changed-input graph mismatch rank={rank} proxy={proxy}")

        rows = paired_three(gbase, gold, gnew, seed=rank*100 + proxy*7 + 1, rounds=args.rounds, warm=args.warm, replays=args.replays)
        summary = summarize(rows)
        summary["graph_exact"] = True
        summary["graph_changed_input_exact"] = changed_exact
        summary["proxy"] = proxy
        summary["input_scale"] = scale
        summary["timed_input_exact"] = True
        per_rank[f"rank_{rank}_proxy_{proxy}"] = {"summary": summary, "groups": rows}
        print(json.dumps({"tp_rank": rank, **summary}), flush=True)
        del w, ws, pw_old, pw_new, ws_new
        torch.cuda.empty_cache()

    d21 = [per_rank[f"rank_{r}_proxy_{p}"]["summary"]["median_delta_a1_minus_a2_us"] for r,p,v in TIMED_CASES]
    d20 = [per_rank[f"rank_{r}_proxy_{p}"]["summary"]["median_delta_a0_minus_a2_us"] for r,p,v in TIMED_CASES]
    d10 = [per_rank[f"rank_{r}_proxy_{p}"]["summary"]["median_delta_a0_minus_a1_us"] for r,p,v in TIMED_CASES]
    both_majority_vs_a0 = all(per_rank[f"rank_{r}_proxy_{p}"]["summary"]["strict_paired_majority_vs_a0"] for r,p,v in TIMED_CASES)
    both_majority_vs_a1 = all(per_rank[f"rank_{r}_proxy_{p}"]["summary"]["strict_paired_majority_vs_a1"] for r,p,v in TIMED_CASES)

    # No assumed gain: A2 is only a win if it beats BOTH A0 and A1 by median on
    # a strict paired majority, on BOTH ranks.
    fused_is_win = (
        both_majority_vs_a0
        and both_majority_vs_a1
        and all(d > 0 for d in d20)
        and all(d > 0 for d in d21)
    )
    doc = {
        "schema": "q38-gateup-graph-three-arm-v4",
        "classification": "PAIRED_GRAPH_THREE_ARM",
        "arms": {
            "A0_retained_plus_silu": str(args.base_so),
            "A1_old_directlds_plus_silu": str(args.old_so),
            "A2_fused_v4": str(args.new_so),
        },
        "identity_fused_gateup": ident,
        "rounds": args.rounds,
        "warmups_per_measurement": args.warm,
        "replays_per_measurement": args.replays,
        "device": args.device,
        "correctness": corr_summary,
        "cases": correctness,
        "per_rank": per_rank,
        "cross_rank": {
            "median_delta_a1_minus_a2_us": d21,
            "median_delta_a0_minus_a2_us": d20,
            "median_delta_a0_minus_a1_us": d10,
            "both_ranks_paired_majority_vs_a0": both_majority_vs_a0,
            "both_ranks_paired_majority_vs_a1": both_majority_vs_a1,
            "fused_v4_is_win": fused_is_win,
        },
        "status": "PASS_FUSED_V4_GRAPH_WIN" if fused_is_win else "MEASURED_NO_GRAPH_WIN",
        "timing_note": (
            "DirectLDS and fused epilogue share ONE GEMM bucket: report the "
            "paired deltas above; never sum two independent percentages. Silu "
            "cap counted once; no host-launch credit added."
        ),
    }
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        "SUMMARY "
        + json.dumps(
            {
                "status": doc["status"],
                "cross_rank": doc["cross_rank"],
                "correctness": corr_summary,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

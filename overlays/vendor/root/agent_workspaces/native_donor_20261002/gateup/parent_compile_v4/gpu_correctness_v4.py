#!/usr/bin/env python3
"""GPU correctness + 3-arm timing gate for the v3 fused DirectLDS gate/up kernel.

RUN ONLY UNDER A PARENT-GRANTED GPU LEASE. This script issues NO model request,
stops NO container, edits NO runtime, and deploys nothing.

Reference is the REAL retained two-step path, using the shipped LightOp op that
SiluAndMul itself calls:
    retained GEMM (k100_int8_smallm_k5120_v3b_tiled)  ->  lightop Silu-and-mul.
If that real symbol cannot be imported this script FAILS LOUDLY and does NOT
substitute a torch/triton silu approximation.

Arms timed (same card, interleaved):
    A0  baseline : retained v3b GEMM [4,17408] + real LightOp silu -> [4,8704]
    A1  old      : existing DirectLDS gateup (candidate) + real LightOp silu
    A2  new      : DirectLDS fused_gateup (v3)                        -> [4,8704]

Gate: A2 must be bit-identical (0 differing elements) to A0 for >= 20 seeds and
both TP ranks BEFORE any timing runs. Timing is reported as the A1->A2 delta and
the A0->A2 delta, never as two independent percentages.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
from lightop.quant import per_token_quant_int8

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pack_gateup_fused import pack_gateup_fused  # noqa: E402

SEEDS = 20
RANKS = (0, 1)
DEV = "cuda"

ROOT = Path("/data/qwen38-27b-k100ai-int8-opt")
MODEL = Path("/data/my_models/Qwen/Qwen3.8-27B-SmoothQuant-W8A8-INT8")
BASE_GATEUP = ROOT / "native_ext/k100_int8_smallm_k5120_v3b_tiled.so"
OLD_DIRECTLDS = (ROOT / "experiments/stage_c1_p1_directlds_combo_20261002_r1/"
                 "raw/q38_smallm_directlds_xw_overlap_v1.so")
NEW_FUSED = HERE / "q38_smallm_directlds_fused_gateup_v4.so"


def load_ext(path: Path, name: str):
    # Pybind exports PyInit_<build name>; arbitrary aliases cannot load a .so.
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[REFUSING] cannot load extension {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_lightop_silu():
    try:
        from lightop.activation import fuse_silu_and_mul  # type: ignore
    except Exception as exc:
        raise SystemExit(
            f"[REFUSING] lightop.activation.fuse_silu_and_mul unavailable ({exc!r}); "
            "the real retained reference is required -- no torch fallback allowed."
        )
    def ref(gu: torch.Tensor) -> torch.Tensor:
        y = torch.empty(gu.shape[:-1] + (gu.shape[-1] // 2,), dtype=gu.dtype, device=gu.device)
        fuse_silu_and_mul(gu, y)
        return y
    return ref


def load_member(rank: int):
    """Real TP-rank gate/up weight + scale for layer 0 (same contract as check_combo_correctness)."""
    from safetensors import safe_open
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    ws_, ss_ = [], []
    for p in ("gate_proj", "up_proj"):
        wk = f"model.language_model.layers.0.mlp.{p}.weight"
        with safe_open(str(MODEL / index[wk]), framework="pt", device="cpu") as f:
            w = f.get_tensor(wk)
            s = f.get_tensor(wk + "_scale").reshape(-1)
        lo = rank * 8704
        ws_.append(w[lo:lo + 8704])
        ss_.append(s[lo:lo + 8704])
    return torch.cat(ws_, 0).contiguous(), torch.cat(ss_, 0).float().contiguous()


def neq(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a.view(torch.int16) != b.view(torch.int16)).sum().item())


def main() -> int:
    base = load_ext(BASE_GATEUP, "k100_v3b_retained")
    old = load_ext(OLD_DIRECTLDS, "directlds_old")
    new = load_ext(NEW_FUSED, "directlds_fused_v3")
    ref_silu = load_lightop_silu()

    if "FUSED_GATEUP_V4" not in new.identity_fused_gateup():
        raise SystemExit(f"[REFUSING] unexpected fused identity {new.identity_fused_gateup()}")

    torch.cuda.set_device(0)
    rows, all_exact = [], True

    for rank in RANKS:
        w_cpu, ws_cpu = load_member(rank)
        for seed in range(SEEDS):
            torch.manual_seed(1000 * rank + seed)
            # real quantized activations from per-token int8 quant
            xf = torch.randn((4, 5120), device=DEV, dtype=torch.bfloat16) * 0.5
            xq, xs = per_token_quant_int8(xf)
            xq, xs = xq.contiguous(), xs.float().reshape(-1).contiguous()

            w = w_cpu.to(DEV)
            ws = ws_cpu.to(DEV)
            pw_old = None
            # old DirectLDS packing: reuse its own check_combo pack (byte-identical pack_x4)
            from pack_gateup_fused import pack_x4, row_permutation
            pw_old = pack_x4(w).to(DEV)
            idx = row_permutation()
            pw_new, ws_new = pack_gateup_fused(w_cpu, ws_cpu)
            pw_new, ws_new = pw_new.to(DEV), ws_new.to(DEV)

            # A0 baseline retained GEMM + real LightOp silu
            a0 = ref_silu(base.matmul(xq, w, xs, ws, 4, 4))
            # A1 old DirectLDS + real LightOp silu
            a1 = ref_silu(old.gateup(xq, pw_old, xs, ws))
            # A2 new DirectLDS fused
            a2 = new.fused_gateup(xq, pw_new, xs, ws_new)
            torch.cuda.synchronize()

            d_new_base = neq(a2, a0)
            d_old_base = neq(a1, a0)
            all_exact &= (d_new_base == 0 and d_old_base == 0)
            rows.append({"tp_rank": rank, "seed": seed,
                         "fused_vs_retained_diff": d_new_base,
                         "old_directlds_vs_retained_diff": d_old_base,
                         "shape": list(a2.shape)})
            print(json.dumps(rows[-1]), flush=True)
            del w, ws, pw_old, pw_new, ws_new
            torch.cuda.empty_cache()

        if not all_exact:
            break

    result = {"seeds": SEEDS, "ranks": list(RANKS), "cases": len(rows),
              "all_exact": all_exact,
              "max_fused_diff": max(r["fused_vs_retained_diff"] for r in rows)}
    if not all_exact:
        print(json.dumps({"verdict": "REJECT_NOT_BIT_EXACT", **result}, indent=2))
        return 1

    # ------------------- timing only after 0-diff on every case -------------------
    w_cpu, ws_cpu = load_member(0)
    from pack_gateup_fused import pack_x4, row_permutation
    torch.manual_seed(777)
    xf = torch.randn((4, 5120), device=DEV, dtype=torch.bfloat16) * 0.5
    xq, xs = per_token_quant_int8(xf)
    xq, xs = xq.contiguous(), xs.float().reshape(-1).contiguous()
    w = w_cpu.to(DEV); ws = ws_cpu.to(DEV)
    pw_old = pack_x4(w).to(DEV)
    pw_new, ws_new = pack_gateup_fused(w_cpu, ws_cpu)
    pw_new, ws_new = pw_new.to(DEV), ws_new.to(DEV)

    arms = {
        "A0_retained_plus_silu": lambda: ref_silu(base.matmul(xq, w, xs, ws, 4, 4)),
        "A1_old_directlds_plus_silu": lambda: ref_silu(old.gateup(xq, pw_old, xs, ws)),
        "A2_new_directlds_fused": lambda: new.fused_gateup(xq, pw_new, xs, ws_new),
    }
    for _ in range(50):
        for fn in arms.values():
            fn()
    torch.cuda.synchronize()

    def timeit(fn, iters=500):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ev[0].record()
        for _ in range(iters):
            fn()
        ev[1].record(); torch.cuda.synchronize()
        return ev[0].elapsed_time(ev[1]) / iters * 1000.0  # us

    reps = {k: [] for k in arms}
    for _ in range(5):                      # interleaved repeats, no arm ordering bias
        for k, fn in arms.items():
            reps[k].append(timeit(fn))
    med = {k: sorted(v)[len(v) // 2] for k, v in reps.items()}
    result["timing_us_median"] = med
    result["delta_us"] = {"A1_minus_A2": med["A1_old_directlds_plus_silu"] - med["A2_new_directlds_fused"],
                          "A0_minus_A2": med["A0_retained_plus_silu"] - med["A2_new_directlds_fused"]}
    result["timing_note"] = ("DirectLDS and fused epilogue share ONE GEMM bucket; report these deltas, "
                             "never sum them as two independent percentages. Silu cap counted once; "
                             "no host-launch credit.")
    print(json.dumps({"verdict": "PASS_BIT_EXACT", **result}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

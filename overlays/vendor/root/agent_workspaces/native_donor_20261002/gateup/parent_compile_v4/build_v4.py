#!/usr/bin/env python3
"""Build the v3 fused-DirectLDS-gateup HIP extension (COMPILE ONLY, no GPU).

Pins gfx928 and MAX_JOBS=2, never imports torch.cuda, never launches a kernel,
never deploys. The parent agent compiles this in an isolated CPU container.
Nothing here reports success unless load() actually returns.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "candidate_family_directlds_combo_fused_gateup_v4.hip"
OUT = HERE / "build_v4"
OUT.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MAX_JOBS", "2")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx928")

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

if not SRC.exists():
    sys.exit(f"[v3 build] missing source: {SRC}")

try:
    mod = load(
        name="q38_smallm_directlds_fused_gateup_v4",
        sources=[str(SRC)],
        build_directory=str(OUT),
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "-fno-gpu-rdc", "--offload-arch=gfx928"],
        with_cuda=True,
        verbose=True,
    )
except Exception as exc:  # fail loudly; never print a fake compiled state
    sys.exit(f"[v3 build] COMPILE FAILED: {type(exc).__name__}: {exc}")

dst = HERE / "q38_smallm_directlds_fused_gateup_v4.so"
shutil.copy2(mod.__file__, dst)
print(f"[v3 build] compiled artifact: {dst}")
print("[v3 build] NOTE: no GPU kernel was launched and nothing was deployed.")

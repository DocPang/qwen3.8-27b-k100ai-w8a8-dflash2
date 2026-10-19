"""Import-safe entry point for the isolated TP2 DFlash2 q8 candidate.

Do not import Torch/FA/LightOp from interpreter helpers: tvm_ffi's
torch C-DLPack probe forks a Python helper that itself imports sitecustomize.
That would recurse indefinitely if the q8 GPU patch loads eagerly.
"""
from __future__ import annotations

import os
import runpy

try:
    _cmd = open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
except OSError:
    _cmd = ""

_HELPER_NAMES = (
    "_build_optional_torch_c_dlpack.py",
    "multiprocessing.resource_tracker",
    "ninja --version",
)
if any(name in _cmd for name in _HELPER_NAMES):
    # Helper process must not initialize GPU kernels.
    pass
else:
    _BASE = "/data/qwen38-27b-k100ai-int8-opt/experiments/dflash_hip_target_sampling_20261012/batch2_raw_q8_overlay/q8_batch2_v6.py"
    if not os.path.isfile(_BASE):
        raise RuntimeError(f"Missing pinned read-only q8 source: {_BASE}")
    runpy.run_path(_BASE, run_name="__q38_dflash2_q8_guarded__")

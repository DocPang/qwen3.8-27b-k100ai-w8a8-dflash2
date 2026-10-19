"""C0 diagnostic adapter: TVM-FFI helper bypass + correctness-only W8A8 shim.

This file intentionally contains no performance optimization. It exists only to
prevent sitecustomize recursion in TVM-FFI/ninja helper subprocesses while
retaining the exact compressed-tensors ignore-name compatibility used by the
official new-base W8A8 baseline.
"""
from __future__ import annotations

import os

try:
    _cmd = open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
except Exception:
    _cmd = ""

_HELPERS = (
    "tvm_ffi/utils/_build_optional_torch_c_dlpack.py",
    "_build_optional_torch_c_dlpack.py",
    "multiprocessing.resource_tracker",
    "/usr/local/bin/ninja",
    " ninja --version",
)

if any(x in _cmd for x in _HELPERS):
    print(f"[C0 helper bypass] pid={os.getpid()} cmd={_cmd[:180]}", flush=True)
else:
    from typing import Iterable
    from sglang.srt.layers.quantization.compressed_tensors import utils as _u

    def _normalize(name: str) -> str:
        if name.startswith("model.language_model."):
            return "model." + name[len("model.language_model."):]
        if name.startswith("model.visual."):
            return "visual." + name[len("model.visual."):]
        return name

    def _exact_match(layer_name: str, targets: Iterable[str]) -> bool:
        layer_name = _normalize(layer_name)
        for target in targets:
            if target.startswith("re:"):
                if _u._is_equal_or_regex_match(layer_name, target):
                    return True
            elif _u._is_equal_or_regex_match(layer_name, _normalize(target)):
                return True
        return False

    _u.check_equal_or_regex_match = _exact_match
    print("[C0 correctness shim] compressed-tensors exact/regex ignore compatibility enabled", flush=True)

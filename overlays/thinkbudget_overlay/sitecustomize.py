import os
import runpy

try:
    with open("/proc/self/cmdline", "rb") as fh:
        _cmd = fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
except OSError:
    _cmd = ""

_HELPER_NAMES = (
    "_build_optional_torch_c_dlpack.py",
    "multiprocessing.resource_tracker",
    "ninja --version",
)
if any(name in _cmd for name in _HELPER_NAMES):
    pass
else:
    _BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pr79_thinkbudget.py")
    if not os.path.isfile(_BASE):
        raise RuntimeError("Missing PR79 overlay: " + _BASE)
    runpy.run_path(_BASE, run_name="__q38_pr79_top__")

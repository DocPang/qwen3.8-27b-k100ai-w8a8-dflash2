"""Q38 integration v7: DirectLDS down overlay on top of integration v6.

Why this exists
---------------
integration v6 installs a complete pinned-source overlay but *deliberately*
keeps the old tiled `k100_int8_smallm_k8704_v1_tiled` down route.  That omits
the already-measured NON-XLDS DirectLDS down (559.09us/cycle from the modeled
combo).  v7 is an INDEPENDENT complement: it never edits v6 files, it runpy-
chains v6, then adds exactly one additional route.

What v7 adds (and nothing else)
-------------------------------
  * `CompressedTensorsW8A8Int8.process_weights_after_loading` is wrapped so the
    M4/K8704/N5120 down_proj is pre-packed (real `pack_x4_tiles` bytes) during
    model init, i.e. BEFORE graph capture.  Packing inside a capture is refused.
  * the module seam `sglang...compressed_tensors.quant_ops.triton_scaled_mm` is
    wrapped again for the exact M4/K8704/N5120 down call only; it calls
    `down.matmul(xq, pw, xs, ws, 2, 4)` on the measured SO and returns the result
    straight back to the seam.  Every other shape delegates to v6's wrapper,
    which in turn delegates to the retained parent.

Authority: the pinned-image sources under `pinned_source/` (real reduce flags,
real LightOp quant) plus the measured DirectLDS down experiment
`experiments/stage_c1_p1_smallm_directlds_20261002_r1/`.  The down SO is
`raw/q38_k8704_directlds_gemm_v1.so` -- NEVER the XLDS variant, and raw `.t()` is
never handed to the packed kernel.

CPU/source only.  No GPU launch, no model request, no service mutation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = os.getenv("Q38_V7_ROOT", os.getenv("Q38_V6_ROOT", "/data/qwen38-27b-k100ai-int8-opt"))

_V6 = os.getenv(
    "Q38_V7_V6",
    f"{_ROOT}/agent_workspaces/native_donor_20261002/integration/parent_v8/sitecustomize.py",
)
_DOWN_SO = os.getenv(
    "Q38_V7_DOWN_SO",
    f"{_ROOT}/experiments/stage_c1_p1_smallm_directlds_20261002_r1/raw/"
    "q38_k8704_directlds_gemm_v1.so",
)
_ROUTING = os.getenv("Q38_V7_ROUTING", str(_HERE / "down_routing_v7.py"))
_MANIFEST = os.getenv("Q38_V7_MANIFEST", str(_HERE / "manifest_hashes.json"))

_LAYERS = 64
_HITS = {"down": 0}
_DOWN_PACKED: dict = {}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_manifest() -> dict:
    """Verify every measured hash on disk.  Never invents a hash."""
    with open(_MANIFEST) as f:
        man = json.load(f)
    checked = {}
    for name, rec in man["files"].items():
        path = rec["path"].format(root=_ROOT, here=str(_HERE))
        want = rec["sha256"]
        if not want or want == "UNSET":
            raise RuntimeError(f"[Q38 V7] manifest entry {name} has no measured sha256")
        required = bool(rec.get("required_at_runtime", True))
        if not os.path.exists(path):
            if required:
                raise RuntimeError(f"[Q38 V7] manifest file missing: {path}")
            print(f"[Q38 V7] audit-only manifest entry absent, not fatal: {name}", flush=True)
            continue
        got = _sha256(path)
        if got != want:
            if required:
                raise RuntimeError(f"[Q38 V7] hash drift for {name}: {got} != {want}")
            print(f"[Q38 V7] audit-only manifest entry drift: {name}", flush=True)
            continue
        checked[name] = got
    return checked


def _load_ext(path: str):
    """Load a torch extension by its real PyInit basename (aliases fail)."""
    name = Path(path).stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"[Q38 V7] cannot load extension: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"[Q38 V7] cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1) chain v6 (which itself runpy-chains the retained M1 full-head parent).
#    Force the v6 gate on so v7 is self-sufficient; v6 keeps its own helper-
#    process guard, so builder helpers (ninja / dlpack) stay untouched.
# ---------------------------------------------------------------------------
if os.getenv("Q38_INTEGRATION_V7", "0") == "1":
    os.environ.setdefault("Q38_INTEGRATION_V6", "1")

runpy.run_path(_V6, run_name="__q38_integration_v7_v6__")

try:
    _CMD = (
        open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
    )
except Exception:
    _CMD = ""

_HELPERS = (
    "tvm_ffi/utils/_build_optional_torch_c_dlpack.py",
    "_build_optional_torch_c_dlpack.py",
    "multiprocessing.resource_tracker",
    "/usr/local/bin/ninja",
    " ninja --version",
)

if os.getenv("Q38_INTEGRATION_V7", "0") == "1" and not any(m in _CMD for m in _HELPERS):
    import torch

    from sglang.srt.layers.quantization.compressed_tensors import quant_ops as _ops
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
        CompressedTensorsW8A8Int8,
    )

    _verify_manifest()

    # v6 already installed its seam; capture it BEFORE we replace it.
    _v6_scaled_mm = _ops.triton_scaled_mm

    _routing = _load_module(_ROUTING, "q38_down_routing_v7")

    _down = _load_ext(_DOWN_SO)
    if not callable(getattr(_down, "matmul", None)):
        raise RuntimeError(f"[Q38 V7] down SO has no matmul(): {_DOWN_SO}")
    _ident = getattr(_down, "identity", None)
    if not callable(_ident) or _ident() != _routing.EXPECTED_DOWN_ID:
        raise RuntimeError(
            f"[Q38 V7] down SO identity drift: "
            f"{_ident() if callable(_ident) else '<no identity()>'}"
        )

    # ------------------------------------------------------------- down pack
    def _weight_of(mod):
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor):
            return w
        w = getattr(w, "data", None)
        return w if isinstance(w, torch.Tensor) else None

    def _down_scale(layer):
        ws = getattr(layer, "weight_scale", None)
        if not isinstance(ws, torch.Tensor) or ws.numel() != _routing.DOWN_N:
            return None
        return ws.to(torch.float32).reshape(-1).contiguous()

    def _build_down(w, ws):
        w_nk = _routing.normalise_down_nk(w.detach())
        packed_cpu = _routing.pack_x4_tiles(w_nk.to("cpu"))
        packed = packed_cpu.contiguous().to(w.device)
        ws_dev = ws.to(torch.float32).reshape(-1).contiguous().to(w.device)
        return packed, ws_dev

    def _prebuild_down(layer):
        w = _weight_of(layer)
        key = _routing.cache_key(w)
        if key in _DOWN_PACKED:
            return
        ws = _down_scale(layer)
        if ws is None:
            raise RuntimeError(
                f"[Q38 V7] down: weight_scale must be fp32 with exactly "
                f"{_routing.DOWN_N} entries; refusing to build a DirectLDS pack"
            )
        _DOWN_PACKED[key] = _build_down(w, ws)
        print(
            f"[Q38 V7] prebuilt down pack: +{_routing.down_cache_bytes() / 2**20:.1f} "
            f"MiB this layer, ~{_routing.down_cache_bytes_per_rank(_LAYERS) / 2**30:.3f} "
            f"GiB/rank over {_LAYERS} layers",
            flush=True,
        )

    _v6_pwal = CompressedTensorsW8A8Int8.process_weights_after_loading

    def _process_weights_after_loading_v7(self, layer):
        _v6_pwal(self, layer)
        w = _weight_of(layer)
        if w is None or w.dtype is not torch.int8:
            return
        lw = getattr(layer, "logical_widths", None)
        if getattr(self,"w8a8_strategy",0)==1 and _routing.down_member_of(tuple(w.shape), lw):
            _prebuild_down(layer)

    CompressedTensorsW8A8Int8.process_weights_after_loading = (
        _process_weights_after_loading_v7
    )

    # --------------------------------------------------------------- down seam
    def _down_scale_from_seam(scale_b):
        ws = scale_b
        if not isinstance(ws, torch.Tensor) or ws.numel() != _routing.DOWN_N:
            raise RuntimeError(
                f"[Q38 V7] down seam: scale_b must have exactly "
                f"{_routing.DOWN_N} entries"
            )
        return ws.to(torch.float32).reshape(-1).contiguous()

    def _v7_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None, best_config=None):
        if _routing.is_exact_down_call(a, b, scale_a, scale_b, out_dtype, bias):
            key = _routing.cache_key(b)
            got = _DOWN_PACKED.get(key)
            if got is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "[Q38 V7] down pack is missing while a CUDA graph is "
                        "being captured; it must be built in "
                        "process_weights_after_loading before capture. "
                        "Refusing to pack inside capture."
                    )
                _DOWN_PACKED[key] = _build_down(b, _down_scale_from_seam(scale_b))
                got = _DOWN_PACKED[key]
            packed, ws = got
            out = _down.matmul(
                a, packed, scale_a.reshape(-1).contiguous(), ws, 2, 4
            )
            _HITS["down"] += 1
            if _HITS["down"] <= 16:
                print(
                    f"[Q38 V7] hit down #{_HITS['down']}: M4 K8704->N5120 "
                    "DirectLDS (pack_x4_tiles, matmul(...,2,4))",
                    flush=True,
                )
            return out
        # gate/up fused, gdn, out, retained NON-XLDS fallbacks: v6 chain.
        return _v6_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias, best_config)

    _ops.triton_scaled_mm = _v7_scaled_mm

    print(
        "[Q38 V7] installed: DirectLDS down overlay on v6 (M4/K8704/N5120 "
        "exact only); v6 fused gate/up + GDN/out + retained parent untouched; "
        f"down cache ~{_routing.down_cache_bytes_per_rank(_LAYERS) / 2**30:.3f} "
        "GiB/rank",
        flush=True,
    )

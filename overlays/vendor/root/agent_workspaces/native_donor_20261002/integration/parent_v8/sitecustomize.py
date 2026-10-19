"""Q38 integration v6: COMPLETE isolated runtime overlay from the PINNED image sources.

What this replaces
------------------
integration v5 was written against the wrong source tree: it assumed
`Qwen2MoeMLP.forward(self, x)` had no allreduce flags and that
`per_token_quant_int8` lived in `sglang.kernels.ops.quantization.int8_kernel`.
The parent extracted the ACTUAL pinned-image sources to
`agent_workspaces/native_donor_20261002/pinned_source/`.  In those authoritative
files:

  * `sglang/srt/models/qwen2_moe.py` `Qwen2MoeMLP.forward` DOES accept
    `should_allreduce_fusion` and `use_reduce_scatter` and forwards them into
    `down_proj(..., skip_all_reduce=...)`.  v5's D2/D6/D8 verdicts are void.
  * `sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py`
    imports `per_token_quant_int8` from **`lightop.quant`**.  The
    `sglang.kernels...` module v5 imported does not exist in the pinned image.
  * `sglang/srt/layers/quantization/compressed_tensors/quant_ops.py` exists and
    is the real seam: the scheme's `apply_weights` calls
    `ops.triton_scaled_mm(...)` on that module object, and every retained
    `runtime_patch_spec_gain_*` overlay patches exactly that attribute.

What v6 does
------------
Retains the M1 full-head parent chain, then installs one exact-shape overlay:

  * M4/K5120/N17408 gate/up + SiLU becomes ONE fused DirectLDS GEMM that
    returns the already-activated [4, 8704] BF16 tensor straight to `down_proj`
    (gateup/v4 `fused_gateup`, real compiled SO, identity FUSED_GATEUP_V4).
  * M4/K5120/N8192 GDN and M4/K3072/N5120 linear-attention out_proj are routed
    to the SAME family SO's `gdn` / `outproj` DirectLDS entries using REAL
    `pack_x4` byte packing, never `raw b.t()` handed to a packed entry.
  * M4/K8704/N5120 down_proj deliberately keeps the already-measured NON-XLDS
    `k100_int8_smallm_k8704_v1_tiled` route from the retained parent chain; the
    DirectLDS/XLDS down variant is NOT re-routed here.
  * Every pack is built in `process_weights_after_loading` (model init), i.e.
    BEFORE graph capture, and the overlay refuses to pack inside a capture.
  * Prefill / MTP / any other shape falls through to the retained parent.

CPU/source only.  No GPU launch, no model request, no service mutation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import runpy
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = os.getenv("Q38_V6_ROOT", "/data/qwen38-27b-k100ai-int8-opt")

_PARENT = os.getenv(
    "Q38_V6_PARENT",
    f"{_ROOT}/runtime_patch_spec_gain_fullhead10347_m1_v1_20260926/sitecustomize.py",
)
_FAMILY_SO = os.getenv(
    "Q38_V6_FAMILY_SO",
    f"{_ROOT}/agent_workspaces/native_donor_20261002/gateup/v4/"
    "q38_smallm_directlds_fused_gateup_v4.so",
)
_PACK_HELPER = os.getenv("Q38_V6_PACK_HELPER", str(_HERE / "pack_v6.py"))
_MANIFEST = os.getenv("Q38_V6_MANIFEST", str(_HERE / "manifest_hashes.json"))

_FAMILY_IDENTITY = "H-SMALLM-W-DIRECTLDS-XW-OVERLAP-V1:FAMILY:GATEUP+GDN+OUT"
_FAMILY_IDENTITY_FUSED = "H-SMALLM-W-DIRECTLDS-XW-OVERLAP-V1:FAMILY:GATEUP+GDN+OUT:FUSED_GATEUP_V4"

_GU_N, _GU_K, _GU_HALF = 17408, 5120, 8704
_GD_N, _GD_K = 8192, 5120
_OT_N, _OT_K = 5120, 3072
_DOWN_N, _DOWN_K = 5120, 8704
_LAYERS = 64

# Extra device bytes per layer per member (packed int8 W + fp32 ws buffer).
_CACHE_PER_LAYER = {
    "gateup": _GU_N * _GU_K + _GU_N * 4,
    "gdn": _GD_N * _GD_K + _GD_N * 4,
    "out": _OT_N * _OT_K + _OT_N * 4,
}
_CACHE_PER_RANK = sum(_CACHE_PER_LAYER.values()) * _LAYERS

_PACKED: dict = {}
_HITS = {"gateup_fused": 0, "gdn": 0, "out": 0}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest() -> dict:
    with open(_MANIFEST) as f:
        return json.load(f)


def _verify_manifest() -> dict:
    """Verify every measured hash on disk.  Never invents a hash."""
    man = _manifest()
    checked = {}
    for name, rec in man["files"].items():
        if not rec.get("required_at_runtime",False):
            continue
        path = rec["path"].format(root=_ROOT, here=str(_HERE))
        want = rec["sha256"]
        if not want or want == "UNSET":
            raise RuntimeError(f"[Q38 V6] manifest entry {name} has no measured sha256")
        required = bool(rec.get("required_at_runtime", True))
        if not os.path.exists(path):
            if required:
                raise RuntimeError(f"[Q38 V6] manifest file missing: {path}")
            print(f"[Q38 V6] audit-only manifest entry absent, not fatal: {name}", flush=True)
            continue
        got = _sha256(path)
        if got != want:
            if required:
                raise RuntimeError(f"[Q38 V6] hash drift for {name}: {got} != {want}")
            print(f"[Q38 V6] audit-only manifest entry drift: {name}", flush=True)
            continue
        checked[name] = got
    return checked


def _load_ext(path: str):
    """Load a torch extension by its real PyInit basename (aliases fail)."""
    name = Path(path).stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"[Q38 V6] cannot load extension: {path}")
    mod = importlib.util.module_from_spec(spec)
    import sys as _sys

    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_per_token_quant_int8():
    """The pinned scheme's REAL quantizer: `from lightop.quant import ...`."""
    try:
        from lightop.quant import per_token_quant_int8

        return per_token_quant_int8
    except Exception as exc:  # pragma: no cover - install-time environment probe
        raise RuntimeError(
            "[Q38 V6] lightop.quant.per_token_quant_int8 is unavailable; the "
            "pinned compressed_tensors_w8a8_int8 scheme imports exactly that "
            "symbol.  Refusing to substitute a different quantizer."
        ) from exc


runpy.run_path(_PARENT, run_name="__q38_integration_v6_parent__")

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

if os.getenv("Q38_INTEGRATION_V6", "0") == "1" and not any(m in _CMD for m in _HELPERS):
    import torch

    from sglang.srt.layers.quantization.compressed_tensors import quant_ops as _ops
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
        CompressedTensorsW8A8Int8,
    )
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

    per_token_quant_int8 = _import_per_token_quant_int8()

    _verify_manifest()

    _fam = _load_ext(_FAMILY_SO)
    for _name in ("fused_gateup", "gdn", "outproj", "identity", "identity_fused_gateup"):
        if not callable(getattr(_fam, _name, None)):
            raise RuntimeError(f"[Q38 V6] family SO missing {_name}(): {_FAMILY_SO}")
    if _fam.identity() != _FAMILY_IDENTITY:
        raise RuntimeError(f"[Q38 V6] family identity drift: {_fam.identity()}")
    if _fam.identity_fused_gateup() != _FAMILY_IDENTITY_FUSED:
        raise RuntimeError(
            f"[Q38 V6] fused identity drift: {_fam.identity_fused_gateup()}"
        )

    _pack = _load_ext(_PACK_HELPER)
    for _name in ("pack_x4", "pack_gateup_fused", "pack_member", "normalise_nk", "member_for_shape"):
        if not callable(getattr(_pack, _name, None)):
            raise RuntimeError(f"[Q38 V6] pack helper missing {_name}(): {_PACK_HELPER}")

    _orig_forward = Qwen2MoeMLP.forward
    _orig_pwal = CompressedTensorsW8A8Int8.process_weights_after_loading

    # ---------------------------------------------------------------- packing
    def _weight_of(mod):
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor):
            return w
        w = getattr(w, "data", None)
        return w if isinstance(w, torch.Tensor) else None

    def _key(mod, member):
        w = _weight_of(mod)
        return (member, id(w), int(w.data_ptr()), tuple(w.shape))

    def _scales_of(mod, member, n):
        ws = getattr(mod, "weight_scale", None)
        if not isinstance(ws, torch.Tensor):
            return None
        ws = ws.reshape(-1)
        if ws.numel() != n:
            return None
        return ws.to(torch.float32)

    def _member_of_layer(layer):
        """DirectLDS member for a quantized linear layer, or None."""
        w = _weight_of(layer)
        if w is None or w.dtype is not torch.int8:
            return None
        lw = getattr(layer, "logical_widths", None)
        return _pack.member_for_shape(tuple(w.shape), lw)

    def _prebuild(layer, member):
        """Pack one layer's DirectLDS weights + permuted scales on the host."""
        key = _key(layer, member)
        if key in _PACKED:
            return
        w = _weight_of(layer)
        if member == "gateup":
            n, k = _GU_N, _GU_K
        elif member == "gdn":
            n, k = _GD_N, _GD_K
        else:
            n, k = _OT_N, _OT_K
        ws = _scales_of(layer, member, n)
        if ws is None:
            raise RuntimeError(
                f"[Q38 V6] {member}: weight_scale must be fp32 with exactly {n} "
                "entries; refusing to build a DirectLDS pack"
            )
        w_nk = _pack.normalise_nk(w.detach(), n, k)
        if member == "gateup":
            packed_cpu, ws_dev_cpu = _pack.pack_gateup_fused(w_nk.to("cpu"), ws.to("cpu"))
        else:
            packed_cpu = _pack.pack_member(member, w_nk.to("cpu"))
            ws_dev_cpu = ws.to("cpu").contiguous()
        packed = packed_cpu.contiguous().to(w.device)
        ws_dev = ws_dev_cpu.contiguous().to(w.device)
        _PACKED[key] = (packed, ws_dev)
        print(
            f"[Q38 V6] prebuilt {member} pack: +{_CACHE_PER_LAYER[member] / 2**20:.1f} "
            f"MiB this layer, ~{_CACHE_PER_RANK / 2**30:.2f} GiB/rank over {_LAYERS} layers",
            flush=True,
        )

    def _lookup(layer, member):
        key = _key(layer, member)
        got = _PACKED.get(key)
        if got is not None:
            return got
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"[Q38 V6] {member} pack is missing while a CUDA graph is being "
                "captured; packing must happen in process_weights_after_loading "
                "before capture. Refusing to pack inside capture."
            )
        _prebuild(layer, member)
        return _PACKED[key]

    def _process_weights_after_loading_v6(self, layer):
        _orig_pwal(self, layer)
        member = _member_of_layer(layer)
        if member is not None and getattr(self,"w8a8_strategy",0)==1:
            _prebuild(layer, member)

    CompressedTensorsW8A8Int8.process_weights_after_loading = _process_weights_after_loading_v6

    # ------------------------------------------------------------- MLP forward
    def _target_mlp_contract(self, x):
        gu = getattr(self, "gate_up_proj", None)
        down = getattr(self, "down_proj", None)
        gu_w = _weight_of(gu) if gu is not None else None
        gu_ws = _scales_of(gu, "gateup", _GU_N) if gu is not None else None
        return bool(
            isinstance(x, torch.Tensor)
            and x.ndim == 2
            and tuple(x.shape) == (4, 5120)
            and x.dtype is torch.bfloat16
            and x.is_contiguous()
            and gu is not None
            and down is not None
            and isinstance(gu_w, torch.Tensor)
            and gu_w.dtype is torch.int8
            and tuple(gu_w.shape) in ((_GU_N, _GU_K), (_GU_K, _GU_N))
            and gu_ws is not None
            and gu_ws.dtype is torch.float32
            # the pinned dynamic per-token scheme has no static input scale
            and getattr(gu, "input_scale", None) is None
            and getattr(getattr(gu,"scheme",None),"input_symmetric",False)
            and not getattr(getattr(gu,"scheme",None),"is_static_input_scheme",True)
            and int(getattr(getattr(gu, "scheme", None), "w8a8_strategy", 0) or 0) == 1
            and int(getattr(down, "tp_size", -1)) == 2
            and bool(getattr(down, "input_is_parallel", False))
            and getattr(down, "bias", None) is None
            and gu_w.device == x.device
        )

    def _integration_v6_forward(
        self,
        x,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        """Signature mirrors the pinned Qwen2MoeMLP.forward exactly."""
        if not _target_mlp_contract(self, x):
            # Prefill, MTP, every other shape: retained parent path unchanged.
            return _orig_forward(
                self,
                x,
                should_allreduce_fusion=should_allreduce_fusion,
                use_reduce_scatter=use_reduce_scatter,
            )
        gu = self.gate_up_proj
        down = self.down_proj
        gu_packed, gu_ws_perm = _lookup(gu, "gateup")
        xq, xs = per_token_quant_int8(x)
        activated = _fam.fused_gateup(
            xq, gu_packed, xs.reshape(-1).contiguous(), gu_ws_perm
        )
        if activated.dtype is not torch.bfloat16 or tuple(activated.shape) != (4, _GU_HALF):
            raise RuntimeError(
                f"[Q38 V6] fused_gateup returned {tuple(activated.shape)} "
                f"{activated.dtype}; refusing to hand a wrong shape to down_proj"
            )
        _HITS["gateup_fused"] += 1
        if _HITS["gateup_fused"] <= 16:
            print(
                f"[Q38 V6] hit gateup_fused #{_HITS['gateup_fused']}: "
                "M4 K5120->[4,8704] SiLU straight into down8704",
                flush=True,
            )
        # Exactly the pinned tail: down_proj with the same skip_all_reduce flag,
        # and down_proj keeps the retained parent's NON-XLDS K8704 route.
        out, _ = down(
            activated,
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
        )
        return out

    Qwen2MoeMLP.forward = _integration_v6_forward

    # ------------------------------------------------------------- quant seam
    _parent_scaled_mm = _ops.triton_scaled_mm

    def _route(member, a, b, scale_a, scale_b, out_dtype, bias, best_config):
        if member == "gdn":
            n, k = _GD_N, _GD_K
            entry = _fam.gdn
        else:
            n, k = _OT_N, _OT_K
            entry = _fam.outproj
        exact = bool(
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.ndim == 2
            and b.ndim == 2
            and tuple(a.shape) == (4, k)
            and tuple(b.shape) == (k, n)
            and a.dtype is torch.int8
            and b.dtype is torch.int8
            and out_dtype is torch.bfloat16
            and bias is None
            and a.is_contiguous()
            and b.t().is_contiguous()
            and getattr(scale_a, "numel", lambda: 0)() >= 4
            and getattr(scale_b, "numel", lambda: -1)() == n
        )
        if not exact:
            return None
        # b is the stored [K, N] int8 parameter; pack from its real [N, K] view.
        if member == "gdn":
            key = ("gdn", id(b), int(b.data_ptr()), tuple(b.shape))
        else:
            key = ("out", id(b), int(b.data_ptr()), tuple(b.shape))
        got = _PACKED.get(key)
        if got is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"[Q38 V6] {member} pack missing during capture; "
                    "process_weights_after_loading must have prebuilt it"
                )
            ws = scale_b.reshape(-1).to(torch.float32)
            packed = _pack.pack_member(member, _pack.normalise_nk(b.detach(), n, k).to("cpu"))
            got = (packed.contiguous().to(b.device), ws.contiguous().to(b.device))
            _PACKED[key] = got
        packed, ws = got
        out = entry(a, packed, scale_a.reshape(-1).contiguous(), ws)
        _HITS[member] += 1
        if _HITS[member] <= 8:
            print(
                f"[Q38 V6] hit {member} #{_HITS[member]}: M4 K{k}->N{n} DirectLDS",
                flush=True,
            )
        return out

    def _v6_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None, best_config=None):
        for member in ("gdn", "out"):
            routed = _route(member, a, b, scale_a, scale_b, out_dtype, bias, best_config)
            if routed is not None:
                return routed
        # down_proj (4,8704)x(8704,5120) and every other shape: parent chain.
        return _parent_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias, best_config)

    _ops.triton_scaled_mm = _v6_scaled_mm

    print(
        "[Q38 V6] installed: fused DirectLDS gate/up+SiLU, DirectLDS GDN/out "
        f"(pack_x4), retained NON-XLDS K8704 down, M1 fullhead parent; "
        f"packed cache budget ~{_CACHE_PER_RANK / 2**30:.2f} GiB/rank",
        flush=True,
    )

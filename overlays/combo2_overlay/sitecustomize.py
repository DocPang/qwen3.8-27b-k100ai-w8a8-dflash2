"""Compose the two retained HCU2/3 candidates into ONE arm (runtime overlay only).

Retained candidate 1 (decode): bit-exact fused GDN in-proj split
    $D/gdn_fuse_overlay/sitecustomize.py (+ $D/gdn_split_fused.py)
Retained candidate 2 (prefill): ported production token-major GDN prefill stack
    $D/prefill_gdn_overlay/sitecustomize.py (+ gdn8192 tile pin + aiter backport)

Both children chain the lane baseline `tail_overlay/sitecustomize.py` themselves.
Chaining it twice would double-apply its patches, so while the children run we
temporarily replace runpy.run_path and turn the duplicate tail_overlay chain into
a no-op.  Everything else is delegated unchanged, so each child keeps its own gate
file, SHA pin and fail-closed guards.

Fail-closed: inert without combo2_cfg.txt; inert if any required path is missing;
one env/gate switch reverts (prefix -> tail_overlay, gates -> 0).
"""
import os
import runpy

_EXP = "/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_w8a8_tune_20261013"
_ON = f"{_EXP}/combo2_cfg.txt"
_TAIL = f"{_EXP}/tail_overlay/sitecustomize.py"
_GDN = f"{_EXP}/gdn_fuse_overlay/sitecustomize.py"
_PRE = f"{_EXP}/prefill_gdn_overlay/sitecustomize.py"
_REQUIRED = (_TAIL, _GDN, _PRE, f"{_EXP}/gdn_split_fused.py")


def _say(m):
    print("[Q38 COMBO2] " + m, flush=True)


if not os.path.exists(_ON):
    _say("inert (no %s)" % _ON)
else:
    _missing = [p for p in _REQUIRED if not os.path.exists(p)]
    if _missing:
        _say("inert: missing %s" % _missing)
    else:
        _real = runpy.run_path
        _tail_abs = os.path.abspath(_TAIL)

        def _guarded(path, init_globals=None, run_name=None):
            try:
                same = os.path.abspath(path) == _tail_abs
            except Exception:
                same = False
            if same:
                _say("skipping duplicate tail_overlay chain")
                return {}
            return _real(path, init_globals, run_name)

        try:
            _real(_TAIL, run_name="__q38_combo2_tail__")
            _say("tail_overlay chained once")
        except Exception as exc:
            _say(f"baseline chain failed: {exc}")

        runpy.run_path = _guarded
        try:
            for _label, _path in (("gdn_fuse", _GDN), ("prefill_gdn", _PRE)):
                try:
                    _real(_path, run_name="__q38_combo2_%s__" % _label)
                except Exception as exc:
                    _say(f"{_label} chain failed: {exc}")
        finally:
            runpy.run_path = _real

        _say("installed: composed arm (bit-exact fused GDN split + production GDN prefill stack)")

"""Modern TP2 DFlash2 narrow page-lifecycle compatibility layer (isolated).

The historical v30 stack repaired req-to-token writes, retained page
ownership, reclamation and native int64 assignment for speculative Verify.
Keep modern combined_v9 unchanged, then install the legacy narrow patches
in their original order with no old TP4 math parents.
"""
from __future__ import annotations
import os
import runpy

_ROOT = "/data/qwen38-dflash2-k100ai"
_MODERN = "/data/qwen38-27b-k100ai-int8-opt/agent_workspaces/native_donor_20261002/integration/combined_v9/sitecustomize.py"

runpy.run_path(_MODERN, run_name="__q38_dflash2_pagefix_modern_parent__")
os.environ["SGLANG_Q38_TP124_V30_SKIP_TP4_PARENT"] = "1"

_PATCHES = (
    ("runtime_patch_dflash_tp4_assign_verify_fix_v2/sitecustomize.py", "assign_verify"),
    ("runtime_patch_dflash_tp4_retained_alloc_fix_v3/sitecustomize.py", "retained_alloc"),
    ("runtime_patch_dflash_tp4_page_reclaim_fix_v4/sitecustomize.py", "page_reclaim"),
    ("runtime_patch_dflash_tp4_abort_nocache_v8/sitecustomize.py", "abort_nocache"),
    ("runtime_patch_dflash_tp4_verify_stockalloc_v9/sitecustomize.py", "verify_stockalloc"),
    ("runtime_patch_dflash_tp4_prepare_native_i64_v29/sitecustomize.py", "prepare_native_i64"),
)
for relative, tag in _PATCHES:
    path = f"{_ROOT}/{relative}"
    if not os.path.isfile(path):
        raise RuntimeError(f"Required v30 pagefix {tag} missing: {path}")
    runpy.run_path(path, run_name=f"__q38_pagefix_{tag}__")

print("[Q38 MODERN DFLASH2 PAGEFIX] exact six audited v30 lifecycle deltas installed; modern M4/W8A8 overlay retained",flush=True)

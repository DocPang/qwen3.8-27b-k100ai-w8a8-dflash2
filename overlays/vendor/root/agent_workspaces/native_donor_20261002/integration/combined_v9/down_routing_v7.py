#!/usr/bin/env python3
"""Q38 integration v7 down routing helpers (CPU only, no device work).

This module holds the PURE, importable pieces of the down overlay so the CPU
interface probe can exercise the real predicates and the real packing function
without executing the full runtime overlay (which needs the pinned sglang image).

`pack_x4_tiles` is copied VERBATIM from the measured DirectLDS down experiment
`experiments/stage_c1_p1_smallm_directlds_20261002_r1/work/check_gemm_correctness.py`
(module constants and the body are character-for-character the same), so the
overlay and the CPU proof exercise the SAME generation function the measured
correctness run used, instead of a second, possibly-different formula.

Nothing here touches a GPU, a service, or a model request.
"""
from __future__ import annotations

import torch

# ---- exact down member contract (measured experiment) -----------------------
#   check_gemm_correctness.py:  M, K, N = 4, 8704, 5120
#                               KC, KCS, NT, GRID = 512, 17, 2, 2560
DOWN_M, DOWN_K, DOWN_N = 4, 8704, 5120
KC, KCS, NT, GRID = 512, 17, 2, 2560

# The retained tiled kernel and the DirectLDS candidate share this identity.
EXPECTED_DOWN_ID = (
    "H-SMALLM-W-DIRECTLDS-V1:M4:K8704:N5120:N2:RW4:K512:X_RETAINED:"
    "W_PACKED_X4_DIRECTLDS"
)
DOWN_SO_MODULE = "q38_k8704_directlds_gemm_v1"


def pack_x4_tiles(w: torch.Tensor) -> torch.Tensor:
    # VERBATIM from check_gemm_correctness.py (measured DirectLDS down pack).
    # Logical tile order is [grid, k_chunk, nt=2, 512B].  Convert to i32 and
    # apply the inverse Q001 x4 placement permutation inside each 16-dword cohort.
    logical = (
        w.view(DOWN_N, KCS, KC)
        .view(GRID, NT, KCS, KC)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(torch.int32)
        .view(GRID, KCS, 16, 16)
    )
    source_for_dest = [
        ((d & 3) << 2) | (((d >> 2) & 1) << 1) | ((d >> 3) & 1)
        for d in range(16)
    ]
    inverse = [0] * 16
    for dest, source in enumerate(source_for_dest):
        inverse[source] = dest
    return logical[..., inverse].contiguous().view(torch.int8)


def normalise_down_nk(b: torch.Tensor) -> torch.Tensor:
    """Return a contiguous [N, K] int8 copy of the stored down weight.

    The measured checker stored the down parameter as [N, K] = [5120, 8704] and
    `pack_x4_tiles` consumed it directly.  A [K, N] orientation is also accepted
    and transposed back; anything else fails closed.  `raw .t()` is NEVER handed
    to the packed kernel -- it is only used to recover the [N, K] logical matrix
    before the measured pack runs.
    """
    if not isinstance(b, torch.Tensor) or b.dtype is not torch.int8:
        raise ValueError(f"down weight must be int8, got {type(b)}")
    shape = tuple(b.shape)
    if shape == (DOWN_N, DOWN_K):
        return b if b.is_contiguous() else b.contiguous()
    if shape == (DOWN_K, DOWN_N):
        return b.t().contiguous()
    raise ValueError(
        f"down weight orientation is neither ({DOWN_N},{DOWN_K}) nor "
        f"({DOWN_K},{DOWN_N}): {shape}"
    )


def down_member_of(weight_shape, logical_widths=None) -> bool:
    """True when a quantized linear layer is the M4/K8704/N5120 down_proj."""
    shape = tuple(weight_shape)
    if shape not in ((DOWN_N, DOWN_K), (DOWN_K, DOWN_N)):
        return False
    if logical_widths is None:
        return True
    return list(logical_widths) == [DOWN_N]


def is_exact_down_call(a, b, scale_a, scale_b, out_dtype, bias) -> bool:
    """Exact-shape gate for the runtime `quant_ops.triton_scaled_mm` seam.

    Mirrors the measured contract: a=[4,8704] int8, b=[5120,8704]/[8704,5120]
    int8, out_dtype=bfloat16, bias=None, per-token scale_a (>=4), channel
    scale_b with exactly 5120 fp32 entries.  Everything else falls through.
    """
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return False
    if a.ndim != 2 or b.ndim != 2:
        return False
    if tuple(a.shape) != (DOWN_M, DOWN_K):
        return False
    if tuple(b.shape) not in ((DOWN_N, DOWN_K), (DOWN_K, DOWN_N)):
        return False
    if a.dtype is not torch.int8 or b.dtype is not torch.int8:
        return False
    if out_dtype is not torch.bfloat16:
        return False
    if bias is not None:
        return False
    if not a.is_contiguous():
        return False
    if getattr(scale_a, "numel", lambda: 0)() < DOWN_M:
        return False
    if getattr(scale_b, "numel", lambda: -1)() != DOWN_N:
        return False
    if getattr(scale_b, "dtype", None) is not torch.float32:
        return False
    if a.device != b.device:
        return False
    return True


def cache_key(b: torch.Tensor) -> tuple:
    """Stable identity of the FINAL stored weight tensor (post pwal)."""
    return (id(b), int(b.data_ptr()), tuple(b.shape))


def down_cache_bytes() -> int:
    """Extra device bytes per layer (packed int8 W + fp32 ws)."""
    return DOWN_N * DOWN_K + DOWN_N * 4


def down_cache_bytes_per_rank(layers: int = 64) -> int:
    return down_cache_bytes() * layers

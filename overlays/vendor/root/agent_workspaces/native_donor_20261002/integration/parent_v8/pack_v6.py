#!/usr/bin/env python3
"""Q38 integration v6 real packing helpers (CPU only, no device work).

`pack_x4` is copied byte-for-byte from the measured DirectLDS combo experiment
(`experiments/stage_c1_p1_directlds_combo_20261002_r1/work/check_combo_correctness.py`)
including that file's per-member CFG, so the runtime overlay and the CPU proof
exercise the SAME generation function instead of a second, possibly-correct
formula.  The gate/up fused entry additionally needs the paired-row permutation
from `gateup/v4/pack_gateup_fused.py`: each CTA owns logical rows
{g[2q], g[2q+1], u[2q], u[2q+1]}.

Nothing here touches a GPU, a service, or a model request.
"""
from __future__ import annotations

import torch

# Exactly the CFG table of check_combo_correctness.py (N, K, KC, NT, KCS, units).
MEMBERS = {
    "gateup": {"K": 5120, "N": 17408, "KC": 1024, "NT": 4, "KCS": 5, "units": 4},
    "gdn": {"K": 5120, "N": 8192, "KC": 512, "NT": 2, "KCS": 10, "units": 1},
    "out": {"K": 3072, "N": 5120, "KC": 1024, "NT": 2, "KCS": 3, "units": 2},
}

GU_N, GU_HALF, GU_K = 17408, 8704, 5120
GU_NT = 4
GU_GRID = GU_N // GU_NT

_SRC_FOR_DEST = [((d & 3) << 2) | (((d >> 2) & 1) << 1) | ((d >> 3) & 1) for d in range(16)]
_INV = [0] * 16
for _dest, _src in enumerate(_SRC_FOR_DEST):
    _INV[_src] = _dest


def pack_x4(member: str, w_nk: torch.Tensor) -> torch.Tensor:
    """check_combo_correctness.pack_x4, byte-for-byte.

    w_nk: [N, K] int8 row-major logical weight.
    Returns the RAW DirectLDS packed tensor, shape
    (N//NT, KCS, NT, 16, 64) int8 -- i.e. exactly what the measured experiment
    handed to the kernel (the runner flattened it).
    """
    c = MEMBERS[member]
    N, K, NT, KC, KCS = c["N"], c["K"], c["NT"], c["KC"], c["KCS"]
    if w_nk.dtype is not torch.int8:
        raise ValueError(f"pack_x4[{member}]: weight must be int8, got {w_nk.dtype}")
    if tuple(w_nk.shape) != (N, K):
        raise ValueError(f"pack_x4[{member}]: weight must be [{N},{K}], got {tuple(w_nk.shape)}")
    if not w_nk.is_contiguous():
        raise ValueError(f"pack_x4[{member}]: weight must be contiguous")
    grid = N // NT
    logical = (
        w_nk.view(N, KCS, KC)
        .view(grid, NT, KCS, KC)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(grid, KCS, c["units"], 1024)
        .view(torch.int32)
        .view(grid, KCS, c["units"], 16, 16)
    )
    return logical[..., _INV].contiguous().view(torch.int8)


def unpack_x4(member: str, packed: torch.Tensor) -> torch.Tensor:
    """Exact inverse of pack_x4; CPU proof only, never used at runtime."""
    c = MEMBERS[member]
    N, K, NT, KC, KCS = c["N"], c["K"], c["NT"], c["KC"], c["KCS"]
    grid = N // NT
    if packed.dtype is not torch.int8 or packed.numel() != N * K:
        raise ValueError(f"unpack_x4[{member}]: want int8 with {N * K} bytes, got {packed.dtype} {packed.numel()}")
    perm = packed.contiguous().view(grid, KCS, c["units"], 16, 16, 4).view(torch.int32).view(
        grid, KCS, c["units"], 16, 16
    )
    logical = torch.empty_like(perm)
    for d in range(16):
        logical[..., d] = perm[..., _SRC_FOR_DEST[d]]
    return (
        logical.contiguous()
        .view(torch.int8)
        .view(grid, KCS, c["units"], 1024)
        .view(grid, KCS, NT, KC)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(N, K)
    )


def pair_n_col(q2: int, nt: int) -> int:
    """Logical gate/up column owned by packed slot nt of CTA (q2 / 2)."""
    return (q2 + nt) if nt < 2 else (GU_HALF + q2 + (nt - 2))


def packed_slot_row(b: int, nt: int) -> int:
    return pair_n_col(b * 2, nt)


def row_permutation() -> torch.Tensor:
    """idx[b*NT+nt] = logical gate/up column placed at packed position b*NT+nt."""
    return torch.tensor(
        [packed_slot_row(b, nt) for b in range(GU_GRID) for nt in range(GU_NT)],
        dtype=torch.long,
    )


def inverse_row_permutation() -> torch.Tensor:
    """inv[c] = packed position of logical column c (exact inverse)."""
    inv = torch.empty(GU_N, dtype=torch.long)
    idx = row_permutation()
    inv[idx] = torch.arange(GU_N, dtype=torch.long)
    return inv


def output_columns() -> torch.Tensor:
    """Logical activated column produced by CTA b, slot j (j in 0..1)."""
    return torch.tensor(
        [b * 2 + j for b in range(GU_GRID) for j in range(2)], dtype=torch.long
    )


def pack_gateup_fused(w_nk: torch.Tensor, ws: torch.Tensor):
    """(packed_flat_int8, ws_perm_fp32) for the v4 fused_gateup entry.

    w_nk: [17408, 5120] int8 in LOGICAL gate(0..8703)+up(8704..17407) order.
    ws:   [17408] fp32 per-output-channel scale in the same logical order.
    """
    if tuple(w_nk.shape) != (GU_N, GU_K) or w_nk.dtype is not torch.int8:
        raise ValueError(f"gateup fused packing needs int8 [{GU_N},{GU_K}], got {tuple(w_nk.shape)} {w_nk.dtype}")
    if not w_nk.is_contiguous():
        raise ValueError("gateup fused packing needs a contiguous [17408,5120] weight")
    ws_flat = ws.reshape(-1)
    if ws_flat.numel() != GU_N or ws_flat.dtype is not torch.float32:
        raise ValueError(f"gateup fused packing needs fp32 ws[{GU_N}], got {tuple(ws.shape)} {ws.dtype}")
    idx = row_permutation()
    w_perm = w_nk.index_select(0, idx).contiguous()
    ws_perm = ws_flat.index_select(0, idx).contiguous()
    return pack_x4("gateup", w_perm).reshape(-1), ws_perm


def normalise_nk(b: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Return a contiguous [N, K] int8 view/copy of a stored W8A8 weight.

    The pinned `CompressedTensorsW8A8Int8.process_weights_after_loading` stores
    the parameter as [K, N] (the transpose of a contiguous [N, K] matrix) on the
    triton path and as [N, K] on the blaslt path.  Both are accepted and refused
    when neither holds.
    """
    if not isinstance(b, torch.Tensor) or b.dtype is not torch.int8:
        raise ValueError(f"weight must be an int8 tensor, got {type(b)}")
    shape = tuple(b.shape)
    if shape == (N, K):
        return b if b.is_contiguous() else b.contiguous()
    if shape == (K, N):
        return b.t().contiguous()
    raise ValueError(f"weight orientation is neither ({N},{K}) nor ({K},{N}): {shape}")


def member_for_shape(shape, logical_widths=None):
    """Map a stored W8A8 parameter shape (+ logical widths) to a DirectLDS member.

    `logical_widths` is the `output_partition_sizes` list the pinned
    CompressedTensorsW8A8Int8.create_weights stores on the layer.  Requiring it
    keeps the prebuild from packing an unrelated parameter that merely happens
    to share a shape (the runtime seam still matches on shape, exactly like the
    retained overlays do).

      gateup: MergedColumnParallelLinear [intermediate]*2 -> [8704, 8704]
      gdn:    qkvz MergedColumnParallelLinear [k,k,v,v]  -> length 4, N 8192
      out:    linear-attn out_proj RowParallelLinear     -> length 1, N 5120
    """
    shape = tuple(shape)
    lw = None if logical_widths is None else [int(x) for x in logical_widths]
    if shape in ((GU_N, GU_K), (GU_K, GU_N)) and lw == [GU_HALF, GU_HALF]:
        return "gateup"
    if shape in ((8192, 5120), (5120, 8192)) and lw is not None and len(lw) == 4:
        return "gdn"
    if shape in ((5120, 3072), (3072, 5120)) and lw == [5120]:
        return "out"
    return None


def pack_member(member: str, b: torch.Tensor) -> torch.Tensor:
    """Flat int8 packed buffer for one DirectLDS member from its stored weight."""
    c = MEMBERS[member]
    w_nk = normalise_nk(b, c["N"], c["K"])
    return pack_x4(member, w_nk).reshape(-1)


def cache_bytes_per_layer():
    """Extra device bytes per layer for each prebuilt pack (int8 W + fp32 ws)."""
    return {m: c["N"] * c["K"] + c["N"] * 4 for m, c in MEMBERS.items()}

#!/usr/bin/env python3
"""Real packing helper for the v3 fused DirectLDS gate/up kernel.

Two steps, in this order:
  1. ROW PERMUTATION: logical gate/up columns are reordered so that CTA b owns
     packed slots {g[2b], g[2b+1], u[2b], u[2b+1]} (see pair_n_col). ws is
     permuted with the identical index vector.
  2. DIRECTLDS PACK: the byte/nibble interleave is byte-for-byte the SAME
     pack_x4 used by check_combo_correctness.py for the existing DirectLDS
     candidate. Nothing about the kernel's contiguous packed-W read changes.

Not CPU-only-safe by itself (imports torch), but performs no device work.
"""
from __future__ import annotations

import torch

N, HALF, K, KC, NT, KCS = 17408, 8704, 5120, 1024, 4, 5
GRID, UNITS = N // NT, NT


def pair_n_col(q2: int, nt: int) -> int:
    """Logical column owned by packed slot nt of CTA (q2/2)."""
    return (q2 + nt) if nt < 2 else (HALF + q2 + (nt - 2))


def packed_slot_row(b: int, nt: int) -> int:
    """Logical column that CTA b must hold in packed slot nt."""
    return pair_n_col(b * 2, nt)


def row_permutation() -> torch.Tensor:
    """idx[b*NT+nt] = logical column for packed position b*NT+nt."""
    return torch.tensor([packed_slot_row(b, nt)
                         for b in range(GRID) for nt in range(NT)], dtype=torch.long)


def inverse_row_permutation() -> torch.Tensor:
    """inv[c] = packed position of logical column c (exact inverse of the above)."""
    inv = torch.empty(N, dtype=torch.long)
    idx = row_permutation()
    inv[idx] = torch.arange(N, dtype=torch.long)
    return inv


def pack_x4(w: torch.Tensor) -> torch.Tensor:
    """Byte-identical to check_combo_correctness.pack_x4 for gateup."""
    logical = (w.reshape(N, KCS, KC).reshape(GRID, NT, KCS, KC)
               .permute(0, 2, 1, 3).contiguous().view(GRID, KCS, UNITS, 1024)
               .view(torch.int32).view(GRID, KCS, UNITS, 16, 16))
    src_for_dest = [((d & 3) << 2) | (((d >> 2) & 1) << 1) | ((d >> 3) & 1) for d in range(16)]
    inv = [0] * 16
    for dest, src_i in enumerate(src_for_dest):
        inv[src_i] = dest
    return logical[..., inv].contiguous().view(torch.int8)


def unpack_x4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_x4 (used by the CPU test to prove the byte round-trip)."""
    logical = (packed.reshape(GRID, KCS, UNITS, 1024).view(torch.int32)
               .view(GRID, KCS, UNITS, 16, 16))
    src_for_dest = [((d & 3) << 2) | (((d >> 2) & 1) << 1) | ((d >> 3) & 1) for d in range(16)]
    out = torch.empty_like(logical)
    # pack maps L[s] -> packed[src_for_dest[s]], so L[s] = packed[src_for_dest[s]]
    for s in range(16):
        out[..., s] = logical[..., src_for_dest[s]]
    return (out.contiguous().view(torch.int8).view(GRID, KCS, UNITS, 1024)
            .permute(0, 2, 1, 3).contiguous().view(N, K))


def pack_gateup_fused(w: torch.Tensor, ws: torch.Tensor):
    """Return (packed_w_int8, ws_perm_fp32) for v3 fused_gateup.

    w: [17408,5120] int8 (logical gate rows 0..8703 then up rows 8704..17407)
    ws: [17408] fp32 per-output-channel scale, logical column order.
    """
    if tuple(w.shape) != (N, K) or w.dtype != torch.int8:
        raise ValueError(f"w must be int8 [{N},{K}], got {tuple(w.shape)} {w.dtype}")
    if ws.numel() != N or ws.dtype != torch.float32:
        raise ValueError(f"ws must be fp32 [{N}], got {tuple(ws.shape)} {ws.dtype}")
    idx = row_permutation()
    w_perm = w.index_select(0, idx).contiguous()
    ws_perm = ws.reshape(-1).index_select(0, idx).contiguous()
    return pack_x4(w_perm).reshape(-1), ws_perm

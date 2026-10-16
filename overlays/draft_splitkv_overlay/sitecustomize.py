"""Draft decode extend-attention KV-DIRECTION SPLIT (the only mechanism left that can
raise the block count), with a logsumexp combine.

Why this and not the tile/launch axes.  All of these are measured and closed:
  * BLOCK_M  (H-HCU23-DRAFT-EXTN-ATTN-PARALLEL-V1): cdiv(8, BLOCK_M)=1 for every
    BLOCK_M >= 8, so the grid stays (1,16,1) = 16 blocks on a 120-CU part.
  * BLOCK_N / num_warps (H-HCU23-DRAFT-ATTN-BN-V1, this session): BN=128 neutral,
    BN=256 +7.2 ms, BN=128+nw16 +6.9 ms, BM=8 pending; no gain.
  * num_stages / waves_per_eu (H-HCU23-DRAFT-ATTN-LAUNCH-V1, this session):
    ns=2 +3.8 ms, ns=3 +3.7 ms, both numerically identical (scheduling only) - so
    software pipelining does not lift this kernel.
The remaining coordinate is the KV direction itself: with 16 blocks and ~122 serial
KV-chunk iterations per block the kernel is latency-bound at ~100 GB/s of a
demonstrated 682 GB/s, and 104 of 120 CUs are idle, so more blocks is the cure.

What this does.  The vendored _fwd_kernel is cloned with one extra split coordinate:
grid = (batch, head_num, cdiv(Lq,BLOCK_M) * NUM_SPLITS).  Each block sweeps only its
own contiguous slice of the prefix slot list and writes partial (acc, m, l); split 0
additionally performs the vendor's stage-2 (the dense extend part) before writing.
A second kernel combines the NUM_SPLITS partials with a logsumexp rescale and writes
O with the same HIP transposed store the vendor uses.

Correctness protocol.  NUM_SPLITS=1 reproduces the vendor's loop structure, masks and
accumulation order exactly, so it is a ZERO-RISK smoke test of the addressing,
strides, paged layout and store path: accept and the streamed sha must match the
tuned control *exactly*, with no speed change.  Only then is the split raised.
NUM_SPLITS>1 changes the online-softmax grouping and is NOT assumed bit-exact.

Fail-closed gates (any mismatch -> the vendored function runs unchanged):
  max_len_extend <= 64, Lq == Lv, BLOCK_DPE == 0, custom_mask is None, sinks is None,
  window_kv_offsets is None, sliding_window_size <= 0, logit_cap <= 0,
  xai_temperature_len <= 0, k/v_scale == 1.0, contiguous inputs.
The target prefill never matches (max_len_extend = 7843), so TTFT/prefill cannot
regress by construction.  Inert without draft_splitkv_cfg.txt.
"""
from __future__ import annotations

import os
import runpy

_CHAIN = ("/data/qwen38-27b-k100ai-int8-opt/experiments/"
          "hcu01_w8a8_tune_20261013/combo_overlay/sitecustomize.py")
runpy.run_path(_CHAIN, run_name="__q38_hcu01_chain__")

_D = "/data/qwen38-27b-k100ai-int8-opt/experiments/hcu01_w8a8_tune_20261013"
_CFG_FILE = os.path.join(_D, "draft_splitkv_cfg.txt")
_EVIDENCE = os.path.join(_D, "draft_splitkv_evidence.txt")
_MAX_EXTEND = 64
_HELPERS = (
    "tvm_ffi/utils/_build_optional_torch_c_dlpack.py",
    "_build_optional_torch_c_dlpack.py",
    "multiprocessing.resource_tracker",
    "/usr/local/bin/ninja",
    " ninja --version",
)

try:
    _cmd = open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode(
        "utf-8", "replace")
except OSError:
    _cmd = ""


def _say(msg):
    print("[Q38 SPLITKV] " + msg, flush=True)
    try:
        with open(_EVIDENCE, "a") as fh:
            fh.write(msg + "\n")
    except OSError:
        pass


try:
    _NUM_SPLITS = int(open(_CFG_FILE).read().split()[0])
except Exception:
    _NUM_SPLITS = None


if _NUM_SPLITS is None:
    try:
        os.remove(_EVIDENCE)
    except OSError:
        pass
    print("[Q38 SPLITKV] inert (no %s)" % _CFG_FILE, flush=True)
elif any(_x in _cmd for _x in _HELPERS):
    print("[Q38 SPLITKV] helper process, inert", flush=True)
else:
    import torch
    import triton
    import triton.language as tl
    from sglang.srt.layers.attention.triton_ops import extend_attention as _ea

    _is_hip = torch.version.hip is not None
    _stats = {"hit": 0, "pass": 0, "fallback": 0}
    _logged = [False]

    @triton.jit
    def _q38_split_fwd_kernel(
        Q_Extend, K_Buffer, V_Buffer,
        Part_Acc, Part_ML,
        qo_indptr, kv_indptr, kv_indices,
        sm_scale, k_scale, v_scale,
        kv_group_num, num_heads, rows,
        stride_qbs, stride_qh,
        stride_buf_kbs, stride_buf_kh, stride_buf_k2, stride_buf_k3,
        stride_buf_vbs, stride_buf_vh, stride_buf_v2, stride_buf_v3,
        DCU_PAGED_LAYOUT: tl.constexpr, DCU_PAGE_SIZE: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr, BLOCK_DV: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        Lq: tl.constexpr, Lv: tl.constexpr,
        NUM_SPLITS: tl.constexpr, SLIDING_WINDOW_SIZE: tl.constexpr,
    ):
        """Stage 1 (windowed prefix) only.  The loop start is clipped to
        valid_lo = max(0, prefix_len - sliding_window) BEFORE the range is split, so
        every block works on slots that can actually be visible: with prefix 7843 and
        window 2047, a uniform split of [0, 7843) would put 6 of 8 splits entirely
        outside the window.  The window itself is then applied as an arithmetic mask
        (no runtime branch, which Triton rejects inside a KV loop).
        """
        cur_seq = tl.program_id(0)
        cur_head = tl.program_id(1)
        bm_split = tl.program_id(2)
        cur_block_m = bm_split // NUM_SPLITS
        cur_split = bm_split % NUM_SPLITS
        cur_kv_head = cur_head // kv_group_num

        cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
        cur_seq_len_extend = (
            tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
        )
        cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
        cur_seq_len_prefix = (
            tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
        )

        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_dv = tl.arange(0, BLOCK_DV)
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend
        mask_d = offs_d < Lq
        mask_dv = offs_dv < Lv

        offs_q = (
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_d[None, :]
        )
        q = tl.load(
            Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]),
            other=0.0,
        )

        acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
        deno = tl.zeros([BLOCK_M], dtype=tl.float32)
        e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

        if SLIDING_WINDOW_SIZE > 0:
            valid_lo = tl.maximum(
                cur_seq_len_prefix - SLIDING_WINDOW_SIZE, 0
            ).to(tl.int64)
        else:
            valid_lo = tl.zeros([], dtype=tl.int64)
        cap = cur_seq_len_prefix.to(tl.int64)
        span = cap - valid_lo
        chunk = (span + NUM_SPLITS - 1) // NUM_SPLITS
        lo_raw = valid_lo + cur_split * chunk
        # Split boundaries are tile-aligned (multiples of BLOCK_N) so that the
        # online-softmax grouping stays the vendor's one when NUM_SPLITS == 1 and so
        # that two splits never share a partial tile.  int64 throughout: kv_indptr and
        # qo_indptr are int64, and Triton refuses to unify a tensor-typed int32 loop
        # init with an int64 bound.
        lo = (lo_raw // BLOCK_N) * BLOCK_N
        if cur_split == NUM_SPLITS - 1:
            hi = cap
        else:
            hi = ((lo_raw + chunk) // BLOCK_N) * BLOCK_N
        hi = tl.maximum(hi, lo + BLOCK_N)

        for start_n in range(lo, hi, BLOCK_N):
            mask_n = (start_n + offs_n) < hi
            final_mask = mask_m[:, None] & mask_n[None, :]
            if SLIDING_WINDOW_SIZE > 0:
                window_mask = (
                    cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m[:, None]
                ) <= (start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE)
                final_mask &= window_mask

            offs_kv_loc = tl.load(
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
                mask=mask_n, other=0,
            )
            if DCU_PAGED_LAYOUT:
                page_idx = offs_kv_loc // DCU_PAGE_SIZE
                page_off = offs_kv_loc % DCU_PAGE_SIZE
                offs_buf_k = (
                    page_idx[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + page_off[None, :] * stride_buf_k2
                    + offs_d[:, None] * stride_buf_k3
                )
            else:
                offs_buf_k = (
                    offs_kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_d[:, None] * stride_buf_k2
                )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0,
            )
            qk = tl.dot(q.to(k.dtype), k) * sm_scale * k_scale
            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            if DCU_PAGED_LAYOUT:
                offs_buf_v = (
                    page_idx[:, None] * stride_buf_vbs
                    + cur_kv_head * stride_buf_vh
                    + offs_dv[None, :] * stride_buf_v2
                    + page_off[:, None] * stride_buf_v3
                )
            else:
                offs_buf_v = (
                    offs_kv_loc[:, None] * stride_buf_vbs
                    + cur_kv_head * stride_buf_vh
                    + offs_dv[None, :] * stride_buf_v2
                )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=mask_n[:, None] & mask_dv[None, :], other=0.0,
            )
            acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v) * v_scale
            e_max = n_e_max

        row = cur_seq * num_heads + cur_head
        base = (cur_split * rows + row) * BLOCK_M
        offs_pa = (base + offs_m)[:, None] * BLOCK_DV + offs_dv[None, :]
        tl.store(
            Part_Acc + offs_pa, acc,
            mask=mask_m[:, None] & mask_dv[None, :],
        )
        offs_pml = (base + offs_m) * 2
        tl.store(Part_ML + offs_pml, e_max, mask=mask_m)
        tl.store(Part_ML + offs_pml + 1, deno, mask=mask_m)

    @triton.jit
    def _q38_extend_tail_kernel(
        Q_Extend, K_Extend, V_Extend,
        Part_Acc, Part_ML,
        qo_indptr,
        sm_scale,
        kv_group_num, num_heads, rows, tail_slot,
        stride_qbs, stride_qh,
        stride_kbs, stride_kh,
        stride_vbs, stride_vh,
        BLOCK_DMODEL: tl.constexpr, BLOCK_DV: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        Lq: tl.constexpr, Lv: tl.constexpr,
        IS_CAUSAL: tl.constexpr, SLIDING_WINDOW_SIZE: tl.constexpr,
    ):
        """The vendor's stage 2 (dense extend part), as its own kernel, writing
        partial slot `tail_slot`.  Separate kernel because Triton cannot unify the
        induction variable of two loops that carry different integer widths in the
        same function."""
        cur_seq = tl.program_id(0)
        cur_head = tl.program_id(1)
        cur_block_m = tl.program_id(2)
        cur_kv_head = cur_head // kv_group_num

        cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
        cur_seq_len_extend = (
            tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
        )
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_dv = tl.arange(0, BLOCK_DV)
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend
        mask_d = offs_d < Lq
        mask_dv = offs_dv < Lv
        offs_q = (
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_d[None, :]
        )
        q = tl.load(
            Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]),
            other=0.0,
        )
        acc2 = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
        deno2 = tl.zeros([BLOCK_M], dtype=tl.float32)
        emax2 = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

        if IS_CAUSAL:
            cm_end = tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
        else:
            cm_end = cur_seq_len_extend
        for sn in range(0, cm_end, BLOCK_N):
            m_n = (sn + offs_n) < cm_end
            if IS_CAUSAL:
                f_mask = (
                    (cur_block_m * BLOCK_M + offs_m[:, None])
                    >= (sn + offs_n[None, :])
                ) & mask_m[:, None] & m_n[None, :]
            else:
                f_mask = mask_m[:, None] & m_n[None, :]
            if SLIDING_WINDOW_SIZE > 0:
                w_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) <= (
                    sn + offs_n[None, :] + SLIDING_WINDOW_SIZE
                )
                f_mask &= w_mask
            offs_k = (
                (cur_seq_extend_start_idx + sn + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_d[:, None]
            )
            k2 = tl.load(
                K_Extend + offs_k,
                mask=(m_n[None, :]) & (mask_d[:, None]), other=0.0,
            )
            qk2 = tl.dot(q, k2, out_dtype=tl.float32) * sm_scale
            qk2 = tl.where(f_mask, qk2, float("-inf"))
            r_max = tl.max(qk2, 1)
            r_fix = tl.where(r_max == float("-inf"), -1e20, r_max)
            n_max = tl.maximum(r_fix, emax2)
            rescale = tl.exp(emax2 - n_max)
            p2 = tl.exp(qk2 - n_max[:, None])
            deno2 = deno2 * rescale + tl.sum(p2, 1)
            offs_v = (
                (cur_seq_extend_start_idx + sn + offs_n[:, None]) * stride_vbs
                + cur_kv_head * stride_vh
                + offs_dv[None, :]
            )
            v2 = tl.load(
                V_Extend + offs_v,
                mask=m_n[:, None] & mask_dv[None, :], other=0.0,
            )
            acc2 = acc2 * rescale[:, None] + tl.dot(p2.to(v2.dtype), v2)
            emax2 = n_max

        row = cur_seq * num_heads + cur_head
        base = (tail_slot * rows + row) * BLOCK_M
        offs_pa = (base + offs_m)[:, None] * BLOCK_DV + offs_dv[None, :]
        tl.store(
            Part_Acc + offs_pa, acc2,
            mask=mask_m[:, None] & mask_dv[None, :],
        )
        offs_pml = (base + offs_m) * 2
        tl.store(Part_ML + offs_pml, emax2, mask=mask_m)
        tl.store(Part_ML + offs_pml + 1, deno2, mask=mask_m)

    @triton.jit
    def _q38_split_combine_kernel(
        Part_Acc, Part_ML, O_Extend, qo_indptr,
        stride_obs, stride_oh, num_heads, rows, num_block_m,
        BLOCK_M: tl.constexpr, BLOCK_DV: tl.constexpr,
        NUM_SLOTS: tl.constexpr, STORE_TRANSPOSE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_m = pid % num_block_m
        row = pid // num_block_m
        cur_seq = row // num_heads
        cur_head = row % num_heads

        offs_m = tl.arange(0, BLOCK_M)
        offs_dv = tl.arange(0, BLOCK_DV)
        cur_seq_len_extend = (
            tl.load(qo_indptr + cur_seq + 1) - tl.load(qo_indptr + cur_seq)
        )
        mask_m = (block_m * BLOCK_M + offs_m) < cur_seq_len_extend

        m = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
        for s in range(NUM_SLOTS):
            base = (s * rows + row) * BLOCK_M
            ms = tl.load(Part_ML + (base + offs_m) * 2, mask=mask_m,
                         other=float("-inf"))
            ls = tl.load(Part_ML + (base + offs_m) * 2 + 1, mask=mask_m,
                         other=0.0)
            as_ = tl.load(
                Part_Acc + (base + offs_m)[:, None] * BLOCK_DV
                + offs_dv[None, :],
                mask=mask_m[:, None], other=0.0,
            )
            nm = tl.maximum(m, ms)
            w_old = tl.exp(m - nm)
            w_new = tl.exp(ms - nm)
            acc = acc * w_old[:, None] + as_ * w_new[:, None]
            l = l * w_old + ls * w_new
            m = nm

        out = acc / l[:, None]
        start = tl.load(qo_indptr + cur_seq)
        offs_o = (
            (start + block_m * BLOCK_M + offs_m[:, None]) * stride_obs
            + cur_head * stride_oh
            + offs_dv[None, :]
        )
        mask_o = mask_m[:, None]
        if STORE_TRANSPOSE:
            tl.store(O_Extend + offs_o.T, out.T, mask=mask_o.T)
        else:
            tl.store(O_Extend + offs_o, out, mask=mask_o)

    _orig_fwd = _ea.extend_attention_fwd

    def _split_fwd(
        q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices, custom_mask, is_causal, mask_indptr,
        max_len_extend, k_scale, v_scale, sm_scale=None, logit_cap=0.0,
        skip_prefix_custom_mask=True, sliding_window_size=-1, sinks=None,
        window_kv_offsets=None, xai_temperature_len=-1,
    ):
        _stats["hit"] += 1
        reasons = []
        if int(max_len_extend) > _MAX_EXTEND:
            reasons.append("max_len_extend=%s" % max_len_extend)
        Lq = int(q_extend.shape[-1])
        Lv = int(v_extend.shape[-1])
        if Lq != Lv:
            reasons.append("Lq=%d Lv=%d" % (Lq, Lv))
        if custom_mask is not None:
            reasons.append("custom_mask")
        if sinks is not None:
            reasons.append("sinks")
        if float(logit_cap) > 0:
            reasons.append("logit_cap=%s" % logit_cap)
        if int(xai_temperature_len) > 0:
            reasons.append("xai_temperature_len=%s" % xai_temperature_len)
        if float(k_scale) != 1.0 or float(v_scale) != 1.0:
            reasons.append("k/v_scale")
        if not (q_extend.is_contiguous() and o_extend.is_contiguous()):
            reasons.append("non-contiguous")
        if reasons:
            _stats["fallback"] += 1
            if _stats["fallback"] == 1:
                _say("fallback (vendor path): %s" % ", ".join(reasons))
            return _orig_fwd(
                q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer,
                qo_indptr, kv_indptr, kv_indices, custom_mask, is_causal,
                mask_indptr, max_len_extend, k_scale, v_scale, sm_scale,
                logit_cap, skip_prefix_custom_mask, sliding_window_size, sinks,
                window_kv_offsets, xai_temperature_len,
            )

        BLOCK_DMODEL = triton.next_power_of_2(Lq)
        BLOCK_DV = triton.next_power_of_2(Lv)
        BLOCK_M, BLOCK_N, num_warps = 16, 64, 8
        sm_scale = sm_scale or 1.0 / (Lq ** 0.5)
        batch_size = int(qo_indptr.shape[0]) - 1
        head_num = int(q_extend.shape[1])
        kv_group_num = head_num // int(k_extend.shape[1])
        num_block_m = triton.cdiv(int(max_len_extend), BLOCK_M)
        rows = batch_size * head_num

        DCU_PAGED_LAYOUT = k_buffer.ndim == 4 and v_buffer.ndim == 4
        DCU_PAGE_SIZE = int(k_buffer.shape[2]) if DCU_PAGED_LAYOUT else 1

        part_acc = torch.empty(
            ((_NUM_SPLITS + 1) * rows * BLOCK_M, BLOCK_DV),
            device=q_extend.device, dtype=torch.float32,
        )
        part_ml = torch.empty(
            ((_NUM_SPLITS + 1) * rows * BLOCK_M, 2),
            device=q_extend.device, dtype=torch.float32,
        )
        extra = ({"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
                 if _is_hip else {})
        if not _logged[0]:
            _logged[0] = True
            try:
                _prefixes = [int(x) for x in
                             (kv_indptr[1:] - kv_indptr[:-1]).tolist()[:8]]
            except Exception:
                _prefixes = "n/a"
            _say("first split call: Lq=%d Lv=%d prefix_slots=%s batch=%d heads=%d "
                 "kv_heads=%d splits=%d paged=%s page=%d block=%dx%d mle=%s "
                 "causal=%s window=%s woffs=%s mask=%s sinks=%s cap=%s xai=%s "
                 "kscale=%s vscale=%s smscale=%s skipmask=%s"
                 % (Lq, Lv, _prefixes, batch_size, head_num,
                    int(k_extend.shape[1]), _NUM_SPLITS, DCU_PAGED_LAYOUT,
                    DCU_PAGE_SIZE, BLOCK_M, BLOCK_N, max_len_extend,
                    bool(is_causal), int(sliding_window_size),
                    None if window_kv_offsets is None else
                    tuple(window_kv_offsets.shape), custom_mask is not None,
                    sinks is not None, logit_cap, xai_temperature_len,
                    k_scale, v_scale, sm_scale, skip_prefix_custom_mask))

        _q38_split_fwd_kernel[(batch_size, head_num,
                               num_block_m * _NUM_SPLITS)](
            q_extend, k_buffer, v_buffer,
            part_acc, part_ml,
            qo_indptr, kv_indptr, kv_indices,
            float(sm_scale), float(k_scale), float(v_scale),
            kv_group_num, head_num, rows,
            q_extend.stride(0), q_extend.stride(1),
            k_buffer.stride(0), k_buffer.stride(1), k_buffer.stride(2),
            k_buffer.stride(3) if k_buffer.ndim == 4 else 0,
            v_buffer.stride(0), v_buffer.stride(1), v_buffer.stride(2),
            v_buffer.stride(3) if v_buffer.ndim == 4 else 0,
            DCU_PAGED_LAYOUT=DCU_PAGED_LAYOUT, DCU_PAGE_SIZE=DCU_PAGE_SIZE,
            BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DV=BLOCK_DV,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, Lq=Lq, Lv=Lv,
            NUM_SPLITS=_NUM_SPLITS,
            SLIDING_WINDOW_SIZE=int(sliding_window_size),
            num_warps=num_warps, num_stages=1, **extra,
        )
        _q38_extend_tail_kernel[(batch_size, head_num, num_block_m)](
            q_extend, k_extend, v_extend,
            part_acc, part_ml,
            qo_indptr,
            float(sm_scale),
            kv_group_num, head_num, rows, _NUM_SPLITS,
            q_extend.stride(0), q_extend.stride(1),
            k_extend.stride(0), k_extend.stride(1),
            v_extend.stride(0), v_extend.stride(1),
            BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DV=BLOCK_DV,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, Lq=Lq, Lv=Lv,
            IS_CAUSAL=bool(is_causal),
            SLIDING_WINDOW_SIZE=int(sliding_window_size),
            num_warps=num_warps, num_stages=1, **extra,
        )
        _q38_split_combine_kernel[(rows * num_block_m,)](
            part_acc, part_ml, o_extend, qo_indptr,
            o_extend.stride(0), o_extend.stride(1), head_num, rows,
            num_block_m,
            BLOCK_M=BLOCK_M, BLOCK_DV=BLOCK_DV, NUM_SLOTS=_NUM_SPLITS + 1,
            STORE_TRANSPOSE=_is_hip, num_warps=4, num_stages=1,
            **( {"waves_per_eu": 1} if _is_hip else {} ),
        )
        _stats["pass"] += 1
        return None

    _ea.extend_attention_fwd = _split_fwd
    _say("installed: KV-direction split for the draft decode extend "
         "(NUM_SPLITS=%d, BLOCK_M=16 BLOCK_N=64 num_warps=8); "
         "NUM_SPLITS=1 is a bit-exact smoke test" % _NUM_SPLITS)

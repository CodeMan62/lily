"""Extend (prefill-with-prefix) attention. Mirrors sglang/srt/layers/attention/triton_ops/extend_attention.py.

Each program handles one (sequence, q head, block of extend queries). It first
attends over the cached prefix (gathered through kv_indices), then causally over
the new extend tokens. `custom_mask` hooks in a per-query tree mask (speculative decoding).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q_Extend, K_Extend, V_Extend, O_Extend,
    K_Buffer, V_Buffer,
    qo_indptr, kv_indptr, kv_indices,
    mask_ptr, mask_indptr,
    sm_scale, kv_group_num,
    stride_qbs, stride_qh,
    stride_kbs, stride_kh,
    stride_vbs, stride_vh,
    stride_obs, stride_oh,
    stride_buf_kbs, stride_buf_kh,
    stride_buf_vbs, stride_buf_vh,
    USE_CUSTOM_MASK: tl.constexpr,
    IEEE: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    if cur_block_m * BLOCK_M >= cur_seq_len_extend:
        return

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    offs_q = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q_Extend + offs_q, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    # ---- stage 1: cached prefix -------------------------------------------
    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix
        offs_kv_loc = tl.load(kv_indices + cur_seq_kv_start_idx + start_n + offs_n, mask=mask_n, other=0)

        offs_buf_k = offs_kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[:, None]
        k = tl.load(K_Buffer + offs_buf_k, mask=mask_n[None, :] & mask_d[:, None], other=0.0)

        if IEEE:
            qk = tl.dot(q, k.to(q.dtype), input_precision="ieee")
        else:
            qk = tl.dot(q, k.to(q.dtype))
        qk *= sm_scale

        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
                + start_n + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :], other=0,
            )
            final_mask = mask_m[:, None] & mask_n[None, :] & (custom_mask != 0)
        else:
            final_mask = mask_m[:, None] & mask_n[None, :]
        qk = tl.where(final_mask, qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_buf_v = offs_kv_loc[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + offs_dv[None, :]
        v = tl.load(V_Buffer + offs_buf_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
        if IEEE:
            acc = acc * re_scale[:, None] + tl.dot(p.to(q.dtype), v.to(q.dtype), input_precision="ieee")
        else:
            acc = acc * re_scale[:, None] + tl.dot(p.to(q.dtype), v.to(q.dtype))
        e_max = n_e_max

    # ---- stage 2: new extend tokens, causal --------------------------------
    extend_end = tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    for start_n in range(0, extend_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_extend

        offs_k = (
            (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None]
        )
        k = tl.load(K_Extend + offs_k, mask=mask_n[None, :] & mask_d[:, None], other=0.0)

        if IEEE:
            qk = tl.dot(q, k.to(q.dtype), input_precision="ieee")
        else:
            qk = tl.dot(q, k.to(q.dtype))
        qk *= sm_scale

        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
                + cur_seq_len_prefix + start_n + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :], other=0,
            )
            final_mask = mask_m[:, None] & mask_n[None, :] & (custom_mask != 0)
        else:
            mask_causal = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (start_n + offs_n[None, :])
            final_mask = mask_m[:, None] & mask_n[None, :] & mask_causal
        qk = tl.where(final_mask, qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_v = (
            (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_dv[None, :]
        )
        v = tl.load(V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
        if IEEE:
            acc = acc * re_scale[:, None] + tl.dot(p.to(q.dtype), v.to(q.dtype), input_precision="ieee")
        else:
            acc = acc * re_scale[:, None] + tl.dot(p.to(q.dtype), v.to(q.dtype))
        e_max = n_e_max

    offs_o = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(O_Extend + offs_o, acc / deno[:, None], mask=mask_m[:, None] & mask_dv[None, :])


def extend_attention_fwd(
    q_extend: torch.Tensor,  # [N_ext, H, Dq]
    k_extend: torch.Tensor,  # [N_ext, Hkv, Dq]
    v_extend: torch.Tensor,  # [N_ext, Hkv, Dv]
    o_extend: torch.Tensor,  # [N_ext, H, Dv]
    k_buffer: torch.Tensor,  # [pool, Hkv, Dq]
    v_buffer: torch.Tensor,  # [pool, Hkv, Dv]
    qo_indptr: torch.Tensor,  # [bs + 1]
    kv_indptr: torch.Tensor,  # [bs + 1]
    kv_indices: torch.Tensor,  # [sum(prefix_lens)]
    custom_mask: Optional[torch.Tensor],  # flattened bool/uint8 [sum(ext_i * (prefix_i + ext_i))]
    mask_indptr: Optional[torch.Tensor],  # [bs + 1]
    max_len_extend: int,
    sm_scale: Optional[float] = None,
) -> None:
    Lq, Lv = q_extend.shape[-1], v_extend.shape[-1]
    assert Lq in (16, 32, 64, 128, 256) and Lv in (16, 32, 64, 128, 256), (Lq, Lv)
    BLOCK_DMODEL = triton.next_power_of_2(Lq)
    BLOCK_DV = triton.next_power_of_2(Lv)
    if Lq >= 256:
        BLOCK_M, BLOCK_N = 32, 32
    else:
        BLOCK_M, BLOCK_N = 64, 64
    num_warps = 4 if Lq <= 64 else 8
    num_stages = 1

    sm_scale = sm_scale if sm_scale is not None else 1.0 / (Lq**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]
    use_custom_mask = custom_mask is not None
    ieee = q_extend.dtype == torch.float32

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    _fwd_kernel[grid](
        q_extend, k_extend, v_extend, o_extend,
        k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices,
        custom_mask if use_custom_mask else q_extend,  # dummy pointers when unused
        mask_indptr if use_custom_mask else qo_indptr,
        sm_scale, kv_group_num,
        q_extend.stride(0), q_extend.stride(1),
        k_extend.stride(0), k_extend.stride(1),
        v_extend.stride(0), v_extend.stride(1),
        o_extend.stride(0), o_extend.stride(1),
        k_buffer.stride(0), k_buffer.stride(1),
        v_buffer.stride(0), v_buffer.stride(1),
        USE_CUSTOM_MASK=use_custom_mask,
        IEEE=ieee,
        BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        Lq=Lq, Lv=Lv,
        num_warps=num_warps, num_stages=num_stages,
    )

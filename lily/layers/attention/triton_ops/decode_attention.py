"""Split-KV decode attention. Mirrors sglang/srt/layers/attention/triton_ops/decode_attention.py.

Stage 1: each program handles (request, group of q heads sharing one kv head, KV split)
and writes a normalised partial output plus its log-sum-exp.
Stage 2: combines the splits per (request, head).

`rows` > 1 (speculative verify): rows come in runs of `rows` consecutive queries of one request
whose lengths grow by one (causal). One program takes the whole run x the head group, so each
KV tile is read once for all of them; splits follow the run's last (longest) row.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_stage1(
    Q, K_Buffer, V_Buffer, sm_scale,
    Req_to_tokens, B_req_idx, B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs, stride_qh,
    stride_buf_kbs, stride_buf_kh,
    stride_buf_vbs, stride_buf_vh,
    stride_mid_ob, stride_mid_oh, stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    VALID_BLOCK_H: tl.constexpr,
    IEEE: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    MIN_SPLIT: tl.constexpr,
    ROWS: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_run = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    offs_h = tl.arange(0, BLOCK_H)
    r = offs_h // VALID_BLOCK_H                      # row within the run
    cur_head = cur_head_id * VALID_BLOCK_H + offs_h % VALID_BLOCK_H
    mask_h = (r < ROWS) & (cur_head < q_head_num)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    cur_row = cur_run * ROWS + r

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    row_seq_len = tl.load(B_Seqlen + cur_row, mask=r < ROWS, other=0)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_run * ROWS + ROWS - 1)
    cur_batch_req_idx = tl.load(B_req_idx + cur_run * ROWS)

    offs_q = cur_row[:, None] * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=mask_h[:, None] & mask_d[None, :], other=0.0)

    kv_len_per_split = tl.maximum(tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS), MIN_SPLIT)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            kv_loc = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n, mask=mask_n, other=0
            )
            offs_buf_k = kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[:, None]
            k = tl.load(K_Buffer + offs_buf_k, mask=mask_n[None, :] & mask_d[:, None], other=0.0)

            if IEEE:
                qk = tl.dot(q, k.to(q.dtype), input_precision="ieee")
            else:
                qk = tl.dot(q, k.to(q.dtype))
            qk *= sm_scale
            qk = tl.where(mask_h[:, None] & (offs_n[None, :] < row_seq_len[:, None]), qk, float("-inf"))

            offs_buf_v = kv_loc[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + offs_dv[None, :]
            v = tl.load(V_Buffer + offs_buf_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            if IEEE:
                acc += tl.dot(p.to(q.dtype), v.to(q.dtype), input_precision="ieee")
            else:
                acc += tl.dot(p.to(q.dtype), v.to(q.dtype))
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # a row whose length ends before this split stores garbage here; stage 2 skips it
        offs_mid_o = (
            cur_row[:, None] * stride_mid_ob + cur_head[:, None] * stride_mid_oh + split_kv_id * stride_mid_os + offs_dv[None, :]
        )
        tl.store(Att_Out + offs_mid_o, acc / e_sum[:, None], mask=mask_h[:, None] & mask_dv[None, :])
        offs_mid_o_1 = cur_row * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os + Lv
        tl.store(Att_Out + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)


@triton.jit
def _fwd_kernel_stage2(
    Mid_O, O, B_Seqlen,
    stride_mid_ob, stride_mid_oh, stride_mid_os,
    stride_obs, stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    MIN_SPLIT: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    row_seq_len = tl.load(B_Seqlen + cur_batch)
    run_seq_len = tl.load(B_Seqlen + (cur_batch // ROWS) * ROWS + ROWS - 1)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.maximum(tl.cdiv(run_seq_len, NUM_KV_SPLITS), MIN_SPLIT)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, row_seq_len)
        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / e_sum, mask=mask_d)


def decode_attention_fwd(
    q: torch.Tensor,  # [bs, H, Dq]
    k_buffer: torch.Tensor,  # [pool, Hkv, Dq]
    v_buffer: torch.Tensor,  # [pool, Hkv, Dv]
    o: torch.Tensor,  # [bs, H, Dv]
    req_to_token: torch.Tensor,  # [max_reqs, max_ctx] int32
    b_req_idx: torch.Tensor,  # [bs]
    b_seq_len: torch.Tensor,  # [bs]
    attn_logits: torch.Tensor,  # [bs, H, NUM_KV_SPLITS, Dv + 1] float32 scratch
    num_kv_splits: int,
    sm_scale: Optional[float] = None,
    rows: int = 1,
    min_split: int = 64,
) -> None:
    Lk, Lv = k_buffer.shape[-1], v_buffer.shape[-1]
    sm_scale = sm_scale if sm_scale is not None else 1.0 / (Lk**0.5)
    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = head_num // k_buffer.shape[1]

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)
    BLOCK_N = 64 if Lk < 256 else 32
    VALID_BLOCK_H = min(max(16, triton.next_power_of_2(kv_group_num)), kv_group_num)
    BLOCK_H = max(16, triton.next_power_of_2(VALID_BLOCK_H * rows))
    assert batch % rows == 0 and (rows == 1 or kv_group_num <= BLOCK_H)
    ieee = q.dtype == torch.float32

    grid1 = (batch // rows, triton.cdiv(head_num, VALID_BLOCK_H), num_kv_splits)
    _fwd_kernel_stage1[grid1](
        q, k_buffer, v_buffer, sm_scale,
        req_to_token, b_req_idx, b_seq_len,
        attn_logits,
        req_to_token.stride(0),
        q.stride(0), q.stride(1),
        k_buffer.stride(0), k_buffer.stride(1),
        v_buffer.stride(0), v_buffer.stride(1),
        attn_logits.stride(0), attn_logits.stride(1), attn_logits.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        VALID_BLOCK_H=VALID_BLOCK_H,
        IEEE=ieee,
        BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=num_kv_splits, MIN_SPLIT=min_split, ROWS=rows,
        Lk=Lk, Lv=Lv,
        num_warps=8 if BLOCK_H >= 64 else 4, num_stages=2,
    )
    grid2 = (batch, head_num)
    _fwd_kernel_stage2[grid2](
        attn_logits, o, b_seq_len,
        attn_logits.stride(0), attn_logits.stride(1), attn_logits.stride(2),
        o.stride(0), o.stride(1),
        NUM_KV_SPLITS=num_kv_splits, MIN_SPLIT=min_split, ROWS=rows,
        BLOCK_DV=BLOCK_DV, Lv=Lv,
        num_warps=4, num_stages=2,
    )

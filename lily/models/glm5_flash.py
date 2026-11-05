"""GLM-5.3-Flash (`Glm5NextForConditionalGeneration`, text part) for lily.

Reference: transformers `models/glm5_next/modeling_glm5_next.py`. Pieces (see docs/glm-5.3-flash-plan.md):
  * mHC hyper-connections: `hc_mult` residual streams [T, N, H]; each sub-block collapses them
    with learned weights, runs, then expands and mixes (Sinkhorn-normalised 4x4);
  * KDA linear attention on `linear_attention` layers (fused CUDA kernel, lily/csrc/kda_fused.cu);
  * MLA + DSA on `indexed_attention` layers;
  * dense clamped-SwiGLU MLP (layers 0-2) or MoE (288 experts, top-8) elsewhere.

This file is built up component by component, each verified against the matching
transformers module with tiny random weights (tests/test_glm5_*.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from lily.distributed import get_tp_info
from lily.layers.layernorm import RMSNorm
from lily.layers.linear import ReplicatedLinear, RowParallelLinear, default_weight_loader
from lily.utils import ceil_div
from lily.layers.moe import NVFP4MoE
from lily.layers.quantization import cuda_ops
from lily.layers.quantization.base import QuantizationConfig
from lily.layers.vocab_embedding import VocabParallelEmbedding
from lily.mem_cache.memory_pool import MLATokenToKVPool
from lily.model_executor.forward_batch_info import ForwardBatch
from lily.models.llama import LlamaForCausalLMBase
from lily.models.model_config import ModelConfig
from lily.models.registry import ModelRegistry



# ---------------------------------------------------------------------------------------------
# mHC hyper-connections
# ---------------------------------------------------------------------------------------------

class HyperConnection(nn.Module):
    """Manifold-constrained hyper-connections (mirrors Glm5NextTextHyperConnection).

    `collapse(streams)` -> (post [T, N], comb [T, N, N], collapsed [T, H]);
    `expand(block_out, post, comb, streams)` -> new streams [T, N, H]:
        streams' = post[:, :, None] * block_out[:, None, :] + comb^T @ streams
    All weight math in fp32 as in the reference.
    """

    def __init__(self, hidden_size: int, hc_mult: int, sinkhorn_iters: int, eps: float, rms_eps: float):
        super().__init__()
        self.n = hc_mult
        self.iters = sinkhorn_iters
        self.eps = eps
        self.rms_eps = rms_eps
        width = (2 + hc_mult) * hc_mult
        self.fn = nn.Parameter(torch.empty(width, hc_mult * hidden_size, dtype=torch.bfloat16), requires_grad=False)
        self.base = nn.Parameter(torch.empty(width, dtype=torch.bfloat16), requires_grad=False)
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.bfloat16), requires_grad=False)

    def collapse(self, streams: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = streams.shape[0]
        n = self.n
        flat = streams.reshape(T, -1).float()
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.rms_eps)
        w = F.linear(flat, self.fn.float())
        pre_w, post_w, comb_w = w.split([n, n, n * n], dim=-1)
        pre_b, post_b, comb_b = self.base.float().split([n, n, n * n])
        pre_s, post_s, comb_s = self.scale.float().unbind(0)
        pre = torch.sigmoid(pre_w * pre_s + pre_b) + self.eps
        post = 2 * torch.sigmoid(post_w * post_s + post_b)
        comb = torch.softmax(comb_w.view(T, n, n) * comb_s + comb_b.view(n, n), dim=-1) + self.eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        collapsed = (pre.unsqueeze(-1) * streams.float()).sum(dim=1).to(streams.dtype)
        return post, comb, collapsed

    @staticmethod
    def expand(block_out: torch.Tensor, post: torch.Tensor, comb: torch.Tensor, streams: torch.Tensor) -> torch.Tensor:
        dtype = streams.dtype
        return post.to(dtype).unsqueeze(-1) * block_out.unsqueeze(1) + torch.matmul(comb.to(dtype).transpose(-1, -2), streams)


# ---------------------------------------------------------------------------------------------
# Dense clamped-SwiGLU MLP
# ---------------------------------------------------------------------------------------------

class ClampedMLP(nn.Module):
    """gate clamped at +limit, up at +-limit (Glm5NextTextMLP). bf16 weights or NVFP4 via quant_config."""

    def __init__(self, hidden_size: int, intermediate_size: int, limit: Optional[float], quant_config=None, prefix: str = "",
                 reduce_results: bool = True):
        super().__init__()
        from lily.layers.linear import MergedColumnParallelLinear

        tp_size, _ = get_tp_info()
        self.limit = limit
        self.intermediate_size = intermediate_size // tp_size      # per rank
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False,
                                                       quant_config=quant_config, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, quant_config=quant_config,
                                           prefix=f"{prefix}.down_proj", reduce_results=reduce_results)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu, _ = self.gate_up_proj(x)
        gate, up = gu[..., :self.intermediate_size], gu[..., self.intermediate_size:]
        if self.limit is not None:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        out, _ = self.down_proj(F.silu(gate) * up)
        return out


# ---------------------------------------------------------------------------------------------
# KDA linear attention
# ---------------------------------------------------------------------------------------------

class KimiDeltaAttention(nn.Module):
    """Glm5NextTextLinearAttention: merged q/k/v projection, three causal convs, per-channel
    forget gate from a low-rank projection, delta-rule recurrence, sigmoid-gated RMSNorm, o_proj.
    The step runs in the fused CUDA kernel; states live in the request's slot of a KDA state pool
    ([slot, 3*H*D, 4] conv windows and [slot, H, D, D] recurrent states)."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, conv_kernel: int, lower_bound: Optional[float],
                 rms_eps: float, layer_id: int, state_layer_idx: int):
        super().__init__()
        tp_size, _ = get_tp_info()
        assert num_heads % tp_size == 0, (num_heads, tp_size)
        num_heads //= tp_size                                 # heads are sharded across ranks
        self.hidden_size, self.num_heads, self.head_dim = hidden_size, num_heads, head_dim
        self.qkv_dim = num_heads * head_dim
        self.conv_kernel = conv_kernel
        self.lower_bound = lower_bound
        self.eps = rms_eps
        self.layer_id = layer_id
        self.state_layer_idx = state_layer_idx
        self.qkv_proj = ReplicatedLinear(hidden_size, 3 * self.qkv_dim)            # checkpoint q_proj | k_proj | v_proj
        self.conv1d = nn.Parameter(torch.empty(3 * self.qkv_dim, conv_kernel, dtype=torch.bfloat16), requires_grad=False)
        self.f_a_proj = ReplicatedLinear(hidden_size, head_dim)
        self.f_b_proj = ReplicatedLinear(head_dim, self.qkv_dim)
        self.dt_bias = nn.Parameter(torch.empty(self.qkv_dim, dtype=torch.float32), requires_grad=False)
        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32), requires_grad=False)
        self.b_proj = ReplicatedLinear(hidden_size, num_heads)
        self.g_a_proj = ReplicatedLinear(hidden_size, head_dim)
        self.g_b_proj = ReplicatedLinear(head_dim, self.qkv_dim)
        self.o_norm_weight = nn.Parameter(torch.ones(head_dim, dtype=torch.float32), requires_grad=False)
        self.o_proj = RowParallelLinear(self.qkv_dim * tp_size, hidden_size, bias=False, params_dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor, forward_batch: ForwardBatch, pool) -> torch.Tensor:
        proj = self.qkv_proj(x)
        gate = self.g_b_proj(self.g_a_proj(x))
        beta = self.b_proj(x)
        fgate = self.f_b_proj(self.f_a_proj(x))
        starts, lens, slots, fresh = forward_batch.linear_attention_meta()
        idx = self.state_layer_idx
        conv_snap = rec_snap = None
        if forward_batch.gdn_snapshot:
            pool.alloc_snapshots()
            conv_snap, rec_snap = pool.conv_snapshots[idx], pool.recurrent_snapshots[idx]
            if forward_batch.gdn_proj_stash is not None:
                forward_batch.gdn_proj_stash[idx] = (proj, beta, fgate)
        o = cuda_ops.kda_fused_forward(
            proj, gate, beta, fgate, self.conv1d.data, self.A_log.data, self.dt_bias.data,
            self._lower_bound, self.o_norm_weight.data, self.eps,
            starts, lens, slots, fresh, pool.conv_states[idx], pool.conv_states[idx],
            pool.recurrent_states[idx], pool.recurrent_states[idx], conv_snap, rec_snap)
        return self.o_proj(o)[0]

    @property
    def _lower_bound(self) -> float:
        return -1e30 if self.lower_bound is None else self.lower_bound

    # ---- speculative rollback (see speculative/base_worker.py) ---------------------------

    @property
    def linear_layer_idx(self) -> int:
        return self.state_layer_idx

    def replay_supported(self, pool) -> bool:
        return cuda_ops.kda_fused_supported(self.head_dim, self.conv_kernel, pool.dtype)

    def replay_table_row(self, stash, pool) -> List[int]:
        """Pointer row for `cuda_ops.kda_fused_replay` (KDA_REPLAY_COLUMNS order); `stash` is
        this layer's (proj, beta, fgate) kept from the verify pass."""
        proj, beta, fgate = stash
        idx = self.state_layer_idx
        return [proj.data_ptr(), beta.data_ptr(), fgate.data_ptr(),
                self.conv1d.data.data_ptr(), self.A_log.data.data_ptr(), self.dt_bias.data.data_ptr(),
                pool.conv_snapshots[idx].data_ptr(), pool.conv_states[idx].data_ptr(),
                pool.recurrent_snapshots[idx].data_ptr(), pool.recurrent_states[idx].data_ptr()]

    def replay(self, table: torch.Tensor, stash, starts: torch.Tensor, lens: torch.Tensor, slots: torch.Tensor, pool) -> None:
        proj, beta, fgate = stash
        cuda_ops.kda_fused_replay(table, proj.stride(0), beta.stride(0), fgate.stride(0), self._lower_bound,
                                  starts, lens, slots, self.num_heads, self.head_dim, self.conv_kernel, pool.dtype)


# ---------------------------------------------------------------------------------------------
# MLA (NoPE) + DeepSeek Sparse Attention indexer with k-pools
# ---------------------------------------------------------------------------------------------

class DSAIndexer(nn.Module):
    """Glm5NextTextIndexer: scores pools of `kpool` consecutive tokens and keeps the top
    `index_topk / kpool` pools (plus the incomplete tail) per query.

    Per token the cache holds its indexer key (LayerNorm(wk(x)), 128) and its pool-gate
    logits (x @ compress_gate^T, 128). With <= index_topk (+kpool-1) cached tokens every pool
    is selected, i.e. attention is dense; the scoring is only needed beyond that.
    """

    def __init__(self, hidden_size: int, q_lora_rank: int, n_heads: int, head_dim: int, index_topk: int,
                 kpool: int, select_tail: bool):
        super().__init__()
        self.n_heads, self.head_dim, self.index_topk, self.kpool, self.select_tail = n_heads, head_dim, index_topk, kpool, select_tail
        self.wq_b = ReplicatedLinear(q_lora_rank, n_heads * head_dim)
        self.wk = ReplicatedLinear(hidden_size, head_dim)
        self.k_norm_weight = nn.Parameter(torch.ones(head_dim, dtype=torch.bfloat16), requires_grad=False)
        self.k_norm_bias = nn.Parameter(torch.zeros(head_dim, dtype=torch.bfloat16), requires_grad=False)
        self.weights_proj = ReplicatedLinear(hidden_size, n_heads)
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(kpool, head_dim, dtype=torch.bfloat16), requires_grad=False)
        self.index_kpool_compress_gate = nn.Parameter(torch.zeros(head_dim, hidden_size, dtype=torch.bfloat16), requires_grad=False)
        self.softmax_scale = head_dim ** -0.5

    @property
    def cache_width(self) -> int:
        return 2 * self.head_dim

    def cache_entry(self, x: torch.Tensor) -> torch.Tensor:
        """[T, 2*head_dim]: indexer key | pool-gate logits, as stored per token."""
        k = F.layer_norm(self.wk(x), (self.head_dim,), self.k_norm_weight, self.k_norm_bias, 1e-6)
        gate = F.linear(x, self.index_kpool_compress_gate)
        return torch.cat([k, gate], dim=-1)

    def dense_enough(self, kv_len: int) -> bool:
        return (kv_len + self.kpool - 1) // self.kpool <= self.index_topk // self.kpool

    @property
    def max_visible(self) -> int:
        """Most tokens one query can see: the selected pools plus the incomplete tail."""
        return self.index_topk + self.kpool - 1

    def decode_visible(self, x: torch.Tensor, q_resid: torch.Tensor, forward_batch: ForwardBatch,
                       index_cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Visible positions of each request's newest token, same selection as `select` for a
        single query at position L-1: (vis_idx int32 [B, max_visible], -1 padded after the
        valid entries; vis_cnt int32 [B]). Pool scores come from the CUDA kernel."""
        B = x.shape[0]
        kp = self.kpool
        device = x.device
        p_cap = (max(forward_batch.seq_lens_cpu) + kp - 1) // kp
        q_idx = self.wq_b(q_resid).view(B, self.n_heads, self.head_dim)
        w = self.weights_proj(x).float() * (self.n_heads ** -0.5)
        scores = cuda_ops.dsa_pool_scores(index_cache, self.index_kpool_compress_ape.data,
                                          forward_batch.req_to_token_pool.req_to_token, forward_batch.req_pool_indices,
                                          forward_batch.seq_lens, q_idx, w, self.softmax_scale, kp, p_cap)
        k = min(self.index_topk // kp, p_cap)
        top = scores.topk(k, dim=1)
        pools = top.indices.masked_fill(~torch.isfinite(top.values), -1)                     # [B, k]
        toks = (pools.unsqueeze(-1) * kp + torch.arange(kp, device=device)).masked_fill(pools.unsqueeze(-1) < 0, -1)
        L = forward_batch.seq_lens
        tail = (L // kp * kp).unsqueeze(1) + torch.arange(kp - 1, device=device)
        tail = tail.masked_fill(tail >= L.unsqueeze(1), -1)
        vis = torch.cat([toks.view(B, -1), tail], dim=1)
        if vis.shape[1] < self.max_visible:
            vis = F.pad(vis, (0, self.max_visible - vis.shape[1]), value=-1)
        vis = torch.sort(vis, dim=1, descending=True).values.to(torch.int32)
        return vis, (vis >= 0).sum(dim=1).to(torch.int32)

    def select(self, x: torch.Tensor, q_resid: torch.Tensor, q_pos: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        """Boolean visibility mask [Tq, L] for the queries of one request.

        x / q_resid: the request's query rows [Tq, ...]; q_pos [Tq] their positions; cache [L, 2*hd]
        the request's cached indexer entries in position order (L = kv length including the queries).
        """
        Tq, L = x.shape[0], cache.shape[0]
        kp = self.kpool
        keys, gates = cache[:, :self.head_dim], cache[:, self.head_dim:]
        P = (L + kp - 1) // kp
        pad = P * kp - L
        if pad:
            keys = F.pad(keys, (0, 0, 0, pad))
            gates = F.pad(gates, (0, 0, 0, pad))
        valid = torch.arange(P * kp, device=x.device) < L
        gk = keys.view(P, kp, -1)
        logits = gates.view(P, kp, -1).float() + self.index_kpool_compress_ape.float()[None]
        logits = logits.masked_fill(~valid.view(P, kp, 1), float("-inf"))
        probs = torch.nan_to_num(logits.softmax(dim=1)).to(gk.dtype)
        pool_keys = (probs * gk).sum(dim=1)                                        # [P, hd]
        pool_valid = valid.view(P, kp).all(dim=1)
        pool_end = torch.arange(P, device=x.device) * kp + (kp - 1)

        q = self.wq_b(q_resid).view(Tq, self.n_heads, self.head_dim)
        scores = F.relu(torch.matmul(q.float(), pool_keys.float().t().unsqueeze(0)) * self.softmax_scale)   # [Tq, Hh, P]
        w = self.weights_proj(x).float() * (self.n_heads ** -0.5)                   # [Tq, Hh]
        index_scores = torch.einsum("th,thp->tp", w, scores)                         # [Tq, P]
        selectable = (pool_end.unsqueeze(0) <= q_pos.unsqueeze(1)) & pool_valid.unsqueeze(0)
        index_scores = index_scores.masked_fill(~selectable, torch.finfo(torch.float32).min)
        select_k = min(self.index_topk // kp, P)
        sel = index_scores.topk(select_k, dim=-1).indices                              # [Tq, k]
        sel_valid = selectable.gather(1, sel)
        mask = torch.zeros(Tq, P, dtype=torch.bool, device=x.device)
        mask.scatter_(1, sel, sel_valid)
        tok_mask = mask.unsqueeze(2).expand(Tq, P, kp).reshape(Tq, P * kp)[:, :L]
        if self.select_tail:
            # the incomplete pool holding the positions after the last complete pool, up to the query
            visible_count = q_pos + 1
            tail_count = visible_count % kp
            pos = torch.arange(L, device=x.device).unsqueeze(0)
            tail = (pos >= (visible_count - tail_count).unsqueeze(1)) & (pos <= q_pos.unsqueeze(1))
            tok_mask = tok_mask | tail
        # causality (the indexer already excluded pools ending after the query; keep it explicit)
        tok_mask = tok_mask & (torch.arange(L, device=x.device).unsqueeze(0) <= q_pos.unsqueeze(1))
        return tok_mask


class MLAAttention(nn.Module):
    """Glm5NextTextAttention: latent-cached multi-head attention without rotary (NoPE).

    q = q_b(norm(q_a(x))) [T, heads, 256]; c = norm(kv_a(x)) [T, 512] is what the cache stores;
    K/V = kv_b(c) split per head. Attention uses the DSA mask when the context is long enough
    to need it. This first version expands K/V from the latents with torch ops and runs SDPA
    per request; the absorbed decode kernel comes next.
    """

    def __init__(self, hidden_size: int, num_heads: int, q_lora_rank: int, kv_lora_rank: int, qk_nope_head_dim: int,
                 v_head_dim: int, rms_eps: float, indexer: Optional[DSAIndexer], layer_id: int, kv_layer_idx: int,
                 max_visible: int):
        super().__init__()
        tp_size, _ = get_tp_info()
        self.max_visible = max_visible      # most cached tokens one decode query attends to (index_topk + kpool - 1)
        assert num_heads % tp_size == 0, (num_heads, tp_size)
        total_heads, num_heads = num_heads, num_heads // tp_size      # heads are sharded across ranks
        self.num_heads, self.qk_dim, self.v_dim, self.kv_lora_rank = num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank
        self.layer_id, self.kv_layer_idx = layer_id, kv_layer_idx
        self.q_a_proj = ReplicatedLinear(hidden_size, q_lora_rank)
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=rms_eps)
        self.q_b_proj = ReplicatedLinear(q_lora_rank, num_heads * qk_nope_head_dim)
        self.kv_a_proj_with_mqa = ReplicatedLinear(hidden_size, kv_lora_rank)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=rms_eps)
        self.kv_b_proj = ReplicatedLinear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))
        self.o_proj = RowParallelLinear(total_heads * v_head_dim, hidden_size, bias=False, params_dtype=torch.bfloat16)
        self.indexer = indexer
        self.scaling = qk_nope_head_dim ** -0.5

    def forward(self, x: torch.Tensor, forward_batch: ForwardBatch, latent_cache: torch.Tensor,
                index_cache: Optional[torch.Tensor], shared=None):
        """latent_cache [slots, kv_lora_rank] and index_cache [slots, 2*hd] are this layer's pools.
        Returns (output, shared): what a following layer without its own indexer (`indexer_types`
        "shared") reuses: per-request visibility masks in extend, (vis_idx, vis_cnt) in sparse
        decode, None in dense decode."""
        T = x.shape[0]
        q_resid = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_resid).view(T, self.num_heads, self.qk_dim)
        c = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))
        loc = forward_batch.out_cache_loc
        latent_cache[loc] = c
        if self.indexer is not None:
            index_cache[loc] = self.indexer.cache_entry(x)
        if cuda_ops.mla_decode_supported(self.kv_lora_rank) and (
                forward_batch.forward_mode.is_decode() or self._kernel_extend_ok(forward_batch)):
            return self._decode(x, q_resid, q, forward_batch, latent_cache, index_cache, shared)
        prev_masks = shared

        req_to_token = forward_batch.req_to_token_pool.req_to_token
        decode = forward_batch.forward_mode.is_decode()
        ext = [1] * forward_batch.batch_size if decode else forward_batch.extend_seq_lens_cpu
        seq_lens = forward_batch.seq_lens_cpu
        out = torch.empty(T, self.num_heads, self.v_dim, dtype=x.dtype, device=x.device)
        masks: List[torch.Tensor] = []
        start = 0
        for b in range(forward_batch.batch_size):
            L, n = seq_lens[b], ext[b]
            rows = slice(start, start + n)
            slots = req_to_token[forward_batch.req_pool_indices[b], :L].long()
            kv = self.kv_b_proj(latent_cache[slots]).view(L, self.num_heads, self.qk_dim + self.v_dim)
            k, v = kv[..., :self.qk_dim], kv[..., self.qk_dim:]
            q_pos = torch.arange(L - n, L, device=x.device)
            if self.indexer is not None and not self.indexer.dense_enough(L):
                mask = self.indexer.select(x[rows], q_resid[rows], q_pos, index_cache[slots])
            elif self.indexer is None and prev_masks is not None:
                mask = prev_masks[b]
            else:
                mask = torch.arange(L, device=x.device).unsqueeze(0) <= q_pos.unsqueeze(1)
            o = F.scaled_dot_product_attention(
                q[rows].transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                attn_mask=mask.unsqueeze(0), scale=self.scaling)
            out[rows] = o.transpose(0, 1)
            masks.append(mask)
            start += n
        return self.o_proj(out.reshape(T, -1))[0], masks

    KERNEL_EXTEND_MAX = 8     # queries per request the kernel path takes in extend (speculative verify)

    def _kernel_extend_ok(self, forward_batch: ForwardBatch) -> bool:
        """Small fixed-shape extends (k+1 verify rows per request) whose every row attends densely
        run through the decode kernel, one row per query, so the verify step can be a CUDA graph."""
        ext = forward_batch.extend_seq_lens_cpu
        return (ext is not None and max(ext) <= self.KERNEL_EXTEND_MAX
                and all(L <= self.max_visible - (self.indexer.kpool - 1 if self.indexer else 0)
                        for L in forward_batch.seq_lens_cpu))

    def _decode(self, x, q_resid, q, forward_batch, latent_cache, index_cache, shared):
        """Absorbed attention, one kernel row per query: q_lat = W_k^T q attends over the cached
        latents of positions <= its own (CUDA kernel, static launch shape), then W_v. Dense while
        every request fits in index_topk tokens; in decode beyond that the indexer picks each
        request's visible positions."""
        T = x.shape[0]
        W = self.kv_b_proj.weight.view(self.num_heads, self.qk_dim + self.v_dim, self.kv_lora_rank)
        Wk, Wv = W[:, :self.qk_dim], W[:, self.qk_dim:]
        q_lat = torch.bmm(q.transpose(0, 1), Wk).transpose(0, 1).contiguous()           # [T, H, lora]
        vis = None
        if forward_batch.forward_mode.is_decode():
            row_req, row_len = forward_batch.req_pool_indices, forward_batch.seq_lens
            if self.indexer is not None:
                if any(not self.indexer.dense_enough(L) for L in forward_batch.seq_lens_cpu):
                    vis = self.indexer.decode_visible(x, q_resid, forward_batch, index_cache)
            else:
                vis = shared
        else:
            # verify rows: row of position p sees positions 0..p (its latent is already cached)
            row_req = forward_batch.req_pool_indices.repeat_interleave(forward_batch.extend_seq_lens, output_size=T)
            row_len = forward_batch.positions + 1
        vis_idx, vis_cnt = vis if vis is not None else (None, None)
        chunks = -(-self.max_visible // cuda_ops.MLA_DECODE_CHUNK)
        out_lat = cuda_ops.mla_decode(q_lat, latent_cache, forward_batch.req_to_token_pool.req_to_token,
                                      row_req, row_len, vis_idx, vis_cnt, self.scaling, chunks)
        o = torch.bmm(out_lat.transpose(0, 1), Wv.transpose(1, 2)).transpose(0, 1)       # [T, H, v]
        return self.o_proj(o.reshape(T, -1))[0], vis


# ---------------------------------------------------------------------------------------------
# Decoder layer, model, causal LM + ModelOpt checkpoint loader
# ---------------------------------------------------------------------------------------------

@dataclass
class Glm5Dims:
    """The glm5_next text-config fields lily needs, with transformers' defaults filled in."""

    hidden_size: int
    num_layers: int
    rms_eps: float
    layer_types: List[str]          # "linear_attention" | MLA (anything else)
    mlp_types: List[str]            # "dense" | "sparse"
    indexer_types: List[str]        # "full" | "shared"
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    swiglu_limit: Optional[float]
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    top_k: int
    n_shared_experts: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    kda_heads: int
    kda_head_dim: int
    kda_conv_kernel: int
    kda_lower_bound: Optional[float]
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    index_select_tail: bool

    @classmethod
    def from_model_config(cls, cfg: ModelConfig) -> "Glm5Dims":
        t = cfg.hf_text_config
        n = cfg.num_hidden_layers
        kda = t.get("linear_attn_config") or {}
        layer_types = t.get("layer_types") or [
            "linear_attention" if i % 4 != 3 else "indexed_attention" for i in range(n)]
        dense = t.get("first_k_dense_replace", 3)
        mlp_types = t.get("mlp_layer_types") or ["dense"] * min(dense, n) + ["sparse"] * max(n - dense, 0)
        lower = t.get("linear_lower_bound", kda.get("gate_lower_bound"))
        if lower is None and kda.get("safe_gate", True):
            lower = -5.0
        return cls(
            hidden_size=cfg.hidden_size, num_layers=n, rms_eps=cfg.rms_norm_eps,
            layer_types=list(layer_types[:n]), mlp_types=list(mlp_types[:n]),
            indexer_types=list((t.get("indexer_types") or ["full"] * n)[:n]),
            hc_mult=t.get("hc_mult", 4), hc_sinkhorn_iters=t.get("hc_sinkhorn_iters", 20), hc_eps=t.get("hc_eps", 1e-6),
            swiglu_limit=t.get("swiglu_limit"), intermediate_size=cfg.intermediate_size,
            moe_intermediate_size=t.get("moe_intermediate_size", cfg.intermediate_size),
            num_experts=t.get("n_routed_experts") or t.get("num_local_experts") or 0,
            top_k=t.get("num_experts_per_tok", 8), n_shared_experts=t.get("n_shared_experts", 1),
            routed_scaling_factor=t.get("routed_scaling_factor", 1.0), norm_topk_prob=t.get("norm_topk_prob", True),
            kda_heads=cfg.linear_num_value_heads, kda_head_dim=cfg.linear_value_head_dim,
            kda_conv_kernel=cfg.linear_conv_kernel_size, kda_lower_bound=lower,
            num_heads=cfg.num_attention_heads, q_lora_rank=t["q_lora_rank"], kv_lora_rank=t["kv_lora_rank"],
            qk_nope_head_dim=t["qk_nope_head_dim"], v_head_dim=t["v_head_dim"],
            index_n_heads=t["index_n_heads"], index_head_dim=t["index_head_dim"], index_topk=t["index_topk"],
            index_kpool=t.get("index_kpool", 16), index_select_tail=t.get("index_kpool_always_select_tail", True),
        )

    def is_linear(self, layer_id: int) -> bool:
        return self.layer_types[layer_id] == "linear_attention"

    @property
    def linear_layer_ids(self) -> List[int]:
        return [i for i in range(self.num_layers) if self.is_linear(i)]

    @property
    def mla_layer_ids(self) -> List[int]:
        return [i for i in range(self.num_layers) if not self.is_linear(i)]

    @property
    def indexed_layer_ids(self) -> List[int]:
        return [i for i in self.mla_layer_ids if self.indexer_types[i] == "full"]


def _mla_block(g: Glm5Dims, layer_id: int, kv_layer_idx: int, with_indexer: bool) -> MLAAttention:
    indexer = DSAIndexer(g.hidden_size, g.q_lora_rank, g.index_n_heads, g.index_head_dim, g.index_topk, g.index_kpool,
                         g.index_select_tail) if with_indexer else None
    return MLAAttention(g.hidden_size, g.num_heads, g.q_lora_rank, g.kv_lora_rank, g.qk_nope_head_dim, g.v_head_dim,
                        g.rms_eps, indexer, layer_id, kv_layer_idx, max_visible=g.index_topk + g.index_kpool - 1)


def _moe_block(g: Glm5Dims, quant_config: Optional[QuantizationConfig], prefix: str) -> NVFP4MoE:
    """Routed experts plus the shared expert; the shared expert's reduce is folded into the MoE's."""
    shared = ClampedMLP(g.hidden_size, g.moe_intermediate_size * g.n_shared_experts, g.swiglu_limit, quant_config=quant_config,
                        prefix=f"{prefix}.mlp.shared_experts", reduce_results=False) if g.n_shared_experts else None
    return NVFP4MoE(g.hidden_size, g.moe_intermediate_size, g.num_experts, g.top_k, g.routed_scaling_factor,
                    g.norm_topk_prob, g.swiglu_limit, shared_expert=shared)


class Glm5FlashDecoderLayer(nn.Module):
    """One GLM-5.3-Flash layer over the `hc_mult` residual streams (see Glm5NextTextDecoderLayer).

    Module names follow the checkpoint (`self_attn`, `mlp`, `attn_hc`, `ffn_hc`, the two norms)."""

    def __init__(self, g: Glm5Dims, layer_id: int, quant_config: Optional[QuantizationConfig], prefix: str):
        super().__init__()
        self.layer_id = layer_id
        self.is_linear = g.is_linear(layer_id)
        H = g.hidden_size
        self.attn_hc = HyperConnection(H, g.hc_mult, g.hc_sinkhorn_iters, g.hc_eps, g.rms_eps)
        self.ffn_hc = HyperConnection(H, g.hc_mult, g.hc_sinkhorn_iters, g.hc_eps, g.rms_eps)
        self.input_layernorm = RMSNorm(H, eps=g.rms_eps)
        self.post_attention_layernorm = RMSNorm(H, eps=g.rms_eps)
        if self.is_linear:
            self.self_attn = KimiDeltaAttention(H, g.kda_heads, g.kda_head_dim, g.kda_conv_kernel, g.kda_lower_bound,
                                                g.rms_eps, layer_id, g.linear_layer_ids.index(layer_id))
        else:
            self.self_attn = _mla_block(g, layer_id, g.mla_layer_ids.index(layer_id), g.indexer_types[layer_id] == "full")
        if g.mlp_types[layer_id] == "sparse":
            self.mlp = _moe_block(g, quant_config, prefix)
        else:
            self.mlp = ClampedMLP(H, g.intermediate_size, g.swiglu_limit, quant_config=quant_config, prefix=f"{prefix}.mlp")

    def forward(self, streams: torch.Tensor, forward_batch: ForwardBatch, pool: MLATokenToKVPool, shared):
        """`shared`: the last indexed MLA layer's selection, for layers that reuse it (see MLAAttention)."""
        post, comb, h = self.attn_hc.collapse(streams)
        h = self.input_layernorm(h)
        if self.is_linear:
            h = self.self_attn(h, forward_batch, pool.mamba_pool)
        else:
            h, shared = self.self_attn(h, forward_batch, pool.latent[self.layer_id], pool.index.get(self.layer_id), shared)
        streams = HyperConnection.expand(h, post, comb, streams)
        post, comb, h = self.ffn_hc.collapse(streams)
        h = self.mlp(self.post_attention_layernorm(h))
        return HyperConnection.expand(h, post, comb, streams), shared


class Glm5FlashModel(nn.Module):
    def __init__(self, config: ModelConfig, attn_cls=None, quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "model"):
        super().__init__()
        self.dims = g = Glm5Dims.from_model_config(config)
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens")
        self.layers = nn.ModuleList([
            Glm5FlashDecoderLayer(g, i, quant_config, f"{prefix}.layers.{i}") for i in range(g.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=g.rms_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch,
                input_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        streams = h.unsqueeze(1).expand(-1, self.dims.hc_mult, -1).contiguous()
        pool = forward_batch.token_to_kv_pool
        shared = None
        for layer in self.layers:
            streams, shared = layer(streams, forward_batch, pool, shared)
        # Glm5NextTextHyperHead: unweighted mean of the streams, then the final norm.
        return self.norm(streams.mean(dim=1))


class _SharedHead(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)


class Glm5FlashMTP(nn.Module):
    """GLM-5.3-Flash multi-token-prediction layer (checkpoint `layers.{num_hidden_layers}`,
    DeepSeek-V3 style): h = eh_proj(cat(enorm(embed(token)), hnorm(target hidden))) -> one
    MLA + MoE decoder layer with plain residual adds (no hyper-connections) -> shared_head.norm;
    logits come from the target's lm_head. Own latent / indexer cache (layer id =
    num_hidden_layers) indexed by the target's KV slots; driven by speculative/mtp_worker.py
    with the same position convention as Qwen3_5MTP."""

    def __init__(self, g: Glm5Dims, quant_config: Optional[QuantizationConfig], prefix: str):
        super().__init__()
        H = g.hidden_size
        self.layer_id = g.num_layers
        self.enorm = RMSNorm(H, eps=g.rms_eps)
        self.hnorm = RMSNorm(H, eps=g.rms_eps)
        self.eh_proj = ReplicatedLinear(2 * H, H)
        self.input_layernorm = RMSNorm(H, eps=g.rms_eps)
        self.post_attention_layernorm = RMSNorm(H, eps=g.rms_eps)
        self.self_attn = _mla_block(g, self.layer_id, 0, with_indexer=True)
        self.mlp = _moe_block(g, quant_config, prefix)
        self.shared_head = _SharedHead(H, g.rms_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, hidden_states: torch.Tensor,
                forward_batch: ForwardBatch, embed_tokens: nn.Module) -> torch.Tensor:
        emb = self.enorm(embed_tokens(input_ids))
        x = self.eh_proj(torch.cat([emb, self.hnorm(hidden_states.to(emb.dtype))], dim=-1))
        pool = forward_batch.token_to_kv_pool
        h, _ = self.self_attn(self.input_layernorm(x), forward_batch, pool.latent[self.layer_id], pool.index[self.layer_id], None)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return self.shared_head.norm(x)


# checkpoint name -> lily parameter name (after the `model.language_model.` -> `model.` remap)
_RENAMES = (
    (".hc_attn_fn", ".attn_hc.fn"), (".hc_attn_base", ".attn_hc.base"), (".hc_attn_scale", ".attn_hc.scale"),
    (".hc_ffn_fn", ".ffn_hc.fn"), (".hc_ffn_base", ".ffn_hc.base"), (".hc_ffn_scale", ".ffn_hc.scale"),
    (".self_attn.o_norm.weight", ".self_attn.o_norm_weight"),
    (".indexer.k_norm.weight", ".indexer.k_norm_weight"), (".indexer.k_norm.bias", ".indexer.k_norm_bias"),
)
_EXPERT_ANY_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
_EXPERT_RE = re.compile(r"^(model\.layers\.\d+|mtp)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(weight_packed|weight_scale|weight_global_scale)$")
_KDA_QKV_RE = re.compile(r"^(model\.layers\.\d+\.self_attn\.)([qkv])_(proj|conv1d)\.weight$")
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


@ModelRegistry.register("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM")
class Glm5FlashForCausalLM(LlamaForCausalLMBase):
    """GLM-5.3-Flash text model from an NVIDIA ModelOpt NVFP4 export.

    Loader: `model.language_model.*` -> `model.*`; ModelOpt `weight` (uint8) / `weight_scale` /
    `weight_scale_2` -> `weight_packed` / `weight_scale` / `weight_global_scale` = 1 / weight_scale_2
    (ModelOpt multiplies by its per-tensor scale, lily divides; `input_scale` is W4A4-only and dropped); `q/k/v_proj` and `q/k/v_conv1d` of KDA layers into the merged
    `qkv_proj` / `conv1d`; per-expert tensors into the stacked NVFP4 expert buffers; the vision
    tower is skipped and the MTP layer (`layers.{num_hidden_layers}`) loads into `mtp` when
    speculative decoding asks for it.

    Tensor parallelism (`tp_plan`): attention heads are sharded (KDA q/k/v/convs/gates/o_proj
    and the per-rank state pool; MLA q_b/kv_b/o_proj), the low-rank projections, norms and
    the DSA indexer are replicated, dense and shared MLPs are column/row sharded, the MoE is
    expert-parallel (each rank owns E/tp experts), embeddings and lm_head vocab-sharded.

    Decode: KDA and MoE kernels plus the absorbed MLA decode kernel, all with static launch
    shapes, so decode batches run as CUDA graphs while every request is within `index_topk`
    tokens (attention exactly dense); longer contexts decode eagerly with the DSA selection
    (pool scores kernel + top-k). Prefill/extend uses torch ops per request.
    """

    model_cls = Glm5FlashModel
    supports_cuda_graph = True
    uses_attention_backend = False

    def __init__(self, config: ModelConfig, quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        # quant prefix = the checkpoint's own layer index so its `ignore` patterns apply
        self.mtp = Glm5FlashMTP(self.dims, quant_config, f"model.layers.{self.dims.num_layers}") \
            if config.enable_mtp and config.mtp_num_hidden_layers >= 1 else None

    # ---- engine hooks ------------------------------------------------------------------------

    def graph_safe_lengths(self, seq_lens_cpu: List[int]) -> bool:
        """Graphs cover the dense attention regime: every request within index_topk tokens."""
        g = self.dims
        return cuda_ops.mla_decode_supported(g.kv_lora_rank) and all(L <= g.index_topk for L in seq_lens_cpu)

    def can_run_cuda_graph(self, forward_batch: ForwardBatch) -> bool:
        return forward_batch.seq_lens_cpu is not None and self.graph_safe_lengths(forward_batch.seq_lens_cpu)

    def recurrent_layers(self) -> List[KimiDeltaAttention]:
        """Layers with per-request recurrent state (snapshot + replay in speculative rounds)."""
        return [layer.self_attn for layer in self.model.layers if layer.is_linear]

    _ATTN_ROWS = (".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".q_conv1d.weight", ".k_conv1d.weight",
                  ".v_conv1d.weight", ".f_b_proj.weight", ".g_b_proj.weight", ".dt_bias", ".A_log", ".b_proj.weight",
                  ".q_b_proj.weight", ".kv_b_proj.weight")

    def tp_plan(self, name: str, shape: Tuple[int, ...]):
        """This rank's slice of a checkpoint tensor: None (replicated), loader.SKIP, or (dim, start, stop)."""
        from lily.models.loader import SKIP

        tp, r = get_tp_info()
        if tp == 1:
            return None
        g = self.dims
        n = name.replace("model.language_model.", "model.", 1)
        if n.startswith("model.visual."):                      # vision tower: not served, never read
            return SKIP

        def rows(total):
            per = total // tp
            return (0, per * r, per * (r + 1))

        def cols(total):
            per = total // tp
            return (1, per * r, per * (r + 1))

        m = _EXPERT_ANY_RE.match(n)
        if m:
            per = g.num_experts // tp
            return None if per * r <= int(m.group(2)) < per * (r + 1) else SKIP
        if n.endswith("embed_tokens.weight") or n.startswith("lm_head."):
            per = ceil_div(shape[0], tp)                      # VocabParallelEmbedding's partition
            return (0, per * r, min(per * (r + 1), shape[0]))
        if ".self_attn." in n:
            if n.endswith(self._ATTN_ROWS):
                return rows(shape[0])
            if n.endswith(".o_proj.weight"):
                return cols(shape[1])
            return None
        if ".mlp." in n and ".mlp.gate." not in n:            # dense and shared MLPs (the router is replicated)
            if n.endswith((".weight_scale_2", ".input_scale")):
                return None
            if ".gate_proj." in n or ".up_proj." in n:
                return rows(shape[0])
            if ".down_proj." in n:
                return cols(shape[1])
        return None

    # ---- pools -----------------------------------------------------------------------------

    @property
    def dims(self) -> Glm5Dims:
        return self.model.dims

    def token_cell_bytes(self, dtype: torch.dtype) -> int:
        g = self.dims
        mtp = 1 if self.mtp is not None else 0
        return MLATokenToKVPool.cell_size_bytes(g.kv_lora_rank, len(g.mla_layer_ids) + mtp, 2 * g.index_head_dim,
                                                len(g.indexed_layer_ids) + mtp, dtype)

    def make_token_to_kv_pool(self, size: int, device, dtype: torch.dtype) -> MLATokenToKVPool:
        g = self.dims
        return MLATokenToKVPool(size, dtype, g.kv_lora_rank, g.mla_layer_ids, 2 * g.index_head_dim,
                                g.indexed_layer_ids, device)

    def make_mtp_pool(self, size: int, device, dtype: torch.dtype) -> MLATokenToKVPool:
        g = self.dims
        return MLATokenToKVPool(size, dtype, g.kv_lora_rank, [g.num_layers], 2 * g.index_head_dim, [g.num_layers], device)

    # ---- loading ---------------------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Tuple[Set[str], Set[str]]:
        params = dict(self.named_parameters())
        loaded: Set[str] = set()
        unexpected: Set[str] = set()
        n_layers = self.dims.num_layers
        tp, rank = get_tp_info()
        qkv = self.dims.kda_heads // tp * self.dims.kda_head_dim          # this rank's heads
        seg = {"q": 0, "k": qkv, "v": 2 * qkv}
        expert_offset = rank * (self.dims.num_experts // tp)
        for name, w in weights:
            if name.endswith(".input_scale"):       # W4A4 activation scales: lily runs W4A16
                continue
            if name.startswith("model.visual."):
                unexpected.add(name)
                continue
            if name.startswith("lm_head.") and self.config.tie_word_embeddings:
                continue
            name = name.replace("model.language_model.", "model.", 1)
            m = _LAYER_RE.match(name)
            if m and int(m.group(1)) >= n_layers:        # the MTP layer
                if self.mtp is None or int(m.group(1)) != n_layers:
                    unexpected.add(name)
                    continue
                name = "mtp." + name[m.end():]
            if name.endswith(".weight_scale_2"):
                # ModelOpt: W = e2m1 * weight_scale * weight_scale_2 (a multiplier, ~amax/2688);
                # lily's NVFP4 layout divides by its global scale, so store the reciprocal.
                name = name[:-len("weight_scale_2")] + "weight_global_scale"
                w = (1.0 / w.float()).reshape(1)
            elif name.endswith(".weight") and w.dtype == torch.uint8:
                name = name[:-len("weight")] + "weight_packed"
            m = _EXPERT_RE.match(name)
            if m:
                owner, e, proj, kind = m.group(1), int(m.group(2)) - expert_offset, m.group(3), m.group(4)
                moe = self.mtp.mlp if owner == "mtp" else self.model.layers[int(owner.rsplit(".", 1)[1])].mlp
                moe.experts.load_expert(e, proj, kind, w)
                tag = "down" if proj == "down" else "gate_up"
                loaded.add(f"{owner}.mlp.experts.{tag}_{ {'weight_packed': 'packed', 'weight_scale': 'scale', 'weight_global_scale': 'gscale'}[kind] }")
                continue
            for src, dst in _RENAMES:
                if name.endswith(src):
                    name = name[:-len(src)] + dst
                    break
            m = _KDA_QKV_RE.match(name)
            if m and self.model.layers[int(_LAYER_RE.match(name).group(1))].is_linear:   # (the MTP layer is MLA)
                base, which, kind = m.groups()
                pname = base + ("qkv_proj.weight" if kind == "proj" else "conv1d")
                param = params[pname]
                src = w if kind == "proj" else w.reshape(w.shape[0], -1)
                param.data[seg[which]: seg[which] + qkv].copy_(src)
                loaded.add(pname)
                continue
            shard_id = None
            for mapped, weight_name, sid in self.stacked_params_mapping:
                if weight_name in name:
                    name = name.replace(weight_name, mapped)
                    shard_id = sid
                    break
            if name not in params:
                unexpected.add(name)
                continue
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            try:
                if shard_id is not None:
                    loader(param, w, shard_id)
                else:
                    loader(param, w)
            except (AssertionError, RuntimeError) as e:
                raise RuntimeError(f"failed loading {name!r} shape={tuple(w.shape)} dtype={w.dtype} into "
                                   f"param shape={tuple(param.shape)} shard={shard_id!r}: {e}") from e
            loaded.add(name)
        return loaded, unexpected

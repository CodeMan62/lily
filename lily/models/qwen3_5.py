"""Qwen3.5 hybrid model (Qwen3_5ForConditionalGeneration / Qwen3_5ForCausalLM).

Text-only serving of gated delta-rule linear-attention layers interleaved
with full-attention layers (config.layer_types). Vision and MTP blocks from
the checkpoint are ignored.

Hybrid specifics:
- full-attention layers use the paged KV pool (HybridTokenToKVPool);
- linear-attention layers keep one (conv, recurrent) state pair per request
  slot in MambaStatePool, addressed 1:1 by req_pool_idx; a fresh request
  (extend_prefix_len == 0) starts from zeroed states;
- radix prefix reuse is not supported for linear-attention states, so serving
  hybrid checkpoints requires --disable-radix-cache (forced by ModelRunner).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Set

import torch
import torch.nn.functional as F
from torch import nn

from lily.distributed import get_tp_info
from lily.layers.attention.gated_delta_rule import (
    causal_conv1d_step,
    causal_conv1d_stream,
    chunked_gated_delta_rule,
    gated_rmsnorm,
    recurrent_gated_delta_rule,
)
from lily.layers.layernorm import RMSNorm
from lily.layers.linear import (
    default_weight_loader,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from lily.layers.quantization import cuda_ops
from lily.layers.quantization.base import QuantizationConfig
from lily.model_executor.forward_batch_info import ForwardBatch
from lily.models.llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLMBase, LlamaMLP, LlamaModel
from lily.models.model_config import ModelConfig
from lily.models.registry import ModelRegistry


class GateQKVParallelLinear(QKVParallelLinear):
    """QKV projection whose q block carries the fused output-per-head gate:
    per-q head rows are [q; gate] (2x head_dim). Shard ids "q"/"k"/"v"."""

    def __init__(self, hidden_size: int, head_size: int, total_num_heads: int,
                 total_num_kv_heads: Optional[int] = None, bias: bool = False, **kw):
        super().__init__(hidden_size, head_size, total_num_heads, total_num_kv_heads, bias,
                         q_size_multiplier=2, **kw)


class _Conv1d(nn.Module):
    """Grouped causal conv1d weights holder (checkpoint name `conv1d.weight`)."""

    def __init__(self, conv_dim: int, kernel_size: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(conv_dim, 1, kernel_size, dtype=torch.bfloat16), requires_grad=False)


class Qwen3_5Attention(LlamaAttention):
    """Full-attention block: per-head q/k RMSNorm, partial NeoX rotary and the
    fused per-head output gate (`attn_output_gate = True`)."""

    qkv_cls = GateQKVParallelLinear
    q_size_multiplier = 2
    use_rope_scaling = False

    def __init__(self, config: ModelConfig, layer_id: int,
                 quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__(config, layer_id, quant_config=quant_config, prefix=prefix)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, zero_centered=True)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, zero_centered=True)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        n = positions.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q2, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # HF layout: per head the q_proj rows are [q (head_dim); gate (head_dim)]
        # (torch.chunk along the last dim of view(..., head_dim * 2)), not interleaved.
        q2 = q2.view(n, self.num_heads, 2, self.head_dim)
        q = self.q_norm(q2[:, :, 0]).reshape(n, -1)
        gate = torch.sigmoid(q2[:, :, 1].reshape(n, -1))
        k = self.k_norm(k.view(n, self.num_kv_heads, self.head_dim)).reshape(n, -1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output * gate)
        return output


class RMSNormGated(nn.Module):
    """Norm * weight (fp32 norm), then silu-gated; parameter name `norm.weight`."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dtype=torch.bfloat16), requires_grad=False)
        self.weight.weight_loader = self._load  # type: ignore[assignment]
        self.eps = eps

    @staticmethod
    def _load(param, loaded_weight, *args) -> None:
        default_weight_loader(param, loaded_weight)

    def forward(self, x, gate) -> torch.Tensor:
        return gated_rmsnorm(x, self.weight.data, gate, self.eps)


class GatedDeltaNet(nn.Module):
    """Qwen3.5 linear-attention block: one merged projection -> causal conv ->
    gated delta rule -> RMSNormGated -> out_proj.

    The checkpoint's four input projections (in_proj_qkv, in_proj_z, in_proj_b,
    in_proj_a) are fused into `in_proj`, whose output columns are laid out as
    [q | k | v | z | b | a] (per rank). The step itself runs as one fused CUDA
    kernel when the pool says so (`pool.fused`), else as the eager reference ops.
    """

    # shard ids of the merged projection, in column order
    SHARD_Q, SHARD_K, SHARD_V, SHARD_Z, SHARD_B, SHARD_A = range(6)
    IN_PROJ_SHARDS = {"z": SHARD_Z, "b": SHARD_B, "a": SHARD_A}

    def __init__(self, config: ModelConfig, layer_id: int,
                 quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__()
        tp_size, _ = get_tp_info()
        self.layer_id = layer_id
        self.linear_layer_idx = config.linear_attention_layer_ids.index(layer_id)
        self.num_k_heads = config.get_linear_key_heads(tp_size)
        self.num_v_heads = config.get_linear_value_heads(tp_size)
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim      # per-rank
        self.value_dim = self.num_v_heads * self.head_v_dim    # per-rank
        self.rank_key_dim = self.key_dim
        self.rank_value_dim = self.value_dim
        self.full_key_dim = config.linear_num_key_heads * self.head_k_dim
        self.full_value_dim = config.linear_num_value_heads * self.head_v_dim
        self.full_conv_dim = self.full_key_dim * 2 + self.full_value_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim      # per-rank
        self.conv_kernel_size = config.linear_conv_kernel_size
        self.eps = config.rms_norm_eps
        self.v_per_k = self.num_v_heads // self.num_k_heads

        # Merged sizes are *full*; MergedColumnParallelLinear divides by tp.
        self.in_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [self.full_key_dim, self.full_key_dim, self.full_value_dim, self.full_value_dim,
             config.linear_num_value_heads, config.linear_num_value_heads],
            bias=False, quant_config=quant_config, prefix=f"{prefix}.in_proj")
        self.out_proj = RowParallelLinear(
            self.full_value_dim, config.hidden_size,
            bias=False, quant_config=quant_config, prefix=f"{prefix}.out_proj")

        self.conv1d = _Conv1d(self.conv_dim, self.conv_kernel_size)
        self.conv1d.weight.weight_loader = self._conv_loader  # type: ignore[union-attr]
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.bfloat16), requires_grad=False)
        self.A_log.weight_loader = lambda p, w, *a: default_weight_loader(p, w)
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.bfloat16), requires_grad=False)
        self.dt_bias.weight_loader = lambda p, w, *a: default_weight_loader(p, w)
        self.norm = RMSNormGated(self.head_v_dim, eps=self.eps)
        # fp32 copies for the fused kernel, made on first use (after loading)
        self._A_log_f32: Optional[torch.Tensor] = None
        self._dt_bias_f32: Optional[torch.Tensor] = None

    def _conv_loader(self, param, loaded_weight, *args) -> None:
        """conv1d.weight channels follow in_proj_qkv rows; slice per TP rank."""
        tp_size, tp_rank = get_tp_info()
        if tp_size == 1:
            param.data.copy_(loaded_weight)
            return
        q = self.full_key_dim
        v = self.full_value_dim
        qs, vs = self.rank_key_dim, self.rank_value_dim
        chunks = [
            loaded_weight[tp_rank * qs: (tp_rank + 1) * qs],
            loaded_weight[q + tp_rank * qs: q + (tp_rank + 1) * qs],
            loaded_weight[2 * q + tp_rank * vs: 2 * q + (tp_rank + 1) * vs],
        ]
        param.data.copy_(torch.cat(chunks, dim=0))

    # ---- forward ------------------------------------------------------------

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        n = positions.shape[0]
        proj, _ = self.in_proj(hidden_states)                 # [n, conv_dim + value_dim + 2 * nv]
        pool = forward_batch.token_to_kv_pool.mamba_pool
        if pool is None:
            raise RuntimeError("hybrid model requires the mamba state pool on the forward batch")
        conv_all = pool.conv_states[self.linear_layer_idx]
        rec_all = pool.recurrent_states[self.linear_layer_idx]
        if pool.fused:
            out = self._forward_fused(proj, forward_batch, pool, conv_all, rec_all)
        else:
            out = self._forward_eager(proj, forward_batch, conv_all, rec_all, n)
        out, _ = self.out_proj(out)
        return out

    # ---- fused CUDA path --------------------------------------------------------

    def _fp32_params(self, device) -> None:
        if self._A_log_f32 is None or self._A_log_f32.device != device:
            self._A_log_f32 = self.A_log.data.float().contiguous()
            self._dt_bias_f32 = self.dt_bias.data.float().contiguous()

    def _forward_fused(self, proj, forward_batch, pool, conv_all, rec_all) -> torch.Tensor:
        starts, lens, slots, fresh = forward_batch.linear_attention_meta()
        self._fp32_params(proj.device)
        conv_snap = rec_snap = None
        if forward_batch.gdn_snapshot:
            # Speculative verify: keep the pre-step states and this layer's inputs so
            # the accepted prefix can be replayed after rejection (see replay_table_row).
            pool.alloc_snapshots()
            conv_snap = pool.conv_snapshots[self.linear_layer_idx]
            rec_snap = pool.recurrent_snapshots[self.linear_layer_idx]
            if forward_batch.gdn_proj_stash is not None:
                forward_batch.gdn_proj_stash[self.linear_layer_idx] = proj
        return cuda_ops.gdn_fused_forward(
            proj, self.conv_weight_2d(), self._A_log_f32, self._dt_bias_f32, self.norm.weight.data, self.eps,
            starts, lens, slots, fresh, conv_all, conv_all, rec_all, rec_all, conv_snap, rec_snap, self.num_k_heads)

    def conv_weight_2d(self) -> torch.Tensor:
        return self.conv1d.weight.data.view(self.conv_dim, self.conv_kernel_size)

    # ---- speculative rollback (see speculative/base_worker.py) ---------------------------

    def replay_supported(self, pool) -> bool:
        return pool.fused

    def replay_table_row(self, proj: torch.Tensor, pool) -> List[int]:
        """Pointer row for `cuda_ops.gdn_fused_replay` (GDN_REPLAY_COLUMNS order):
        replay this layer from its snapshot into the live pool over rows of `proj`."""
        idx = self.linear_layer_idx
        self._fp32_params(proj.device)
        return [proj.data_ptr(), self.conv_weight_2d().data_ptr(), self._A_log_f32.data_ptr(), self._dt_bias_f32.data_ptr(),
                pool.conv_snapshots[idx].data_ptr(), pool.conv_states[idx].data_ptr(),
                pool.recurrent_snapshots[idx].data_ptr(), pool.recurrent_states[idx].data_ptr()]

    def replay(self, table: torch.Tensor, stash, starts: torch.Tensor, lens: torch.Tensor, slots: torch.Tensor, pool) -> None:
        """Replay every GDN layer (rows of `table`) over the accepted prefixes; `stash` is any
        layer's stashed projection (all share one row stride)."""
        cuda_ops.gdn_fused_replay(table, stash.stride(0), starts, lens, slots, self.num_k_heads, self.num_v_heads,
                                  self.head_v_dim, self.conv_kernel_size, pool.dtype)

    # ---- eager reference path ---------------------------------------------------

    def _forward_eager(self, proj, forward_batch, conv_all, rec_all, n) -> torch.Tensor:
        qkv, z, b, a = proj.split([self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads], dim=-1)
        slots = forward_batch.req_pool_indices
        w = self.conv1d.weight.data

        output = torch.empty((n, self.num_v_heads, self.head_v_dim), dtype=qkv.dtype, device=qkv.device)
        if forward_batch.forward_mode.is_decode():
            idx = slots.long()
            conv_state = conv_all[idx]                       # [bs, conv_dim, kernel]
            mixed = causal_conv1d_step(qkv, conv_state, w)   # [bs, conv_dim]
            conv_all[idx] = conv_state
            state = rec_all[idx]
            o = self._step(mixed, b, a, state)
            rec_all[idx] = state
            output = o.to(output.dtype)
        else:
            ext_lens = forward_batch.extend_seq_lens_cpu
            starts: List[int] = []
            cursor = 0
            for ext in ext_lens:
                starts.append(cursor)
                cursor += ext
            decode_rows: List[int] = []
            prefix_lens = forward_batch.extend_prefix_lens_cpu or [1] * len(ext_lens)
            for i, ext_len in enumerate(ext_lens):
                rows = slice(starts[i], starts[i] + ext_len)
                if ext_len == 1:
                    decode_rows.append(starts[i])
                    continue
                prev = None if prefix_lens[i] == 0 else conv_all[slots[i:i + 1]].squeeze(0)
                mixed, new_state = causal_conv1d_stream(qkv[rows].t(), prev, w, self.conv_kernel_size)
                conv_all[slots[i:i + 1]] = new_state.unsqueeze(0).to(conv_all.dtype)
                state = rec_all[slots[i:i + 1]]                      # [1, h, k, v]
                initial = state[0]
                output[rows] = self._chunk(mixed.t(), b[rows], a[rows], initial).to(output.dtype)
                rec_all[slots[i:i + 1]] = initial.unsqueeze(0).to(rec_all.dtype)
            if decode_rows:
                idx = torch.tensor(decode_rows, dtype=torch.int64, device=qkv.device)
                row_slots = slots[idx]
                conv_state = conv_all[row_slots]
                mixed = causal_conv1d_step(qkv[idx], conv_state, w)
                conv_all[row_slots] = conv_state
                state = rec_all[row_slots]
                o = self._step(mixed, b[idx], a[idx], state)
                rec_all[row_slots] = state
                output[idx] = o.to(output.dtype)

        out = self.norm(output.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return out.reshape(n, -1)

    # ---- sub-paths -----------------------------------------------------------

    def _split(self, mixed_qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor):
        """mixed_qkv [m, conv_dim] -> (q, k, v [m, h, dim], g, beta [m, h])."""
        key, value = torch.split(mixed_qkv, [self.key_dim * 2, self.value_dim], dim=-1)
        q, k = torch.split(key, [self.key_dim, self.key_dim], dim=-1)
        m = mixed_qkv.shape[0]
        q = q.view(m, self.num_k_heads, self.head_k_dim)
        k = k.view(m, self.num_k_heads, self.head_k_dim)
        v = value.view(m, self.num_v_heads, self.head_v_dim)
        if self.v_per_k > 1:
            q = q.repeat_interleave(self.v_per_k, dim=1)
            k = k.repeat_interleave(self.v_per_k, dim=1)
        beta = torch.sigmoid(b.float())
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        return q, k, v, g, beta

    def _step(self, mixed_qkv, b, a, state):
        """Single-token recurrence (decode or 1-token extend rows). Mutates
        `state` in place; returns o [m, h, v_dim] in fp32."""
        q, k, v, g, beta = self._split(mixed_qkv, b, a)
        return recurrent_gated_delta_rule(q, k, v, g, beta, state)

    def _chunk(self, mixed_qkv, b, a, init_state):
        """Multi-token chunked scan for one request. `init_state` [h,k,v] is
        both the initial state (read-only) and the final-state write target."""
        q, k, v, g, beta = self._split(mixed_qkv, b, a)
        out = chunked_gated_delta_rule(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
            g.unsqueeze(0), beta.unsqueeze(0), init_state,
            final_state_out=init_state)
        return out.squeeze(0)


class Qwen3_5DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: ModelConfig, layer_id: int, attn_cls=None,
                 quant_config: Optional[QuantizationConfig] = None, prefix: str = "",
                 block_type: Optional[str] = None):
        nn.Module.__init__(self)
        block_type = block_type or config.layer_types[layer_id]
        self.hidden_size = config.hidden_size
        # Module attribute names match the checkpoint (`linear_attn` / `self_attn`)
        # so weight streaming needs no remapping between them.
        self.block_type = block_type
        if block_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, layer_id, quant_config=quant_config, prefix=f"{prefix}.linear_attn")
        elif block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_id, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        else:
            raise ValueError(f"unknown Qwen3.5 layer type {block_type!r}")
        self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size, config.hidden_act,
                            quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, zero_centered=True)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, zero_centered=True)

    @property
    def attn_layer(self) -> nn.Module:
        return self.linear_attn if self.block_type == "linear_attention" else self.self_attn


class Qwen3_5Model(LlamaModel):
    def __init__(self, config: ModelConfig, attn_cls=None,
                 quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__(config, quant_config=quant_config, prefix=prefix,
                         layer_cls=Qwen3_5DecoderLayer, zero_centered=True)

    def _prepare_forward(self, input_ids, positions, forward_batch, input_embeds) -> None:
        # Fresh requests (no cached prefix) start from zeroed linear-attention
        # states; continuation chunks keep theirs. Skipped in decode mode so
        # CUDA-graph replay never bakes a zero pass in.
        pool = forward_batch.token_to_kv_pool.mamba_pool
        if pool is None or pool.fused or forward_batch.extend_prefix_lens_cpu is None \
                or forward_batch.forward_mode.is_decode():
            return
        fresh = [i for i, p in enumerate(forward_batch.extend_prefix_lens_cpu) if p == 0]
        if fresh:
            fresh_idx = torch.tensor(fresh, dtype=torch.int64, device=positions.device)
            pool.zero_slots(forward_batch.req_pool_indices[fresh_idx])



class Qwen3_5MTP(nn.Module):
    """Qwen3.5/3.8 multi-token-prediction head (checkpoint `mtp.*`, bf16).

    hidden = fc(cat(norm_e(embed(token)), norm_h(target_hidden))) -> one gated
    full-attention decoder layer (own KV cache, layer id = num_hidden_layers) ->
    norm. Logits come from the target's lm_head. The pair (token at position p,
    target hidden at position p-1) sits at the draft layer's position p-1, i.e.
    the draft cache is shifted one position left of the target's.
    """

    def __init__(self, config: ModelConfig, prefix: str = "mtp"):
        super().__init__()
        self.layer_id = config.num_hidden_layers
        self.pre_fc_norm_embedding = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, zero_centered=True)
        self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, zero_centered=True)
        self.fc = ReplicatedLinear(2 * config.hidden_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(
            config, self.layer_id, quant_config=None, prefix=f"{prefix}.layers.0", block_type="full_attention")])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, zero_centered=True)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, hidden_states: torch.Tensor,
                forward_batch: ForwardBatch, embed_tokens: nn.Module) -> torch.Tensor:
        emb = self.pre_fc_norm_embedding(embed_tokens(input_ids))
        hid = self.pre_fc_norm_hidden(hidden_states.to(emb.dtype))
        x = self.fc(torch.cat([emb, hid], dim=-1))
        h, residual = self.layers[0](positions, x, forward_batch, None)
        h, _ = self.norm(h, residual)
        return h


def _rank_segments(w: torch.Tensor, n_k: int, n_v: int, hd: int, vd: int, tp_rank: int, tp_size: int):
    """Rows of the fused [q | k | v] layout for a given TP rank."""
    k_per, v_per = n_k // tp_size, n_v // tp_size
    q_full = n_k * hd
    q1 = (tp_rank + 1) * k_per * hd
    k0 = q_full + tp_rank * k_per * hd
    v0 = 2 * q_full + tp_rank * v_per * vd
    return [w[tp_rank * k_per * hd: q1], w[k0: q_full + q1], w[v0: v0 + v_per * vd]]


@ModelRegistry.register("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")
class Qwen3_5ForCausalLM(LlamaForCausalLMBase):
    """Remaps `model.language_model.*` checkpoint names, ignores vision/MTP
    tensors, merges the four linear-attention input projections into `in_proj`
    and slices the fused `in_proj_qkv`/`conv1d` checkpoints per rank."""

    model_cls = Qwen3_5Model

    def __init__(self, config: ModelConfig, quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.mtp = Qwen3_5MTP(config) if config.enable_mtp and config.mtp_num_hidden_layers >= 1 else None

    def load_weights(self, weights: Iterable[tuple]) -> tuple[Set[str], Set[str]]:
        params_dict = dict(self.named_parameters())
        loaded: Set[str] = set()
        unexpected: Set[str] = set()
        for name, loaded_weight in weights:
            if name.startswith("lm_head.") and self.config.tie_word_embeddings:
                continue
            if name.startswith("model.visual.") or (name.startswith("mtp.") and self.mtp is None):
                unexpected.add(name)
                continue
            name = name.replace("model.language_model.", "model.", 1)

            # The checkpoint's in_proj_qkv / in_proj_z / in_proj_b / in_proj_a
            # all land in the merged `in_proj`: qkv is split per rank into its
            # q/k/v row segments (shards 0..2), z/b/a arrive pre-sliced (3..5).
            in_proj_part = None
            if ".linear_attn.in_proj_" in name:
                in_proj_part = name.split(".linear_attn.in_proj_", 1)[1].split(".", 1)[0]
                name = name.replace(f".in_proj_{in_proj_part}.", ".in_proj.", 1)
            shard_id = None
            for mapped_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name in name:
                    name = name.replace(weight_name, mapped_name)
                    break
            else:
                shard_id = None
            if name not in params_dict:
                unexpected.add(name)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if in_proj_part == "qkv":
                tp_size, tp_rank = get_tp_info()
                if loaded_weight.numel() == 1:
                    # Per-tensor quant scalar (NVFP4 weight_global_scale): the
                    # same value applies to every q/k/v row segment.
                    segs = [loaded_weight] * 3
                else:
                    segs = _rank_segments(
                        loaded_weight,
                        self.config.linear_num_key_heads, self.config.linear_num_value_heads,
                        self.config.linear_key_head_dim, self.config.linear_value_head_dim,
                        tp_rank, tp_size)
                for sid, partial in enumerate(segs):
                    weight_loader(param, partial, sid)
            elif in_proj_part is not None:
                weight_loader(param, loaded_weight, GatedDeltaNet.IN_PROJ_SHARDS[in_proj_part])
            else:
                try:
                    if shard_id is not None:
                        weight_loader(param, loaded_weight, shard_id)
                    else:
                        weight_loader(param, loaded_weight)
                except (AssertionError, RuntimeError) as e:
                    raise RuntimeError(
                        f"failed loading checkpoint tensor {name!r} shape={tuple(loaded_weight.shape)} "
                        f"dtype={loaded_weight.dtype} into param shape={tuple(param.shape)} "
                        f"shard={shard_id!r}: {e}") from e
            loaded.add(name)
        return loaded, unexpected

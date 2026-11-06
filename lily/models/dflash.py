"""DFlash 2 block-diffusion drafter (z-lab / Inco AI), e.g. `z-lab/Qwen3.8-27B-DFlash2`.

Reference: https://github.com/z-lab/dflash (dflash/model.py), which SGLang and vLLM port.
Not a language model on its own: it drafts a whole block of tokens for the target
to verify, in one forward pass.

Per round, for a request whose target has x_1..x_m cached and whose last committed
(uncached) token is x_{m+1}:
  * block input   = [x_{m+1}, mask, ..., mask] (block_size rows) at positions m..m+block_size-1,
    embedded with the target's embedding table;
  * context       = the target's residual stream after `target_layer_ids`, concatenated,
    `fc` -> `hidden_norm`; every drafter layer projects it with its own k/v projections and
    uses the result as extra K/V entries (cached per position in a dedicated KV pool);
  * attention     = bidirectional inside the block, plus context within `sliding_window`;
  * each layer wraps attention and MLP in two-tap grouped dynamic causal convolutions
    (within the block): out_t = (base_0 + dyn_t,0) * x_t + (base_1 + dyn_t,1) * x_{t-1};
  * rows 1.. go through the target's lm_head; the candidate selector keeps top-k tokens
    per position and traces one path left to right with a low-rank bilinear pair score.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from lily.distributed.distributed import get_tp_info, tensor_model_parallel_all_reduce
from lily.layers.layernorm import RMSNorm
from lily.layers.linear import ReplicatedLinear
from lily.layers.rotary import get_rope


@dataclass
class DFlashConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    max_position_embeddings: int
    sliding_window: Optional[int]
    is_causal: bool
    block_size: int
    conv_group_size: int
    conv_kernel_size: int
    mask_token_id: int
    selector_rank: int
    selector_top_k: int
    target_layer_ids: List[int]
    input_embedding_scale: float = 1.0
    output_multiplier: float = 1.0
    final_logit_softcapping: Optional[float] = None
    layer_types: List[str] = field(default_factory=list)

    @classmethod
    def from_dir(cls, path: str) -> "DFlashConfig":
        with open(os.path.join(path, "config.json")) as f:
            c = json.load(f)
        d = c.get("dflash_config", {})
        rope = c.get("rope_parameters") or {}
        layer_types = c.get("layer_types") or ["full_attention"] * c["num_hidden_layers"]
        sliding = c.get("sliding_window") if (c.get("use_sliding_window", True) and "sliding_attention" in layer_types) else None
        is_causal = c.get("is_causal")
        if is_causal is None:
            # reference: sliding layers default to causal, else bidirectional
            is_causal = layer_types[0] == "sliding_attention"
        return cls(
            hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"], num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c["num_key_value_heads"],
            head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
            rms_norm_eps=c.get("rms_norm_eps", 1e-6),
            rope_theta=float(rope.get("rope_theta", c.get("rope_theta", 10000.0))),
            vocab_size=c["vocab_size"], max_position_embeddings=c.get("max_position_embeddings", 262144),
            sliding_window=sliding, is_causal=bool(is_causal),
            block_size=int(d.get("block_size", c.get("block_size", 8))),
            conv_group_size=int(d.get("conv_group_size", 16)), conv_kernel_size=int(d.get("conv_kernel_size", 2)),
            mask_token_id=int(d["mask_token_id"]), selector_rank=int(d["selector_rank"]),
            selector_top_k=int(d["selector_top_k"]), target_layer_ids=list(d["target_layer_ids"]),
            input_embedding_scale=float(d.get("input_embedding_scale", 1.0)),
            output_multiplier=float(d.get("output_multiplier", 1.0)),
            final_logit_softcapping=d.get("final_logit_softcapping"),
            layer_types=layer_types,
        )



class GroupedDynamicCausalConv(nn.Module):
    """Two-tap depthwise conv along the block with per-group content-adaptive kernels.

    `prepare(h)` convolves the layer input with base_kernel[0] + dyn_0 and also returns
    dyn_1 (computed from the same input); `finish(y, dyn_1)` convolves the sub-layer
    output with base_kernel[1] + dyn_1. Causal within the block: x_{t-1} is zero at t=0.
    """

    def __init__(self, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, hidden_size, dtype=torch.bfloat16), requires_grad=False)
        self.kernel_projection = ReplicatedLinear(hidden_size, 2 * kernel_size * groups)

    def _convolve(self, hidden: torch.Tensor, dynamic: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        # hidden [B, L, H]; dynamic [B, L, K, groups]; base [K, H]
        B, L, H = hidden.shape
        groups = H // self.group_size
        blocks = hidden.view(B, L, groups, self.group_size)
        out = torch.zeros_like(blocks)
        for offset in range(self.kernel_size):
            values = blocks if offset == 0 else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
            kernel = base[offset].view(1, 1, groups, self.group_size).to(hidden.dtype)
            out = out + kernel * values
            out = torch.addcmul(out, dynamic[:, :, offset].unsqueeze(-1).to(hidden.dtype), values)
        return out.view(B, L, H)

    def prepare(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).view(*hidden.shape[:-1], 2, self.kernel_size, groups)
        return self._convolve(hidden, dynamic[..., 0, :, :], self.base_kernel[0]), dynamic[..., 1, :, :]

    def finish(self, hidden: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        return self._convolve(hidden, dynamic, self.base_kernel[1])


class DFlashAttention(nn.Module):
    def __init__(self, cfg: DFlashConfig, layer_idx: int):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scaling = cfg.head_dim ** -0.5
        self.q_proj = ReplicatedLinear(cfg.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = ReplicatedLinear(cfg.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = ReplicatedLinear(cfg.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = ReplicatedLinear(self.num_heads * self.head_dim, cfg.hidden_size)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.rotary = get_rope(self.head_dim, self.head_dim, cfg.max_position_embeddings, cfg.rope_theta, is_neox_style=True)
        layer_type = cfg.layer_types[layer_idx] if cfg.layer_types else "full_attention"
        self.sliding_window = cfg.sliding_window if layer_type == "sliding_attention" else None
        self.is_causal = cfg.is_causal

    def context_kv(self, features: torch.Tensor, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projected (normed, roped) K and V for context rows: features [T, H] -> [T, kvh, hd] each."""
        T = features.shape[0]
        k = self.k_norm(self.k_proj(features).view(T, self.num_kv_heads, self.head_dim))
        v = self.v_proj(features).view(T, self.num_kv_heads, self.head_dim)
        # rotary expects [N, heads*hd]; rotate k only (dummy q)
        _, k = self.rotary(positions, k.new_zeros(T, self.head_dim), k.reshape(T, -1))
        return k.view(T, self.num_kv_heads, self.head_dim), v

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor, ctx_k: torch.Tensor, ctx_v: torch.Tensor,
                ctx_pos: torch.Tensor) -> torch.Tensor:
        """hidden [B, S, H] block rows at `positions` [B, S]; ctx_k/ctx_v [B, L, kvh, hd] gathered
        context entries at `ctx_pos` [B, L] (-1 = padding)."""
        B, S, _ = hidden.shape
        L = ctx_k.shape[1]
        q = self.q_norm(self.q_proj(hidden).view(B * S, self.num_heads, self.head_dim))
        k_blk = self.k_norm(self.k_proj(hidden).view(B * S, self.num_kv_heads, self.head_dim))
        v_blk = self.v_proj(hidden).view(B, S, self.num_kv_heads, self.head_dim)
        q, k_blk = self.rotary(positions.reshape(-1), q.reshape(B * S, -1), k_blk.reshape(B * S, -1))
        q = q.view(B, S, self.num_heads, self.head_dim)
        k_blk = k_blk.view(B, S, self.num_kv_heads, self.head_dim)
        k = torch.cat([ctx_k, k_blk], dim=1)          # [B, L+S, kvh, hd]
        v = torch.cat([ctx_v, v_blk], dim=1)
        rep = self.num_heads // self.num_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
        # mask [B, 1, S, L+S]: context within the window, block fully visible (non-causal)
        qpos = positions.unsqueeze(2)                                   # [B, S, 1]
        kpos = torch.cat([ctx_pos, positions], dim=1).unsqueeze(1)     # [B, 1, L+S]
        visible = kpos >= 0
        if self.is_causal:
            visible = visible & (kpos <= qpos)
        if self.sliding_window is not None:
            visible = visible & (qpos - kpos < self.sliding_window)
            if not self.is_causal:
                visible = visible & (kpos - qpos < self.sliding_window)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=visible.unsqueeze(1), scale=self.scaling)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, -1))


class DFlashMLP(nn.Module):
    def __init__(self, cfg: DFlashConfig):
        super().__init__()
        self.gate_proj = ReplicatedLinear(cfg.hidden_size, cfg.intermediate_size)
        self.up_proj = ReplicatedLinear(cfg.hidden_size, cfg.intermediate_size)
        self.down_proj = ReplicatedLinear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return tensor_model_parallel_all_reduce(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class DFlashLayer(nn.Module):
    def __init__(self, cfg: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(cfg, layer_idx)
        self.mlp = DFlashMLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attention_conv = GroupedDynamicCausalConv(cfg.hidden_size, cfg.conv_kernel_size, cfg.conv_group_size)
        self.mlp_conv = GroupedDynamicCausalConv(cfg.hidden_size, cfg.conv_kernel_size, cfg.conv_group_size)

    def forward(self, hidden, positions, ctx_k, ctx_v, ctx_pos):
        residual = hidden
        h = self.input_layernorm(hidden)
        h, attn_dyn = self.attention_conv.prepare(h)
        h = self.self_attn(h, positions, ctx_k, ctx_v, ctx_pos)
        h = self.attention_conv.finish(h, attn_dyn)
        hidden = residual + h
        residual = hidden
        h = self.post_attention_layernorm(hidden)
        h, mlp_dyn = self.mlp_conv.prepare(h)
        h = self.mlp(h)
        h = self.mlp_conv.finish(h, mlp_dyn)
        return residual + h


class CandidateSelector(nn.Module):
    """Top-k candidates per position; one left-to-right path scored by
    logit(cand) + (pred_codebook[prev] * proj(h_t)) . succ_codebook[cand]."""

    def __init__(self, cfg: DFlashConfig):
        super().__init__()
        self.top_k = cfg.selector_top_k
        self.predecessor_codebook = nn.Parameter(torch.empty(cfg.vocab_size, cfg.selector_rank, dtype=torch.bfloat16), requires_grad=False)
        self.successor_codebook = nn.Parameter(torch.empty(cfg.vocab_size, cfg.selector_rank, dtype=torch.bfloat16), requires_grad=False)
        self.hidden_projection = ReplicatedLinear(cfg.hidden_size, cfg.selector_rank)

    def select(self, hidden: torch.Tensor, logits: torch.Tensor, anchor_ids: torch.Tensor,
               temperatures: Sequence[float], generators: Sequence[torch.Generator]):
        """hidden [B, S, H], logits [B, S, V] (fp32), anchor_ids [B].

        Returns (path [B, S] int64, candidates [B, S, K] int64, q [B, S, K] fp32): the chosen
        token per position, the candidate set, and the selector's distribution over it
        (one-hot for temperature 0). q is the draft distribution for rejection sampling.
        """
        B, S, _ = hidden.shape
        unary, candidates = torch.topk(logits, self.top_k, dim=-1, sorted=False)   # [B, S, K]
        proj = self.hidden_projection(hidden).float()                               # [B, S, r]
        sampling_rows = [i for i, t in enumerate(temperatures) if t > 0]
        temps = torch.tensor([float(t) for t in temperatures], device=hidden.device) if sampling_rows else None
        predecessor = anchor_ids
        path, qs = [], []
        for pos in range(S):
            cand = candidates[:, pos]                                                # [B, K]
            gate = F.embedding(predecessor, self.predecessor_codebook).float() * proj[:, pos]   # [B, r]
            pair = torch.einsum("br,bkr->bk", gate, F.embedding(cand, self.successor_codebook).float())
            scores = unary[:, pos] + pair                                            # [B, K]
            if sampling_rows:
                q = torch.softmax(scores / temps.clamp_min(1e-5).unsqueeze(1), dim=-1)
                onehot = F.one_hot(scores.argmax(-1), self.top_k).float()
                q = torch.where((temps > 0).unsqueeze(1), q, onehot)
                index = scores.argmax(-1)
                for i in sampling_rows:
                    index[i] = torch.multinomial(q[i], 1, generator=generators[i])[0]
            else:
                index = scores.argmax(-1)
                q = F.one_hot(index, self.top_k).float()
            predecessor = cand.gather(1, index.unsqueeze(1)).squeeze(1)
            path.append(predecessor)
            qs.append(q)
        return torch.stack(path, dim=1), candidates, torch.stack(qs, dim=1)


class DFlash2Drafter(nn.Module):
    """The drafter stack. Embeddings and lm_head are the target's (passed in)."""

    def __init__(self, cfg: DFlashConfig, target_hidden_size: int):
        super().__init__()
        self.cfg = cfg
        self.fc = ReplicatedLinear(len(cfg.target_layer_ids) * target_hidden_size, cfg.hidden_size)
        self.hidden_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.layers = nn.ModuleList([DFlashLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.candidate_selector = CandidateSelector(cfg)

    @property
    def block_size(self) -> int:
        return self.cfg.block_size

    @property
    def num_drafts(self) -> int:
        return self.cfg.block_size - 1

    # ---- context ---------------------------------------------------------------

    def fuse_context(self, aux_hidden: torch.Tensor) -> torch.Tensor:
        """Target residual streams [T, len(target_layer_ids) * H_target] -> context features [T, H]."""
        return self.hidden_norm(self.fc(aux_hidden.to(self.fc.weight.dtype)))

    def context_kv(self, features: torch.Tensor, positions: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Per layer (k, v) [T, kvh, hd] for context rows, ready to be written to the KV pool."""
        return [layer.self_attn.context_kv(features, positions) for layer in self.layers]

    # ---- block draft -------------------------------------------------------------

    def forward(self, block_ids: torch.Tensor, positions: torch.Tensor, embed_tokens: nn.Module,
                ctx_kv: List[Tuple[torch.Tensor, torch.Tensor]], ctx_pos: torch.Tensor) -> torch.Tensor:
        """block_ids / positions [B, S]; ctx_kv per layer ([B, L, kvh, hd], [B, L, kvh, hd]);
        ctx_pos [B, L] (-1 padding). Returns normed hidden [B, S, H]."""
        h = embed_tokens(block_ids.reshape(-1)).view(*block_ids.shape, -1)
        if self.cfg.input_embedding_scale != 1.0:
            h = h * self.cfg.input_embedding_scale
        for layer, (ck, cv) in zip(self.layers, ctx_kv):
            h = layer(h, positions, ck, cv, ctx_pos)
        return self.norm(h)

    def logits(self, hidden: torch.Tensor, lm_head) -> torch.Tensor:
        logits = lm_head.compute_logits(hidden.reshape(-1, hidden.shape[-1])).view(*hidden.shape[:-1], -1)
        if self.cfg.output_multiplier != 1.0:
            logits = logits * self.cfg.output_multiplier
        cap = self.cfg.final_logit_softcapping
        if cap:
            logits = torch.tanh(logits / float(cap)) * float(cap)
        return logits

    # ---- loading ---------------------------------------------------------------

    @classmethod
    def load(cls, path: str, target_hidden_size: int, device) -> "DFlash2Drafter":
        from safetensors.torch import load_file

        from lily.utils import set_default_torch_dtype

        cfg = DFlashConfig.from_dir(path)
        with set_default_torch_dtype(torch.bfloat16):   # norm weights etc. in the checkpoint dtype
            model = cls(cfg, target_hidden_size)
        state = {}
        for fn in sorted(os.listdir(path)):
            if fn.endswith(".safetensors"):
                state.update(load_file(os.path.join(path, fn)))
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [m for m in missing if not m.endswith("rotary.cos_sin_cache")]
        if missing or unexpected:
            raise RuntimeError(f"DFlash2 checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
        tp_size, tp_rank = get_tp_info()
        for layer in model.layers:      # the MLP is most of the drafter's weights: each rank reads a slice
            for proj, dim in ((layer.mlp.gate_proj, 0), (layer.mlp.up_proj, 0), (layer.mlp.down_proj, 1)):
                proj.weight.data = proj.weight.data.chunk(tp_size, dim)[tp_rank].contiguous()
        return model.to(device).eval()


def resolve_draft_path(path: str) -> str:
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download

    return snapshot_download(path)

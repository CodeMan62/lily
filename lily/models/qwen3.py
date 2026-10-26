"""Qwen3 dense model: Llama with per-head q/k RMSNorm before RoPE. Mirrors sglang/srt/models/qwen3.py."""

from __future__ import annotations

from typing import Optional

import torch

from lily.models.model_config import ModelConfig
from lily.layers.layernorm import RMSNorm
from lily.layers.quantization.base import QuantizationConfig
from lily.model_executor.forward_batch_info import ForwardBatch
from lily.models.llama import LlamaAttention, LlamaForCausalLMBase
from lily.models.registry import ModelRegistry


class Qwen3Attention(LlamaAttention):
    def __init__(self, config: ModelConfig, layer_id: int,
                 quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__(config, layer_id, quant_config=quant_config, prefix=prefix)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        q, k, v = self._qkv(hidden_states)
        n = q.shape[0]
        q = self.q_norm(q.view(n, self.num_heads, self.head_dim)).view(n, -1)
        k = self.k_norm(k.view(n, self.num_kv_heads, self.head_dim)).view(n, -1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


@ModelRegistry.register("Qwen3ForCausalLM")
class Qwen3ForCausalLM(LlamaForCausalLMBase):
    attn_cls = Qwen3Attention

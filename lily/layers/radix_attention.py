"""RadixAttention: the attention layer that delegates to the active backend.
Mirrors sglang/srt/layers/radix_attention.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

if TYPE_CHECKING:
    from lily.model_executor.forward_batch_info import ForwardBatch


class RadixAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        v_head_dim: int = -1,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim
        self.scaling = scaling
        self.layer_id = layer_id

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        return forward_batch.attn_backend.forward(q, k, v, self, forward_batch, save_kv_cache)

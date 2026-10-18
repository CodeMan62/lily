"""Attention backend interface. Mirrors sglang/srt/layers/attention/base_attn_backend.py."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from lula.layers.radix_attention import RadixAttention
    from lula.model_executor.forward_batch_info import ForwardBatch


class AttentionBackend(ABC):
    """One instance per model runner. Called by every RadixAttention layer."""

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        """Precompute per-batch metadata (indices, masks) before the forward pass."""

    # ---- CUDA-graph hooks (Phase 7) ---------------------------------------

    def supports_cuda_graph(self) -> bool:
        """Whether this backend can be captured. Backends with data-dependent
        shapes (e.g. the torch-native reference) return False."""
        return True

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        """Allocate the shared buffers the capture/replay metadata points at."""
        raise NotImplementedError

    def init_forward_metadata_capture_cuda_graph(
        self, bs: int, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor
    ) -> None:
        raise NotImplementedError

    def init_forward_metadata_replay_cuda_graph(
        self, bs: int, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor
    ) -> None:
        raise NotImplementedError

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return 1

    # ---- dispatch ----------------------------------------------------------

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(q, k, v, layer, forward_batch, save_kv_cache)
        return self.forward_extend(q, k, v, layer, forward_batch, save_kv_cache)

    @abstractmethod
    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True) -> torch.Tensor: ...

    @abstractmethod
    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True) -> torch.Tensor: ...

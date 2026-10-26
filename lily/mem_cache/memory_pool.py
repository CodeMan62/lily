"""Memory pools. Mirrors sglang/srt/mem_cache/memory_pool.py (reduced).

Two levels, as in SGLang:
  ReqToTokenPool   : request slot -> row of KV-pool indices, one per token position.
  MHATokenToKVPool : KV-pool index -> K and V vectors for every layer (page_size = 1).
Slot 0 of the KV pool is reserved so that "no token" can be encoded as index 0.
"""

from __future__ import annotations

import abc
import logging
from typing import List, Optional, Union

import torch

logger = logging.getLogger(__name__)


class ReqToTokenPool:
    def __init__(self, size: int, max_context_len: int, device: torch.device | str):
        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        self.req_to_token = torch.zeros((size, max_context_len), dtype=torch.int32, device=device)
        self.free_slots: List[int] = list(range(size))

    def write(self, indices, values) -> None:
        """indices: (req_idx, slice) or tensor; values: KV-pool indices."""
        self.req_to_token[indices] = values

    def available_size(self) -> int:
        return len(self.free_slots)

    def alloc(self, need_size: int = 1) -> Optional[List[int]]:
        if need_size > len(self.free_slots):
            return None
        out = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return out

    def free(self, free_index: Union[int, List[int]]) -> None:
        if isinstance(free_index, int):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self) -> None:
        self.free_slots = list(range(self.size))


class KVCache(abc.ABC):
    size: int
    page_size: int
    dtype: torch.dtype
    layer_num: int

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor: ...

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor: ...

    @abc.abstractmethod
    def set_kv_buffer(self, layer_id: int, loc: torch.Tensor, cache_k: torch.Tensor, cache_v: torch.Tensor) -> None: ...

    @abc.abstractmethod
    def get_kv_size_bytes(self) -> int: ...


class MHATokenToKVPool(KVCache):
    """Per-layer K and V buffers of shape [size + page_size, head_num, head_dim]."""

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: torch.device | str,
        v_head_dim: Optional[int] = None,
        start_layer: int = 0,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim or head_dim
        self.layer_num = layer_num
        self.device = device
        self.start_layer = start_layer
        self.store_dtype = dtype if dtype != torch.float8_e5m2 else torch.uint8

        self.k_buffer = [
            torch.zeros((size + page_size, head_num, head_dim), dtype=self.store_dtype, device=device)
            for _ in range(layer_num)
        ]
        self.v_buffer = [
            torch.zeros((size + page_size, head_num, self.v_head_dim), dtype=self.store_dtype, device=device)
            for _ in range(layer_num)
        ]
        logger.info(
            "KV cache allocated: %d tokens x %d layers, %.2f GB",
            size, layer_num, self.get_kv_size_bytes() / (1 << 30),
        )

    def get_kv_size_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self.k_buffer) + sum(
            b.numel() * b.element_size() for b in self.v_buffer
        )

    @staticmethod
    def cell_size_bytes(head_num: int, head_dim: int, layer_num: int, dtype: torch.dtype, v_head_dim: Optional[int] = None) -> int:
        """Bytes of K+V for one token across all layers."""
        itemsize = torch.empty((), dtype=dtype).element_size()
        return head_num * (head_dim + (v_head_dim or head_dim)) * layer_num * itemsize

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffer[layer_id - self.start_layer]

    def set_kv_buffer(self, layer_id: int, loc: torch.Tensor, cache_k: torch.Tensor, cache_v: torch.Tensor) -> None:
        L = layer_id - self.start_layer
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        self.k_buffer[L][loc] = cache_k.view(-1, self.head_num, self.head_dim)
        self.v_buffer[L][loc] = cache_v.view(-1, self.head_num, self.v_head_dim)

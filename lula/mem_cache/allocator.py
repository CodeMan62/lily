"""KV slot allocator. Mirrors sglang/srt/mem_cache/allocator.py (page_size = 1 variant)."""

from __future__ import annotations

import abc
from typing import Optional

import torch

from lula.mem_cache.memory_pool import KVCache


class BaseTokenToKVPoolAllocator(abc.ABC):
    def __init__(self, size: int, page_size: int, dtype: torch.dtype, device: torch.device | str, kvcache: KVCache):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache

    @abc.abstractmethod
    def available_size(self) -> int: ...

    @abc.abstractmethod
    def alloc(self, need_size: int) -> Optional[torch.Tensor]: ...

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor) -> None: ...

    @abc.abstractmethod
    def clear(self) -> None: ...


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Free list over KV-pool indices 1..size (index 0 is reserved)."""

    def __init__(self, size: int, dtype: torch.dtype, device: torch.device | str, kvcache: KVCache):
        super().__init__(size, 1, dtype, device, kvcache)
        self.clear()

    def available_size(self) -> int:
        return len(self.free_pages)

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > len(self.free_pages):
            return None
        select = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select

    def free(self, free_index: torch.Tensor) -> None:
        if free_index.numel() == 0:
            return
        self.free_pages = torch.cat((self.free_pages, free_index.to(self.free_pages.dtype)))

    def clear(self) -> None:
        self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64, device=self.device)

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
    mamba_pool: Optional["MambaStatePool"] = None

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
            torch.zeros((size + page_size, head_num, self.head_dim), dtype=self.store_dtype, device=device)
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

    def _layer_index(self, layer_id: int) -> int:
        return layer_id - self.start_layer

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.k_buffer[self._layer_index(layer_id)]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffer[self._layer_index(layer_id)]

    def set_kv_buffer(self, layer_id: int, loc: torch.Tensor, cache_k: torch.Tensor, cache_v: torch.Tensor) -> None:
        L = self._layer_index(layer_id)
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        self.k_buffer[L][loc] = cache_k.view(-1, self.head_num, self.head_dim)
        self.v_buffer[L][loc] = cache_v.view(-1, self.head_num, self.v_head_dim)


class HybridTokenToKVPool(MHATokenToKVPool):
    """KV pool for hybrid linear-attention models: only the `full_attention`
    layers own KV buffers, addressed by their global layer id."""

    def __init__(self, full_attention_layer_ids: List[int], **kw):
        self.layer_ids = list(full_attention_layer_ids)
        self.layer_index_map = {g: i for i, g in enumerate(self.layer_ids)}
        kv_kwargs = dict(kw)
        super().__init__(layer_num=len(self.layer_ids), **kv_kwargs)
        logger.info(
            "hybrid KV cache: %d full-attention layers (ids %d..%d), %.2f GB",
            len(self.layer_ids), self.layer_ids[0], self.layer_ids[-1],
            self.get_kv_size_bytes() / (1 << 30),
        )

    def _layer_index(self, layer_id: int) -> int:
        try:
            return self.layer_index_map[layer_id]
        except KeyError:
            raise KeyError(f"layer {layer_id} is not a full-attention layer ({self.layer_ids})")


class MambaStatePool:
    """Per-request linear-attention states for hybrid models.

    One slot per ReqToTokenPool slot (states[a] lives for the request holding
    req_pool_idx == a). Slot 0 is reserved as the all-zero state, so a fresh
    request with a zeroed slot starts from zero without extra init work.

    conv_states[L] stores the last `kernel_size` mixed-qkv tokens (H * W),
    recurrent_states[L] : [size, num_v_heads, k_head_dim, v_head_dim]
    """

    def __init__(self, size: int, num_layers: int, conv_dim: int, kernel_size: int,
                 num_v_heads: int, k_head_dim: int, v_head_dim: int,
                 dtype: torch.dtype, device: torch.device | str):
        self.size = size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.device = device
        self.dtype = dtype
        self.conv_states = [
            torch.zeros((size, conv_dim, kernel_size), dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.recurrent_states = [
            torch.zeros((size, num_v_heads, k_head_dim, v_head_dim), dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        gb = self.get_size_bytes() / (1 << 30)
        logger.info(
            "mamba state pool: %d slots x %d linear-attention layers, %.2f GB",
            size, num_layers, gb,
        )

    def get_size_bytes(self) -> int:
        conv = sum(b.numel() * b.element_size() for b in self.conv_states)
        rec = sum(b.numel() * b.element_size() for b in self.recurrent_states)
        return conv + rec

    def slots(self, loc: torch.Tensor) -> torch.Tensor:
        """Mamba-slot indices for request-row indices (1:1 mapping)."""
        return loc

    def zero_slots(self, loc: torch.Tensor) -> None:
        """Reset conv + recurrent states of the given slots (fresh prefill)."""
        idx = loc.to(self.device).long()
        for L in range(self.num_layers):
            self.conv_states[L][idx] = 0
            self.recurrent_states[L][idx] = 0

    def reset(self) -> None:
        self.zero_slots(torch.arange(self.size, device=self.device))

    @staticmethod
    def _dims_for_model_config(cfg):
        """(v_heads, k_head_dim, v_head_dim, conv_dim, kernel_size, num_layers) per rank."""
        from lily.distributed import get_tp_info

        tp_size, _ = get_tp_info()
        k_heads = cfg.get_linear_key_heads(tp_size)
        v_heads = cfg.get_linear_value_heads(tp_size)
        hd, vd = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        conv_dim = 2 * k_heads * hd + v_heads * vd
        return v_heads, hd, vd, conv_dim, cfg.linear_conv_kernel_size or 4, len(cfg.linear_attention_layer_ids)

    @classmethod
    def for_model_config(cls, cfg, size: int, device: torch.device | str, dtype: Optional[torch.dtype] = None):
        """Build a pool sized for this model (linear-attention layers, per rank)."""
        v_heads, hd, vd, conv_dim, kernel_size, num_layers = cls._dims_for_model_config(cfg)
        return cls(
            size=size,
            num_layers=num_layers,
            conv_dim=conv_dim,
            kernel_size=kernel_size,
            num_v_heads=v_heads,
            k_head_dim=hd,
            v_head_dim=vd,
            dtype=dtype or cfg.mamba_state_dtype,
            device=device,
        )

    def per_slot_bytes(self) -> int:
        conv = self.conv_states[0].numel() * self.conv_states[0].element_size() if self.conv_states else 0
        rec = self.recurrent_states[0].numel() * self.recurrent_states[0].element_size() if self.recurrent_states else 0
        return (conv + rec) * self.num_layers

    @classmethod
    def per_slot_bytes_for_model_config(cls, cfg) -> int:
        """Bytes per request slot without allocating the pool."""
        v_heads, hd, vd, conv_dim, kernel_size, num_layers = cls._dims_for_model_config(cfg)
        itemsize = cfg.mamba_state_dtype.itemsize
        conv = conv_dim * kernel_size * itemsize
        rec = v_heads * hd * vd * itemsize
        return (conv + rec) * num_layers

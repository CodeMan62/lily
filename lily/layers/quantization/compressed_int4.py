"""compressed-tensors `pack-quantized` checkpoints: symmetric int4 (e.g. AWQ+GPTQ, G128).

Storage per quantized linear layer:
  - `weight_packed` int32 [N, K/8]: raw int4 codes packed 8 per little-endian int32
  - `weight_scale`  bf16 [N, K/group_size]: one scale per group of `group_size` columns
  - `weight_shape`  int64 [2]: original (N, K)
Layers in the checkpoint's `ignore` list (lm_head, embed, linear-attention,
vision) fall through to the unquantized method.

Bytes-level view: LE int32 -> uint8 keeps each int4 low/high nibble pair in the
same byte order as the NVFP4/uint8 packing, so the fused decode kernel
(FAMILY_INT4) and torch unpack path need no special casing — see kernels.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from lily.layers.quantization.base import LinearMethodBase, QuantizationConfig, prefix_ignored
from lily.layers.quantization.kernels import FAMILY_INT4, weight_only_matmul

INT32_PACK = 8  # int4 values per little-endian int32


class CompressedInt4LinearMethod(LinearMethodBase):
    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    def create_weights(self, layer, input_size, output_size, params_dtype):
        assert input_size % (2 * INT32_PACK) == 0 and input_size % self.group_size == 0, (
            f"pack-quantized int4 requires K={input_size} divisible by {self.group_size}")
        layer.weight_packed = nn.Parameter(
            torch.empty(output_size, input_size // INT32_PACK, dtype=torch.int32), requires_grad=False)
        layer.weight_scale = nn.Parameter(
            torch.empty(output_size, input_size // self.group_size, dtype=torch.bfloat16), requires_grad=False)
        layer.weight_shape = nn.Parameter(torch.zeros(2, dtype=torch.int64), requires_grad=False)

    # nibble unpack identical to the uint8-packed W4A16 scheme; grouped scale along K
    def dequantized_weight(self, layer) -> torch.Tensor:
        packed = layer.weight_packed.data.view(torch.uint8)  # [N, K/2], n-th byte twin columns
        n, k_half = packed.shape
        scale = layer.weight_scale.data.to(torch.bfloat16).repeat_interleave(self.group_size, dim=1)
        u = torch.stack(((packed & 0x0F).to(torch.int8), (packed >> 4).to(torch.int8)), dim=-1)
        w = (u.view(n, -1) - 8).to(torch.bfloat16).mul_(scale)
        return w

    def apply(self, layer, x, bias=None):
        packed = layer.weight_packed.data.view(torch.uint8)
        out = weight_only_matmul(x, packed, layer.weight_scale.data, FAMILY_INT4, self.group_size)
        if out is None:
            out = F.linear(x, self.dequantized_weight(layer))
        if bias is not None:
            out.add_(bias)
        return out


class CompressedInt4Config(QuantizationConfig):
    """Checkpoint config for quant_method=compressed-tensors, format=pack-quantized.

    `tp_pack_factors`: stored-elements per original-K-column when a shard is
    sliced along the K dim (weight_packed packs 8 int4 per int32; scales one
    per group).
    """

    def __init__(self, group_size: int = 128, ignore: Optional[List[str]] = None):
        self.group_size = group_size
        self.ignore = ignore or []
        self.tp_pack_factors = {"weight_packed": INT32_PACK, "weight_scale": self.group_size}

    @classmethod
    def get_name(cls) -> str:
        return "int4-packed"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CompressedInt4Config":
        group = config.get("config_groups", {}).get("group_0", {})
        group_size = int(group.get("weights", {}).get("group_size", 128))
        return cls(group_size=group_size, ignore=list(config.get("ignore") or []))

    def get_quant_method(self, layer: nn.Module, prefix: str) -> Optional[LinearMethodBase]:
        if prefix_ignored(prefix, self.ignore):
            return None
        return CompressedInt4LinearMethod(self.group_size)

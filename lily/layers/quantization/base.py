"""Quantization plugin interface. Mirrors sglang/srt/layers/quantization/base_config.py.

Phase 1 ships only the unquantized method; Phase 9 adds FP8/AWQ/GPTQ/online quant
behind the same two classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from lily.layers.quantization.kernels import FAMILY_INT4, FAMILY_NVFP4, weight_only_matmul


class QuantizeMethodBase(ABC):
    @abstractmethod
    def create_weights(self, layer: nn.Module, *args, **kwargs) -> None: ...

    @abstractmethod
    def apply(self, layer: nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor: ...

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        return


class LinearMethodBase(QuantizeMethodBase):
    @abstractmethod
    def create_weights(
        self,
        layer: nn.Module,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> None: ...


class PackedWeightLinearMethod(LinearMethodBase):
    """Shared `apply` for methods that can use the fused dequant matmul.

    Subclasses return the kernel output from `_matmul`; when the kernel cannot
    handle the shape/dtype, apply falls back to a materialized dequantized weight.
    """

    @abstractmethod
    def _matmul(self, layer: nn.Module, x: torch.Tensor) -> Optional[torch.Tensor]: ...

    def dequantized_weight(self, layer: nn.Module) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def apply(self, layer, x, bias=None):
        out = self._matmul(layer, x)
        if out is None:
            out = F.linear(x, self.dequantized_weight(layer))
        if bias is not None:
            out.add_(bias)
        return out


class UnquantizedLinearMethod(LinearMethodBase):
    def create_weights(self, layer, input_size, output_size, params_dtype):
        weight = nn.Parameter(torch.empty(output_size, input_size, dtype=params_dtype), requires_grad=False)
        layer.register_parameter("weight", weight)

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)


class QuantizationConfig(ABC):
    """Describes a quantization scheme and hands out per-layer methods."""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig": ...

    @abstractmethod
    def get_quant_method(self, layer: nn.Module, prefix: str) -> Optional[QuantizeMethodBase]: ...


def prefix_ignored(prefix: str, patterns) -> bool:
    """True when `prefix` matches one of the checkpoint's `ignore` patterns.

    Checkpoints name modules like `model.language_model.layers...` while lily's
    own prefix is `model.layers...` (see Qwen3_5 name remap), so both forms are
    accepted.
    """
    for pattern in patterns:
        for p in (pattern, pattern.replace("model.language_model.", "model.", 1)):
            base = p.rstrip(".")
            if prefix == base or prefix.startswith(base + "."):
                return True
    return False


def unpack_int4_weight(packed: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Unpack uint8-packed symmetric int4 `[N, K/2]` with a per-group scale `[N, K/g]`."""
    n = packed.shape[0]
    scale = scale.to(torch.bfloat16).repeat_interleave(group_size, dim=1)
    u = torch.stack(((packed & 0x0F).to(torch.int8), (packed >> 4).to(torch.int8)), dim=-1)
    return (u.view(n, -1) - 8).to(torch.bfloat16).mul_(scale)


def get_linear_method(quant_config: Optional[QuantizationConfig], layer: nn.Module, prefix: str) -> LinearMethodBase:
    if quant_config is None:
        return UnquantizedLinearMethod()
    m = quant_config.get_quant_method(layer, prefix)
    return m if m is not None else UnquantizedLinearMethod()


# ---- pre-quantized checkpoint formats --------------------------------------

class _IgnoredPrefixConfig(QuantizationConfig):
    """Checkpoint config whose `ignore` patterns bypass quantization."""

    def __init__(self, ignore: Optional[List[str]] = None):
        self.ignore = ignore or []

    @abstractmethod
    def _linear_method(self) -> LinearMethodBase: ...

    def get_quant_method(self, layer: nn.Module, prefix: str) -> Optional[LinearMethodBase]:
        return None if prefix_ignored(prefix, self.ignore) else self._linear_method()


# NVFP4 (compressed-tensors `nvfp4-pack-quantized`, ModelOpt compatible, 4.5 bpw):
#   `weight_packed`       uint8 [N, K/2], E2M1 codes two per byte, low nibble first
#   `weight_scale`        float8_e4m3fn [N, K/16], one E4M3 block scale per 16 elements
#   `weight_global_scale` float32 [N] (checkpoint stores [1]; shard loaders broadcast)
# Dequantized value: e2m1 * block_scale / global_scale (global_scale = 448 * 6 / amax(W)).
NVFP4_GROUP_SIZE = 16


class NVFP4LinearMethod(PackedWeightLinearMethod):
    def create_weights(self, layer, input_size, output_size, params_dtype):
        assert input_size % (2 * NVFP4_GROUP_SIZE) == 0, (
            f"NVFP4 requires K={input_size} divisible by {2 * NVFP4_GROUP_SIZE}")
        layer.weight_packed = nn.Parameter(
            torch.empty(output_size, input_size // 2, dtype=torch.uint8), requires_grad=False)
        layer.weight_scale = nn.Parameter(
            torch.empty(output_size, input_size // NVFP4_GROUP_SIZE, dtype=torch.float8_e4m3fn),
            requires_grad=False)
        layer.weight_global_scale = nn.Parameter(torch.ones(output_size, dtype=torch.float32), requires_grad=False)

    def _matmul(self, layer, x):
        return weight_only_matmul(
            x, layer.weight_packed.data,
            layer.weight_scale.data.view(torch.uint8), FAMILY_NVFP4,
            global_scale=layer.weight_global_scale.data)

    def dequantized_weight(self, layer) -> torch.Tensor:
        """Reference decode (uint8 packed [N, K/2] -> bf16 [N, K])."""
        packed = layer.weight_packed.data
        codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
        a = codes & 0x07
        m = (a & 0x01).float()
        e = (a >> 1).float()
        val = torch.where(a < 2, m * 0.5, (1.0 + 0.5 * m) * torch.exp2(e - 1.0))
        val = val * torch.where((codes & 0x08) != 0, -1.0, 1.0)
        scales = layer.weight_scale.data.to(torch.float32)
        w = val * scales.repeat_interleave(NVFP4_GROUP_SIZE, dim=1)
        return (w / layer.weight_global_scale.data.to(w.device).view(-1, 1)).to(torch.bfloat16)


class NVFP4Config(_IgnoredPrefixConfig):
    """compressed-tensors `nvfp4-pack-quantized` checkpoint config."""

    @classmethod
    def get_name(cls) -> str:
        return "nvfp4"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NVFP4Config":
        return cls(ignore=list(config.get("ignore") or []))

    def _linear_method(self) -> LinearMethodBase:
        return NVFP4LinearMethod()


# compressed-tensors `pack-quantized` symmetric int4 (e.g. AWQ+GPTQ, G128):
#   `weight_packed` int32 [N, K/8] (8 int4 per little-endian int32)
#   `weight_scale`  bf16 [N, K/group_size]
#   `weight_shape`  int64 [2] original (N, K)
# LE int32 -> uint8 keeps nibble pairs byte-identical to the NVFP4/uint8 packing.
INT32_PACK = 8  # int4 values per little-endian int32


class CompressedInt4LinearMethod(PackedWeightLinearMethod):
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

    def _matmul(self, layer, x):
        packed = layer.weight_packed.data.view(torch.uint8)
        return weight_only_matmul(x, packed, layer.weight_scale.data, FAMILY_INT4, self.group_size)

    def dequantized_weight(self, layer) -> torch.Tensor:
        packed = layer.weight_packed.data.view(torch.uint8)  # [N, K/2], n-th byte twin columns
        return unpack_int4_weight(packed, layer.weight_scale.data, self.group_size)


class CompressedInt4Config(_IgnoredPrefixConfig):
    """Checkpoint config for quant_method=compressed-tensors, format=pack-quantized.

    `tp_pack_factors`: stored-elements per original-K-column when a shard is
    sliced along the K dim (weight_packed packs 8 int4 per int32; scales one
    per group).
    """

    def __init__(self, group_size: int = 128, ignore: Optional[List[str]] = None):
        super().__init__(ignore)
        self.group_size = group_size
        self.tp_pack_factors = {"weight_packed": INT32_PACK, "weight_scale": self.group_size}

    @classmethod
    def get_name(cls) -> str:
        return "int4-packed"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CompressedInt4Config":
        group = config.get("config_groups", {}).get("group_0", {})
        group_size = int(group.get("weights", {}).get("group_size", 128))
        return cls(group_size=group_size, ignore=list(config.get("ignore") or []))

    def _linear_method(self) -> LinearMethodBase:
        return CompressedInt4LinearMethod(self.group_size)

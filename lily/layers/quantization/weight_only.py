"""Weight-only quantization methods, applied online at load time.

Weights stream in as usual (bf16 per-rank TP shards) into a full-size
`weight` parameter; `process_weights_after_loading` converts it to the
quantized storage. `apply` dequantizes against the activation dtype and runs
plain F.linear, so everything works on any CUDA arch (no FP8/int matmul
hardware required) and stays CUDA-graph capturable.

Storage per method (weights quantized symmetrically along the output dim):
  fp8   : float8_e4m3fn weight [N, K] + fp32 per-channel scale [N] (max 448)
  w8a16 : int8 weight [N, K] + fp32 per-channel scale [N] (range [-127, 127])
  w4a16 : uint8 packed int4 [N, K/2] (two nibbles per byte, low=2i, high=2i+1,
          stored as q+8) + fp32 group scale [N, K/group_size]
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from lily.layers.quantization.base import LinearMethodBase, QuantizationConfig
from lily.layers.quantization.kernels import FAMILY_FP8, FAMILY_INT4, FAMILY_INT8, weight_only_matmul

FP8_MAX = 448.0  # float8_e4m3fn range
INT8_MAX = 127.0
INT4_MIN, INT4_MAX = -8, 7


def _replace_weight(layer: nn.Module, data: torch.Tensor) -> None:
    layer._parameters.pop("weight", None)
    layer.register_parameter("weight", nn.Parameter(data, requires_grad=False))


class WeightOnlyLinearMethod(LinearMethodBase):
    """Common scaffold: plain fp weights at load, converted after loading."""

    def create_weights(self, layer, input_size, output_size, params_dtype):
        weight = nn.Parameter(torch.empty(output_size, input_size, dtype=params_dtype), requires_grad=False)
        layer.register_parameter("weight", weight)

    def apply(self, layer, x, bias=None):
        out = F.linear(x, self.dequantized_weight(layer))
        if bias is not None:
            out.add_(bias)
        return out

    def dequantized_weight(self, layer) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class W8A16LinearMethod(WeightOnlyLinearMethod):
    """int8 + per-channel scale. `y = (x @ Wq^T) * scale`."""

    def process_weights_after_loading(self, layer):
        with torch.no_grad():
            w = layer.weight.data.float()
            scale = (w.abs().amax(dim=1) / INT8_MAX).clamp_(min=1e-12).float()
            q = torch.round(w / scale.unsqueeze(1)).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
            _replace_weight(layer, q)
            layer.weight_scale = nn.Parameter(scale, requires_grad=False)

    def apply(self, layer, x, bias=None):
        out = weight_only_matmul(x, layer.weight.data, layer.weight_scale.data, FAMILY_INT8)
        if out is None:
            out = F.linear(x, layer.weight.to(x.dtype))
            out.mul_(layer.weight_scale.to(out.dtype))
        if bias is not None:
            out.add_(bias)
        return out

    def dequantized_weight(self, layer):  # pragma: no cover
        return layer.weight.to(torch.float32) * layer.weight_scale.unsqueeze(1)


class FP8LinearMethod(WeightOnlyLinearMethod):
    """float8_e4m3fn + per-channel scale, same math as W8A16."""

    def process_weights_after_loading(self, layer):
        with torch.no_grad():
            w = layer.weight.data.float()
            scale = (w.abs().amax(dim=1) / FP8_MAX).clamp_(min=1e-12).float()
            q = (w / scale.unsqueeze(1)).to(torch.float8_e4m3fn)
            _replace_weight(layer, q)
            layer.weight_scale = nn.Parameter(scale, requires_grad=False)

    def apply(self, layer, x, bias=None):
        # byte view: the kernel decodes e4m3 manually (fp8 dtype pointers are sm89+
        out = weight_only_matmul(x, layer.weight.data.view(torch.uint8), layer.weight_scale.data, FAMILY_FP8)
        if out is None:
            out = F.linear(x, layer.weight.to(x.dtype))
            out.mul_(layer.weight_scale.to(out.dtype))
        if bias is not None:
            out.add_(bias)
        return out

    def dequantized_weight(self, layer):  # pragma: no cover
        return layer.weight.to(torch.float32) * layer.weight_scale.unsqueeze(1)


class W4A16LinearMethod(WeightOnlyLinearMethod):
    """Grouped int4 (group_size along the input dim) packed into uint8."""

    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    def process_weights_after_loading(self, layer):
        with torch.no_grad():
            g = self.group_size
            w = layer.weight.data.float()
            n, k = w.shape
            pad = -k % g
            if pad:
                w = F.pad(w, (0, pad))  # zero rows dequantize to zero; trimmed in apply
            groups = w.view(n, -1, g)
            amax = groups.abs().amax(dim=-1, keepdim=True)
            scale = (amax / (INT4_MAX + 1)).clamp_(min=1e-12).float()  # [N, K/g, 1]
            q = torch.round(groups / scale).clamp(INT4_MIN, INT4_MAX).to(torch.int16) + 8  # -> 0..15
            packed = (q[..., 0::2] | (q[..., 1::2] << 4)).view(n, -1).to(torch.uint8)  # [N, K_pad/2]
            _replace_weight(layer, packed)
            layer.weight_scale = nn.Parameter(scale.squeeze(-1).float(), requires_grad=False)  # [N, K/g]

    def dequantized_weight(self, layer):
        packed = layer.weight.data  # [N, K_pad/2] uint8
        n, k_half = packed.shape
        # nibble unpack + group rescale, bf16 throughout (fp32 here shows up in decode latency)
        scale = layer.weight_scale.data.to(torch.bfloat16).repeat_interleave(self.group_size, dim=1)  # [N, K_pad]
        u = torch.stack(((packed & 0x0F).to(torch.int8), (packed >> 4).to(torch.int8)), dim=-1)
        w = (u.view(n, -1) - 8).to(torch.bfloat16).mul_(scale)
        return w[:, : layer.input_size]

    def apply(self, layer, x, bias=None):
        out = weight_only_matmul(x, layer.weight.data, layer.weight_scale.data,
                                 FAMILY_INT4, self.group_size)
        if out is None:
            out = F.linear(x, self.dequantized_weight(layer))
        if bias is not None:
            out.add_(bias)
        return out


class W8A16Config(QuantizationConfig):
    @classmethod
    def get_name(cls) -> str:
        return "w8a16"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "W8A16Config":
        return cls()

    def get_quant_method(self, layer: nn.Module, prefix: str) -> Optional[LinearMethodBase]:
        return W8A16LinearMethod()


class FP8Config(QuantizationConfig):
    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FP8Config":
        return cls()

    def get_quant_method(self, layer: nn.Module, prefix: str) -> Optional[LinearMethodBase]:
        return FP8LinearMethod()


class W4A16Config(QuantizationConfig):
    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    @classmethod
    def get_name(cls) -> str:
        return "w4a16"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "W4A16Config":
        return cls(group_size=int(config.get("group_size", 128)))

    def get_quant_method(self, layer: nn.Module, prefix: str) -> Optional[LinearMethodBase]:
        return W4A16LinearMethod(self.group_size)

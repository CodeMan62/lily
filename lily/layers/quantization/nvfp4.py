"""NVFP4 (compressed-tensors `nvfp4-pack-quantized`) pre-quantized checkpoints.

Format per linear layer (ModelOpt / compressed-tensors compatible, 4.5 bpw):
  - `weight_packed`   uint8 [N, K/2]: E2M1 codes packed two per byte, low nibble first
  - `weight_scale`    float8_e4m3fn [N, K/16]: one E4M3 block scale per 16 elements
  - `weight_global_scale` float32 [1] (checkpoint name `weight_global_scale`)
Dequantized value: e2m1 * block_scale / global_scale  (compressed-tensors /
ModelOpt convention: global_scale = 448 * 6 / amax(W), so it is a large number
such as 6000 and block scales sit in [1, 448]).

The checkpoint stores one global scale per linear, but fused layers
(qkv_proj, gate_up_proj, in_proj_qkv) merge several checkpoint linears that
each carry their own scalar. We therefore keep the global scale per output
row ([N] fp32); the shard loaders broadcast each incoming scalar over its
shard's rows and the kernel reads it per output column.

Layers in the checkpoint's `ignore` list (lm_head, embed, vision, mtp) fall
through to the unquantized method and stream in as bf16.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from lily.layers.quantization.base import LinearMethodBase, QuantizationConfig, prefix_ignored
from lily.layers.quantization.kernels import FAMILY_NVFP4, weight_only_matmul

GROUP_SIZE = 16

# E2M1 code -> value (bit 3 = sign); same LUT as hollowquant's reference.
_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


class NVFP4LinearMethod(LinearMethodBase):
    def create_weights(self, layer, input_size, output_size, params_dtype):
        assert input_size % (2 * GROUP_SIZE) == 0, (
            f"NVFP4 requires K={input_size} divisible by {2 * GROUP_SIZE}")
        layer.weight_packed = nn.Parameter(
            torch.empty(output_size, input_size // 2, dtype=torch.uint8), requires_grad=False)
        layer.weight_scale = nn.Parameter(
            torch.empty(output_size, input_size // GROUP_SIZE, dtype=torch.float8_e4m3fn),
            requires_grad=False)
        # Per output row (see module docstring); loaders broadcast the checkpoint's [1] scalar.
        layer.weight_global_scale = nn.Parameter(torch.ones(output_size, dtype=torch.float32), requires_grad=False)

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
        w = val * scales.repeat_interleave(GROUP_SIZE, dim=1)
        return (w / layer.weight_global_scale.data.to(w.device).view(-1, 1)).to(torch.bfloat16)

    def apply(self, layer, x, bias=None):
        out = weight_only_matmul(
            x, layer.weight_packed.data,
            layer.weight_scale.data.view(torch.uint8), FAMILY_NVFP4,
            global_scale=layer.weight_global_scale.data)
        if out is None:
            out = F.linear(x, self.dequantized_weight(layer))
        if bias is not None:
            out.add_(bias)
        return out


class NVFP4Config(QuantizationConfig):
    """compressed-tensors `nvfp4-pack-quantized` checkpoint config."""

    def __init__(self, ignore: Optional[List[str]] = None):
        self.ignore = ignore or []

    @classmethod
    def get_name(cls) -> str:
        return "nvfp4"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NVFP4Config":
        return cls(ignore=list(config.get("ignore") or []))

    def get_quant_method(self, layer: nn.Module, prefix: str) -> Optional[LinearMethodBase]:
        if prefix_ignored(prefix, self.ignore):
            return None
        return NVFP4LinearMethod()

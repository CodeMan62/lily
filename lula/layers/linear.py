"""Linear layers with SGLang/vLLM names and weight-loader semantics.

Single-GPU for now (tp_size = 1), but shard ids and loaders are kept so the
weight-loading code in models/ is the same one that will work under TP.
Every forward returns `(output, output_bias)` like SGLang; `output_bias` is
only non-None when `skip_bias_add=True`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn

from lula.layers.quantization.base import LinearMethodBase, QuantizationConfig, get_linear_method


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    assert param.shape == loaded_weight.shape, f"shape mismatch {tuple(param.shape)} vs {tuple(loaded_weight.shape)}"
    param.data.copy_(loaded_weight)


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype or torch.get_default_dtype()
        self.prefix = prefix
        self.quant_method: LinearMethodBase = get_linear_method(quant_config, self, prefix)
        self.quant_method.create_weights(self, input_size, output_size, self.params_dtype)
        self.weight.weight_loader = self.weight_loader  # type: ignore[attr-defined]
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size, dtype=self.params_dtype), requires_grad=False)
            self.bias.weight_loader = self.weight_loader  # type: ignore[attr-defined]
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor, *args) -> None:
        default_weight_loader(param, loaded_weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bias = self.bias if not self.skip_bias_add else None
        out = self.quant_method.apply(self, x, bias)
        return out, (self.bias if self.skip_bias_add else None)


class MergedColumnParallelLinear(LinearBase):
    """Several column-parallel linears fused along the output dim (e.g. gate_proj + up_proj)."""

    def __init__(self, input_size: int, output_sizes: List[int], bias: bool, **kw):
        self.output_sizes = list(output_sizes)
        super().__init__(input_size, sum(output_sizes), bias, **kw)

    def weight_loader(self, param, loaded_weight, loaded_shard_id: Optional[int] = None):
        if loaded_shard_id is None:
            default_weight_loader(param, loaded_weight)
            return
        offset = sum(self.output_sizes[:loaded_shard_id])
        size = self.output_sizes[loaded_shard_id]
        assert loaded_weight.shape[0] == size, (loaded_weight.shape, size)
        param.data.narrow(0, offset, size).copy_(loaded_weight)


class QKVParallelLinear(LinearBase):
    """Fused q/k/v projection. Shard ids are "q", "k", "v"."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = False,
        **kw,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.num_heads = self.total_num_heads
        self.num_kv_heads = self.total_num_kv_heads
        self.q_size = self.num_heads * head_size
        self.kv_size = self.num_kv_heads * head_size
        super().__init__(hidden_size, self.q_size + 2 * self.kv_size, bias, **kw)

    def weight_loader(self, param, loaded_weight, loaded_shard_id: Optional[str] = None):
        if loaded_shard_id is None:
            default_weight_loader(param, loaded_weight)
            return
        offsets = {"q": 0, "k": self.q_size, "v": self.q_size + self.kv_size}
        sizes = {"q": self.q_size, "k": self.kv_size, "v": self.kv_size}
        off, size = offsets[loaded_shard_id], sizes[loaded_shard_id]
        assert loaded_weight.shape[0] == size, (loaded_shard_id, loaded_weight.shape, size)
        param.data.narrow(0, off, size).copy_(loaded_weight)

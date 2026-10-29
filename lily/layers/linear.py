"""Linear layers with SGLang/vLLM names and weight-loader semantics.

Under TP the loader hands each layer only its rank's shard of the checkpoint
(see models/loader.py), so layers just size their params per rank.
Every forward returns `(output, output_bias)` like SGLang; `output_bias` is
only non-None when `skip_bias_add=True`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn

from lily.distributed import get_tp_info, tensor_model_parallel_all_reduce
from lily.layers.quantization.base import LinearMethodBase, QuantizationConfig, get_linear_method


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    if loaded_weight.numel() == 1 and param.numel() != 1:
        # Per-tensor quant scalar (e.g. NVFP4 weight_global_scale) stored per row.
        param.data.fill_(loaded_weight.reshape(()).item())
        return
    assert param.shape == loaded_weight.shape, f"shape mismatch {tuple(param.shape)} vs {tuple(loaded_weight.shape)}"
    param.data.copy_(loaded_weight)


def _broadcast_scalar(loaded_weight: torch.Tensor, size: int) -> torch.Tensor:
    """A checkpoint [1] scalar (per-tensor quant scale) fills every row of its shard."""
    if loaded_weight.numel() == 1 and size != 1:
        return loaded_weight.reshape(1).expand(size)
    return loaded_weight


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
        for param in self._parameters.values():
            if getattr(param, "weight_loader", None) is None:
                param.weight_loader = self.weight_loader  # type: ignore[attr-defined]
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
        tp_size, _ = get_tp_info()
        self.output_sizes = [size // tp_size for size in output_sizes]
        super().__init__(input_size, sum(self.output_sizes), bias, **kw)

    def weight_loader(self, param, loaded_weight, loaded_shard_id: Optional[int] = None):
        if loaded_shard_id is None:
            default_weight_loader(param, loaded_weight)
            return
        offset = sum(self.output_sizes[:loaded_shard_id])
        size = self.output_sizes[loaded_shard_id]
        loaded_weight = _broadcast_scalar(loaded_weight, size)
        assert loaded_weight.shape[0] == size, (self.prefix, loaded_shard_id, loaded_weight.shape, size)
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
        loaded_weight = _broadcast_scalar(loaded_weight, size)
        assert loaded_weight.shape[0] == size, (loaded_shard_id, loaded_weight.shape, size)
        param.data.narrow(0, off, size).copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """Input dim split across TP ranks (o_proj, down_proj); partial outputs are all-reduced."""

    def __init__(self, input_size: int, output_size: int, bias: bool, **kw):
        tp_size, _ = get_tp_info()
        super().__init__(input_size // tp_size, output_size, bias, **kw)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = tensor_model_parallel_all_reduce(self.quant_method.apply(self, x))
        # Bias is replicated, so add it once after the reduce.
        if self.bias is not None and not self.skip_bias_add:
            out = out + self.bias
        return out, (self.bias if self.skip_bias_add else None)

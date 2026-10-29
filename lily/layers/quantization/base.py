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


def get_linear_method(quant_config: Optional[QuantizationConfig], layer: nn.Module, prefix: str) -> LinearMethodBase:
    if quant_config is None:
        return UnquantizedLinearMethod()
    m = quant_config.get_quant_method(layer, prefix)
    return m if m is not None else UnquantizedLinearMethod()

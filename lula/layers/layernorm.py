from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm with optional fused residual add (SGLang/vLLM convention).

    forward(x) -> normed
    forward(x, residual) -> (normed(x + residual), x + residual)
    Numerics follow HF: the sum is taken in the activation dtype, the variance in fp32.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            x = x + residual
            residual = x
        orig_dtype = x.dtype
        xf = x.to(torch.float32)
        variance = xf.pow(2).mean(dim=-1, keepdim=True)
        xf = xf * torch.rsqrt(variance + self.variance_epsilon)
        out = self.weight * xf.to(orig_dtype)
        if residual is None:
            return out
        return out, residual

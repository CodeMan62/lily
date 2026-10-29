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

    def __init__(self, hidden_size: int, eps: float = 1e-6, zero_centered: bool = False):
        super().__init__()
        # zero_centered (Qwen3.5 `RMSNormZeroCentered`): out = norm(x) * (1 + weight),
        # computed in fp32; the checkpoint weight is an offset around zero.
        self.zero_centered = zero_centered
        self.weight = nn.Parameter(torch.zeros(hidden_size) if zero_centered else torch.ones(hidden_size))
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
        if self.zero_centered:
            out = (xf * (1.0 + self.weight.float())).to(orig_dtype)
        else:
            out = self.weight * xf.to(orig_dtype)
        if residual is None:
            return out
        return out, residual

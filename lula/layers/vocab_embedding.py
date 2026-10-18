from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class VocabParallelEmbedding(nn.Module):
    """Token embedding table (vocab would be sharded under TP)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, params_dtype: Optional[torch.dtype] = None, prefix: str = ""):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        dtype = params_dtype or torch.get_default_dtype()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype), requires_grad=False)
        self.weight.weight_loader = self.weight_loader  # type: ignore[attr-defined]

    def weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        # Checkpoints may carry fewer rows than the (padded) vocab size.
        n = loaded_weight.shape[0]
        assert n <= param.shape[0] and loaded_weight.shape[1] == param.shape[1], (loaded_weight.shape, param.shape)
        param.data[:n].copy_(loaded_weight)
        if n < param.shape[0]:
            param.data[n:].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)


class ParallelLMHead(VocabParallelEmbedding):
    """Output projection. Used by LogitsProcessor via `.weight` / `.bias`; never called directly."""

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False, params_dtype: Optional[torch.dtype] = None, prefix: str = ""):
        super().__init__(num_embeddings, embedding_dim, params_dtype, prefix)
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_embeddings, dtype=self.weight.dtype), requires_grad=False)
            self.bias.weight_loader = self.weight_loader  # type: ignore[attr-defined]
        else:
            self.register_parameter("bias", None)

    def tie_weights(self, embed_tokens: VocabParallelEmbedding) -> "ParallelLMHead":
        self.weight = embed_tokens.weight
        return self

    def forward(self, input_ids):  # pragma: no cover
        raise RuntimeError("ParallelLMHead is applied by LogitsProcessor, not called directly")

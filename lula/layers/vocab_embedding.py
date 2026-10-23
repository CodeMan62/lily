from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from lula.distributed import get_tp_info, tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce
from lula.utils import ceil_div


class VocabParallelEmbedding(nn.Module):
    """Token embedding table, sharded by vocab range under TP: rank r holds rows [vocab_start, vocab_end)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, params_dtype: Optional[torch.dtype] = None, prefix: str = ""):
        super().__init__()
        size, rank = get_tp_info()
        self.tp_size = size
        self.num_embeddings = num_embeddings
        # Ceil-divide so only the last rank's shard is short (zero-padded by weight_loader).
        self.num_embeddings_per_partition = ceil_div(num_embeddings, size)
        dtype = params_dtype or torch.get_default_dtype()
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim, dtype=dtype), requires_grad=False)
        self.weight.weight_loader = self.weight_loader  # type: ignore[attr-defined]
        self.vocab_start = self.num_embeddings_per_partition * rank
        self.vocab_end = self.vocab_start + self.num_embeddings_per_partition

    def weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        # Checkpoints may carry fewer rows than the (padded) vocab size.
        n = loaded_weight.shape[0]
        assert n <= param.shape[0] and loaded_weight.shape[1] == param.shape[1], (loaded_weight.shape, param.shape)
        param.data[:n].copy_(loaded_weight)
        if n < param.shape[0]:
            param.data[n:].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return F.embedding(input_ids, self.weight)
        # Ids owned by other ranks embed to zero here; the all-reduce fills them in.
        mask = (input_ids < self.vocab_start) | (input_ids >= self.vocab_end)
        out = F.embedding((input_ids - self.vocab_start).masked_fill(mask, 0), self.weight)
        out.masked_fill_(mask.unsqueeze(-1), 0)
        return tensor_model_parallel_all_reduce(out)


class ParallelLMHead(VocabParallelEmbedding):
    """Output projection, applied via `compute_logits`; never called directly."""

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False, params_dtype: Optional[torch.dtype] = None, prefix: str = ""):
        super().__init__(num_embeddings, embedding_dim, params_dtype, prefix)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.num_embeddings_per_partition, dtype=self.weight.dtype), requires_grad=False)
            self.bias.weight_loader = self.weight_loader  # type: ignore[attr-defined]
        else:
            self.register_parameter("bias", None)

    def tie_weights(self, embed_tokens: VocabParallelEmbedding) -> "ParallelLMHead":
        self.weight = embed_tokens.weight
        return self

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[n, hidden] -> [n, vocab] float32, full vocab on every rank."""
        logits = F.linear(hidden_states, self.weight, self.bias)
        logits = tensor_model_parallel_all_gather(logits, dim=-1)
        return logits[:, : self.num_embeddings].float()

    def forward(self, input_ids):  # pragma: no cover
        raise RuntimeError("ParallelLMHead is applied by LogitsProcessor, not called directly")

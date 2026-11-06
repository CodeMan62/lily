from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from lily.distributed import get_tp_info, tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce
from lily.utils import ceil_div


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
    """Output projection, applied via `compute_logits`; never called directly.

    With a `quant_method` (checkpoint-quantized `lm_head.weight_packed`, or
    online `--quantize-lm-head`) the projection runs through the quantized
    linear path: on a 248k vocab that is a 0.72 GB read per token instead of
    2.5 GB in bf16.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False, params_dtype: Optional[torch.dtype] = None,
                 prefix: str = "", quant_method=None):
        super().__init__(num_embeddings, embedding_dim, params_dtype, prefix)
        self.quant_method = quant_method
        if quant_method is not None:
            # Replace the bf16 table with the method's storage (vocab rows per rank).
            self._parameters.pop("weight", None)
            self.input_size = embedding_dim
            quant_method.create_weights(self, embedding_dim, self.num_embeddings_per_partition, params_dtype or torch.get_default_dtype())
            for param in self._parameters.values():
                param.weight_loader = self.quant_weight_loader  # type: ignore[attr-defined]
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.num_embeddings_per_partition, dtype=params_dtype or torch.get_default_dtype()), requires_grad=False)
            self.bias.weight_loader = self.weight_loader  # type: ignore[attr-defined]
        else:
            self.register_parameter("bias", None)

    def quant_weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor, *args) -> None:
        """Rows are vocab entries (zero-padded like the embedding); a [1] tensor is a per-tensor scale."""
        if loaded_weight.numel() == 1 and param.numel() != 1:
            param.data.fill_(loaded_weight.reshape(()).item())
            return
        n = loaded_weight.shape[0]
        assert n <= param.shape[0] and loaded_weight.shape[1:] == param.shape[1:], (loaded_weight.shape, param.shape)
        param.data[:n].copy_(loaded_weight)
        if n < param.shape[0]:
            param.data[n:].zero_()

    def tie_weights(self, embed_tokens: VocabParallelEmbedding) -> "ParallelLMHead":
        assert self.quant_method is None, "a tied lm_head shares the embedding table and cannot be quantized separately"
        self.weight = embed_tokens.weight
        return self

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[n, hidden] -> [n, vocab] float32, full vocab on every rank."""
        if self.quant_method is not None:
            logits = self.quant_method.apply(self, hidden_states, self.bias)
        else:
            logits = F.linear(hidden_states, self.weight, self.bias)
        logits = tensor_model_parallel_all_gather(logits, dim=-1)
        return logits[:, : self.num_embeddings].float()

    def compute_topk(self, hidden_states: torch.Tensor, k: int):
        """Top-k of the full-vocab logits without gathering them: each rank's top-k of its shard, then the
        top-k of the gathered candidates -> (values float32 [n, k], token ids int64 [n, k])."""
        if self.quant_method is not None:
            logits = self.quant_method.apply(self, hidden_states, self.bias)
        else:
            logits = F.linear(hidden_states, self.weight, self.bias)
        logits = logits.float()
        valid = self.num_embeddings - self.vocab_start            # padded rows of the last shard never win
        if valid < logits.shape[-1]:
            logits[:, max(valid, 0):] = float("-inf")
        values, ids = torch.topk(logits, min(k, logits.shape[-1]), dim=-1, sorted=False)
        ids = ids + self.vocab_start
        if self.tp_size > 1:
            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            values, pick = torch.topk(values, k, dim=-1, sorted=False)
            ids = ids.gather(1, pick)
        return values, ids

    def forward(self, input_ids):  # pragma: no cover
        raise RuntimeError("ParallelLMHead is applied by LogitsProcessor, not called directly")

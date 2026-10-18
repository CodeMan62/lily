"""Token sampler. Mirrors sglang/srt/layers/sampler.py (reduced).

Order of operations per request, matching HF's `LogitsProcessor` pipeline:
  1. logit bias (additive), repetition / frequency / presence penalties
  2. greedy fast path (argmax) when every request in the batch is greedy
  3. temperature scaling
  4. top-k, then top-p and min-p over the surviving candidates
  5. multinomial draw with the request's own seeded generator

`top_k` uses `torch.topk`, so a small top-k never sorts the full vocab (matters
at 248k). `top_p` and `min_p` are applied on the top-k candidates only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from lula.sampling.sampling_batch_info import SamplingBatchInfo
from lula.sampling.sampling_params import SamplingParams


@dataclass
class SamplerOutput:
    next_token_ids: torch.Tensor
    # [bs] log-probability of the chosen token, when requested.
    next_token_logprobs: Optional[torch.Tensor] = None
    # [bs] list of (token_id, logprob) for the most likely candidates.
    top_logprobs: Optional[List[List[Tuple[int, float]]]] = None


class Sampler:
    def __call__(self, logits: torch.Tensor, info: SamplingBatchInfo) -> SamplerOutput:
        if logits.numel() == 0 or len(info) == 0:
            return SamplerOutput(torch.empty(0, dtype=torch.int64, device=logits.device))

        device = logits.device
        assert logits.shape[0] == len(info), (logits.shape, len(info))
        logits = logits.float()
        self._apply_penalties(logits, info)

        need_logprobs = info.need_logprobs
        if info.all_greedy:
            ids = torch.argmax(logits, dim=-1)
            logprobs = torch.log_softmax(logits, dim=-1) if need_logprobs else None
            return self._finish(ids, logprobs, info)

        bs = logits.shape[0]
        ids = torch.empty(bs, dtype=torch.int64, device=device)
        chosen_logprobs: List[float] = []
        for i, param in enumerate(info.params):
            token, logprob = self._sample_one(logits[i], param, info.generators[i])
            ids[i] = token
            chosen_logprobs.append(logprob)

        next_logprobs = torch.tensor(chosen_logprobs, device=device) if need_logprobs else None
        return SamplerOutput(next_token_ids=ids, next_token_logprobs=next_logprobs)

    # ---- penalties ---------------------------------------------------------

    def _apply_penalties(self, logits: torch.Tensor, info: SamplingBatchInfo) -> None:
        for i, param in enumerate(info.params):
            token_ids = info.token_ids[i]
            if param.repetition_penalty != 1.0 and token_ids:
                unique = torch.tensor(sorted(set(token_ids)), device=logits.device)
                values = logits[i, unique]
                logits[i, unique] = torch.where(
                    values > 0, values / param.repetition_penalty, values * param.repetition_penalty
                )
            if (param.frequency_penalty or param.presence_penalty) and token_ids:
                counts = torch.bincount(
                    torch.tensor(token_ids, device=logits.device), minlength=logits.shape[-1]
                ).to(logits.dtype)
                if param.frequency_penalty:
                    logits[i] -= counts * param.frequency_penalty
                if param.presence_penalty:
                    logits[i] -= (counts > 0).to(logits.dtype) * param.presence_penalty
            if param.logit_bias:
                bias_ids = torch.tensor(list(param.logit_bias.keys()), device=logits.device)
                bias_values = torch.tensor(
                    list(param.logit_bias.values()), dtype=logits.dtype, device=logits.device
                )
                logits[i, bias_ids] += bias_values

    # ---- sampling ----------------------------------------------------------

    def _sample_one(
        self, logits_row: torch.Tensor, param: SamplingParams, generator: torch.Generator
    ) -> Tuple[int, float]:
        if param.greedy:
            token = int(torch.argmax(logits_row))
            return token, float(torch.log_softmax(logits_row, dim=-1)[token])

        vocab = logits_row.shape[-1]
        x = logits_row / max(param.temperature, 1e-5)

        if 0 < param.top_k < vocab:
            values, indices = torch.topk(x, param.top_k)
        else:
            values, indices = torch.sort(x, descending=True)

        probs = torch.softmax(values, dim=-1)
        if param.top_p < 1.0:
            cumulative = torch.cumsum(probs, dim=-1)
            keep = cumulative - probs < param.top_p
            keep[0] = True
            probs = torch.where(keep, probs, torch.zeros_like(probs))
        if param.min_p > 0.0:
            probs = torch.where(probs >= param.min_p * probs.max(), probs, torch.zeros_like(probs))
        probs = probs / probs.sum()

        sampled = int(torch.multinomial(probs, num_samples=1, generator=generator))
        token = int(indices[sampled])
        return token, float(torch.log(probs[sampled]))

    # ---- logprobs ----------------------------------------------------------

    def _finish(self, ids: torch.Tensor, logprobs: Optional[torch.Tensor],
                info: SamplingBatchInfo) -> SamplerOutput:
        chosen = logprobs.gather(1, ids.unsqueeze(1)).squeeze(1) if logprobs is not None else None
        top: Optional[List[List[Tuple[int, float]]]] = None
        if logprobs is not None and any(p.top_logprobs for p in info.params):
            top = []
            for i, param in enumerate(info.params):
                k = param.top_logprobs or 0
                if k <= 0:
                    top.append([])
                    continue
                values, indices = torch.topk(logprobs[i], min(k, logprobs.shape[-1]))
                top.append([(int(t), float(v)) for t, v in zip(indices.tolist(), values.tolist())])
        return SamplerOutput(next_token_ids=ids, next_token_logprobs=chosen, top_logprobs=top)

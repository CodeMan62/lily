"""Batched sampling state. Mirrors sglang/srt/sampling/sampling_batch_info.py.

Holds one `SamplingParams` per request, the token ids used for penalties, and a
per-request `torch.Generator` so a seeded request samples reproducibly across
steps. Generators live on the `Req` (see `schedule_batch.Req`) and are carried
into the batch, so they keep advancing as the request is decoded.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import torch

from lula.sampling.sampling_params import SamplingParams
from lula.utils import new_generator


class SamplingBatchInfo:
    def __init__(
        self,
        params: Sequence[SamplingParams],
        token_ids: Optional[Sequence[Sequence[int]]] = None,
        generators: Optional[Sequence[torch.Generator]] = None,
        device: torch.device | str = "cuda",
    ):
        self.params: List[SamplingParams] = list(params)
        self.token_ids: List[List[int]] = [list(x) for x in (token_ids or [[] for _ in self.params])]
        if generators is None:
            generators = [new_generator(p.seed, device) for p in self.params]
        self.generators: List[torch.Generator] = list(generators)

    @classmethod
    def from_reqs(cls, reqs: Sequence[Any]) -> "SamplingBatchInfo":
        return cls(
            [r.sampling_params for r in reqs],
            [r.origin_input_ids + r.output_ids for r in reqs],
            [r.generator for r in reqs],
        )

    def refresh_from_reqs(self, reqs: Sequence[Any]) -> None:
        """Re-read the token ids used for penalties (prompt + generated so far)."""
        self.token_ids = [r.origin_input_ids + r.output_ids for r in reqs]

    def __len__(self) -> int:
        return len(self.params)

    @property
    def all_greedy(self) -> bool:
        return all(p.greedy for p in self.params)

    @property
    def need_logprobs(self) -> bool:
        return any(p.logprobs for p in self.params)

    def filter_batch(self, indices: Sequence[int]) -> "SamplingBatchInfo":
        return SamplingBatchInfo(
            [self.params[i] for i in indices],
            [self.token_ids[i] for i in indices],
            [self.generators[i] for i in indices],
        )

    def merge_batch(self, other: "SamplingBatchInfo") -> "SamplingBatchInfo":
        return SamplingBatchInfo(
            self.params + other.params,
            self.token_ids + other.token_ids,
            self.generators + other.generators,
        )

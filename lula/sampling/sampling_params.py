"""Per-request sampling parameters. Mirrors sglang/srt/sampling/sampling_params.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class SamplingParams:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: Optional[Dict[int, float]] = None
    stop_token_ids: List[int] = field(default_factory=list)
    stop: Optional[Union[str, List[str]]] = None
    ignore_eos: bool = False
    seed: Optional[int] = None
    stream: bool = False
    n: int = 1
    logprobs: bool = False
    top_logprobs: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not (0 < self.top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError("top_k must be -1 or a positive integer")
        if not (0 <= self.min_p < 1):
            raise ValueError("min_p must be in [0, 1)")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if isinstance(self.stop, str):
            self.stop = [self.stop]
        if self.top_logprobs is not None:
            self.logprobs = True
        if self.logit_bias is not None:
            self.logit_bias = {int(k): float(v) for k, v in self.logit_bias.items()}

    @property
    def greedy(self) -> bool:
        """No randomness: a single deterministic token is produced."""
        return self.temperature == 0 or self.top_k == 1

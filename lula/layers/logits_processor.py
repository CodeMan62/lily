"""Final projection to logits output container."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LogitsProcessorOutput:
    # [num_output_tokens, vocab] in float32. One row per request for EXTEND/DECODE.
    next_token_logits: torch.Tensor
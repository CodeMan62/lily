"""Final projection to logits output container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LogitsProcessorOutput:
    # [num_output_tokens, vocab] in float32. One row per request for EXTEND/DECODE,
    next_token_logits: torch.Tensor
    # [num_tokens, hidden] final-norm hidden states, when the batch asked for them
    hidden_states: Optional[torch.Tensor] = None
    # [num_tokens, len(capture_aux_layers) * hidden]: residual stream after the requested layers.
    aux_hidden_states: Optional[torch.Tensor] = None
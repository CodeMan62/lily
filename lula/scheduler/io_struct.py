"""Request/response dataclasses exchanged between processes. Mirrors sglang/srt/managers/io_struct.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lula.sampling.sampling_params import SamplingParams


@dataclass
class GenerateReqInput:
    """A user request as it arrives at the tokenizer frontend (pre-tokenization)."""

    text: Optional[str] = None
    input_ids: Optional[List[int]] = None
    messages: Optional[List[Dict[str, Any]]] = None
    sampling_params: Optional[Dict[str, Any]] = None
    stream: bool = False
    rid: Optional[str] = None
    is_chat: bool = False
    chat_template_kwargs: Optional[Dict[str, Any]] = None

    def to_sampling_params(self) -> SamplingParams:
        params = dict(self.sampling_params or {})
        params.setdefault("stream", self.stream)
        return SamplingParams(**params)


@dataclass
class TokenizedGenerateReqInput:
    """Tokenized request sent tokenizer frontend -> scheduler."""

    rid: str
    input_ids: List[int]
    sampling_params: SamplingParams
    stream: bool = False


@dataclass
class BatchTokenIDOut:
    """Scheduler -> detokenizer: newly sampled token ids for a batch of requests."""

    rids: List[str]
    output_ids: List[List[int]]
    finished_reasons: List[Optional[dict]]
    prompt_tokens: List[int]
    completion_tokens: List[int]
    stop_strs: List[List[str]]
    logprobs: Optional[List[Optional[List[float]]]] = None


@dataclass
class BatchStrOut:
    """Detokenizer -> tokenizer frontend: incremental text per request."""

    rids: List[str]
    output_strs: List[str]
    finished_reasons: List[Optional[dict]]
    meta_info: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AbortReq:
    rid: str


@dataclass
class FlushCacheReq:
    pass

"""Single-process, single-sequence engine. Mirrors sglang/srt/entrypoints/engine.py (reduced).

Drives one request at a time directly through a `ModelRunner`, without the
scheduler/detokenizer processes. It plays the scheduler's role for KV
bookkeeping: allocate a request slot, allocate KV slots for new tokens, write
them into `req_to_token`, run, and free everything at the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoTokenizer

from lily.models.model_config import ModelConfig
from lily.distributed import DEFAULT_DIST_INIT_ADDR
from lily.model_executor.forward_batch_info import ForwardBatch
from lily.model_executor.model_runner import ModelRunner
from lily.server_args import ServerArgs


@dataclass
class EngineConfig:
    model_path: str
    dtype: str = "auto"
    attention_backend: str = "triton"
    quantization: Optional[str] = None
    device: str = "cuda"
    context_length: Optional[int] = None
    max_running_requests: int = 2
    max_total_tokens: Optional[int] = None
    mem_fraction_static: Optional[float] = None
    tp_size: int = 1
    rank: int = 0
    dist_init_addr: str = DEFAULT_DIST_INIT_ADDR
    # Any other ServerArgs field, e.g. {"mamba_ssm_dtype": "bfloat16", "quantize_lm_head": "nvfp4"}.
    server_args: Optional[dict] = None

class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.server_args = ServerArgs(model_path=config.model_path, dtype=config.dtype,
                                      attention_backend=config.attention_backend,
                                      quantization=config.quantization,
                                      context_length=config.context_length,
                                      max_running_requests=config.max_running_requests,
                                      max_total_tokens=config.max_total_tokens,
                                      mem_fraction_static=config.mem_fraction_static,
                                      tp_size=config.tp_size,
                                      dist_init_addr=config.dist_init_addr,
                                      **(config.server_args or {}))
        self.model_config = ModelConfig.from_pretrained(config.model_path, dtype=config.dtype,
                                                        context_length=config.context_length)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_path)
        self.runner = ModelRunner(self.model_config, self.server_args, device=config.device, tp_rank=config.rank)
        self.device = self.runner.device

    # ---- KV bookkeeping -----------------------------------------------------
    # Per-request state is just (req_idx, seq_len): a pool slot and committed length.

    def _new_seq(self) -> int:
        slots = self.runner.req_to_token_pool.alloc(1)
        assert slots is not None, "no free request slots"
        return slots[0]

    def _alloc(self, req_idx: int, seq_len: int, n: int) -> torch.Tensor:
        loc = self.runner.token_to_kv_pool_allocator.alloc(n)
        assert loc is not None, "KV pool exhausted"
        self.runner.req_to_token_pool.write((req_idx, slice(seq_len, seq_len + n)), loc.to(torch.int32))
        return loc

    def _release(self, req_idx: int, seq_len: int) -> None:
        locs = self.runner.req_to_token_pool.req_to_token[req_idx, :seq_len].long()
        self.runner.token_to_kv_pool_allocator.free(locs)
        self.runner.req_to_token_pool.free(req_idx)

    def _extend(self, req_idx: int, seq_len: int, ids: List[int]):
        loc = self._alloc(req_idx, seq_len, len(ids))
        fb = ForwardBatch.for_extend([ids], [seq_len], [req_idx], self.device, out_cache_loc=loc)
        return self.runner.forward(fb), seq_len + len(ids)

    def _decode(self, req_idx: int, seq_len: int, tok: int):
        loc = self._alloc(req_idx, seq_len, 1)
        fb = ForwardBatch.for_decode([tok], [seq_len + 1], [req_idx], self.device, out_cache_loc=loc)
        return self.runner.forward(fb)

    # ---- public helpers ------------------------------------------------------

    @torch.inference_mode()
    def prefill_logits(self, prompt_ids: List[int]) -> torch.Tensor:
        req_idx = self._new_seq()
        try:
            logits, _ = self._extend(req_idx, 0, prompt_ids)
            return logits.next_token_logits[0]
        finally:
            self._release(req_idx, len(prompt_ids))

    @torch.inference_mode()
    def all_prefill_logits(self, prompt_ids: List[int]) -> torch.Tensor:
        """Logits for every prompt position, [len, vocab] float32."""
        req_idx = self._new_seq()
        try:
            loc = self._alloc(req_idx, 0, len(prompt_ids))
            fb = ForwardBatch.for_extend([prompt_ids], [0], [req_idx], self.device, out_cache_loc=loc)
            self.runner._attach(fb)
            hidden = self.runner.model.model(fb.input_ids, fb.positions, fb)
        finally:
            self._release(req_idx, len(prompt_ids))
        return self.runner.model.lm_head.compute_logits(hidden)

    @torch.inference_mode()
    def generate_greedy(self, prompt_ids: List[int], max_new_tokens: int, ignore_eos: bool = True) -> List[int]:
        eos = set(self.model_config.eos_token_ids)
        req_idx = self._new_seq()
        seq_len = 0
        try:
            logits, seq_len = self._extend(req_idx, 0, prompt_ids)
            logits = logits.next_token_logits[0]
            out: List[int] = []
            for _ in range(max_new_tokens):
                tok = int(torch.argmax(logits, dim=-1))
                out.append(tok)
                if not ignore_eos and tok in eos:
                    break
                logits = self._decode(req_idx, seq_len, tok).next_token_logits[0]
                seq_len += 1
            return out
        finally:
            self._release(req_idx, seq_len)

    def generate_text(self, prompt: str, max_new_tokens: int = 64, chat: bool = True) -> str:
        if chat and self.tokenizer.chat_template:
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
            if not isinstance(ids, list):
                ids = ids["input_ids"]
        else:
            ids = self.tokenizer.encode(prompt)
        out = self.generate_greedy(list(ids), max_new_tokens, ignore_eos=False)
        return self.tokenizer.decode(out, skip_special_tokens=True)

    def kv_available(self) -> int:
        return self.runner.token_to_kv_pool_allocator.available_size()

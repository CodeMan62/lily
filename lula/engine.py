"""Single-process, single-sequence engine. Mirrors sglang/srt/entrypoints/engine.py (reduced).

Drives one request at a time directly through a `ModelRunner`, without the
scheduler/detokenizer processes. It plays the scheduler's role for KV
bookkeeping: allocate a request slot, allocate KV slots for new tokens, write
them into `req_to_token`, run, and free everything at the end.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from lula.models.model_config import ModelConfig
from lula.model_executor.forward_batch_info import ForwardBatch
from lula.model_executor.model_runner import ModelRunner
from lula.server_args import ServerArgs


class _Seq:
    """Minimal per-request state: pool slot and committed length."""

    def __init__(self, req_idx: int):
        self.req_idx = req_idx
        self.seq_len = 0


class Engine:
    def __init__(self, model_path: str, dtype: str = "auto", attention_backend: str = "triton",
                 device: str = "cuda", context_length: Optional[int] = None, max_running_requests: int = 2,
                 max_total_tokens: Optional[int] = None, mem_fraction_static: Optional[float] = None):
        self.server_args = ServerArgs(model_path=model_path, dtype=dtype, attention_backend=attention_backend,
                                      context_length=context_length, max_running_requests=max_running_requests,
                                      max_total_tokens=max_total_tokens, mem_fraction_static=mem_fraction_static)
        self.model_config = ModelConfig.from_pretrained(model_path, dtype=dtype, context_length=context_length)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_path)
        self.runner = ModelRunner(self.model_config, self.server_args, device=device)
        self.device = self.runner.device

    # ---- KV bookkeeping -----------------------------------------------------

    def _new_seq(self) -> _Seq:
        slots = self.runner.req_to_token_pool.alloc(1)
        assert slots is not None, "no free request slots"
        return _Seq(slots[0])

    def _alloc(self, seq: _Seq, n: int) -> torch.Tensor:
        loc = self.runner.token_to_kv_pool_allocator.alloc(n)
        assert loc is not None, "KV pool exhausted"
        self.runner.req_to_token_pool.write((seq.req_idx, slice(seq.seq_len, seq.seq_len + n)), loc.to(torch.int32))
        return loc

    def _release(self, seq: _Seq) -> None:
        locs = self.runner.req_to_token_pool.req_to_token[seq.req_idx, : seq.seq_len].long()
        self.runner.token_to_kv_pool_allocator.free(locs)
        self.runner.req_to_token_pool.free(seq.req_idx)

    def _extend(self, seq: _Seq, ids: List[int]):
        loc = self._alloc(seq, len(ids))
        fb = ForwardBatch.for_extend([ids], [seq.seq_len], [seq.req_idx], self.device, out_cache_loc=loc)
        seq.seq_len += len(ids)
        return self.runner.forward(fb)

    def _decode(self, seq: _Seq, tok: int):
        loc = self._alloc(seq, 1)
        seq.seq_len += 1
        fb = ForwardBatch.for_decode([tok], [seq.seq_len], [seq.req_idx], self.device, out_cache_loc=loc)
        return self.runner.forward(fb)

    # ---- public helpers ------------------------------------------------------

    @torch.inference_mode()
    def prefill_logits(self, prompt_ids: List[int]) -> torch.Tensor:
        seq = self._new_seq()
        try:
            return self._extend(seq, prompt_ids).next_token_logits[0]
        finally:
            self._release(seq)

    @torch.inference_mode()
    def all_prefill_logits(self, prompt_ids: List[int]) -> torch.Tensor:
        """Logits for every prompt position, [len, vocab] float32."""
        seq = self._new_seq()
        try:
            loc = self._alloc(seq, len(prompt_ids))
            fb = ForwardBatch.for_extend([prompt_ids], [0], [seq.req_idx], self.device, out_cache_loc=loc)
            seq.seq_len += len(prompt_ids)
            self.runner._attach(fb)
            hidden = self.runner.model.model(fb.input_ids, fb.positions, fb)
        finally:
            self._release(seq)
        lm_head = self.runner.model.lm_head
        return F.linear(hidden, lm_head.weight, lm_head.bias).float()

    @torch.inference_mode()
    def generate_greedy(self, prompt_ids: List[int], max_new_tokens: int, ignore_eos: bool = True) -> List[int]:
        eos = set(self.model_config.eos_token_ids)
        seq = self._new_seq()
        try:
            logits = self._extend(seq, prompt_ids).next_token_logits[0]
            out: List[int] = []
            for _ in range(max_new_tokens):
                tok = int(torch.argmax(logits, dim=-1))
                out.append(tok)
                if not ignore_eos and tok in eos:
                    break
                logits = self._decode(seq, tok).next_token_logits[0]
            return out
        finally:
            self._release(seq)

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

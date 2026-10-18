"""Request and batch bookkeeping. Mirrors sglang/srt/managers/schedule_batch.py (reduced).

KV ownership
------------
A request's committed KV lives in its `req_to_token` row. The first
`tree_prefix_len` entries are borrowed from the radix tree (the tree owns them,
the request holds a lock); the rest are owned by the request. `seq_len` is the
number of committed tokens, `prefix_indices` are the borrowed tree indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from lula.model_executor.forward_batch_info import ForwardMode
from lula.sampling.sampling_batch_info import SamplingBatchInfo
from lula.sampling.sampling_params import SamplingParams
from lula.utils import new_generator


@dataclass
class Req:
    rid: str
    origin_input_ids: List[int]
    sampling_params: SamplingParams
    eos_token_ids: List[int]
    stream: bool = False
    output_ids: List[int] = field(default_factory=list)
    fill_ids: List[int] = field(default_factory=list)
    prefix_indices: Optional[List[int]] = None
    extend_input_len: int = 0
    finished_reason: Optional[dict] = None
    req_pool_idx: Optional[int] = None
    seq_len: int = 0
    last_node: Any = None
    is_chunked: bool = False
    prefix_locked: bool = False
    tree_prefix_len: int = 0
    generator: torch.Generator = None  # type: ignore[assignment]
    output_logprobs: Optional[List[float]] = None
    top_logprobs: Optional[List[List[tuple]]] = None
    pending_free: bool = False
    lookahead: int = 0
    future_valid: bool = False

    def __post_init__(self) -> None:
        self.origin_input_ids = list(self.origin_input_ids)
        self.eos_token_ids = list(self.eos_token_ids)
        self.fill_ids = list(self.origin_input_ids)
        # Moved to the model device by the scheduler.
        self.generator = new_generator(self.sampling_params.seed)
        if self.sampling_params.logprobs:
            self.output_logprobs = []
            if self.sampling_params.top_logprobs:
                self.top_logprobs = []

    def init_next_round_input(self, tree_cache: Any = None) -> None:
        self.fill_ids = self.origin_input_ids + self.output_ids
        if self.prefix_indices is None:
            if tree_cache is None:
                matched, last = [], None
            else:
                matched, last = tree_cache.match_prefix(self.fill_ids)
            self.prefix_indices = list(matched)
            self.last_node = last
            self.tree_prefix_len = len(matched)
            self.seq_len = len(matched)
        self.extend_input_len = len(self.fill_ids) - self.seq_len
        if self.extend_input_len <= 0 and self.fill_ids:
            self.seq_len -= 1
            self.tree_prefix_len = min(self.tree_prefix_len, self.seq_len)
            self.prefix_indices = self.prefix_indices[: self.seq_len]
            self.extend_input_len = 1

    def check_finished(self) -> Optional[dict]:
        if self.finished_reason is not None:
            return self.finished_reason
        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = {"type": "length"}
        elif self.output_ids and not self.sampling_params.ignore_eos:
            stop_ids = set(self.eos_token_ids) | set(self.sampling_params.stop_token_ids)
            if self.output_ids[-1] in stop_ids:
                self.finished_reason = {"type": "stop", "value": self.output_ids[-1]}
        return self.finished_reason

    @property
    def finished(self) -> bool:
        return self.finished_reason is not None

    def next_output_finishes_by_length(self) -> bool:
        """True if committing one more output token would hit max_new_tokens."""
        return len(self.output_ids) + 1 >= self.sampling_params.max_new_tokens


@dataclass
class ModelWorkerBatch:
    forward_mode: ForwardMode
    input_ids: torch.Tensor
    positions: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor
    seq_lens_sum: int
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_prefix_lens: Optional[torch.Tensor] = None
    extend_start_loc: Optional[torch.Tensor] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    extend_prefix_lens_cpu: Optional[List[int]] = None
    seq_lens_cpu: Optional[List[int]] = None
    sampling_info: Optional[SamplingBatchInfo] = None
    batch_size: int = 0


class ScheduleBatch:
    def __init__(self, reqs: List[Req], model_runner: Any, tree_cache: Any = None):
        self.reqs = list(reqs)
        self.model_runner = model_runner
        self.tree_cache = tree_cache
        self.device = model_runner.device
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.forward_mode = ForwardMode.IDLE
        self.input_ids = torch.empty(0, dtype=torch.int64, device=self.device)
        self.positions = torch.empty(0, dtype=torch.int64, device=self.device)
        self.req_pool_indices = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens = torch.empty(0, dtype=torch.int64, device=self.device)
        self.out_cache_loc = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens_cpu: List[int] = []
        self.extend_seq_lens_cpu: List[int] = []
        self.extend_prefix_lens_cpu: List[int] = []
        self.extend_seq_lens: Optional[torch.Tensor] = None
        self.extend_prefix_lens: Optional[torch.Tensor] = None
        self.extend_start_loc: Optional[torch.Tensor] = None
        self.extend_num_tokens = 0
        self.seq_lens_sum = 0
        self.sampling_info = SamplingBatchInfo.from_reqs(self.reqs)
        # In a MIXED batch the first `num_decode_reqs` requests are decoding.
        self.num_decode_reqs = 0

    # ---- KV allocation -----------------------------------------------------

    def _alloc_tokens(self, count: int) -> torch.Tensor:
        if count == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        allocator = self.token_to_kv_pool_allocator
        loc = allocator.alloc(count)
        if loc is None and self.tree_cache is not None:
            self.tree_cache.evict(count)
            loc = allocator.alloc(count)
        if loc is None:
            raise RuntimeError(
                f"KV cache exhausted: requested {count}, available {allocator.available_size()}"
            )
        return loc

    def _ensure_req_slots(self) -> None:
        missing = [r for r in self.reqs if r.req_pool_idx is None]
        if not missing:
            return
        slots = self.req_to_token_pool.alloc(len(missing))
        if slots is None:
            raise RuntimeError("request pool exhausted")
        for req, slot in zip(missing, slots):
            req.req_pool_idx = slot

    def _finalize(
        self,
        ext_lens: List[int],
        input_lists: Optional[List[List[int]]],
        mode: ForwardMode,
        input_ids_override: Optional[torch.Tensor] = None,
    ) -> None:
        self._ensure_req_slots()
        total = sum(ext_lens)
        out_loc = self._alloc_tokens(total)

        starts: List[int] = []
        cursor = 0
        for ext in ext_lens:
            starts.append(cursor)
            cursor += ext

        flat_ids: List[int] = []
        positions: List[int] = []
        seq_lens: List[int] = []
        for i, req in enumerate(self.reqs):
            prefix = req.seq_len
            ext = ext_lens[i]
            if req.tree_prefix_len:
                values = torch.tensor(req.prefix_indices[: req.tree_prefix_len], dtype=torch.int32, device=self.device)
                self.req_to_token_pool.write((req.req_pool_idx, slice(0, req.tree_prefix_len)), values)
            if ext:
                values = out_loc[starts[i] : starts[i] + ext].to(torch.int32)
                self.req_to_token_pool.write((req.req_pool_idx, slice(prefix, prefix + ext)), values)
            req.seq_len = prefix + ext
            if input_lists is not None:
                flat_ids.extend(input_lists[i])
            positions.extend(range(prefix, prefix + ext))
            seq_lens.append(req.seq_len)

        self.forward_mode = mode
        if input_ids_override is not None:
            self.input_ids = input_ids_override
        else:
            self.input_ids = torch.tensor(flat_ids, dtype=torch.int64, device=self.device)
        self.positions = torch.tensor(positions, dtype=torch.int64, device=self.device)
        self.req_pool_indices = torch.tensor(
            [r.req_pool_idx for r in self.reqs], dtype=torch.int64, device=self.device
        )
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=self.device)
        self.out_cache_loc = out_loc
        self.seq_lens_cpu = seq_lens
        self.extend_seq_lens_cpu = ext_lens
        self.extend_prefix_lens_cpu = [s - e for s, e in zip(seq_lens, ext_lens)]
        self.extend_seq_lens = torch.tensor(ext_lens, dtype=torch.int64, device=self.device)
        self.extend_prefix_lens = torch.tensor(self.extend_prefix_lens_cpu, dtype=torch.int64, device=self.device)
        self.extend_start_loc = torch.tensor(starts, dtype=torch.int64, device=self.device)
        self.extend_num_tokens = total
        self.seq_lens_sum = sum(seq_lens)

    def prepare_for_extend(self) -> None:
        ext_lens = [r.extend_input_len for r in self.reqs]
        input_lists = [
            r.fill_ids[r.seq_len : r.seq_len + r.extend_input_len] for r in self.reqs
        ]
        for req, ext, ids in zip(self.reqs, ext_lens, input_lists):
            assert ext == len(ids), (req.rid, ext, len(ids))
        mode = ForwardMode.PREFILL if all(r.seq_len == 0 for r in self.reqs) else ForwardMode.EXTEND
        self._finalize(ext_lens, input_lists, mode)

    def prepare_for_decode(self, future_token_ids: Optional[torch.Tensor] = None) -> None:
        self._ensure_req_slots()
        ext_lens = [1] * len(self.reqs)
        if future_token_ids is not None:
            # Overlap mode: the step's input token is still on the GPU (the sampler
            # of the previous batch wrote it). Gather it without a device sync.
            indices = torch.tensor(
                [r.req_pool_idx for r in self.reqs], dtype=torch.int64, device=self.device
            )
            input_ids = future_token_ids[indices]
            self._finalize(ext_lens, None, ForwardMode.DECODE, input_ids_override=input_ids)
            for r in self.reqs:
                r.lookahead += 1
        else:
            input_lists = [[r.output_ids[-1]] for r in self.reqs]
            self._finalize(ext_lens, input_lists, ForwardMode.DECODE)

    def filter_batch(self, keep_indices: List[int]) -> "ScheduleBatch":
        out = ScheduleBatch([self.reqs[i] for i in keep_indices], self.model_runner, self.tree_cache)
        out.sampling_info = self.sampling_info.filter_batch(keep_indices)
        return out

    def merge_batch(self, other: "ScheduleBatch") -> "ScheduleBatch":
        merged = ScheduleBatch(self.reqs + other.reqs, self.model_runner, self.tree_cache)
        merged.forward_mode = ForwardMode.MIXED
        merged.num_decode_reqs = len(self.reqs)
        merged.input_ids = torch.cat((self.input_ids, other.input_ids))
        merged.positions = torch.cat((self.positions, other.positions))
        merged.req_pool_indices = torch.cat((self.req_pool_indices, other.req_pool_indices))
        merged.seq_lens = torch.cat((self.seq_lens, other.seq_lens))
        merged.out_cache_loc = torch.cat((self.out_cache_loc, other.out_cache_loc))
        merged.seq_lens_cpu = self.seq_lens_cpu + other.seq_lens_cpu
        merged.extend_seq_lens_cpu = self.extend_seq_lens_cpu + other.extend_seq_lens_cpu
        merged.extend_prefix_lens_cpu = self.extend_prefix_lens_cpu + other.extend_prefix_lens_cpu
        merged.extend_seq_lens = torch.cat((self.extend_seq_lens, other.extend_seq_lens))
        merged.extend_prefix_lens = torch.cat((self.extend_prefix_lens, other.extend_prefix_lens))
        starts: List[int] = []
        cursor = 0
        for length in merged.extend_seq_lens_cpu:
            starts.append(cursor)
            cursor += length
        merged.extend_start_loc = torch.tensor(starts, dtype=torch.int64, device=self.device)
        merged.extend_num_tokens = cursor
        merged.seq_lens_sum = sum(merged.seq_lens_cpu)
        merged.sampling_info = self.sampling_info.merge_batch(other.sampling_info)
        return merged

    def get_model_worker_batch(self) -> ModelWorkerBatch:
        self.sampling_info.refresh_from_reqs(self.reqs)
        return ModelWorkerBatch(
            forward_mode=self.forward_mode,
            input_ids=self.input_ids,
            positions=self.positions,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            out_cache_loc=self.out_cache_loc,
            seq_lens_sum=self.seq_lens_sum,
            extend_seq_lens=self.extend_seq_lens,
            extend_prefix_lens=self.extend_prefix_lens,
            extend_start_loc=self.extend_start_loc,
            extend_seq_lens_cpu=list(self.extend_seq_lens_cpu),
            extend_prefix_lens_cpu=list(self.extend_prefix_lens_cpu),
            seq_lens_cpu=list(self.seq_lens_cpu),
            sampling_info=self.sampling_info,
            batch_size=len(self.reqs),
        )

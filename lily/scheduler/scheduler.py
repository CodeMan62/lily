"""Scheduler"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import torch
import zmq
from torch.profiler import record_function

from lily.scheduler.io_struct import (
    AbortReq,
    BatchTokenIDOut,
    FlushCacheReq,
    TokenizedGenerateReqInput,
)
from lily.scheduler.schedule_batch import Req, ScheduleBatch
from lily.layers.logits_processor import LogitsProcessorOutput
from lily.layers.sampler import Sampler, SamplerOutput
from lily.mem_cache.radix_cache import ChunkCache, RadixCache
from lily.model_executor.forward_batch_info import ForwardBatch
from lily.model_executor.model_runner import ModelRunner
from lily.sampling.sampling_params import SamplingParams
from lily.distributed import broadcast_pyobj, tensor_model_parallel_broadcast
from lily.profiler import DecodeProfiler
from lily.utils import int_tensor, new_generator

logger = logging.getLogger(__name__)


class SchedulePolicy:
    def __init__(self, policy: str = "fcfs"):
        if policy not in ("fcfs", "lpm"):
            raise ValueError(f"unknown schedule policy: {policy}")
        self.policy = policy

    def sort(self, reqs: Iterable[Req]) -> List[Req]:
        values = list(reqs)
        if self.policy == "fcfs":
            return values
        return sorted(values, key=lambda req: len(req.prefix_indices or []), reverse=True)


@dataclass
class PrefillAdder:
    token_budget: int
    max_running_requests: int
    chunked_prefill_size: int
    running_requests: int = 0

    def add(self, reqs: Iterable[Req]) -> List[Req]:
        selected: List[Req] = []
        budget = min(self.token_budget, self.chunked_prefill_size)
        slots = max(0, self.max_running_requests - self.running_requests)
        for req in reqs:
            if slots <= 0 or budget <= 0:
                break
            needed = max(1, req.extend_input_len)
            take = min(needed, budget)
            req.extend_input_len = take
            req.is_chunked = take < needed
            selected.append(req)
            budget -= take
            slots -= 1
        return selected


@dataclass
class GenerationBatchResult:
    logits_output: Optional[LogitsProcessorOutput]
    next_token_ids: Optional[torch.Tensor]
    sampler_output: Optional[SamplerOutput] = None
    copy_done: Optional[torch.cuda.Event] = None
    # Speculative round: committed tokens per request (variable length) and the
    # number of verify slots that were allocated per request.
    accepted: Optional[List[List[int]]] = None
    spec_slots: int = 0


class TpModelWorker:
    """One model replica plus the sampler. Mirrors sglang/srt/managers/tp_worker.py.

    `forward_batch_generation` is the single entry point the scheduler uses; the
    scheduler never sees whether a drafter is running (Phase 8 plugs in here).
    """

    def __init__(self, model_config, server_args, device: str = "cuda", tp_rank: int = 0):
        self.model_runner = ModelRunner(model_config, server_args, device=device, tp_rank=tp_rank)
        self.sampler = Sampler()
        self.spec = None
        if server_args.speculative_algorithm is not None:
            from lily.speculative.base_worker import build_spec_worker

            self.spec = build_spec_worker(self.model_runner, server_args)

    def _sync_token_ids(self, next_token_ids: torch.Tensor) -> torch.Tensor:
        # Logits are identical on every rank, but sampling RNG is per-process;
        # rank 0's tokens are authoritative so every rank's scheduler agrees.
        return tensor_model_parallel_broadcast(next_token_ids.contiguous(), src=0)

    @torch.inference_mode()
    def forward_batch_generation(self, model_worker_batch) -> GenerationBatchResult:
        if self.spec is not None and model_worker_batch.forward_mode.is_decode():
            return self._spec_round(model_worker_batch)
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        if self.spec is not None:
            forward_batch.capture_hidden_states = True   # the draft head needs the prompt's hidden states
            forward_batch.capture_aux_layers = self.spec.aux_layers
        logits_output = self.model_runner.forward(forward_batch)
        sampling_info = model_worker_batch.sampling_info
        if sampling_info is None or len(sampling_info) == 0:
            next_token_ids = self._sync_token_ids(torch.argmax(logits_output.next_token_logits, dim=-1))
            result = GenerationBatchResult(logits_output, next_token_ids)
        else:
            sampler_output = self.sampler(logits_output.next_token_logits, sampling_info)
            sampler_output.next_token_ids = self._sync_token_ids(sampler_output.next_token_ids)
            result = GenerationBatchResult(logits_output, sampler_output.next_token_ids, sampler_output)
        if self.spec is not None:
            self.spec.init_after_extend(model_worker_batch.reqs, model_worker_batch, logits_output, result.next_token_ids)
        return result

    def _spec_round(self, model_worker_batch) -> GenerationBatchResult:
        res = self.spec.decode_round(model_worker_batch.reqs, model_worker_batch, self.sampler)
        if self.spec.rounds % 200 == 0:
            logger.info("speculative decoding: %d rounds, mean accepted length %.2f tokens/round (k=%d)",
                        self.spec.rounds, self.spec.acceptance_length(), self.spec.k)
        return GenerationBatchResult(None, None, res.sampler_output, accepted=res.accepted, spec_slots=res.num_draft + 1)


@dataclass
class SchedulerOutput:
    rid: str
    output_ids: List[int]
    new_token_ids: List[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: Optional[dict] = None
    logprobs: Optional[List[float]] = None
    top_logprobs: Optional[List[List[tuple]]] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stop_strs: List[str] = field(default_factory=list)


@dataclass
class _RequestItem:
    rid: str
    input_ids: List[int]
    sampling_params: SamplingParams
    stream: bool = False
    finish_event: Optional[threading.Event] = None
    result: Optional[SchedulerOutput] = None


class Scheduler:
    def __init__(self, model_config, server_args, device: str = "cuda", tp_rank: int = 0):
        self.model_config = model_config
        self.server_args = server_args

        self.worker = TpModelWorker(model_config, server_args, device=device, tp_rank=tp_rank)
        self.model_runner = self.worker.model_runner
        self.device = self.model_runner.device
        self.req_to_token = self.model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = self.model_runner.token_to_kv_pool_allocator

        self.disable_radix_cache = server_args.disable_radix_cache
        self.tree_cache = (
            ChunkCache() if self.disable_radix_cache else RadixCache(self.token_to_kv_pool_allocator)
        )
        self.policy = SchedulePolicy(server_args.schedule_policy)

        self.waiting_queue: List[Req] = []
        self.running_batch: Optional[ScheduleBatch] = None
        self.output_queue: "queue.Queue[SchedulerOutput]" = queue.Queue()

        self.max_running_requests = self.model_runner.max_running_requests
        self.chunked_prefill_size = server_args.chunked_prefill_size
        self.prefill_token_budget = server_args.max_prefill_tokens
        # Speculative decoding: draft tokens per round (0 = off). Prefill and decode
        # run as separate batches and overlap scheduling stays off.
        # (the runner may have adjusted num_steps to the drafter's block size)
        self.spec_k = int(server_args.speculative_num_steps or 0) if server_args.speculative_algorithm else 0
        self.spec_on = server_args.speculative_algorithm is not None

        self._requests: Dict[str, _RequestItem] = {}
        self.prefill_tokens_computed = 0
        self.decode_steps = 0
        self._profiler = DecodeProfiler.from_env(tp_rank)

        # ---- overlap scheduling (Phase 7) ---------------------------------
        self.enable_overlap = (
            not server_args.disable_overlap_schedule
            and not self.spec_on
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        self._batch_overlap = False
        self._inflight_batch: Optional[ScheduleBatch] = None
        self._overlap_loop = False
        if self.enable_overlap:
            # Index by req-pool slot, so size it like `req_to_token_pool`.
            n_slots = self.model_runner.req_to_token_pool.size
            self.future_token_ids = torch.zeros(n_slots, dtype=torch.int64, device=self.device)
            # Ping-pong pinned host buffers: the batch launched at step k copies
            # its tokens into one while the previous batch's tokens (in the other)
            # are consumed by the CPU without a device-wide sync.
            self._pinned = [
                torch.zeros(n_slots, dtype=torch.int64, pin_memory=True) for _ in range(2)
            ]
            self._pinned_idx = 0
            self.copy_stream = torch.cuda.Stream(device=self.device)
            self._scatter_guard: Optional[torch.cuda.Event] = None

    # ---- request intake ----------------------------------------------------

    def add_request(
        self,
        rid: str,
        input_ids: List[int],
        sampling_params: SamplingParams,
        stream: bool = False,
        finish_event: Optional[threading.Event] = None,
    ) -> Req:
        item = _RequestItem(rid, list(input_ids), sampling_params, stream, finish_event)
        self._requests[rid] = item
        req = Req(
            rid=rid,
            origin_input_ids=list(input_ids),
            sampling_params=sampling_params,
            eos_token_ids=self.model_config.eos_token_ids,
            stream=stream,
        )
        req.generator = new_generator(sampling_params.seed, self.device)
        self.waiting_queue.append(req)
        return req

    def abort_request(self, rid: str) -> None:
        item = self._requests.pop(rid, None)
        for r in self.waiting_queue:
            if r.rid == rid:
                self._free_req_kv(r)
                self._free_req_slot(r)
        self.waiting_queue = [r for r in self.waiting_queue if r.rid != rid]
        if self.running_batch is not None:
            for r in self.running_batch.reqs:
                if r.rid == rid:
                    self._free_req_kv(r)
                    self._free_req_slot(r)
            kept = [r for r in self.running_batch.reqs if r.rid != rid]
            self.running_batch = (
                ScheduleBatch(kept, self.model_runner, self.tree_cache) if kept else None
            )
        if item is not None and item.finish_event is not None:
            item.finish_event.set()

    # ---- KV ownership helpers ---------------------------------------------

    def _free_req_slot(self, req: Req) -> None:
        if req.req_pool_idx is not None:
            self.model_runner.req_to_token_pool.free(req.req_pool_idx)
            req.req_pool_idx = None

    def _free_req_kv(self, req: Req) -> None:
        """Release the request's owned KV and its lock on the borrowed prefix."""
        if req.req_pool_idx is None:
            return
        if req.prefix_locked and req.last_node is not None:
            self.tree_cache.dec_lock_ref(req.last_node)
            req.prefix_locked = False
        owned = self.req_to_token[req.req_pool_idx, req.tree_prefix_len : req.seq_len].clone()
        if owned.numel():
            self.token_to_kv_pool_allocator.free(owned)

    def cache_finished_req(self, req: Req) -> None:
        if req.req_pool_idx is None:
            return
        token_ids = (req.origin_input_ids + req.output_ids)[: req.seq_len]
        kv = self.req_to_token[req.req_pool_idx, : req.seq_len].clone()
        if self.disable_radix_cache:
            if kv.numel():
                self.token_to_kv_pool_allocator.free(kv)
        else:
            duplicated, _ = self.tree_cache.insert(token_ids, kv)
            if duplicated > req.tree_prefix_len:
                self.token_to_kv_pool_allocator.free(kv[req.tree_prefix_len : duplicated])
            if req.prefix_locked and req.last_node is not None:
                self.tree_cache.dec_lock_ref(req.last_node)
                req.prefix_locked = False
        self._free_req_slot(req)

    def cache_unfinished_req(self, req: Req) -> None:
        """Insert the tokens computed so far and re-point the request at the tree."""
        if self.disable_radix_cache or req.req_pool_idx is None:
            return
        token_ids = (req.origin_input_ids + req.output_ids)[: req.seq_len]
        kv = self.req_to_token[req.req_pool_idx, : req.seq_len].clone()
        duplicated, _ = self.tree_cache.insert(token_ids, kv)
        if duplicated > req.tree_prefix_len:
            self.token_to_kv_pool_allocator.free(kv[req.tree_prefix_len : duplicated])
        prefix_indices, last_node = self.tree_cache.match_prefix(token_ids)
        if req.prefix_locked and req.last_node is not None:
            self.tree_cache.dec_lock_ref(req.last_node)
        req.prefix_indices = list(prefix_indices)
        req.tree_prefix_len = len(prefix_indices)
        req.last_node = last_node
        self.tree_cache.inc_lock_ref(last_node)
        req.prefix_locked = True

    def _lock_prefix(self, req: Req) -> None:
        if req.last_node is not None and not req.prefix_locked:
            self.tree_cache.inc_lock_ref(req.last_node)
            req.prefix_locked = True

    # ---- batch construction -----------------------------------------------

    def _get_new_prefill_batch(self) -> Optional[ScheduleBatch]:
        if not self.waiting_queue:
            return None
        for req in self.waiting_queue:
            req.init_next_round_input(self.tree_cache)

        running = len(self.running_batch.reqs) if self.running_batch else 0
        # Reserve one slot per running request for the next decode step, so that
        # prefilling never starves decoding into repeated retraction.
        available = (
            self.token_to_kv_pool_allocator.available_size() + self.tree_cache.evictable_size - running
        )
        budget = min(self.prefill_token_budget, self.chunked_prefill_size, max(0, available))
        if budget <= 0:
            return None

        ordered = self.policy.sort(self.waiting_queue)
        adder = PrefillAdder(
            token_budget=budget,
            max_running_requests=self.max_running_requests,
            chunked_prefill_size=self.chunked_prefill_size,
            running_requests=running,
        )
        selected = adder.add(ordered)
        if not selected:
            return None

        selected_ids = {id(r) for r in selected}
        self.waiting_queue = [r for r in self.waiting_queue if id(r) not in selected_ids]

        batch = ScheduleBatch(selected, self.model_runner, self.tree_cache)
        try:
            batch.prepare_for_extend()
        except RuntimeError:
            self.waiting_queue[:0] = selected
            self.tree_cache.evict(batch.extend_num_tokens or 1)
            return None
        for req in selected:
            self._lock_prefix(req)
        return batch

    def _prepare_decode_batch(
        self, batch: ScheduleBatch, future_token_ids: Optional[torch.Tensor] = None
    ) -> Optional[ScheduleBatch]:
        if future_token_ids is not None:
            finishing = self._reqs_finishing_inflight()
            if finishing:
                finishing_ids = {id(r) for r in finishing}
                keep = [i for i, r in enumerate(batch.reqs) if id(r) not in finishing_ids]
                batch = batch.filter_batch(keep)
                if not batch.reqs:
                    return None
        k = self._spec_draft_count(batch) if self.spec_on else 0
        need = len(batch.reqs) * (k + 1)
        if self.token_to_kv_pool_allocator.available_size() < need:
            deficit = need - self.token_to_kv_pool_allocator.available_size()
            self.tree_cache.evict(deficit)
        if self.token_to_kv_pool_allocator.available_size() < need:
            batch = self.retract_decode(batch, k + 1)
            if batch is None or not batch.reqs:
                return None
        if self.spec_on:
            batch.prepare_for_spec_decode(k)
        else:
            batch.prepare_for_decode(future_token_ids)
        return batch

    def _spec_draft_count(self, batch: ScheduleBatch) -> int:
        """Drafts this round: the configured k, or 0 when a request needs the plain
        sampler (penalties, logprobs) or has no draft state yet."""
        from lily.speculative.base_worker import needs_plain_sampler

        if self.spec_k <= 0:
            return 0
        for r in batch.reqs:
            if not self.worker.spec.ready(r) or needs_plain_sampler(r.sampling_params):
                return 0
        return self.spec_k

    def retract_decode(self, batch: ScheduleBatch, slots_per_req: int = 1) -> Optional[ScheduleBatch]:
        """Evict the youngest requests back to the waiting queue until a decode fits."""
        keep: List[Req] = list(batch.reqs)
        while keep and self.token_to_kv_pool_allocator.available_size() < len(keep) * slots_per_req:
            req = keep.pop()
            self._free_req_kv(req)
            self._free_req_slot(req)
            req.clear_spec_state()
            req.prefix_indices = None
            req.last_node = None
            req.tree_prefix_len = 0
            req.seq_len = 0
            req.extend_input_len = 0
            req.is_chunked = False
            self.waiting_queue.insert(0, req)
            logger.info("retract_decode: evicted %s (%d kept)", req.rid, len(keep))
        if not keep:
            return None
        return batch.filter_batch(list(range(len(keep))))

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        running = self.running_batch
        if self.spec_on:
            # Speculative rounds and prefills are separate batches: a waiting prefill
            # runs first (the running requests keep their state and go next step).
            new_batch = self._get_new_prefill_batch()
            if new_batch is not None and new_batch.reqs:
                self._batch_overlap = False
                return new_batch
            if running is None or not running.reqs:
                return None
            self.running_batch = running = self._prepare_decode_batch(running)
            return running
        running_overlap = self._use_overlap_for(running)
        future_ids = self.future_token_ids if running_overlap else None
        if running is not None and running.reqs:
            running = self._prepare_decode_batch(running, future_token_ids=future_ids)
            self.running_batch = running

        new_batch = self._get_new_prefill_batch()

        if running is not None and running.reqs:
            if new_batch is not None and new_batch.reqs:
                self._batch_overlap = False
                return running.merge_batch(new_batch)
            self._batch_overlap = running_overlap
            return running
        self._batch_overlap = False
        return new_batch

    # ---- execution ---------------------------------------------------------

    def run_batch(self, batch: ScheduleBatch):
        if batch.forward_mode.is_decode():
            if not self.spec_on:
                self.decode_steps += len(batch.reqs)
            if self._profiler is not None:
                self._profiler.step(len(batch.reqs))
        else:
            self.prefill_tokens_computed += batch.extend_num_tokens - batch.num_decode_reqs
        return self.worker.forward_batch_generation(batch.get_model_worker_batch())

    # ---- overlap scheduling (Phase 7) -------------------------------------

    def _use_overlap_for(self, batch: Optional[ScheduleBatch]) -> bool:
        """Overlap is only safe for the running (decode) batch whose input tokens
        are already in the GPU future buffer and whose sampling does not depend
        on history (penalties) or need per-token logprobs. The running batch's
        `forward_mode` is still IDLE here; `prepare_for_decode` sets DECODE."""
        if not (self.enable_overlap and self._overlap_loop):
            return False
        if batch is None or not batch.reqs:
            return False
        for r in batch.reqs:
            if not r.future_valid:
                return False
            sp = r.sampling_params
            if sp.logprobs or sp.top_logprobs:
                return False
            if (
                sp.repetition_penalty != 1.0
                or sp.frequency_penalty != 0.0
                or sp.presence_penalty != 0.0
            ):
                return False
        return True

    def _scatter_future_tokens(self, batch: ScheduleBatch, tokens: torch.Tensor) -> None:
        self.future_token_ids[batch.req_pool_indices] = tokens.to(torch.int64)
        for r in batch.reqs:
            r.future_valid = True

    def _stage_overlap_result(self, batch: ScheduleBatch, result):
        """Scatter sampled tokens into the shared GPU buffer, then copy that
        buffer to pinned host memory on the copy stream so the next step's build
        can read this batch's tokens without syncing the current forward."""
        # WAR: do not overwrite `future_token_ids` until the previous copy has
        # finished reading it.
        if self._scatter_guard is not None:
            torch.cuda.current_stream().wait_event(self._scatter_guard)
        self._scatter_future_tokens(batch, result.next_token_ids)
        scatter_done = torch.cuda.Event()
        scatter_done.record()

        self.copy_stream.wait_event(scatter_done)
        cpu = self._pinned[self._pinned_idx]
        self._pinned_idx ^= 1
        with torch.cuda.stream(self.copy_stream):
            cpu.copy_(self.future_token_ids, non_blocking=True)
        copy_done = torch.cuda.Event()
        copy_done.record(self.copy_stream)

        self._scatter_guard = copy_done
        result.copy_done = copy_done
        result.overlap_cpu = cpu
        return result

    def _reqs_finishing_inflight(self) -> List[Req]:
        """Reqs whose in-flight (launched, not yet processed) decode result
        commits their final token by max_new_tokens. They must not get an extra
        step, so they are excluded from the next decode batch."""
        if self._inflight_batch is None or self.running_batch is None:
            return []
        queued = {id(r) for r in self._inflight_batch.reqs}
        return [
            r
            for r in self.running_batch.reqs
            if id(r) in queued and r.next_output_finishes_by_length()
        ]

    def process_batch_result_overlap(self, batch: ScheduleBatch, result) -> None:
        result.copy_done.synchronize()
        cpu = result.overlap_cpu
        running: List[Req] = []
        for i, req in enumerate(batch.reqs):
            if req.pending_free:
                # This in-flight step was over-allocated: undo it and, once no
                # step is outstanding, release the request's KV and slot.
                self.token_to_kv_pool_allocator.free(batch.out_cache_loc[i : i + 1])
                req.lookahead -= 1
                req.seq_len -= 1
                if req.lookahead == 0:
                    self.cache_finished_req(req)
                    req.pending_free = False
                continue
            req.lookahead -= 1
            token = int(cpu[req.req_pool_idx])
            self._append_token(req, token, running)
        if batch.forward_mode.is_decode():
            self.decode_steps += len(batch.reqs)
        self.running_batch = (
            ScheduleBatch(running, self.model_runner, self.tree_cache) if running else None
        )

    def process_batch_result(self, batch: ScheduleBatch, result) -> None:
        running: List[Req] = []
        split = batch.num_decode_reqs
        if batch.forward_mode.is_decode():
            split = len(batch.reqs)
        if split:
            self.process_batch_result_decode(batch, result, 0, split, running)
        if split < len(batch.reqs):
            self.process_batch_result_prefill(batch, result, split, len(batch.reqs), running)
        if self.spec_on and self.running_batch is not None:
            # prefill batches run alone: keep the requests that were not in this batch
            in_batch = {id(r) for r in batch.reqs}
            running = [r for r in self.running_batch.reqs if id(r) not in in_batch] + running
        self.running_batch = (
            ScheduleBatch(running, self.model_runner, self.tree_cache) if running else None
        )
        # Seed the future buffer so the following decode step can overlap.
        if self._overlap_loop and batch.forward_mode.is_decode():
            self._scatter_future_tokens(batch, result.next_token_ids)

    def process_batch_result_prefill(self, batch, result, lo: int, hi: int, running: List[Req]) -> None:
        next_token_ids = result.next_token_ids
        for i in range(lo, hi):
            req = batch.reqs[i]
            if req.is_chunked:
                self.cache_unfinished_req(req)
                req.extend_input_len = max(1, len(req.fill_ids) - req.seq_len)
                req.is_chunked = False
                self.waiting_queue.insert(0, req)
                continue
            self._append_token(req, int(next_token_ids[i]), running, result, i)

    def process_batch_result_decode(self, batch, result, lo: int, hi: int, running: List[Req]) -> None:
        if result.accepted is not None:
            self._process_spec_result(batch, result, running)
            return
        next_token_ids = result.next_token_ids
        for i in range(lo, hi):
            self._append_token(batch.reqs[i], int(next_token_ids[i]), running, result, i)

    def _process_spec_result(self, batch, result, running: List[Req]) -> None:
        """Commit each request's accepted run (stopping at EOS / max tokens), advance its
        cached length by the inputs it consumed, and free the unused verify slots. Host work
        per round, not per token or request: one output message per request, one free."""
        n = result.spec_slots
        per_token = result.sampler_output is not None          # logprobs ride on the plain sampler (k = 0)
        unused: List[int] = []                                  # offsets into the round's verify slots
        for i, req in enumerate(batch.reqs):
            committed: List[int] = []
            for tok in result.accepted[i]:
                if per_token:
                    self._append_token(req, tok, [], result, i)   # emits, checks stop conditions
                else:
                    req.output_ids.append(tok)
                    req.check_finished()
                committed.append(tok)
                if req.finished:
                    break
            if not per_token:
                self._emit_many(req, committed)
                if req.finished:
                    self._finish(req)
            # Inputs now cached: the previous tail token plus len(committed)-1 accepted
            # drafts, i.e. len(committed) of the n verify slots; the rest go back.
            start = req.seq_len
            if len(committed) < n:
                unused.extend(range(i * n + len(committed), (i + 1) * n))
            req.seq_len = start + len(committed)
            if req.finished:
                self.cache_finished_req(req)
            else:
                running.append(req)
        if unused:
            self.token_to_kv_pool_allocator.free(batch.out_cache_loc[int_tensor(unused, batch.out_cache_loc.device)])
        self.decode_steps += len(batch.reqs)

    def _append_token(self, req: Req, token: int, running: List[Req], result=None, index: int = 0) -> None:
        req.output_ids.append(token)
        if req.sampling_params.logprobs and result is not None and result.sampler_output is not None:
            sampler_output = result.sampler_output
            if sampler_output.next_token_logprobs is not None:
                req.output_logprobs.append(float(sampler_output.next_token_logprobs[index]))
            if sampler_output.top_logprobs is not None:
                req.top_logprobs.append(sampler_output.top_logprobs[index])
        self._emit(req, token)
        if req.check_finished():
            # In overlap mode a later decode step is already in flight for this
            # request; defer the KV/slot release until that step is retired.
            if req.lookahead > 0:
                req.pending_free = True
                self._finish(req)
            elif result is not None and result.accepted is not None:
                self._finish(req)   # the spec path frees KV once seq_len is settled
            else:
                self.cache_finished_req(req)
                self._finish(req)
        else:
            running.append(req)

    def _publish(self, req: Req, out: SchedulerOutput) -> None:
        item = self._requests.get(req.rid)
        if item is not None:
            item.result = out
            if out.finished and item.finish_event is not None:
                item.finish_event.set()
        self.output_queue.put(out)

    def _output(self, req: Req, **kwargs) -> SchedulerOutput:
        return SchedulerOutput(
            rid=req.rid,
            output_ids=list(req.output_ids),
            prompt_tokens=len(req.origin_input_ids),
            completion_tokens=len(req.output_ids),
            stop_strs=list(req.sampling_params.stop or []),
            **kwargs,
        )

    def _emit(self, req: Req, token: int) -> None:
        out = self._output(
            req,
            new_token_ids=[token],
            logprobs=[req.output_logprobs[-1]] if req.output_logprobs else None,
        )
        self._publish(req, out)

    def _emit_many(self, req: Req, tokens: List[int]) -> None:
        """A round's committed tokens of one request in one message (the detokenizer takes chunks)."""
        if tokens:
            self._publish(req, self._output(req, new_token_ids=list(tokens)))

    def _finish(self, req: Req) -> None:
        out = self._output(
            req,
            new_token_ids=[],
            finished=True,
            finish_reason=req.finished_reason,
            logprobs=list(req.output_logprobs) if req.output_logprobs else None,
            top_logprobs=list(req.top_logprobs) if req.top_logprobs else None,
        )
        self._publish(req, out)

    # ---- loops -------------------------------------------------------------

    def step(self) -> bool:
        """Run one scheduler step. Returns False when there is nothing to do."""
        self._overlap_loop = False
        with record_function("lily.schedule"):
            batch = self.get_next_batch_to_run()
        if batch is None or not batch.reqs:
            return False
        with record_function("lily.forward"):
            result = self.run_batch(batch)
        with record_function("lily.commit"):
            self.process_batch_result(batch, result)
        return True

    def overlap_step(self, prev):
        """One overlap iteration. `prev` is the (batch, result) launched but not
        yet retired; returns the new pending pair (or None)."""
        self._overlap_loop = True
        launched_seq = self.get_next_batch_to_run()
        launched = launched_seq is not None and launched_seq.reqs
        if launched:
            result = self.run_batch(launched_seq)
            if self._batch_overlap:
                self._stage_overlap_result(launched_seq, result)
                if prev is not None:
                    self.process_batch_result_overlap(*prev)
                prev = (launched_seq, result)
                self._inflight_batch = launched_seq
            else:
                # A non-overlapped batch (prefill/mixed) must be retired before
                # we touch the running state it depends on.
                if prev is not None:
                    self.process_batch_result_overlap(*prev)
                    prev = None
                self._inflight_batch = None
                self.process_batch_result(launched_seq, result)
        else:
            if prev is not None:
                self.process_batch_result_overlap(*prev)
                prev = None
            self._inflight_batch = None
        return prev

    def flush_cache(self) -> None:
        if self.waiting_queue or (self.running_batch is not None and self.running_batch.reqs):
            logger.warning("flush_cache called while requests are running; ignored")
            return
        self.tree_cache.clear()


# ---- process entry point ---------------------------------------------------


MSG_POLL_INTERVAL = 4


def _receive_messages(tp_rank: int, poller, recv) -> List:
    """New requests on rank 0, then the broadcast that gives every rank the same ones in the same order."""
    msgs: List = []
    if tp_rank == 0 and dict(poller.poll(timeout=0)).get(recv) == zmq.POLLIN:
        while True:
            try:
                msgs.append(recv.recv_pyobj(zmq.NOBLOCK))
            except zmq.Again:
                break
    with record_function("lily.broadcast_requests"):
        return broadcast_pyobj(msgs, tp_rank, src=0)


def run_scheduler_process(
    model_config,
    server_args,
    recv_endpoint: str,
    send_endpoint: str,
    stop_event: Optional[threading.Event] = None,
    tp_rank: int = 0,
) -> None:
    # Spawned process: the launcher's logging config is not inherited, so without this only
    # warnings reach the server log (no load / memory / "scheduler ready" lines).
    logging.basicConfig(level=getattr(logging, str(getattr(server_args, "log_level", "info")).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    # Only rank 0 is wired to the tokenizer/detokenizer; the other ranks get
    # their inputs from rank 0's broadcast each iteration.
    context = zmq.Context()
    recv = send = poller = None
    if tp_rank == 0:
        recv = context.socket(zmq.PULL)
        recv.bind(f"ipc://{recv_endpoint}")
        send = context.socket(zmq.PUSH)
        send.connect(f"ipc://{send_endpoint}")
        poller = zmq.Poller()
        poller.register(recv, zmq.POLLIN)

    scheduler = Scheduler(model_config, server_args, tp_rank=tp_rank)
    prev = None
    logger.info("scheduler ready (tp rank %d/%d)", tp_rank, server_args.tp_size)
    step_count = 0

    try:
        while stop_event is None or not stop_event.is_set():
            # The lockstep broadcast below is a gloo round trip: while decoding, take new messages
            # every few iterations only. `running_batch` and the counter agree on every rank.
            step_count += 1
            if scheduler.running_batch is not None and step_count % MSG_POLL_INTERVAL:
                msgs = []
            else:
                msgs = _receive_messages(tp_rank, poller, recv)
            for msg in msgs:
                if isinstance(msg, TokenizedGenerateReqInput):
                    scheduler.add_request(
                        msg.rid, msg.input_ids, msg.sampling_params, msg.stream
                    )
                elif isinstance(msg, AbortReq):
                    scheduler.abort_request(msg.rid)
                elif isinstance(msg, FlushCacheReq):
                    scheduler.flush_cache()

            if scheduler.enable_overlap:
                prev = scheduler.overlap_step(prev)
                ran = prev is not None
            else:
                ran = scheduler.step()
            outs: List[SchedulerOutput] = []
            while True:
                try:
                    outs.append(scheduler.output_queue.get_nowait())
                except Exception:
                    break
            if outs and tp_rank == 0:
              with record_function("lily.outputs"):
                send.send_pyobj(BatchTokenIDOut(
                    rids=[o.rid for o in outs],
                    output_ids=[o.new_token_ids for o in outs],
                    finished_reasons=[o.finish_reason for o in outs],
                    prompt_tokens=[o.prompt_tokens for o in outs],
                    completion_tokens=[o.completion_tokens for o in outs],
                    stop_strs=[o.stop_strs for o in outs],
                    logprobs=[o.logprobs for o in outs],
                ))
            if not ran:
                time.sleep(0.0005)
    finally:
        if recv is not None:
            recv.close(linger=0)
        if send is not None:
            send.close(linger=0)
        context.term()

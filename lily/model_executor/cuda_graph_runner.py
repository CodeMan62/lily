"""Bucketed CUDA-graph decode runner.

The runner captures one decode graph per bucketed batch size. Decode batches
always contain one token per request, so static buffers are small: input ids,
positions, request-row indices, sequence lengths, and KV write locations. The
model's logits output is allocated during capture and reused on replay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch

from lily.layers.logits_processor import LogitsProcessorOutput
from lily.model_executor.forward_batch_info import ForwardBatch, ForwardMode

logger = logging.getLogger(__name__)


@dataclass
class _StaticDecodeBuffers:
    input_ids: torch.Tensor
    positions: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor


class CudaGraphRunner:
    def __init__(self, model_runner):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs require CUDA")
        self.model_runner = model_runner
        self.device = model_runner.device
        args = model_runner.server_args
        configured_max = args.cuda_graph_max_bs
        if configured_max is None:
            configured_max = min(model_runner.max_running_requests, 16)
        self.max_bs = max(1, int(configured_max))
        self.capture_bs = self._get_capture_bs(args.cuda_graph_bs)
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.buffers: Dict[int, _StaticDecodeBuffers] = {}
        self.outputs: Dict[int, object] = {}
        self._captured = False

        backend = model_runner.attn_backend
        if not getattr(backend, "supports_cuda_graph", lambda: True)():
            raise RuntimeError(f"{type(backend).__name__} does not support CUDA graphs")
        backend.init_cuda_graph_state(self.max_bs, self.max_bs)
        self.capture()

    def _get_capture_bs(self, configured: List[int] | None) -> List[int]:
        if configured:
            values = sorted({int(x) for x in configured if 0 < int(x) <= self.max_bs})
            if self.max_bs not in values:
                values.append(self.max_bs)
            return values
        values: List[int] = []
        bs = 1
        while bs < self.max_bs:
            values.append(bs)
            bs *= 2
        values.append(self.max_bs)
        return sorted(set(values))

    def _bucket(self, batch_size: int) -> int:
        for bs in self.capture_bs:
            if bs >= batch_size:
                return bs
        raise ValueError(f"decode batch size {batch_size} exceeds cuda graph max {self.max_bs}")

    def _make_buffers(self, bs: int) -> _StaticDecodeBuffers:
        fill = self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        return _StaticDecodeBuffers(
            input_ids=torch.zeros(bs, dtype=torch.int64, device=self.device),
            positions=torch.zeros(bs, dtype=torch.int64, device=self.device),
            req_pool_indices=torch.zeros(bs, dtype=torch.int64, device=self.device),
            seq_lens=torch.full((bs,), fill, dtype=torch.int64, device=self.device),
            out_cache_loc=torch.zeros(bs, dtype=torch.int64, device=self.device),
        )

    def _make_forward_batch(self, buffers: _StaticDecodeBuffers, bs: int) -> ForwardBatch:
        return ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=bs,
            input_ids=buffers.input_ids,
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            positions=buffers.positions,
            out_cache_loc=buffers.out_cache_loc,
            seq_lens_cpu=[self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()] * bs,
            seq_lens_sum=bs,
            attn_backend=self.model_runner.attn_backend,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
        )

    @torch.inference_mode()
    def _capture_one(self, bs: int) -> None:
        buffers = self._make_buffers(bs)
        forward_batch = self._make_forward_batch(buffers, bs)
        backend = self.model_runner.attn_backend
        backend.init_forward_metadata_capture_cuda_graph(bs, buffers.req_pool_indices, buffers.seq_lens)

        # Warm up on a side stream so Triton autotuning and lazy CUDA allocations
        # happen before capture.
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(2):
                self.model_runner.model(
                    buffers.input_ids, buffers.positions, forward_batch
                )
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(self.device)
        with torch.cuda.graph(graph):
            output = self.model_runner.model(
                buffers.input_ids, buffers.positions, forward_batch
            )
        self.graphs[bs] = graph
        self.buffers[bs] = buffers
        self.outputs[bs] = output
        logger.debug("captured decode CUDA graph bs=%d", bs)

    def capture(self) -> None:
        for bs in self.capture_bs:
            self._capture_one(bs)
        self._captured = True
        logger.info("captured decode CUDA graphs for batch sizes %s", self.capture_bs)

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        return (
            self._captured
            and forward_batch.forward_mode.is_decode()
            and 0 < forward_batch.batch_size <= self.max_bs
        )

    @torch.inference_mode()
    def replay(self, forward_batch: ForwardBatch):
        if not self.can_run(forward_batch):
            raise RuntimeError("forward batch cannot run in CUDA graph runner")
        bs = forward_batch.batch_size
        bucket = self._bucket(bs)
        buffers = self.buffers[bucket]

        buffers.input_ids[:bs].copy_(forward_batch.input_ids)
        buffers.positions[:bs].copy_(forward_batch.positions)
        buffers.req_pool_indices[:bs].copy_(forward_batch.req_pool_indices)
        buffers.seq_lens[:bs].copy_(forward_batch.seq_lens)
        buffers.out_cache_loc[:bs].copy_(forward_batch.out_cache_loc)

        self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bucket, buffers.req_pool_indices, buffers.seq_lens
        )
        self.graphs[bucket].replay()
        captured = self.outputs[bucket]
        # Return a view over the graph-owned logits buffer; the padded rows are
        # discarded. The view is consumed before the next replay overwrites it.
        return LogitsProcessorOutput(next_token_logits=captured.next_token_logits[:bs])

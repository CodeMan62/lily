"""CUDA graphs for the speculative round: the target's fixed-shape verify extend and the
MTP draft-layer decode step, both bucketed by batch size.

A round of the 27B issues ~1800 kernels; eager, the CPU would need ~2-3x the GPU time
to launch them, so both fixed-shape forwards are captured once per bucket and replayed.
The refresh extend (variable tokens per request) stays eager: it is a single layer.

Verify graphs: T = bs * (k + 1) tokens, k + 1 per request, prefix lengths and slots vary
per replay through static buffers. Attention metadata (kv_indptr/kv_indices/qo_indptr)
is computed inside the graph from those buffers (see TritonAttnBackend.init_forward_metadata).
The GDN layers snapshot their states and stash their projections into graph-owned
tensors whose addresses are stable across replays, so the rollback pointer table is
built once per bucket.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from lily.layers.logits_processor import LogitsProcessorOutput
from lily.model_executor.cuda_graph_runner import capture_buckets, seq_len_fill, warmup_and_capture
from lily.model_executor.forward_batch_info import ForwardBatch, ForwardMode

logger = logging.getLogger(__name__)


def _graph_safe_lengths(model, seq_lens_cpu: List[int]) -> bool:
    """Models whose graphs only cover some context lengths (GLM: dense attention regime) say so."""
    check = getattr(model, "graph_safe_lengths", None)
    return check is None or check(seq_lens_cpu)




@dataclass
class _VerifyBuffers:
    input_ids: torch.Tensor        # [T]
    positions: torch.Tensor        # [T]
    out_cache_loc: torch.Tensor    # [T]
    req_pool_indices: torch.Tensor  # [bs]
    seq_lens: torch.Tensor         # [bs]
    extend_prefix_lens: torch.Tensor  # [bs]


class VerifyGraphRunner:
    """Graphs of the target's verify extend, one per batch-size bucket."""

    def __init__(self, model_runner, k: int, num_gdn_layers: int, aux_layers=None):
        self.runner = model_runner
        self.aux_layers = list(aux_layers) if aux_layers else None
        self.device = model_runner.device
        self.k = k
        self.n = k + 1
        args = model_runner.server_args
        max_bs = args.cuda_graph_max_bs or min(model_runner.max_running_requests, 16)
        self.max_bs = max(1, int(max_bs))
        self.capture_bs = capture_buckets(self.max_bs, args.cuda_graph_bs)
        self.num_gdn_layers = num_gdn_layers
        ctx = model_runner.req_to_token_pool.max_context_len
        # shared across buckets: one graph replays at a time
        self.static_meta = {
            "kv_indptr": torch.zeros(self.max_bs + 1, dtype=torch.int32, device=self.device),
            "kv_indices": torch.zeros(self.max_bs * ctx, dtype=torch.int32, device=self.device),
            "qo_indptr": torch.arange(0, (self.max_bs + 1) * self.n, self.n, dtype=torch.int32, device=self.device),
        }
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.buffers: Dict[int, _VerifyBuffers] = {}
        self.batches: Dict[int, ForwardBatch] = {}
        self.outputs: Dict[int, LogitsProcessorOutput] = {}
        self.replay_tables: Dict[int, Optional[torch.Tensor]] = {}
        if model_runner.mamba_pool is not None and num_gdn_layers:
            model_runner.mamba_pool.alloc_snapshots()
        for bs in self.capture_bs:
            self._capture(bs)
        logger.info("captured speculative verify graphs (k=%d) for batch sizes %s", k, self.capture_bs)

    def _make_batch(self, bs: int):
        n, dev = self.n, self.device
        T = bs * n
        fill = seq_len_fill(self.runner.attn_backend)
        buf = _VerifyBuffers(
            input_ids=torch.zeros(T, dtype=torch.int64, device=dev),
            positions=torch.zeros(T, dtype=torch.int64, device=dev),
            out_cache_loc=torch.zeros(T, dtype=torch.int64, device=dev),
            req_pool_indices=torch.zeros(bs, dtype=torch.int64, device=dev),
            seq_lens=torch.full((bs,), fill + n, dtype=torch.int64, device=dev),
            extend_prefix_lens=torch.full((bs,), fill, dtype=torch.int64, device=dev),
        )
        hybrid = self.num_gdn_layers > 0
        fb = ForwardBatch(
            forward_mode=ForwardMode.EXTEND, batch_size=bs,
            input_ids=buf.input_ids, req_pool_indices=buf.req_pool_indices, seq_lens=buf.seq_lens,
            positions=buf.positions, out_cache_loc=buf.out_cache_loc,
            extend_seq_lens=torch.full((bs,), n, dtype=torch.int64, device=dev),
            extend_prefix_lens=buf.extend_prefix_lens,
            extend_start_loc=torch.arange(0, T, n, dtype=torch.int64, device=dev),
            extend_num_tokens=T, extend_seq_lens_cpu=[n] * bs, extend_prefix_lens_cpu=[fill] * bs,
            seq_lens_cpu=[fill + n] * bs, seq_lens_sum=bs * (fill + n),
            attn_backend=self.runner.attn_backend, req_to_token_pool=self.runner.req_to_token_pool,
            token_to_kv_pool=self.runner.token_to_kv_pool,
            return_all_logits=True, capture_hidden_states=True, capture_aux_layers=self.aux_layers,
            gdn_snapshot=hybrid, gdn_proj_stash=[None] * self.num_gdn_layers if hybrid else None,
            graph_static=self.static_meta,
        )
        return buf, fb

    @torch.inference_mode()
    def _capture(self, bs: int) -> None:
        buf, fb = self._make_batch(bs)
        runner = self.runner

        def run():
            if runner.attn_backend is not None:
                runner.attn_backend.init_forward_metadata(fb)
            return runner.model(fb.input_ids, fb.positions, fb)

        graph, out = warmup_and_capture(run)
        self.graphs[bs], self.buffers[bs], self.batches[bs], self.outputs[bs] = graph, buf, fb, out
        self.replay_tables[bs] = None

    def can_run(self, bs: int, k: int, prefix_lens: List[int]) -> bool:
        return (k == self.k and 0 < bs <= self.max_bs
                and _graph_safe_lengths(self.runner.model, [p + self.n for p in prefix_lens]))

    def _bucket(self, bs: int) -> int:
        return next(b for b in self.capture_bs if b >= bs)

    @torch.inference_mode()
    def replay(self, input_ids: torch.Tensor, req_pool_indices: torch.Tensor, prefix_lens: List[int],
               out_cache_loc: torch.Tensor):
        """Returns (logits [bs*(k+1), V], hidden [bs*(k+1), H], aux hidden [bs*(k+1), A] or None,
        proj stash list or None) as views over graph-owned buffers; consume before the next replay."""
        bs = len(prefix_lens)
        bucket = self._bucket(bs)
        buf, n = self.buffers[bucket], self.n
        T = bs * n
        dev = self.device
        prefix = torch.tensor(prefix_lens, dtype=torch.int64, device=dev)
        buf.input_ids[:T].copy_(input_ids)
        buf.out_cache_loc[:T].copy_(out_cache_loc)
        buf.req_pool_indices[:bs].copy_(req_pool_indices)
        buf.extend_prefix_lens[:bs].copy_(prefix)
        buf.seq_lens[:bs].copy_(prefix + n)
        buf.positions[:T].copy_((prefix.unsqueeze(1) + torch.arange(n, device=dev)).reshape(-1))
        if bucket > bs:
            # padding rows: request slot 0, prefix 1, KV slot 0 (all reserved)
            buf.req_pool_indices[bs:].zero_()
            buf.extend_prefix_lens[bs:].fill_(1)
            buf.seq_lens[bs:].fill_(1 + n)
            buf.out_cache_loc[T:].zero_()
        self.graphs[bucket].replay()
        out = self.outputs[bucket]
        aux = out.aux_hidden_states[:T] if out.aux_hidden_states is not None else None
        return out.next_token_logits[:T], out.hidden_states[:T], aux, self.batches[bucket].gdn_proj_stash

    def replay_table(self, bs: int, gdn_layers, pool) -> torch.Tensor:
        """Pointer table for the rollback replay of this bucket (stash addresses are stable)."""
        bucket = self._bucket(bs)
        table = self.replay_tables[bucket]
        if table is None:
            stash = self.batches[bucket].gdn_proj_stash
            rows = [layer.replay_table_row(stash[layer.linear_layer_idx], pool) for layer in gdn_layers]
            table = torch.tensor(rows, dtype=torch.int64, device=self.device)
            self.replay_tables[bucket] = table
        return table



@dataclass
class _DraftBuffers:
    input_ids: torch.Tensor      # [bs]
    positions: torch.Tensor      # [bs]
    out_cache_loc: torch.Tensor  # [bs]
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    hidden: torch.Tensor         # [bs, H] draft-layer input hidden states


class MTPDecodeGraphRunner:
    """Graphs of one MTP draft-layer decode step (bs tokens, bs requests)."""

    def __init__(self, model_runner):
        self.runner = model_runner
        self.device = model_runner.device
        args = model_runner.server_args
        max_bs = args.cuda_graph_max_bs or min(model_runner.max_running_requests, 16)
        self.max_bs = max(1, int(max_bs))
        self.capture_bs = capture_buckets(self.max_bs, args.cuda_graph_bs)
        self.hidden_size = model_runner.model_config.hidden_size
        if model_runner.mtp_attn_backend is not None:
            model_runner.mtp_attn_backend.init_cuda_graph_state(self.max_bs, self.max_bs)
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.buffers: Dict[int, _DraftBuffers] = {}
        self.outputs: Dict[int, torch.Tensor] = {}
        for bs in self.capture_bs:
            self._capture(bs)
        logger.info("captured MTP draft decode graphs for batch sizes %s", self.capture_bs)

    @torch.inference_mode()
    def _capture(self, bs: int) -> None:
        dev = self.device
        backend = self.runner.mtp_attn_backend
        fill = seq_len_fill(backend)
        buf = _DraftBuffers(
            input_ids=torch.zeros(bs, dtype=torch.int64, device=dev),
            positions=torch.zeros(bs, dtype=torch.int64, device=dev),
            out_cache_loc=torch.zeros(bs, dtype=torch.int64, device=dev),
            req_pool_indices=torch.zeros(bs, dtype=torch.int64, device=dev),
            seq_lens=torch.full((bs,), fill, dtype=torch.int64, device=dev),
            hidden=torch.zeros(bs, self.hidden_size, dtype=self.runner.dtype, device=dev),
        )
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE, batch_size=bs, input_ids=buf.input_ids,
            req_pool_indices=buf.req_pool_indices, seq_lens=buf.seq_lens, positions=buf.positions,
            out_cache_loc=buf.out_cache_loc, seq_lens_cpu=[fill] * bs, seq_lens_sum=bs * fill,
            attn_backend=backend, req_to_token_pool=self.runner.req_to_token_pool,
            token_to_kv_pool=self.runner.mtp_kv_pool,
        )
        if backend is not None:
            backend.init_forward_metadata_capture_cuda_graph(bs, buf.req_pool_indices, buf.seq_lens)
        model = self.runner.model

        def run():
            return model.mtp(fb.input_ids, fb.positions, buf.hidden, fb, model.model.embed_tokens)

        graph, out = warmup_and_capture(run)
        self.graphs[bs], self.buffers[bs], self.outputs[bs] = graph, buf, out

    def can_run(self, bs: int, seq_lens_cpu: List[int]) -> bool:
        return 0 < bs <= self.max_bs and _graph_safe_lengths(self.runner.model, seq_lens_cpu)

    @torch.inference_mode()
    def replay(self, input_ids: torch.Tensor, positions: torch.Tensor, seq_lens: torch.Tensor,
               req_pool_indices: torch.Tensor, out_cache_loc: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        bs = input_ids.shape[0]
        bucket = next(b for b in self.capture_bs if b >= bs)
        buf = self.buffers[bucket]
        buf.input_ids[:bs].copy_(input_ids)
        buf.positions[:bs].copy_(positions)
        buf.seq_lens[:bs].copy_(seq_lens)
        buf.req_pool_indices[:bs].copy_(req_pool_indices)
        buf.out_cache_loc[:bs].copy_(out_cache_loc)
        buf.hidden[:bs].copy_(hidden)
        if bucket > bs:
            buf.req_pool_indices[bs:].zero_()
            buf.seq_lens[bs:].fill_(1)
            buf.out_cache_loc[bs:].zero_()
        if self.runner.mtp_attn_backend is not None:
            self.runner.mtp_attn_backend.init_forward_metadata_replay_cuda_graph(bucket, buf.req_pool_indices, buf.seq_lens)
        self.graphs[bucket].replay()
        return self.outputs[bucket][:bs]

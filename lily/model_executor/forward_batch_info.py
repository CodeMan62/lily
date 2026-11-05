"""ForwardMode / ForwardBatch. Mirrors sglang/srt/model_executor/forward_batch_info.py.

`ForwardBatch` is the GPU-side description of one model call. The scheduler
builds a `ModelWorkerBatch`; `ForwardBatch.init_new` turns it into device tensors.
`for_extend` / `for_decode` build one directly from python lists (single-process
drivers and tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, List, Optional

import torch

if TYPE_CHECKING:
    from lily.layers.attention.base_attn_backend import AttentionBackend


class ForwardMode(IntEnum):
    # Prefill of a brand-new request (no cached prefix).
    PREFILL = auto()
    # Prefill with a cached prefix; new tokens are "extended" onto it. Also used for chunked prefill.
    EXTEND = auto()
    # One new token per request.
    DECODE = auto()
    # Extend and decode requests in the same batch.
    MIXED = auto()
    # No requests to run (kept for TP/overlap symmetry).
    IDLE = auto()

    def is_decode(self) -> bool:
        return self == ForwardMode.DECODE

    def is_idle(self) -> bool:
        return self == ForwardMode.IDLE


@dataclass
class ForwardBatch:
    forward_mode: ForwardMode
    batch_size: int
    # Flattened token ids of all requests in the batch, shape [num_tokens].
    input_ids: torch.Tensor
    # Index of each request's row in ReqToTokenPool (and in the temporary native cache), [bs].
    req_pool_indices: torch.Tensor
    # Total sequence length per request after this step, [bs].
    seq_lens: torch.Tensor
    # Position ids for every token in input_ids, [num_tokens].
    positions: torch.Tensor
    # KV-pool slots to write this step's K/V into, [num_tokens].
    out_cache_loc: Optional[torch.Tensor] = None

    # Extend-only metadata.
    extend_seq_lens: Optional[torch.Tensor] = None  # [bs] new tokens per request
    extend_prefix_lens: Optional[torch.Tensor] = None  # [bs] cached tokens per request
    extend_start_loc: Optional[torch.Tensor] = None  # [bs] offset of each request in input_ids
    extend_num_tokens: Optional[int] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    extend_prefix_lens_cpu: Optional[List[int]] = None
    seq_lens_cpu: Optional[List[int]] = None
    seq_lens_sum: int = 0

    # Attached at construction by the model runner.
    attn_backend: Optional["AttentionBackend"] = None
    req_to_token_pool: Any = None
    token_to_kv_pool: Any = None

    # Hybrid (Gated DeltaNet) layers: (cu_seqlens int32 [bs+1], slots int64 [bs],
    # fresh uint8 [bs] | None), built by the first linear-attention layer and
    # shared by the rest. `gdn_snapshot` asks the fused kernel to keep a copy of
    # the pre-step states (speculative verify rollback).
    gdn_meta: Any = None
    gdn_snapshot: bool = False
    # With gdn_snapshot: each linear-attention layer stores its merged projection here
    # (list indexed by linear layer index) so the accepted prefix can be replayed.
    gdn_proj_stash: Optional[List[Any]] = None
    # Speculative decoding: logits for every token (verify) and/or the final hidden
    # states for every token (draft-head inputs).
    return_all_logits: bool = False
    capture_hidden_states: bool = False
    # Residual stream after these layer ids, concatenated per token (DFlash drafter context).
    capture_aux_layers: Optional[List[int]] = None
    aux_hidden_states: Any = None   # filled by the model when capture_aux_layers is set
    # Graph-captured fixed-shape extend (speculative verify): preallocated attention
    # metadata buffers {kv_indptr, kv_indices, qo_indptr} so no host sync is needed.
    graph_static: Any = None

    def linear_attention_meta(self):
        """(tok_starts int32 [bs], tok_lens int32 [bs], slots int64 [bs], fresh uint8 [bs] | None) for the
        fused linear-attention kernels (GDN, KDA); built once per batch and cached in `gdn_meta`."""
        if self.gdn_meta is not None:
            return self.gdn_meta
        device = self.req_pool_indices.device
        bs = self.batch_size
        slots = self.req_pool_indices if self.req_pool_indices.dtype is torch.int64 else self.req_pool_indices.long()
        if self.forward_mode.is_decode():
            starts = torch.arange(bs, dtype=torch.int32, device=device)
            lens = torch.ones(bs, dtype=torch.int32, device=device)
            fresh = None
        else:
            lens_cpu = self.extend_seq_lens_cpu
            starts_cpu = [0]
            for L in lens_cpu[:-1]:
                starts_cpu.append(starts_cpu[-1] + L)
            starts = torch.tensor(starts_cpu, dtype=torch.int32, device=device)
            lens = torch.tensor(lens_cpu, dtype=torch.int32, device=device)
            prefix = self.extend_prefix_lens_cpu or [1] * len(lens_cpu)
            fresh = torch.tensor([int(p == 0) for p in prefix], dtype=torch.uint8, device=device) \
                if any(p == 0 for p in prefix) else None
        self.gdn_meta = (starts, lens, slots.contiguous(), fresh)
        return self.gdn_meta

    @classmethod
    def init_new(cls, model_worker_batch: Any, model_runner: Any) -> "ForwardBatch":
        return cls(
            forward_mode=model_worker_batch.forward_mode,
            batch_size=model_worker_batch.batch_size,
            input_ids=model_worker_batch.input_ids,
            req_pool_indices=model_worker_batch.req_pool_indices,
            seq_lens=model_worker_batch.seq_lens,
            positions=model_worker_batch.positions,
            out_cache_loc=model_worker_batch.out_cache_loc,
            extend_seq_lens=model_worker_batch.extend_seq_lens,
            extend_prefix_lens=model_worker_batch.extend_prefix_lens,
            extend_start_loc=model_worker_batch.extend_start_loc,
            extend_num_tokens=int(model_worker_batch.input_ids.numel()),
            extend_seq_lens_cpu=model_worker_batch.extend_seq_lens_cpu,
            extend_prefix_lens_cpu=model_worker_batch.extend_prefix_lens_cpu,
            seq_lens_cpu=model_worker_batch.seq_lens_cpu,
            seq_lens_sum=model_worker_batch.seq_lens_sum,
            attn_backend=model_runner.attn_backend,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool=model_runner.token_to_kv_pool,
        )

    @classmethod
    def for_extend(
        cls,
        input_ids: List[List[int]],
        prefix_lens: List[int],
        req_pool_indices: List[int],
        device: torch.device | str,
        out_cache_loc: Optional[torch.Tensor] = None,
    ) -> "ForwardBatch":
        assert len(input_ids) == len(prefix_lens) == len(req_pool_indices)
        ext_lens = [len(x) for x in input_ids]
        seq_lens = [p + e for p, e in zip(prefix_lens, ext_lens)]
        flat = [t for seq in input_ids for t in seq]
        positions = [p + i for p, e in zip(prefix_lens, ext_lens) for i in range(e)]
        start_loc = [0]
        for e in ext_lens[:-1]:
            start_loc.append(start_loc[-1] + e)
        mode = ForwardMode.PREFILL if all(p == 0 for p in prefix_lens) else ForwardMode.EXTEND
        return cls(
            forward_mode=mode,
            batch_size=len(input_ids),
            input_ids=torch.tensor(flat, dtype=torch.int64, device=device),
            req_pool_indices=torch.tensor(req_pool_indices, dtype=torch.int64, device=device),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int64, device=device),
            positions=torch.tensor(positions, dtype=torch.int64, device=device),
            out_cache_loc=out_cache_loc,
            extend_seq_lens=torch.tensor(ext_lens, dtype=torch.int64, device=device),
            extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int64, device=device),
            extend_start_loc=torch.tensor(start_loc, dtype=torch.int64, device=device),
            extend_num_tokens=len(flat),
            extend_seq_lens_cpu=ext_lens,
            extend_prefix_lens_cpu=list(prefix_lens),
            seq_lens_cpu=seq_lens,
            seq_lens_sum=sum(seq_lens),
        )

    @classmethod
    def for_decode(
        cls,
        next_tokens: List[int],
        seq_lens: List[int],
        req_pool_indices: List[int],
        device: torch.device | str,
        out_cache_loc: Optional[torch.Tensor] = None,
    ) -> "ForwardBatch":
        """`seq_lens` are the lengths *after* appending `next_tokens`."""
        assert len(next_tokens) == len(seq_lens) == len(req_pool_indices)
        return cls(
            forward_mode=ForwardMode.DECODE,
            batch_size=len(next_tokens),
            input_ids=torch.tensor(next_tokens, dtype=torch.int64, device=device),
            req_pool_indices=torch.tensor(req_pool_indices, dtype=torch.int64, device=device),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int64, device=device),
            positions=torch.tensor([s - 1 for s in seq_lens], dtype=torch.int64, device=device),
            out_cache_loc=out_cache_loc,
            seq_lens_cpu=list(seq_lens),
            seq_lens_sum=sum(seq_lens),
        )

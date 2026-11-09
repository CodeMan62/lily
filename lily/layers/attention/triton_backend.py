"""Triton attention backend over the paged KV pool. Mirrors sglang/srt/layers/attention/triton_backend.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from lily.layers.attention.base_attn_backend import AttentionBackend
from lily.layers.attention.triton_ops.decode_attention import decode_attention_fwd
from lily.layers.attention.triton_ops.extend_attention import extend_attention_fwd
from lily.layers.attention.utils import create_flashinfer_kv_indices_triton

if TYPE_CHECKING:
    from lily.layers.radix_attention import RadixAttention
    from lily.model_executor.forward_batch_info import ForwardBatch
    from lily.model_executor.model_runner import ModelRunner


class TritonAttnBackend(AttentionBackend):
    def __init__(self, model_runner: "ModelRunner", num_kv_splits: int = 32, kv_pool=None):
        self.device = model_runner.device
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        # `kv_pool` lets a second backend serve another KV pool over the same
        # request rows (the MTP draft layer's cache).
        self.token_to_kv_pool = kv_pool or model_runner.token_to_kv_pool
        self.num_q_heads = model_runner.num_attention_heads  # per TP rank
        self.v_head_dim = model_runner.model_config.head_dim
        self.num_kv_splits = num_kv_splits
        self.forward_metadata = None
        self.cuda_graph_attn_logits: Optional[torch.Tensor] = None

    # ---- CUDA-graph hooks --------------------------------------------------

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        self.cuda_graph_attn_logits = torch.empty(
            (max_bs, self.num_q_heads, self.num_kv_splits, self.v_head_dim + 1),
            dtype=torch.float32, device=self.device,
        )

    def init_forward_metadata_capture_cuda_graph(self, bs, req_pool_indices, seq_lens) -> None:
        self.forward_metadata = (self.cuda_graph_attn_logits, None, None, None, None)

    def init_forward_metadata_replay_cuda_graph(self, bs, req_pool_indices, seq_lens) -> None:
        self.forward_metadata = (self.cuda_graph_attn_logits, None, None, None, None)

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        bs = forward_batch.batch_size
        if forward_batch.forward_mode.is_decode():
            attn_logits = torch.empty(
                (bs, self.num_q_heads, self.num_kv_splits, self.v_head_dim + 1),
                dtype=torch.float32, device=self.device,
            )
            self.forward_metadata = (attn_logits, None, None, None, None)
            return

        prefix_lens = forward_batch.extend_prefix_lens
        static = forward_batch.graph_static
        if static is not None and "split_kv_rows" in static:
            # Speculative verify (n tokens per request, cache rows already written): each token is a
            # decode row over prefix + i + 1 keys, so the split-KV decode kernel spreads a long context
            # over the whole GPU instead of one block per head scanning it all.
            n = static["n"]
            if static.get("attn_logits") is None:
                static["attn_logits"] = torch.empty((static["split_kv_rows"], self.num_q_heads, self.num_kv_splits,
                                                     self.v_head_dim + 1), dtype=torch.float32, device=self.device)
            rows = bs * n
            seq = static["row_seq_lens"][:rows].view(bs, n)
            seq.copy_(prefix_lens.view(bs, 1) + static["row_offsets"])
            self.forward_metadata = ("rows", static["attn_logits"][:rows],
                                     forward_batch.req_pool_indices.repeat_interleave(n), seq.view(-1), n)
            return
        if forward_batch.extend_as_decode_rows:
            T = forward_batch.extend_num_tokens
            lens = forward_batch.extend_seq_lens
            start = forward_batch.extend_start_loc.repeat_interleave(lens, output_size=T)
            seq = prefix_lens.repeat_interleave(lens, output_size=T) + torch.arange(T, device=self.device) - start + 1
            attn_logits = torch.empty((T, self.num_q_heads, self.num_kv_splits, self.v_head_dim + 1),
                                      dtype=torch.float32, device=self.device)
            self.forward_metadata = ("rows", attn_logits, forward_batch.req_pool_indices.repeat_interleave(lens, output_size=T),
                                     seq, 1)
            return
        if static is not None:
            # Fixed-shape extend inside a CUDA graph: cumsum + gather into preallocated
            # buffers, no host sync (kv_indices is sized for the whole context).
            kv_indptr = static["kv_indptr"][: bs + 1]       # index 0 stays zero from allocation
            kv_indptr[1:] = torch.cumsum(prefix_lens, dim=0)
            kv_indices = static["kv_indices"]
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token, forward_batch.req_pool_indices, prefix_lens,
                kv_indptr, None, kv_indices, self.req_to_token.stride(0),
            )
            qo_indptr = static["qo_indptr"][: bs + 1]
            self.forward_metadata = (None, kv_indptr, kv_indices, qo_indptr, max(forward_batch.extend_seq_lens_cpu))
            return
        kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        kv_indptr[1:] = torch.cumsum(prefix_lens, dim=0)
        total_prefix = int(kv_indptr[-1])
        kv_indices = torch.empty(max(total_prefix, 1), dtype=torch.int32, device=self.device)
        if total_prefix > 0:
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token, forward_batch.req_pool_indices, prefix_lens,
                kv_indptr, None, kv_indices, self.req_to_token.stride(0),
            )
        qo_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        qo_indptr[1:] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
        max_extend_len = max(forward_batch.extend_seq_lens_cpu)
        self.forward_metadata = (None, kv_indptr, kv_indices, qo_indptr, max_extend_len)

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(layer.layer_id, forward_batch.out_cache_loc, k, v)
        if self.forward_metadata[0] == "rows":
            _, attn_logits, row_req_idx, row_seq_lens, rows = self.forward_metadata
            o = torch.empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim), dtype=q.dtype, device=q.device)
            decode_attention_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                self.token_to_kv_pool.get_key_buffer(layer.layer_id),
                self.token_to_kv_pool.get_value_buffer(layer.layer_id),
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                self.req_to_token, row_req_idx, row_seq_lens, attn_logits, self.num_kv_splits, layer.scaling, rows=rows,
            )
            return o
        _, kv_indptr, kv_indices, qo_indptr, max_extend_len = self.forward_metadata
        o = torch.empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim), dtype=q.dtype, device=q.device)
        extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
            v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            qo_indptr, kv_indptr, kv_indices,
            None, None,
            max_extend_len, layer.scaling,
        )
        return o

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(layer.layer_id, forward_batch.out_cache_loc, k, v)
        attn_logits = self.forward_metadata[0]
        o = torch.empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim), dtype=q.dtype, device=q.device)
        decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            attn_logits,
            self.num_kv_splits,
            layer.scaling,
        )
        return o

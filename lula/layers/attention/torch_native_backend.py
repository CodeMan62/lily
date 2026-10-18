"""Reference attention backend in plain PyTorch over the paged KV pool.

Slow and simple on purpose: it is the oracle the Triton backend is tested against.
Reads K/V through req_to_token -> token_to_kv_pool exactly like the Triton kernels do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from lula.layers.attention.base_attn_backend import AttentionBackend

if TYPE_CHECKING:
    from lula.model_executor.model_runner import ModelRunner


class TorchNativeAttnBackend(AttentionBackend):
    def __init__(self, model_runner: "ModelRunner"):
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool

    def supports_cuda_graph(self) -> bool:
        # Data-dependent shapes (python loop over the batch, dynamic masks)
        # make this backend unsuitable for capture.
        return False

    @staticmethod
    def _sdpa(q, k, v, scaling, mask=None):
        """q: [heads, Lq, d]; k, v: [kv_heads, Lk, d]. Returns [heads, Lq, d]."""
        if k.shape[0] != q.shape[0]:
            rep = q.shape[0] // k.shape[0]
            k = k.repeat_interleave(rep, dim=0)
            v = v.repeat_interleave(rep, dim=0)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scaling)

    def _kv(self, layer_id: int, req_idx: int, seq_len: int):
        locs = self.req_to_token[req_idx, :seq_len].long()
        k = self.token_to_kv_pool.get_key_buffer(layer_id)[locs]  # [seq_len, kv_heads, d]
        v = self.token_to_kv_pool.get_value_buffer(layer_id)[locs]
        return k.transpose(0, 1), v.transpose(0, 1)

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(layer.layer_id, forward_batch.out_cache_loc, k, v)
        n_tok = q.shape[0]
        q = q.view(n_tok, layer.tp_q_head_num, layer.qk_head_dim)
        out = torch.empty((n_tok, layer.tp_q_head_num, layer.v_head_dim), dtype=q.dtype, device=q.device)
        start = 0
        for i in range(forward_batch.batch_size):
            r = int(forward_batch.req_pool_indices[i])
            prefix = forward_batch.extend_prefix_lens_cpu[i]
            ext = forward_batch.extend_seq_lens_cpu[i]
            seq_len = prefix + ext
            ki, vi = self._kv(layer.layer_id, r, seq_len)
            qi = q[start : start + ext].transpose(0, 1)
            key_pos = torch.arange(seq_len, device=q.device)
            q_pos = torch.arange(ext, device=q.device) + prefix
            mask = key_pos[None, :] <= q_pos[:, None]
            out[start : start + ext] = self._sdpa(qi, ki, vi, layer.scaling, mask).transpose(0, 1)
            start += ext
        return out.view(n_tok, -1)

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(layer.layer_id, forward_batch.out_cache_loc, k, v)
        bs = forward_batch.batch_size
        q = q.view(bs, layer.tp_q_head_num, layer.qk_head_dim)
        out = torch.empty((bs, layer.tp_q_head_num, layer.v_head_dim), dtype=q.dtype, device=q.device)
        for i in range(bs):
            r = int(forward_batch.req_pool_indices[i])
            seq_len = forward_batch.seq_lens_cpu[i]
            ki, vi = self._kv(layer.layer_id, r, seq_len)
            qi = q[i : i + 1].transpose(0, 1)
            out[i] = self._sdpa(qi, ki, vi, layer.scaling).transpose(0, 1)[0]
        return out.view(bs, -1)

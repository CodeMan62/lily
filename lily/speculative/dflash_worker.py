"""Speculative decoding with a DFlash 2 block drafter (`--speculative-algorithm DFLASH`).

Drafting is one drafter pass per round: the block [x_{m+1}, mask x k] at positions
m..m+k attends to the drafter's context cache (the target's residual streams after
`target_layer_ids`, projected per drafter layer) within the sliding window, and the
candidate selector turns the k mask rows into k draft tokens plus their distribution.

The drafter's context cache (`ModelRunner.dflash_kv_pool`, one entry per cached target
position, indexed by the same KV slots as the target) is written after every prefill
chunk and after every verify for the accepted rows, straight from the target's
auxiliary hidden states. There is no per-request draft state.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from lily.speculative.base_worker import SpecWorkerBase

logger = logging.getLogger(__name__)


class DFlashSpecWorker(SpecWorkerBase):
    def __init__(self, model_runner, server_args):
        drafter = model_runner.drafter
        if drafter is None or model_runner.dflash_kv_pool is None:
            raise RuntimeError("DFLASH needs a drafter: --speculative-draft-model-path")
        super().__init__(model_runner, server_args, drafter.num_drafts)
        self.drafter = drafter
        self.ctx_pool = model_runner.dflash_kv_pool
        self.aux_layers = list(drafter.cfg.target_layer_ids)
        self.window = drafter.cfg.sliding_window or (1 << 30)
        self.mask_id = drafter.cfg.mask_token_id
        self.embed = self.model.model.embed_tokens
        self._init_verify_graphs(server_args)

    # ---- context cache ---------------------------------------------------------------

    def _write_context(self, aux: torch.Tensor, positions: torch.Tensor, loc: torch.Tensor) -> None:
        """aux [T, len(target_layer_ids) * H] -> per-layer K/V at `loc` (positions `positions`)."""
        feats = self.drafter.fuse_context(aux)
        for layer_id, (k, v) in enumerate(self.drafter.context_kv(feats, positions)):
            self.ctx_pool.set_kv_buffer(layer_id, loc, k, v)

    @torch.inference_mode()
    def init_after_extend(self, reqs, worker_batch, logits_output, next_token_ids: torch.Tensor) -> None:
        aux = logits_output.aux_hidden_states
        if aux is None:
            raise RuntimeError("DFLASH prefill needs capture_aux_layers on the forward batch")
        self._write_context(aux, worker_batch.positions, worker_batch.out_cache_loc)

    def _select(self, hidden: torch.Tensor, logits: torch.Tensor, anchor: torch.Tensor, temps, gens):
        """Hook for tests (an oracle can replace it); returns (path, candidates, q)."""
        return self.drafter.candidate_selector.select(hidden, logits, anchor, temps, gens)

    # ---- one drafter pass ----------------------------------------------------------------

    def draft(self, reqs, k: int, m: List[int], slots: torch.Tensor, req_idx: torch.Tensor, sampling: bool):
        B = len(reqs)
        dev = self.device
        S = k + 1
        anchor = torch.tensor([r.output_ids[-1] for r in reqs], dtype=torch.int64, device=dev)
        block_ids = torch.cat([anchor.unsqueeze(1), torch.full((B, k), self.mask_id, dtype=torch.int64, device=dev)], dim=1)
        m_dev = torch.tensor(m, dtype=torch.int64, device=dev)
        positions = m_dev.unsqueeze(1) + torch.arange(S, device=dev)                      # [B, S]

        # context: the last min(m, window) cached positions of each request
        L = max(1, min(max(m), self.window))
        pos = m_dev.unsqueeze(1) - L + torch.arange(L, device=dev)                            # [B, L]
        valid = (pos >= 0) & (pos >= (m_dev - self.window).unsqueeze(1))
        ctx_pos = torch.where(valid, pos, torch.full_like(pos, -1))
        rows = self.req_to_token[req_idx.unsqueeze(1), pos.clamp_min(0)].long()             # [B, L] KV slots
        rows = torch.where(valid, rows, torch.zeros_like(rows))                             # slot 0 is reserved/zero
        ctx_kv = [(self.ctx_pool.get_key_buffer(i)[rows], self.ctx_pool.get_value_buffer(i)[rows])
                  for i in range(self.drafter.cfg.num_hidden_layers)]                        # [B, L, kvh, hd]

        hidden = self.drafter(block_ids, positions, self.embed, ctx_kv, ctx_pos)            # [B, S, H]
        rows_h = hidden[:, 1:]                                                              # mask rows
        logits = self.drafter.logits(rows_h, self.model.lm_head)                            # [B, k, V] fp32
        temps = [0.0 if r.sampling_params.greedy else r.sampling_params.temperature for r in reqs]
        path, cands, q = self._select(rows_h, logits, anchor, temps, [r.generator for r in reqs])
        dprobs = torch.zeros(B, k, logits.shape[-1], device=dev)
        dprobs.scatter_add_(2, cands, q)
        return path, dprobs

    # ---- after accept: context entries for the accepted rows --------------------------------

    def after_accept(self, reqs, accepted, num_acc: List[int], hidden, aux: Optional[torch.Tensor],
                     slots: torch.Tensor, m: List[int], k: int, sampling: bool) -> None:
        if aux is None:
            raise RuntimeError("DFLASH verify needs capture_aux_layers on the forward batch")
        dev = self.device
        rows: List[int] = []
        positions: List[int] = []
        locs = []
        for i, j in enumerate(num_acc):
            rows.extend(range(i * (k + 1), i * (k + 1) + j + 1))      # accepted inputs: positions m..m+j
            positions.extend(range(m[i], m[i] + j + 1))
            locs.append(slots[i, :j + 1])
        self._write_context(aux.index_select(0, torch.tensor(rows, dtype=torch.int64, device=dev)),
                            torch.tensor(positions, dtype=torch.int64, device=dev), torch.cat(locs))

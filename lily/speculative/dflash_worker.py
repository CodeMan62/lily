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
from lily.utils import int_tensor

logger = logging.getLogger(__name__)


class DFlashSpecWorker(SpecWorkerBase):
    def __init__(self, model_runner, server_args):
        drafter = model_runner.drafter
        if drafter is None or model_runner.dflash_kv_pool is None:
            raise RuntimeError("DFLASH needs a drafter: --speculative-draft-model-path")
        super().__init__(model_runner, server_args, server_args.speculative_num_steps)
        self.drafter = drafter
        self.ctx_pool = model_runner.dflash_kv_pool
        self.aux_layers = list(drafter.cfg.target_layer_ids)
        self.window = drafter.cfg.sliding_window or (1 << 30)
        self.mask_id = drafter.cfg.mask_token_id
        self.embed = self.model.model.embed_tokens
        self.draft_graphs = None
        self.prefetched = None      # (draft tokens, request ids, cached lengths they were drafted for)
        self._init_verify_graphs(server_args)
        if self.verify_graphs is not None:
            from lily.speculative.graph_runner import DFlashDraftGraphRunner

            try:
                self.draft_graphs = DFlashDraftGraphRunner(self, model_runner)
            except Exception as exc:  # pragma: no cover
                logger.warning("DFlash draft graph capture failed; draft passes run eagerly: %s", exc)

    def verify_hook(self, out, forward_batch) -> None:
        """Inside the verify graph: context entries for every verify row. Rows that end up rejected
        only touch slots the scheduler frees, and a later round rewrites whatever it reuses."""
        self._write_context(out.aux_hidden_states, forward_batch.positions, forward_batch.out_cache_loc)

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

    def draft_pass(self, anchor: torch.Tensor, m_dev: torch.Tensor, req_idx: torch.Tensor, L: int, k: int):
        """The drafter over a whole block [anchor, mask, ...] with `L` context rows: (hidden [B, k, H]
        of the first k mask rows, their logits [B, k, V]); verifying fewer drafts than the block holds
        only drops the least likely ones. Device tensors and fixed shapes: the draft graphs capture it."""
        B = anchor.shape[0]
        dev = self.device
        S = self.drafter.block_size
        block_ids = torch.cat([anchor.unsqueeze(1), torch.full((B, S - 1), self.mask_id, dtype=torch.int64, device=dev)], dim=1)
        positions = m_dev.unsqueeze(1) + torch.arange(S, device=dev)                      # [B, S]

        # context: the last min(m, window) cached positions of each request
        pos = m_dev.unsqueeze(1) - L + torch.arange(L, device=dev)                            # [B, L]
        valid = (pos >= 0) & (pos >= (m_dev - self.window).unsqueeze(1))
        ctx_pos = torch.where(valid, pos, torch.full_like(pos, -1))
        rows = self.req_to_token[req_idx.unsqueeze(1), pos.clamp_min(0)].long()             # [B, L] KV slots
        rows = torch.where(valid, rows, torch.zeros_like(rows))                             # slot 0 is reserved/zero
        ctx_kv = [(self.ctx_pool.get_key_buffer(i)[rows], self.ctx_pool.get_value_buffer(i)[rows])
                  for i in range(self.drafter.cfg.num_hidden_layers)]                        # [B, L, kvh, hd]
        hidden = self.drafter(block_ids, positions, self.embed, ctx_kv, ctx_pos)            # [B, S, H]
        rows_h = hidden[:, 1:k + 1]                                                         # mask rows
        return rows_h, self.drafter.logits(rows_h, self.model.lm_head)                      # [B, k, V] fp32

    def can_prefetch(self, batch_size: int) -> bool:
        return self.draft_graphs is not None and self.draft_graphs.can_run(batch_size) and "_select" not in self.__dict__

    def prefetch_draft(self, reqs, k: int, m: List[int], req_idx: torch.Tensor, anchor: torch.Tensor, num_acc: torch.Tensor) -> None:
        """Reads the context and writes only the graph's output: if the batch changes it is just dropped."""
        next_m = int_tensor(m, self.device) + num_acc + 1
        L = max(1, min(max(m) + k + 1, self.window))
        self.prefetched = (self.draft_graphs.replay(anchor.contiguous(), next_m, req_idx, L), [id(r) for r in reqs], None)

    def note_prefetch(self, next_m: List[int]) -> None:
        self.prefetched = (self.prefetched[0], self.prefetched[1], next_m)

    def draft(self, reqs, k: int, m: List[int], slots: torch.Tensor, req_idx: torch.Tensor, sampling: bool):
        dev = self.device
        ready, self.prefetched = self.prefetched, None
        if ready is not None and not sampling and ready[1] == [id(r) for r in reqs] and ready[2] == m:
            return ready[0], None
        anchor = int_tensor([r.output_ids[-1] for r in reqs], dev)
        m_dev = int_tensor(m, dev)
        L = max(1, min(max(m), self.window))
        if self.draft_graphs is not None and self.draft_graphs.can_run(len(reqs)) and not sampling and "_select" not in self.__dict__:
            return self.draft_graphs.replay(anchor, m_dev, req_idx, L), None
        rows_h, logits = self.draft_pass(anchor, m_dev, req_idx, L, k)
        temps = [0.0 if r.sampling_params.greedy else r.sampling_params.temperature for r in reqs]
        path, cands, q = self._select(rows_h, logits, anchor, temps, [r.generator for r in reqs])
        dprobs = torch.zeros(len(reqs), k, logits.shape[-1], device=dev)
        dprobs.scatter_add_(2, cands, q)
        return path, dprobs

    # ---- after accept: context entries for the accepted rows --------------------------------

    def after_accept(self, reqs, accepted, num_acc: List[int], hidden, aux: Optional[torch.Tensor],
                     slots: torch.Tensor, m: List[int], k: int, sampling: bool) -> None:
        if self.verified_in_graph:      # verify_hook wrote them
            return
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

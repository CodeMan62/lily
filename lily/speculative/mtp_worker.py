"""Speculative decoding with the checkpoint's MTP head (Qwen3.5/3.8 `mtp.*`), single chain.

Drafting: k-1 MTP decode steps continue the chain whose first token the previous
round's refresh already produced. After verify, one MTP EXTEND over the accepted pairs
with the *target's* hidden states rewrites the draft cache and yields next round's d_1.

The draft layer keeps its own K/V in `ModelRunner.mtp_kv_pool`, indexed by the same
slots as the target: MTP pair P(p) = (token at position p+1, target hidden at position p)
lives at MTP position p, in the slot of target position p. So the draft cache runs one
position behind the target and every slot it needs is either already cached or one of
this round's verify slots.

Per-request state (on `Req`): `spec_draft_token` (d_1) and `spec_draft_hidden` (the
MTP hidden that produced it), plus `spec_draft_probs` (q for d_1) for sampling.
"""

from __future__ import annotations

import logging
from typing import List

import torch

from lily.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from lily.speculative.base_worker import SpecWorkerBase, processed_probs

logger = logging.getLogger(__name__)


class MTPSpecWorker(SpecWorkerBase):
    def __init__(self, model_runner, server_args):
        super().__init__(model_runner, server_args, server_args.speculative_num_steps)
        if self.model.mtp is None or model_runner.mtp_kv_pool is None:
            raise RuntimeError("MTP speculative decoding needs a model with an MTP head")
        self.draft_graphs = None
        self._init_verify_graphs(server_args)
        if self.verify_graphs is not None:
            from lily.speculative.graph_runner import MTPDecodeGraphRunner

            try:
                self.draft_graphs = MTPDecodeGraphRunner(model_runner)
            except Exception as exc:  # pragma: no cover
                logger.warning("MTP draft graph capture failed; draft steps run eagerly: %s", exc)

    def ready(self, req) -> bool:
        return req.spec_draft_token is not None

    # ---- draft token choice --------------------------------------------------------

    def _draft_tokens(self, logits: torch.Tensor, reqs, sampling: bool, ahead: List[int]):
        """Choose one draft token per request from MTP logits [B, V].

        Returns (tokens int64 [B] on device, processed draft probs [B, V]).
        `ahead[i]` is how many tokens beyond `output_ids` request i has already
        decided (this round's committed tokens and/or the drafts so far); the real
        chooser ignores it, tests use it to substitute an oracle.
        """
        # Always keep the processed distribution (one-hot for greedy rows): a request's
        # state must stay usable when it later shares a round with sampling requests.
        probs = processed_probs(logits, [r.sampling_params for r in reqs])
        if not sampling:
            return torch.argmax(logits, dim=-1), probs
        toks = torch.stack([
            torch.multinomial(probs[i], 1, generator=r.generator)[0] if not r.sampling_params.greedy else probs[i].argmax()
            for i, r in enumerate(reqs)])
        return toks, probs

    def _store_draft(self, reqs, hidden_last: torch.Tensor, sampling: bool, ahead: List[int]) -> None:
        """Logits of the MTP's last rows -> next round's first draft token per request.
        The token stays on the device (a 0-d int64 tensor): nothing needs it on the host."""
        toks, probs = self._draft_tokens(self._lm_logits(hidden_last), reqs, sampling, ahead)
        for i, r in enumerate(reqs):
            r.spec_draft_token = toks[i]
            r.spec_draft_hidden = hidden_last[i]
            r.spec_draft_probs = probs[i] if probs is not None else None

    def _draft_probs_of(self, r, vocab: int) -> torch.Tensor:
        """Stored draft distribution of d_1, or one-hot at d_1 if none was kept (e.g. an
        oracle / test chooser): with p one-hot too, that is exact greedy matching."""
        if r.spec_draft_probs is not None:
            return r.spec_draft_probs
        onehot = torch.zeros(vocab, device=self.device)
        onehot[r.spec_draft_token] = 1.0
        return onehot

    # ---- after a prefill / extend of the target -------------------------------------

    @torch.inference_mode()
    def init_after_extend(self, reqs, worker_batch, logits_output, next_token_ids: torch.Tensor) -> None:
        """Write the draft cache for the chunk the target just prefilled and produce d_1.

        Pairs P(p) for the chunk's positions p use the chunk's tokens shifted by one:
        the token after the chunk is the next chunk's first token (chunked prefill) or
        the token just sampled (final chunk). Slots, positions and lengths are the
        prefill batch's own.
        """
        hidden_states = logits_output.hidden_states
        ext_lens = worker_batch.extend_seq_lens_cpu
        prefix_lens = worker_batch.extend_prefix_lens_cpu
        next_cpu = next_token_ids.tolist()
        shifted: List[int] = []
        for i, req in enumerate(reqs):
            p0, L = prefix_lens[i], ext_lens[i]
            chunk = req.fill_ids[p0:p0 + L]
            nxt = req.fill_ids[p0 + L] if p0 + L < len(req.fill_ids) else int(next_cpu[i])
            shifted.extend(chunk[1:] + [nxt])
        fb = self._extend_batch(shifted, worker_batch.req_pool_indices, prefix_lens, ext_lens, worker_batch.out_cache_loc)
        h = self.runner.forward_mtp(fb, hidden_states)
        last = fb.extend_start_loc + fb.extend_seq_lens - 1
        sampling = not all(r.sampling_params.greedy for r in reqs)
        self._store_draft(reqs, h[last], sampling, [1] * len(reqs))

    # ---- drafting ----------------------------------------------------------------------

    def draft(self, reqs, k: int, m: List[int], slots: torch.Tensor, req_idx: torch.Tensor, sampling: bool):
        """d_1 from the last refresh, d_2..d_k from MTP decode steps; all on the device."""
        B = len(reqs)
        dev = self.device
        vocab = self.model.config.vocab_size
        draft_cols: List[torch.Tensor] = [torch.stack([r.spec_draft_token for r in reqs])]
        draft_probs: List[torch.Tensor] = [torch.stack([self._draft_probs_of(r, vocab) for r in reqs])]
        mtp_hidden = torch.stack([r.spec_draft_hidden for r in reqs])
        m_dev = torch.tensor(m, dtype=torch.int64, device=dev)
        use_graph = self.draft_graphs is not None and self.draft_graphs.can_run(B, [mi + k - 1 for mi in m])
        for j in range(2, k + 1):
            seq_lens = m_dev + (j - 1)
            positions = m_dev + (j - 2)
            loc = slots[:, j - 2].contiguous()
            if use_graph:
                mtp_hidden = self.draft_graphs.replay(draft_cols[-1], positions, seq_lens, req_idx, loc, mtp_hidden)
            else:
                fb = ForwardBatch(
                    forward_mode=ForwardMode.DECODE, batch_size=B, input_ids=draft_cols[-1],
                    req_pool_indices=req_idx, seq_lens=seq_lens, positions=positions, out_cache_loc=loc,
                    seq_lens_cpu=[mi + j - 1 for mi in m], seq_lens_sum=sum(m) + B * (j - 1),
                )
                mtp_hidden = self.runner.forward_mtp(fb, mtp_hidden)
            toks, q = self._draft_tokens(self._lm_logits(mtp_hidden), reqs, sampling, [j - 1] * B)
            if q is None:  # a chooser without a distribution (test oracle): one-hot = exact match
                q = torch.zeros(B, vocab, device=dev).scatter_(1, toks.unsqueeze(1), 1.0)
            draft_probs.append(q)
            draft_cols.append(toks)
        return torch.stack(draft_cols, dim=1), torch.stack(draft_probs, dim=1)

    # ---- after accept: refresh the draft cache, produce next d_1 ------------------------

    def after_accept(self, reqs, accepted, num_acc: List[int], hidden, aux, slots, m: List[int], k: int, sampling: bool) -> None:
        B = len(reqs)
        dev = self.device
        refresh_ids: List[int] = []
        hidden_rows: List[int] = []
        for i in range(B):
            j = num_acc[i]
            refresh_ids.extend(accepted[i])                               # tokens at positions m+1..m+j+1
            hidden_rows.extend(range(i * (k + 1), i * (k + 1) + j + 1))   # target hiddens at m..m+j
        ext = [j + 1 for j in num_acc]
        loc = torch.cat([slots[i, :ext[i]] for i in range(B)])
        req_idx = torch.tensor([r.req_pool_idx for r in reqs], dtype=torch.int64, device=dev)
        fb_r = self._extend_batch(refresh_ids, req_idx, m, ext, loc)
        h = self.runner.forward_mtp(fb_r, hidden.index_select(0, torch.tensor(hidden_rows, dtype=torch.int64, device=dev)))
        last = fb_r.extend_start_loc + fb_r.extend_seq_lens - 1
        self._store_draft(reqs, h[last], sampling, [len(a) for a in accepted])

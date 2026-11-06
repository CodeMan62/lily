"""Shared machinery of a speculative round: verify, accept, roll back.

A round for B running requests with k drafts each:
  1. draft     (subclass) k tokens per request, plus the draft distribution q
  2. verify    one target EXTEND over [x_{m+1}, d_1..d_k] (k+1 rows per request) with
               logits for every row, the GDN states snapshotted, and whatever extra
               hidden states the drafter consumes
  3. accept    rejection sampling against q (exact greedy match when the request is greedy)
  4. rollback  the scheduler frees KV slots j+1..k; GDN layers replay the accepted prefix
  5. after     (subclass) update the drafter's own state from the verify outputs

Positions: the target has x_1..x_m cached (positions 0..m-1); x_{m+1} is the last
committed, uncached token (position m). Verify slot i is target position m+i.
Drafting strategies: `mtp_worker.MTPSpecWorker` (the checkpoint's MTP head, chain),
`dflash_worker.DFlashSpecWorker` (DFlash 2 block drafter).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from lily.distributed import tensor_model_parallel_broadcast
from lily.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from lily.sampling.sampling_params import SamplingParams
from lily.utils import int_tensor

logger = logging.getLogger(__name__)


@dataclass
class SpecRoundResult:
    accepted: List[List[int]]        # committed tokens per request (1..k+1 each)
    num_draft: int                   # k used this round (0 = plain decode)
    draft_tokens: int = 0            # k * B, for acceptance-rate accounting
    accepted_drafts: int = 0
    sampler_output: object = None    # when the ordinary sampler ran (k = 0 rounds)


def needs_plain_sampler(params: SamplingParams) -> bool:
    """Requests whose sampling cannot be reproduced on a draft distribution run as k=0
    rounds through the ordinary Sampler (penalties, logit bias, logprobs)."""
    return bool(
        params.repetition_penalty != 1.0 or params.frequency_penalty != 0.0 or params.presence_penalty != 0.0
        or params.logit_bias or params.logprobs or params.top_logprobs
    )


def processed_probs(logits: torch.Tensor, params: Sequence[SamplingParams]) -> torch.Tensor:
    """[N, V] fp32 logits -> the distributions the Sampler draws from, row by row:
    greedy rows become one-hot at the argmax; others apply temperature, top-k, top-p
    and min-p exactly like `Sampler._sample_one` (which this mirrors)."""
    probs = torch.zeros_like(logits)
    vocab = logits.shape[-1]
    # greedy rows (the common case) in one shot, no host syncs
    probs.scatter_(1, logits.argmax(-1, keepdim=True), 1.0)
    for i, p in enumerate(params):
        if p.greedy:
            continue
        probs[i].zero_()
        row = logits[i]
        x = row / max(p.temperature, 1e-5)
        if 0 < p.top_k < vocab:
            values, indices = torch.topk(x, p.top_k)
        else:
            values, indices = torch.sort(x, descending=True)
        pr = torch.softmax(values, dim=-1)
        if p.top_p < 1.0:
            cumulative = torch.cumsum(pr, dim=-1)
            keep = cumulative - pr < p.top_p
            keep[0] = True
            pr = torch.where(keep, pr, torch.zeros_like(pr))
        if p.min_p > 0.0:
            pr = torch.where(pr >= p.min_p * pr.max(), pr, torch.zeros_like(pr))
        pr = pr / pr.sum()
        probs[i].scatter_(0, indices, pr)
    return probs


def rejection_sample(target: torch.Tensor, draft: torch.Tensor, draft_tokens: torch.Tensor,
                     generators: Sequence[torch.Generator]):
    """Standard speculative sampling (Leviathan et al.) for B chains of k drafts.

    target: [B, k+1, V] processed target distributions (row i scores draft i+1; row k
    is the bonus position); draft: [B, k, V] draft distributions; draft_tokens: [B, k].
    Returns (num_accepted [B], correction [B]) on device: draft i is accepted with
    probability min(1, p_i(d)/q_i(d)); the first rejection draws from
    norm(max(p_i - q_i, 0)); full acceptance draws the bonus from p_k. Greedy rows
    (one-hot p and q) reduce to exact argmax matching.
    """
    B, k, V = draft.shape
    device = target.device
    if k == 0:
        corr = torch.stack([torch.multinomial(target[b, 0], 1, generator=g)[0] for b, g in enumerate(generators)])
        return torch.zeros(B, dtype=torch.int64, device=device), corr
    p_d = target[:, :k].gather(2, draft_tokens.unsqueeze(2)).squeeze(2)          # [B, k]
    q_d = draft.gather(2, draft_tokens.unsqueeze(2)).squeeze(2).clamp_min(1e-20)
    u = torch.stack([torch.rand(k, generator=g, device=device) for g in generators])   # [B, k]
    accept = u < (p_d / q_d)
    rejected = ~accept
    first_reject = torch.where(rejected.any(1), rejected.float().argmax(1), torch.full((B,), k, device=device))
    idx = first_reject.clamp_max(k)
    ar = torch.arange(B, device=device)
    p_at = target[ar, idx]                                                       # [B, V]
    q_at = torch.where((idx < k).unsqueeze(1), draft[ar, idx.clamp_max(k - 1)], torch.zeros_like(p_at))
    resid = (p_at - q_at).clamp_min(0)
    bad = resid.sum(1) <= 0
    resid = torch.where(bad.unsqueeze(1), p_at, resid)
    resid = resid / resid.sum(1, keepdim=True)
    corr = torch.stack([torch.multinomial(resid[b], 1, generator=g)[0] for b, g in enumerate(generators)])
    return first_reject, corr


class _StageTimer:
    """`LILY_SPEC_TIMING=1`: mean wall time of each stage of a round (synchronizes between stages)."""

    enabled = bool(os.environ.get("LILY_SPEC_TIMING"))

    def __init__(self):
        self.total: dict = {}
        self.rounds = 0
        self.t = 0.0

    def start(self) -> None:
        if self.enabled:
            torch.cuda.synchronize()
            now = time.perf_counter()
            if self.t:      # between rounds: the scheduler
                self.total["outside"] = self.total.get("outside", 0.0) + now - self.t
            self.t = now

    def mark(self, stage: str) -> None:
        if self.enabled:
            torch.cuda.synchronize()
            now = time.perf_counter()
            self.total[stage] = self.total.get(stage, 0.0) + now - self.t
            self.t = now

    def end_round(self) -> None:
        if self.enabled:
            self.rounds += 1
            if self.rounds % 200 == 0:
                logger.info("spec round ms: %s", {k: round(v / self.rounds * 1e3, 2) for k, v in self.total.items()})


class SpecWorkerBase:
    """Verify / accept / rollback shared by every drafting strategy."""

    #: target residual streams (layer ids) the drafter needs from every verify / prefill
    aux_layers: Optional[List[int]] = None
    #: called inside the verify graph with (output, forward batch), for fixed-shape drafter updates
    verify_hook = None

    def __init__(self, model_runner, server_args, k: int):
        self.runner = model_runner
        self.model = model_runner.model
        self.device = model_runner.device
        self.k = int(k)
        self.pool = model_runner.mamba_pool
        # recurrent-state layers that must be snapshotted and replayed on rejection
        self.gdn_layers = self.model.recurrent_layers() if hasattr(self.model, "recurrent_layers") else [
            layer.linear_attn for layer in self.model.model.layers if getattr(layer, "block_type", None) == "linear_attention"]
        if self.pool is not None and any(not layer.replay_supported(self.pool) for layer in self.gdn_layers):
            raise RuntimeError("speculative decoding on a hybrid model needs the fused linear-attention CUDA kernel "
                               "(rollback replays through it); the CUDA extension did not load")
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.verify_graphs = None
        self.draft_tokens_total = 0
        self.accepted_drafts_total = 0
        self.request_rounds = 0
        self.rounds = 0
        self.timer = _StageTimer()
        self.verified_in_graph = False

    def _init_verify_graphs(self, server_args) -> None:
        backend = self.runner.attn_backend
        graph_ok = getattr(backend, "supports_cuda_graph", lambda: False)() if backend is not None \
            else getattr(self.model, "supports_cuda_graph", False)      # models running their own attention
        if server_args.disable_cuda_graph or self.device.type != "cuda" or not graph_ok:
            return
        from lily.speculative.graph_runner import VerifyGraphRunner

        try:
            self.verify_graphs = VerifyGraphRunner(self.runner, self.k, len(self.gdn_layers), self.aux_layers, self.verify_hook)
        except Exception as exc:  # pragma: no cover - fall back to eager rounds
            logger.warning("speculative verify graph capture failed; rounds run eagerly: %s", exc)
            self.verify_graphs = None

    # ---- subclass API ----------------------------------------------------------------

    def ready(self, req) -> bool:
        """Whether `req` can take part in a drafted round (else it runs k=0)."""
        return True

    def draft(self, reqs, k: int, m: List[int], slots: torch.Tensor, req_idx: torch.Tensor, sampling: bool):
        """-> (draft tokens int64 [B, k] on device, draft distributions [B, k, V] fp32)."""
        raise NotImplementedError

    def can_prefetch(self, batch_size: int) -> bool:
        """Whether `prefetch_draft` can start the next round's draft for this batch."""
        return False

    def prefetch_draft(self, reqs, k: int, m: List[int], req_idx: torch.Tensor, anchor: torch.Tensor, num_acc: torch.Tensor) -> None:
        """Start the next draft from device values, before the host knows the accepted counts."""
        raise NotImplementedError

    def note_prefetch(self, next_m: List[int]) -> None:
        """The cached lengths the prefetched draft assumed, now known on the host."""
        raise NotImplementedError

    def after_accept(self, reqs, accepted, num_acc: List[int], hidden: torch.Tensor,
                     aux: Optional[torch.Tensor], slots: torch.Tensor, m: List[int], k: int, sampling: bool) -> None:
        """Update the drafter's state from the verify outputs (rows are request-major, k+1 each)."""
        raise NotImplementedError

    def init_after_extend(self, reqs, worker_batch, logits_output, next_token_ids: torch.Tensor) -> None:
        """Build the drafter's state for the chunk the target just prefilled."""
        raise NotImplementedError

    # ---- helpers ---------------------------------------------------------------------

    def _lm_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head.compute_logits(hidden)

    def _extend_batch(self, input_ids, req_pool_indices: torch.Tensor, prefix_lens: List[int],
                      ext_lens: List[int], out_cache_loc: torch.Tensor, **flags) -> ForwardBatch:
        """EXTEND ForwardBatch from host lists (positions prefix..prefix+ext-1 per request)."""
        positions: List[int] = []
        starts: List[int] = []
        cursor = 0
        for p, e in zip(prefix_lens, ext_lens):
            positions.extend(range(p, p + e))
            starts.append(cursor)
            cursor += e
        seq_lens = [p + e for p, e in zip(prefix_lens, ext_lens)]
        dev = self.device
        ids = input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids, dtype=torch.int64, device=dev)
        return ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=len(ext_lens),
            input_ids=ids,
            req_pool_indices=req_pool_indices,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int64, device=dev),
            positions=torch.tensor(positions, dtype=torch.int64, device=dev),
            out_cache_loc=out_cache_loc,
            extend_seq_lens=torch.tensor(ext_lens, dtype=torch.int64, device=dev),
            extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int64, device=dev),
            extend_start_loc=torch.tensor(starts, dtype=torch.int64, device=dev),
            extend_num_tokens=cursor,
            extend_seq_lens_cpu=list(ext_lens),
            extend_prefix_lens_cpu=list(prefix_lens),
            seq_lens_cpu=seq_lens,
            seq_lens_sum=sum(seq_lens),
            **flags,
        )

    # ---- one decode round -------------------------------------------------------------

    @torch.inference_mode()
    def decode_round(self, reqs, worker_batch, sampler) -> SpecRoundResult:
        """`worker_batch` was prepared by `ScheduleBatch.prepare_for_spec_decode(k)`:
        k+1 KV slots per request in `out_cache_loc`, req_to_token rows written for
        positions m..m+k, `seq_lens` = m+k+1. Returns the committed tokens per request."""
        k = int(worker_batch.spec_num_draft)
        B = len(reqs)
        dev = self.device
        m = [r.seq_len for r in reqs]
        slots = worker_batch.out_cache_loc.view(B, k + 1)
        req_idx = worker_batch.req_pool_indices
        params = [r.sampling_params for r in reqs]
        sampling = not all(p.greedy for p in params)

        # ---- 1. draft ----
        self.timer.start()
        if k > 0:
            dtoks, dprobs = self.draft(reqs, k, m, slots, req_idx, sampling)
        else:
            dtoks = torch.zeros(B, 0, dtype=torch.int64, device=dev)
            dprobs = None

        self.timer.mark("draft")

        # ---- 2. verify: target extend over [x_{m+1}, d_1..d_k] ----
        tail = int_tensor([r.output_ids[-1] for r in reqs], dev)
        verify_ids = torch.cat([tail.unsqueeze(1), dtoks], dim=1).reshape(-1)
        hybrid = bool(self.gdn_layers) and k > 0
        self.verified_in_graph = self.verify_graphs is not None and self.verify_graphs.can_run(B, k, m)
        if self.verified_in_graph:
            logits, hidden, aux, stash = self.verify_graphs.replay(verify_ids, req_idx, m, worker_batch.out_cache_loc)
            logits = logits.view(B, k + 1, -1)
            table_fn = (lambda: self.verify_graphs.replay_table(B, self.gdn_layers, self.pool)) if hybrid else None
        else:
            fb_v = self._extend_batch(
                verify_ids, req_idx, m, [k + 1] * B, worker_batch.out_cache_loc,
                return_all_logits=True, capture_hidden_states=True, capture_aux_layers=self.aux_layers,
                gdn_snapshot=hybrid, gdn_proj_stash=[None] * len(self.gdn_layers) if hybrid else None)
            out = self.runner.forward(fb_v)
            logits = out.next_token_logits.view(B, k + 1, -1)
            hidden = out.hidden_states                                   # [B*(k+1), H]
            aux = out.aux_hidden_states
            stash = fb_v.gdn_proj_stash
            table_fn = (lambda: torch.tensor(
                [layer.replay_table_row(stash[layer.linear_layer_idx], self.pool) for layer in self.gdn_layers],
                dtype=torch.int64).to(dev, non_blocking=True)) if hybrid else None

        self.timer.mark("verify")

        # ---- 3. accept ----
        sampler_output = None
        if k == 0 and worker_batch.sampling_info is not None and any(needs_plain_sampler(p) for p in params):
            # penalties / logprobs: the ordinary sampler on the single target row
            so = sampler(logits[:, 0], worker_batch.sampling_info)
            num_acc = torch.zeros(B, dtype=torch.int64, device=dev)
            corr = so.next_token_ids
            sampler_output = so
        elif not sampling:
            # all greedy: rejection sampling reduces to matching the target's argmax
            best = logits.argmax(dim=-1)                                              # [B, k+1]
            num_acc = (best[:, :k] == dtoks).long().cumprod(dim=1).sum(dim=1)
            corr = best.gather(1, num_acc.unsqueeze(1)).squeeze(1)
        else:
            target_probs = processed_probs(logits.reshape(B * (k + 1), -1), [p for p in params for _ in range(k + 1)])
            target_probs = target_probs.view(B, k + 1, -1)
            if dprobs is None:
                dprobs = torch.zeros(B, k, logits.shape[-1], device=dev)
                if k > 0:
                    dprobs.scatter_(2, dtoks.unsqueeze(2), 1.0)
            num_acc, corr = rejection_sample(target_probs, dprobs, dtoks, [r.generator for r in reqs])
        # the round's single device -> host transfer: drafts, acceptance counts, corrections;
        # rank 0's are authoritative, so every rank commits and rolls back the same tokens
        result = tensor_model_parallel_broadcast(torch.cat([dtoks, num_acc.unsqueeze(1), corr.unsqueeze(1)], dim=1))
        # All greedy with the verify in a graph: the next round's inputs are on the device already, so
        # queue the rollback and the next draft now; the host commits tokens while the GPU drafts.
        pipelined = k > 0 and not sampling and self.verified_in_graph and self.can_prefetch(B)
        if pipelined:
            host = torch.empty(result.shape, dtype=result.dtype, pin_memory=True)
            host.copy_(result, non_blocking=True)
            copied = torch.cuda.Event()
            copied.record()
            if hybrid:
                self.gdn_layers[0].replay(table_fn(), stash[0], int_tensor([i * (k + 1) for i in range(B)], dev, torch.int32),
                                          (result[:, k] + 1).to(torch.int32), req_idx.contiguous(), self.pool)
            self.prefetch_draft(reqs, k, m, req_idx, result[:, k + 1], result[:, k])
            copied.synchronize()
            packed = host.tolist()
        else:
            packed = result.tolist()
        drafts = [row[:k] for row in packed]
        num_acc_cpu = [row[k] for row in packed]
        accepted = [drafts[i][:num_acc_cpu[i]] + [row[k + 1]] for i, row in enumerate(packed)]

        if pipelined:
            self.note_prefetch([mi + j + 1 for mi, j in zip(m, num_acc_cpu)])
        self.timer.mark("accept")

        # ---- 4. GDN rollback: replay accepted prefixes from the snapshots (one launch) ----
        if hybrid and not pipelined and any(j < k for j in num_acc_cpu):
            table = table_fn()
            starts = int_tensor([i * (k + 1) for i in range(B)], dev, torch.int32)
            lens = int_tensor([j + 1 for j in num_acc_cpu], dev, torch.int32)
            self.gdn_layers[0].replay(table, stash[0], starts, lens, req_idx.contiguous(), self.pool)

        self.timer.mark("rollback")

        # ---- 5. drafter state from the accepted rows ----
        self.after_accept(reqs, accepted, num_acc_cpu, hidden, aux, slots, m, k, sampling)
        self.timer.mark("after_accept")
        self.timer.end_round()

        self.rounds += 1
        self.request_rounds += B
        self.draft_tokens_total += k * B
        self.accepted_drafts_total += sum(num_acc_cpu)
        return SpecRoundResult(accepted=accepted, num_draft=k, draft_tokens=k * B,
                               accepted_drafts=sum(num_acc_cpu), sampler_output=sampler_output)

    def acceptance_length(self) -> float:
        """Mean committed tokens per request-round so far (1 + accepted drafts)."""
        if self.request_rounds == 0:
            return 0.0
        return 1.0 + self.accepted_drafts_total / self.request_rounds


def build_spec_worker(model_runner, server_args):
    algo = server_args.speculative_algorithm
    if algo == "MTP":
        from lily.speculative.mtp_worker import MTPSpecWorker

        return MTPSpecWorker(model_runner, server_args)
    if algo == "DFLASH":
        from lily.speculative.dflash_worker import DFlashSpecWorker

        return DFlashSpecWorker(model_runner, server_args)
    raise ValueError(f"unknown speculative algorithm {algo!r}")

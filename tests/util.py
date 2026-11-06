"""Small helpers shared across the test suite."""

from __future__ import annotations

import gc

from typing import Optional

import torch


def free_cuda() -> None:
    """Drop unreferenced objects and return cached blocks to the CUDA allocator."""
    gc.collect()
    torch.cuda.empty_cache()


TIE_GAP = 0.03  # bf16 logit noise between kernels (decode vs extend, GEMV vs cuBLAS) is ~0.01


ROUTING_TIE = 0.01  # MoE corrected-score margin below which a routing flip is bf16 noise


def _routing_margin_hooks(model, margins: list):
    """Hook every MoE block so each forward appends the smallest top-k margin over the layers (the
    last position's) to `margins`; returns the handles. Hooks the block's input, not the router
    module, so the fused router kernel path is covered too."""
    from lily.layers.moe import NVFP4MoE

    per_forward: list = []

    def hook(mod, inp):
        x, gate = inp[0], mod.gate
        scores = torch.sigmoid(torch.nn.functional.linear(x.float(), gate.weight)) + gate.e_score_correction_bias
        top = scores.topk(gate.top_k + 1, dim=-1).values
        per_forward.append((top[-1, -2] - top[-1, -1]).item())

    handles = [m.register_forward_pre_hook(hook) for m in model.modules() if isinstance(m, NVFP4MoE)]

    def flush():
        if per_forward:
            margins.append(min(per_forward))
            per_forward.clear()
        elif handles:
            margins.append(float("inf"))

    return handles, flush


def greedy_reference(path: str, prompt, n: int, repetition_penalty: float = 1.0, with_routing_margins: bool = False,
                     **engine_kwargs):
    """Plain greedy tokens for `prompt` with, per position, the runner-up token and the
    top-2 logit gap, from a single-request Engine. Lets tests tolerate argmax flips at
    (near-)ties, which different but equally valid kernels can legitimately produce.
    `repetition_penalty` applies the sampler's penalty to the logits first, so the
    gaps describe the distribution that was actually sampled from. With
    `with_routing_margins` a fourth list gives, per position, the smallest MoE routing
    margin of the forward that produced it (inf for dense models): below `ROUTING_TIE`
    another implementation may pick a different expert and the continuation legitimately
    diverges from there."""
    from lily.engine import Engine, EngineConfig

    if with_routing_margins:   # forward hooks do not fire inside CUDA-graph replays
        engine_kwargs = {**engine_kwargs, "server_args": {**engine_kwargs.get("server_args", {}), "disable_cuda_graph": True}}
    eng = Engine(EngineConfig(path, dtype="bfloat16", max_total_tokens=1024, max_running_requests=1, **engine_kwargs))
    try:
        margins: list = []
        handles, flush = _routing_margin_hooks(eng.runner.model, margins) if with_routing_margins else ([], lambda: None)
        req = eng._new_seq()
        logits, seq_len = eng._extend(req, 0, list(prompt))
        flush()
        row = logits.next_token_logits[0]
        toks, second, gaps = [], [], []
        for i in range(n):
            if repetition_penalty != 1.0:
                seen = torch.tensor(sorted(set(list(prompt) + toks)), device=row.device)
                vals = row[seen]
                row = row.clone()
                row[seen] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
            top2 = torch.topk(row, 2)
            toks.append(int(top2.indices[0]))
            second.append(int(top2.indices[1]))
            gaps.append(float(top2.values[0] - top2.values[1]))
            if i + 1 < n:
                row = eng._decode(req, seq_len, toks[-1]).next_token_logits[0]
                flush()
                seq_len += 1
        eng._release(req, seq_len)
        for h in handles:
            h.remove()
        if with_routing_margins:
            return toks, second, gaps, margins
        return toks, second, gaps
    finally:
        del eng
        free_cuda()


def assert_matches_up_to_tie_flip(got, ref, second, gaps, min_agree: int = 3, tie_gap: float = TIE_GAP) -> int:
    """`got` must equal `ref` until the first mismatch, which must be a near-tie flip
    (ref's runner-up with a top-2 gap below `tie_gap`); after a flip the two runs are
    different sequences. Returns the number of positions verified."""
    for i, (g, r) in enumerate(zip(got, ref)):
        if g == r:
            continue
        assert gaps[i] < tie_gap and g == second[i], (
            f"position {i}: got {g} vs reference {r} is not a near-tie flip (gap {gaps[i]:.4f}, runner-up {second[i]})\n"
            f"got {got}\nref {ref}")
        assert i >= min_agree, f"diverged too early at {i} to be informative"
        return i
    assert len(got) == len(ref), (len(got), len(ref))
    return len(got)


# ---- tensor parallelism on one GPU ------------------------------------------------------------

def _tp_rank(path: str, tp: int, rank: int, port: int, ids, steps: int, out, extra: Optional[dict] = None):
    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from lily.engine import Engine, EngineConfig

    args = {"dist_backend": "gloo", "disable_cuda_graph": True} if tp > 1 else {}
    args.update(extra or {})
    eng = Engine(EngineConfig(path, dtype="bfloat16", max_total_tokens=256, max_running_requests=1, tp_size=tp, rank=rank,
                              dist_init_addr=f"127.0.0.1:{port}", device="cuda:0", server_args=args or None))
    try:
        logits = eng.all_prefill_logits(ids).float().cpu()
        toks = eng.generate_greedy(ids, steps, ignore_eos=True)
    finally:
        del eng
        free_cuda()
    if out is not None:
        out.put((rank, logits.tolist(), toks))     # plain lists: no shared-memory handles across exit
    return logits, toks


def assert_tp2_matches_tp1(path: str, ids, steps: int, strict: torch.Tensor, atol: float, extra: Optional[dict] = None) -> None:
    """Two ranks over gloo on this GPU (per-rank weight slicing + collectives) reproduce the single-rank
    prefill logits (at the `strict` positions, within `atol`) and the greedy tokens up to near-ties.
    `extra`: more server args for the two ranks (e.g. a speculative drafter: its sharded top-k)."""
    import multiprocessing as mp
    import queue
    import socket
    import time

    ref_logits, _ = _tp_rank(path, 1, 0, 0, ids, steps, None)
    ref_toks, second, gaps = greedy_reference(path, ids, steps)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_tp_rank, args=(path, 2, r, port, ids, steps, q, extra), daemon=True) for r in range(2)]
    for p in procs:
        p.start()
    results, deadline = {}, time.time() + 900
    while len(results) < 2:
        try:
            rank, logits, toks = q.get(timeout=5)
            results[rank] = (torch.tensor(logits), toks)
        except queue.Empty:
            assert not [p.exitcode for p in procs if p.exitcode not in (None, 0)], "a TP rank died"
            assert time.time() < deadline, "TP ranks timed out"
    for p in procs:
        p.join(timeout=60)
    for rank, (logits, toks) in results.items():
        diff = (logits - ref_logits).abs().max(dim=-1).values
        assert diff[strict].max() < atol, f"rank {rank}: per-position max logit diff {diff.tolist()}"
        assert_matches_up_to_tie_flip(toks, ref_toks, second, gaps, min_agree=1)

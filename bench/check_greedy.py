#!/usr/bin/env python
"""Is a greedy run really the target's greedy output? Teacher-forces saved outputs through the plain
model (prefill path) and reports, per run, the positions where the emitted token is not the argmax
and the top-1 / emitted logit gap there. Paths differ in rounding, so tiny gaps are expected: compare
a speculative run against a plain run, which is the noise floor.

    python bench/check_greedy.py --model-path ... --runs runs.json      # {name: [[prompt, output], ...]}
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import torch


def _worker(rank: int, args) -> None:
    from lily.engine import Engine, EngineConfig

    eng = Engine(EngineConfig(model_path=args.model_path, tp_size=args.tp, rank=rank, context_length=8192, max_running_requests=2,
                              max_total_tokens=32768, mem_fraction_static=0.8,
                              server_args={"quantize_lm_head": "nvfp4", "mamba_ssm_dtype": "bfloat16", "disable_cuda_graph": True}))
    runs = json.load(open(args.runs))
    for name, pairs in runs.items():
        total, gaps = 0, []
        for prompt, out in pairs:
            with torch.inference_mode():
                logits = eng.all_prefill_logits(prompt + out[:-1])[len(prompt) - 1:]
            emitted = torch.tensor(out, device=logits.device)
            top = logits.max(dim=-1).values
            gap = top - logits.gather(1, emitted[:, None]).squeeze(1)
            total += len(out)
            gaps += [round(g, 3) for g in gap[gap > 0].tolist()]
        if rank == 0:
            gaps.sort()
            print(f"=== {name}: {total} tokens, {len(gaps)} not argmax ({100 * len(gaps) / total:.2f}%), "
                  f"max gap {max(gaps, default=0):.3f}, gaps {gaps[-12:]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tp", type=int, default=4)
    args = ap.parse_args()
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(r, args)) for r in range(args.tp)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        if p.exitcode:
            raise SystemExit(f"rank failed: exit {p.exitcode}")


if __name__ == "__main__":
    main()

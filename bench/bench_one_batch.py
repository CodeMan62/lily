"""Fixed-batch latency/throughput benchmark. Argument-compatible with sglang's bench_one_batch.

Runs one batch of `--batch-size` prompts of `--input-len` tokens each, decodes
`--output-len` tokens, and reports prefill latency, decode latency per step and
per token, and aggregate throughput.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from typing import List

import torch

from lula.models.model_config import ModelConfig
from lula.scheduler.scheduler import Scheduler
from lula.sampling.sampling_params import SamplingParams
from lula.server_args import ServerArgs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("lula bench_one_batch")
    parser.add_argument("--model-path", "--model", dest="model_path", required=True)
    parser.add_argument("--backend", default="triton", choices=["triton", "torch_native"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--num-iters", type=int, default=3, help="measured iterations after a warmup")
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--disable-overlap", action="store_true")
    parser.add_argument("--result-file", default=None)
    return parser


def make_prompts(batch_size: int, input_len: int, vocab: int, seed: int = 0) -> List[List[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(100, vocab - 100) for _ in range(input_len)] for _ in range(batch_size)]


@torch.inference_mode()
def run_once(scheduler, prompts, output_len):
    sp = SamplingParams(max_new_tokens=output_len, ignore_eos=True)
    for i, p in enumerate(prompts):
        scheduler.add_request(f"r{i}", p, sp)

    t0 = time.perf_counter()
    first = True
    prefill_latency = 0.0
    while scheduler.waiting_queue or (
        scheduler.running_batch is not None and scheduler.running_batch.reqs
    ):
        if not scheduler.step():
            break
        if first:
            torch.cuda.synchronize()
            prefill_latency = time.perf_counter() - t0
            first = False
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    return prefill_latency, total


def main() -> None:
    args = build_parser().parse_args()

    cfg = ModelConfig.from_pretrained(args.model_path, dtype=args.dtype)
    sa = ServerArgs(
        model_path=args.model_path, dtype=args.dtype, attention_backend=args.backend,
        max_running_requests=args.batch_size, mem_fraction_static=args.mem_fraction_static,
        context_length=max(2048, args.input_len + args.output_len + 16),
        disable_cuda_graph=args.disable_cuda_graph, disable_overlap_schedule=args.disable_overlap,
    )
    scheduler = Scheduler(cfg, sa)
    prompts = make_prompts(args.batch_size, args.input_len, cfg.vocab_size)

    use_overlap = scheduler.enable_overlap

    def one(prompts, out_len):
        if not use_overlap:
            return run_once(scheduler, prompts, out_len)
        sp = SamplingParams(max_new_tokens=out_len, ignore_eos=True)
        for i, p in enumerate(prompts):
            scheduler.add_request(f"r{i}", p, sp)
        t0 = time.perf_counter()
        prev = None
        prefill_latency = 0.0
        first = True
        while True:
            prev = scheduler.overlap_step(prev)
            if first and prev is not None:
                torch.cuda.synchronize()
                prefill_latency = time.perf_counter() - t0
                first = False
            if not scheduler.waiting_queue and (
                scheduler.running_batch is None or not scheduler.running_batch.reqs
            ) and prev is None:
                break
        torch.cuda.synchronize()
        return prefill_latency, time.perf_counter() - t0

    # Warmup (compiles Triton kernels, captures graphs, sizes allocations).
    one(prompts, min(4, args.output_len))
    gc.collect()
    torch.cuda.empty_cache()

    prefill_times, totals = [], []
    for _ in range(args.num_iters):
        prefill, total = one(prompts, args.output_len)
        prefill_times.append(prefill)
        totals.append(total)

    prefill = sum(prefill_times) / len(prefill_times)
    total = sum(totals) / len(totals)
    decode = max(total - prefill, 1e-9)
    n_decode_tokens = args.batch_size * args.output_len
    result = {
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "backend": args.backend,
        "dtype": str(cfg.dtype),
        "prefill_latency_s": round(prefill, 4),
        "decode_latency_s": round(decode, 4),
        "decode_per_token_ms": round(decode / args.output_len * 1000, 3),
        "decode_tokens_per_s": round(n_decode_tokens / decode, 2),
        "total_latency_s": round(total, 4),
        "total_tokens_per_s": round((args.batch_size * (args.input_len + args.output_len)) / total, 2),
    }
    print(json.dumps(result, indent=2))
    if args.result_file:
        with open(args.result_file, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

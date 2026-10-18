"""Online serving benchmark. Argument-compatible with sglang's bench_serving.

Sends `--num-prompts` requests to a running lula server at `--request-rate`
requests/second (use `inf` to fire them all at once), streaming the responses to
measure TTFT and TPOT, and reports request/token throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from typing import List, Optional

import httpx
from transformers import AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("lula bench_serving")
    parser.add_argument("--backend", default="vllm", help="accepted for SGLang compatibility")
    parser.add_argument("--base-url", "--host", dest="base_url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="lula")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-file", default=None)
    return parser


def make_prompts(tokenizer, n: int, input_len: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    vocab = tokenizer.get_vocab()
    words = [w for w in vocab if w.strip() and not w.startswith("<")]
    prompts = []
    for _ in range(n):
        text = " ".join(rng.choice(words) for _ in range(input_len))
        prompts.append(text)
    return prompts


async def one_request(client: httpx.AsyncClient, base_url: str, model: str, prompt: str,
                      output_len: int) -> dict:
    payload = {
        "model": model, "prompt": prompt, "max_tokens": output_len,
        "temperature": 0.0, "stream": True,
    }
    start = time.perf_counter()
    first = None
    tokens = 0
    async with client.stream("POST", f"{base_url}/v1/completions", json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk["choices"][0]["text"]:
                if first is None:
                    first = time.perf_counter()
                tokens += 1
    end = time.perf_counter()
    ttft = (first - start) if first is not None else (end - start)
    return {"ttft": ttft, "latency": end - start, "tokens": tokens}


async def run(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    prompts = make_prompts(tokenizer, args.num_prompts, args.input_len, args.seed)

    interval = 0.0 if args.request_rate == float("inf") else 1.0 / args.request_rate
    results: List[dict] = []
    limits = httpx.Limits(max_connections=args.num_prompts + 8)
    async with httpx.AsyncClient(timeout=600.0, limits=limits) as client:
        tasks = []
        start = time.perf_counter()
        for i, prompt in enumerate(prompts):
            target = i * interval
            now = time.perf_counter() - start
            if target > now:
                await asyncio.sleep(target - now)
            tasks.append(asyncio.create_task(
                one_request(client, args.base_url, args.model, prompt, args.output_len)
            ))
        results = await asyncio.gather(*tasks)
    total = time.perf_counter() - start

    ttfts = sorted(r["ttft"] for r in results)
    tpots = [
        (r["latency"] - r["ttft"]) / max(r["tokens"] - 1, 1) for r in results if r["tokens"] > 1
    ]
    out_tokens = sum(r["tokens"] for r in results)

    def pct(values, q):
        if not values:
            return 0.0
        return values[min(len(values) - 1, int(q * len(values)))]

    return {
        "num_prompts": args.num_prompts,
        "request_rate": args.request_rate,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "completed": len(results),
        "total_latency_s": round(total, 3),
        "request_throughput": round(len(results) / total, 3),
        "output_token_throughput": round(out_tokens / total, 2),
        "mean_ttft_ms": round(1000 * sum(ttfts) / len(ttfts), 2),
        "p90_ttft_ms": round(1000 * pct(ttfts, 0.9), 2),
        "mean_tpot_ms": round(1000 * sum(tpots) / len(tpots), 2) if tpots else 0.0,
        "mean_latency_ms": round(1000 * sum(r["latency"] for r in results) / len(results), 2),
    }


def main() -> None:
    args = build_parser().parse_args()
    result = asyncio.run(run(args))
    print(json.dumps(result, indent=2))
    if args.result_file:
        with open(args.result_file, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Throughput sweep: start a lula server, fire 1/8/32/64 concurrent requests, print a table.
#
#   bench/bench.sh Qwen/Qwen3-0.6B
#   bench/bench.sh Qwen/Qwen2.5-0.5B-Instruct --max-running-requests 128
#
# Extra args go straight to lula-serve. Env: PORT (30000), MAX_TOKENS (256).
# Overlap scheduling is disabled by default (it currently crashes the scheduler).
set -euo pipefail

MODEL="${1:?usage: bench/bench.sh <model-path> [lula-serve args...]}"
shift
PORT="${PORT:-30000}"
MAX_TOKENS="${MAX_TOKENS:-256}"
LOG="$(mktemp -t lula-bench.XXXXXX.log)"
cd "$(dirname "$0")/.."

# New session so the whole server tree (uv, uvicorn, scheduler workers) dies together.
setsid uv run lula-serve --model-path "$MODEL" --dtype bfloat16 --port "$PORT" \
  --disable-overlap-schedule "$@" >"$LOG" 2>&1 &
SERVER_PID=$!
# SIGKILL: uvicorn waits forever on in-flight requests after SIGTERM.
trap 'kill -9 -- -"$SERVER_PID" 2>/dev/null || true' EXIT
crashed() { grep -q Traceback "$LOG"; }
die() { echo "server crashed:"; tail -20 "$LOG"; exit 1; }

echo "starting server for $MODEL (log: $LOG)"
until curl -sf "localhost:$PORT/health" >/dev/null; do
  kill -0 "$SERVER_PID" 2>/dev/null && ! crashed || die
  sleep 2
done

# The HTTP server stays up when the scheduler process dies, so watch the log
# and kill the client instead of letting it wait on requests that never finish.
MODEL="$MODEL" PORT="$PORT" MAX_TOKENS="$MAX_TOKENS" uv run python - <<'EOF' &
import asyncio, os, time
import openai

model, max_tokens = os.environ["MODEL"], int(os.environ["MAX_TOKENS"])
client = openai.AsyncClient(base_url=f"http://127.0.0.1:{os.environ['PORT']}/v1", api_key="none")

async def one(i):
    t0 = time.perf_counter()
    stream = await client.completions.create(
        model=model, prompt=f"Write a long story about robot number {i}.",
        max_tokens=max_tokens, temperature=0.0, stream=True,
        extra_body={"ignore_eos": True})
    ttft = None
    toks = 0
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].text:
            if ttft is None:
                ttft = time.perf_counter() - t0
            toks += 1
    return ttft, toks

async def run(n):
    t = time.perf_counter()
    results = await asyncio.gather(*(one(i) for i in range(n)))
    ttfts = [ttft for ttft, _ in results if ttft is not None]
    toks = sum(toks for _, toks in results)
    p99 = sorted(ttfts)[int(0.99 * (len(ttfts) - 1))] if ttfts else 0.0
    return toks, time.perf_counter() - t, sum(ttfts) / len(ttfts) if ttfts else 0.0, p99

async def main():
    await asyncio.wait_for(run(1), timeout=300)  # warmup
    print(f"\n{model}  ({max_tokens} output tokens/request)\n")
    print("| concurrency | total tok/s | tok/s per request | ttft avg (ms) | ttft p99 (ms) | time (s) |")
    print("|---|---|---|---|---|---|")
    for n in (1, 8, 32, 64):
        toks, dt, ttft_avg, ttft_p99 = await asyncio.wait_for(run(n), timeout=600)
        print(f"| {n} | {toks / dt:.1f} | {toks / dt / n:.1f} | "
              f"{ttft_avg * 1000:.1f} | {ttft_p99 * 1000:.1f} | {dt:.2f} |", flush=True)

asyncio.run(main())
EOF
CLIENT_PID=$!
while kill -0 "$CLIENT_PID" 2>/dev/null; do
  crashed && { kill "$CLIENT_PID"; die; }
  sleep 1
done
wait "$CLIENT_PID"

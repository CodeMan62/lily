# Benchmarks

Machine: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB), Linux, PyTorch 2.14+cu130.
Model: `HuggingFaceTB/SmolLM2-135M-Instruct`, fp32, Triton attention backend,
radix cache on. Dates are the run date.

Reproduce:

```bash
uv run python bench/bench_one_batch.py \
    --model-path HuggingFaceTB/SmolLM2-135M-Instruct --dtype float32 \
    --batch-size 1 --input-len 128 --output-len 32 --num-iters 3

# with a server running (`uv run lula-serve --model-path ...`)
uv run python bench/bench_serving.py --base-url http://127.0.0.1:30000 \
    --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --num-prompts 16 --request-rate inf --input-len 64 --output-len 32
```

## Offline: `bench_one_batch` (2026-10-15)

| batch | input | output | prefill (s) | decode/token (ms) | decode tok/s | total tok/s |
|---|---|---|---|---|---|---|
| 1 | 128 | 32 | 0.041 | 13.34 | 74.95 | 341.70 |
| 8 | 128 | 32 | 0.208 | 21.33 | 375.03 | 1437.15 |

Eager execution (the pre-Phase-7 baseline). Decode is dominated by the per-step
Python and kernel-launch overhead at bs=1, which is what CUDA graphs and overlap
scheduling remove.

### Phase 7: CUDA graphs + overlap (2026-10-15)

Same model/config, `--dtype float32`, `bench_one_batch`, 128-in/32-out. `graphs`
= CUDA graph replay; `overlap` = future-token overlap scheduling.

| batch | graphs | overlap | decode/token (ms) | decode tok/s | total tok/s |
|---|---|---|---|---|---|
| 1 | off | off | 12.74 | 78.52 | 368.14 |
| 1 | **on** | off | **5.34** | **187.45** | **827.62** |
| 16 | on | off | 11.86 | 1348.59 | 4152.63 |
| 16 | on | **on** | 10.97 | 1458.29 | 4186.26 |

Reproduce:

```bash
# graphs off (eager)
uv run python bench/bench_one_batch.py --model-path HuggingFaceTB/SmolLM2-135M-Instruct \
    --dtype float32 --batch-size 1 --input-len 128 --output-len 32 \
    --disable-cuda-graph --disable-overlap --num-iters 3
# graphs on (default)
uv run python bench/bench_one_batch.py --model-path HuggingFaceTB/SmolLM2-135M-Instruct \
    --dtype float32 --batch-size 1 --input-len 128 --output-len 32 --disable-overlap --num-iters 3
```

**bs=1 decode speedup: 12.74 → 5.34 ms/token = 2.39×** (Phase 7 target ≥2×; met).
At bs=16 the batch is already launch-amortized, so overlap scheduling adds a
further ~8% decode throughput (1348.59 → 1458.29 tok/s).

## Online: `bench_serving` (2026-10-15)

16 concurrent requests, 64-token prompts, 32 output tokens.

| request rate | req/s | output tok/s | mean TTFT (ms) | p90 TTFT (ms) | mean TPOT (ms) | mean latency (ms) |
|---|---|---|---|---|---|---|
| inf (all at once) | 6.57 | 186.85 | 1770.13 | 1899.49 | 24.70 | 2424.74 |
| 8 req/s | 3.93 | 117.75 | 1084.47 | 1627.61 | 70.25 | 3095.16 |

## SGLang comparison (2026-10-15)

SGLang 0.5.21 was installed into a separate venv (`/home/rox/.sglang-venv`).
SGLang's default flashinfer attention backend does **not** support fp32, so the
side-by-side was run in **fp16** for both engines (SGLang cannot serve this model
in fp32). lula was therefore re-run with `--dtype float16` too. Same machine,
same model, same 16 concurrent requests / 64-in / 32-out, CUDA graphs on for both;
lula also has overlap scheduling on.

| engine | request rate | req/s | output tok/s | mean TTFT (ms) | p90 TTFT (ms) | mean TPOT (ms) | mean latency (ms) |
|---|---|---|---|---|---|---|---|
| lula (fp16) | inf | 47.19 | 1389.25 | 156.07 | 176.07 | 6.27 | 329.20 |
| SGLang (fp16) | inf | 34.35 | 1013.31 | 164.81 | 184.83 | 10.72 | 454.74 |
| lula (fp16) | 8 req/s | 7.95 | 228.56 | 21.30 | 24.65 | 4.97 | 151.36 |
| SGLang (fp16) | 8 req/s | 8.13 | 217.92 | 22.03 | 64.86 | 3.41 | 104.75 |

Reproduce (SGLang, separate venv; `ninja` must be on `PATH` and `CUDA_HOME` set
so flashinfer can JIT):

```bash
export PATH=/home/rox/.sglang-venv/.venv/bin:$PATH CUDA_HOME=/opt/cuda
python -m sglang.launch_server --model-path HuggingFaceTB/SmolLM2-135M-Instruct \
    --dtype float16 --port 30001 --host 127.0.0.1 --mem-fraction-static 0.5
uv run python bench/bench_serving.py --base-url http://127.0.0.1:30001 \
    --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --num-prompts 16 --input-len 64 --output-len 32 --request-rate inf
```

lula leads on saturation throughput (~1.37× at `inf`) and TPOT at saturation;
SGLang is marginally ahead at the fixed 8 req/s rate. Note SGLang's first-ever
request pays a one-off flashinfer JIT compile (~4.5 s TTFT) — warm the server
before measuring.

# lily

lily is an LLM inference engine. It has an OpenAI-compatible HTTP server, a
batched scheduler, CUDA graphs, tensor and expert parallelism, NVFP4 weights, and speculative
decoding (MTP and DFlash2). The hot paths use custom CUDA kernels.

## Models

| Model | Hardware | Notes |
|---|---|---|
| `nvidia/GLM-5.3-Flash-NVFP4` (320B, 18B active MoE) | 4x RTX PRO 6000 | TP=4 + EP, DFlash2 drafter |
| Qwen3.5 / qwen3.8-27B NVFP4 (hybrid linear attention) | RTX 5090 | MTP or DFlash2 |
| Qwen3, SmolLM2 and other Llama-style models | any CUDA GPU | bf16 |

## Results: GLM-5.3-Flash on 4x RTX PRO 6000

DFlash2 with k = 7, real prompts, 256-token outputs.

| Concurrency | tok/s | ms/token |
|---|---|---|
| 1 | 218 | 4.66 |
| 8 | 511 | 13.7 |
| 16 | 777 | 18.3 |
| 32 | 1034 | 27.0 |

What gets there: `docs/techniques-207.md`.

## Install

```bash
uv sync
```

CUDA kernels in `lily/csrc/` are compiled on first import and cached in `~/.cache/lily/cuda`.

## Serve

```bash
uv run lily-serve --model-path Qwen/Qwen3-0.6B --dtype bfloat16 --port 30000
```

GLM-5.3-Flash with the DFlash2 drafter on 4 GPUs:

```bash
uv run lily-serve --model-path nvidia/GLM-5.3-Flash-NVFP4 --tp-size 4 \
  --quantize-lm-head nvfp4 --mamba-ssm-dtype bfloat16 \
  --speculative-algorithm DFLASH --speculative-draft-model-path incoai/GLM-5.3-Flash-DFlash2 \
  --speculative-num-steps 7 --cuda-graph-max-bs 32 --max-running-requests 32
```

Main flags: `--tp-size`, `--speculative-algorithm MTP|DFLASH`, `--speculative-num-steps`,
`--cuda-graph-max-bs`, `--max-running-requests`, `--context-length`, `--mem-fraction-static`.

## Use

I have wrote a small chat.py file to chat with a model served for quick testing purpose.

```bash
uv run python example/chat.py --model Qwen/Qwen3-0.6B      # streaming chat, shows reasoning
```

## Benchmark and profile

```bash
uv run python bench/bench.py --model-path <path> --batch-sizes 1,8 --concurrency 1,8
modal run scripts/modal_glm5.py::bench --spec dflash --k 7 --concurrency 1,8,16,32 --graph-bs 32
```

- `LILY_PROFILE_BS=1,8` logs a per-kernel GPU-time table for 20 decode steps at each batch size.
- `LILY_PROFILE_TRACE=<dir>` also writes a chrome trace. Summarise it with `bench/profile_report.py` or open it in Perfetto.
- Note that profiler is still experimental i am trying to work on it 

## TODO
1. Multimodal
- [ ] VLM
- [ ] MLLM
- [ ] Audio
- [ ] Multimodal perceptron
2. Hardware
- [ ] AMD goal is to run kimi-k3.1pro on a node of MI350x
- [ ] Tenstorrent(no much sure but will try my best)

## acknowledgement
lily is inspired primarily from sglang. very very big thanks to their developers.

"""GLM-5.3-Flash (nvidia/GLM-5.3-Flash-NVFP4) bring-up on Modal, cheapest possible.

    modal run scripts/modal_glm5.py::download     # CPU only: 190 GiB checkpoint -> Volume `lily-models`
    modal run scripts/modal_glm5.py::check_load   # CPU only: one rank's slice loads with 0 missing params
    modal run scripts/modal_glm5.py::bench        # 4x RTX PRO 6000 ($3.03/h each): serve, check one
                                                  # reply, measure tok/s at concurrency 1 and 8, exit

The image compiles lily's CUDA kernels for sm_120 at build time (CPU, no GPU billed), so the GPU
run only loads weights (~5 min), warms up and benchmarks. `bench` is hard-capped by its timeout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
MODEL_REPO = "nvidia/GLM-5.3-Flash-NVFP4"
MODEL_DIR = "/models/glm53-flash-nvfp4"
GPU = "RTX-PRO-6000:4"
GPU_HOUR_USD = 4 * 3.03
PORT = 30000

app = modal.App("lily-glm53-flash")
vol = modal.Volume.from_name("lily-models", create_if_missing=True)

build_env = {"TORCH_CUDA_ARCH_LIST": "12.0", "LILY_CUDA_BUILD_DIR": "/opt/lily-cuda", "CUDA_HOME": "/usr/local/cuda",
             "TOKENIZERS_PARALLELISM": "false"}

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git")
    .env(build_env)
    .pip_install("uv")
    .add_local_file(ROOT / "pyproject.toml", "/app/pyproject.toml", copy=True)
    .add_local_file(ROOT / "uv.lock", "/app/uv.lock", copy=True)
    .run_commands(
        "cd /app && uv export --frozen --no-hashes --no-emit-project --group dev -o /app/requirements.txt"
        " && uv pip install --system -r /app/requirements.txt")
    .add_local_dir(ROOT / "lily", "/app/lily", copy=True, ignore=["__pycache__", "*.pyc"])
    .add_local_dir(ROOT / "bench", "/app/bench", copy=True, ignore=["__pycache__", "logs", "results"])
    .add_local_file(ROOT / "README.md", "/app/README.md", copy=True)
    .run_commands(
        "cd /app && uv pip install --system --no-deps -e /app",
        # sm_120 kernels compiled here, on the CPU builder; the same TORCH_CUDA_ARCH_LIST at run
        # time makes ninja find them up to date
        "LILY_CUDA_BUILD_ONLY=1 python -c 'from lily.layers.quantization import cuda_ops; assert cuda_ops.available()'",
    )
    .workdir("/app")
)


@app.function(image=image, volumes={"/models": vol}, cpu=8, memory=16384, timeout=3 * 3600)
def download() -> dict:
    """Checkpoint shards + config/tokenizer into the Volume (skips files already there)."""
    t0 = time.time()
    cmd = ["hf", "download", MODEL_REPO, "--local-dir", MODEL_DIR, "--max-workers", "16"]
    for pattern in ("*.safetensors", "*.json", "*.jinja", "*.txt", "*.model"):
        cmd += ["--include", pattern]
    print(" ".join(cmd), flush=True)
    env = {**os.environ, "HF_XET_HIGH_PERFORMANCE": "1"}
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    subprocess.run(cmd, check=True, env=env)
    vol.commit()
    shards = sorted(p.name for p in Path(MODEL_DIR).glob("*.safetensors"))
    index = json.load(open(f"{MODEL_DIR}/model.safetensors.index.json"))["weight_map"]
    missing = sorted(set(index.values()) - set(shards))
    size = sum(p.stat().st_size for p in Path(MODEL_DIR).iterdir()) / 2**30
    print(f"{len(shards)} shards, {size:.1f} GiB, missing {missing}, {time.time() - t0:.0f}s", flush=True)
    return {"shards": len(shards), "gib": round(size, 1), "missing": missing, "seconds": round(time.time() - t0)}


def _tee(proc: subprocess.Popen, log_path: str, t0: float) -> None:
    with open(log_path, "w") as f:
        for line in proc.stdout:
            sys.stdout.write(f"[serve +{time.time() - t0:5.0f}s] " + line)
            sys.stdout.flush()
            f.write(line)


def _generate_probe(base: str) -> bool:
    """The HTTP front end answers /health before the TP ranks have loaded; a real one-token
    request only returns once the scheduler is serving."""
    import httpx

    try:
        r = httpx.post(f"{base}/generate", timeout=15.0,
                       json={"input_ids": [1000, 1001, 1002], "sampling_params": {"max_new_tokens": 1, "ignore_eos": True}})
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


@app.function(image=image, volumes={"/models": vol}, gpu=GPU, cpu=16, memory=65536, timeout=22 * 60)
def bench(input_len: int = 256, output_len: int = 128, concurrency: str = "1,8", health_timeout: int = 12 * 60) -> dict:
    import httpx

    sys.path.insert(0, "/app")
    from bench.bench import _http_line, _one_request, _random_ids, _sweep_level, http_table

    t_start = time.time()
    base = f"http://127.0.0.1:{PORT}"
    log_path = "/tmp/lily-serve.log"
    cmd = ["lily-serve", "--model-path", MODEL_DIR, "--served-model-name", "glm", "--tp-size", "4",
           "--quantize-lm-head", "nvfp4", "--mamba-ssm-dtype", "bfloat16", "--context-length", "32768",
           "--max-running-requests", "16", "--max-total-tokens", "131072", "--cuda-graph-max-bs", "8",
           "--mem-fraction-static", "0.8", "--port", str(PORT)]
    print(" ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    threading.Thread(target=_tee, args=(proc, log_path, t_start), daemon=True).start()
    result: dict = {"gpu": GPU}
    try:
        # ---- wait until a real request is served; abort early on a crash ----
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"lily-serve exited with {proc.returncode}")
            if os.path.exists(log_path) and "Traceback" in open(log_path, errors="replace").read():
                time.sleep(3)
                raise RuntimeError("a lily-serve process crashed during startup (see [serve] log above)")
            if _generate_probe(base):
                break
            if time.time() - t_start > health_timeout:
                raise RuntimeError(f"server not healthy after {health_timeout}s")
            time.sleep(5.0)
        result["startup_s"] = round(time.time() - t_start)
        print(f"\n=== server ready after {result['startup_s']}s ===\n", flush=True)

        # ---- one real reply: is the model coherent? ----
        body = {"model": "glm", "max_tokens": 96, "temperature": 0,
                "messages": [{"role": "user", "content": "In two sentences: what is a mixture-of-experts model?"}]}
        t0 = time.time()
        r = httpx.post(f"{base}/v1/chat/completions", json=body, timeout=600).json()
        dt = time.time() - t0
        msg = r["choices"][0]["message"]
        text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        n_out = r.get("usage", {}).get("completion_tokens", 0)
        result["chat"] = {"text": text, "completion_tokens": n_out, "seconds": round(dt, 2),
                          "tok_s_incl_prefill": round(n_out / dt, 1) if dt else None}
        print(f"=== chat ({n_out} tokens in {dt:.1f}s) ===\n{text}\n", flush=True)

        # ---- throughput sweep (bench/bench.py's HTTP phase against this server) ----
        import asyncio

        vocab = 154880
        url = f"{base}/generate"
        rows = []

        async def go():
            async with httpx.AsyncClient() as client:
                await _one_request(client, url, _random_ids(input_len, vocab, seed=7), output_len)   # warmup
                for c in [int(x) for x in concurrency.split(",")]:
                    prompts = [_random_ids(input_len, vocab, seed=10_000 + i) for i in range(max(8, c))]
                    row = await _sweep_level(client, url, prompts, output_len, c)
                    rows.append(row)
                    print("  " + _http_line(row), flush=True)

        asyncio.run(go())
        print("\n" + http_table(rows) + "\n", flush=True)
        result["http"] = [row.__dict__ for row in rows]
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except ProcessLookupError:
            pass
        total = time.time() - t_start
        result["total_s"] = round(total)
        result["est_cost_usd"] = round(total / 3600 * GPU_HOUR_USD, 2)
        print(f"=== total {total:.0f}s on {GPU} ~ ${result['est_cost_usd']:.2f} ===", flush=True)
        try:
            os.makedirs("/models/results", exist_ok=True)
            out = f"/models/results/glm53-flash-{int(t_start)}.json"
            json.dump({**result, "server_log_tail": open(log_path).read()[-20000:]}, open(out, "w"), indent=1)
            vol.commit()
            print(f"saved {out}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"could not save results: {exc}", flush=True)
    return result



@app.function(image=image, volumes={"/models": vol}, cpu=8, memory=98304, timeout=3600)
def check_load(rank: int = 0, tp: int = 4) -> dict:
    """Load one TP rank's slice of the real checkpoint on the CPU (no GPU billed): catches loader
    name/shape mismatches before the multi-GPU run. Rank 0 of 4 holds ~45 GiB."""
    import torch
    import lily.distributed.distributed as dist_mod
    from lily.layers.quantization import build_quant_config
    from lily.models.loader import get_model
    from lily.models.model_config import ModelConfig

    dist_mod._TP_SIZE, dist_mod._TP_RANK = tp, rank      # slicing only; no process group on the CPU
    cfg = ModelConfig.from_pretrained(MODEL_DIR, context_length=4096)
    cfg.lm_head_quantization = None
    t0 = time.time()
    model = get_model(cfg, device="cpu", dtype=torch.bfloat16, quant_config=build_quant_config(None, cfg))
    n_params = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
    print(f"rank {rank}/{tp}: loaded {n_params:.1f} GiB of parameters in {time.time() - t0:.0f}s", flush=True)
    return {"gib": round(n_params, 1), "seconds": round(time.time() - t0)}


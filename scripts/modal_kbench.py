"""Kernel tuning loop on Modal for a few cents a run: edit lily/csrc or a bench, run, read the table, repeat.

    modal run scripts/modal_kbench.py --bench glm_moe [--filter "T=8"]
    KBENCH_GPU=B200 KBENCH_BUDGET=0.05 modal run scripts/modal_kbench.py --bench glm_moe

1. `build` (CPU container, ~$0.05/h per core): syncs lily/ into the Volume, giving unchanged files an old mtime and
   changed ones the current time, so ninja recompiles only the .cu files you touched. Unchanged sources skip the
   build entirely.
2. `measure` (GPU): checks the binary matches your csrc, imports it directly (no compiler on the GPU clock),
   runs bench/kbench.py. Its timeout comes from the budget, so a run cannot cost more than KBENCH_BUDGET.
3. Locally: the table with the change vs your previous run, saved under bench/results/kbench/.

Source and bench files are mounted per run, so edits never rebuild the image. Triton's cache lives on the
Volume: the first run of a bench pays Triton's JIT, later runs do not.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
GPU = os.environ.get("KBENCH_GPU", "RTX-PRO-6000")
BUDGET_USD = float(os.environ.get("KBENCH_BUDGET", "0.05"))
# Modal list prices (USD); check modal.com/pricing. GPU containers also pay for their CPU cores and memory.
GPU_HOUR_USD = {"RTX-PRO-6000": 3.03, "B200": 6.25, "H100": 3.95, "L40S": 1.95}
ARCH = {"RTX-PRO-6000": "12.0", "B200": "10.0", "H100": "9.0", "L40S": "8.9"}[GPU]
CORE_S_USD, GIB_S_USD = 0.0000131, 0.00000222
BUILD_CORES, BUILD_GIB, GPU_CORES, GPU_GIB = 8, 16, 2, 8
GPU_S_USD = GPU_HOUR_USD[GPU] / 3600 + GPU_CORES * CORE_S_USD + GPU_GIB * GIB_S_USD
GPU_TIMEOUT = max(10, int(BUDGET_USD / GPU_S_USD))

app = modal.App("lily-kbench")
vol = modal.Volume.from_name("lily-kbench", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .env({"KBENCH_GPU": GPU, "KBENCH_BUDGET": str(BUDGET_USD), "CUDA_HOME": "/usr/local/cuda",
          "TRITON_CACHE_DIR": "/cache/triton", "TOKENIZERS_PARALLELISM": "false"})
    .pip_install("uv")
    .add_local_file(ROOT / "pyproject.toml", "/app/pyproject.toml", copy=True)
    .add_local_file(ROOT / "uv.lock", "/app/uv.lock", copy=True)
    .run_commands("cd /app && uv export --frozen --no-hashes --no-emit-project -o /app/requirements.txt"
                  " && uv pip install --system -r /app/requirements.txt")
    .add_local_dir(ROOT / "lily", "/src/lily", ignore=["__pycache__", "*.pyc"])
    .add_local_dir(ROOT / "bench", "/src/bench", ignore=["__pycache__", "logs", "results"])
)


def csrc_digest(csrc: str) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(csrc).iterdir()):
        h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()


def sync_tree(src: str, dst: str) -> list:
    """Mirror src into dst; unchanged files get mtime 1 so ninja treats their objects as up to date
    whatever mtimes the Volume keeps, and changed files get now. Returns the changed paths."""
    changed, now = [], time.time()
    for path in Path(src).rglob("*"):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        out = Path(dst) / path.relative_to(src)
        data = path.read_bytes()
        if out.exists() and out.read_bytes() == data:
            os.utime(out, (1, 1))
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        os.utime(out, (now, now))
        changed.append(str(path.relative_to(src)))
    return changed


@app.function(image=image, volumes={"/cache": vol}, cpu=BUILD_CORES, memory=BUILD_GIB * 1024, timeout=30 * 60)
def build() -> dict:
    t0 = time.time()
    changed = sync_tree("/src/lily", "/cache/src/lily")
    digest = csrc_digest("/cache/src/lily/csrc")
    out = f"/cache/build/sm{ARCH}"
    binary = Path(f"/cache/so/sm{ARCH}/{digest}.so")       # one binary per exact csrc: never a stale pairing
    if binary.exists():
        print("csrc unchanged: no build", flush=True)
    else:
        print("changed: " + ", ".join(c for c in changed if c.startswith("csrc/")), flush=True)
        env = {**os.environ, "TORCH_CUDA_ARCH_LIST": ARCH, "LILY_CUDA_BUILD_DIR": out, "LILY_CUDA_BUILD_ONLY": "1",
               "LILY_CUDA_VERBOSE": "1", "MAX_JOBS": str(BUILD_CORES * 2)}
        subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, '/cache/src'); "
                        "from lily.layers.quantization import cuda_ops; assert cuda_ops.available(), 'build failed'"],
                       check=True, env=env)
        binary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f"{out}/lily_cuda.so", binary)
    vol.commit()
    return {"digest": digest, "seconds": time.time() - t0}


@app.function(image=image, volumes={"/cache": vol}, gpu=GPU, cpu=GPU_CORES, memory=GPU_GIB * 1024, timeout=GPU_TIMEOUT)
def measure(bench: str, pattern: str, digest: str) -> dict:
    t0 = time.time()
    binary = f"/cache/so/sm{ARCH}/{digest}.so"
    if csrc_digest("/src/lily/csrc") != digest or not os.path.exists(binary):
        raise RuntimeError("no binary built from this lily/csrc; run build first")
    sys.path[:0] = ["/src", "/src/bench"]
    import torch  # noqa: F401 - the extension needs torch's libraries loaded

    spec = importlib.util.spec_from_file_location("lily_cuda", binary)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from lily.layers.quantization import cuda_ops

    cuda_ops._module = lambda: mod
    import kbench

    res = kbench.run_bench(bench, pattern)
    vol.commit()                     # Triton cache
    return {**res, "seconds": time.time() - t0}


@app.local_entrypoint()
def main(bench: str = "glm_moe", filter: str = ""):
    print(f"{GPU}: budget ${BUDGET_USD:.2f} -> GPU timeout {GPU_TIMEOUT}s", flush=True)
    built = build.remote()
    res = measure.remote(bench, filter, built["digest"])
    sys.path.insert(0, str(ROOT / "bench"))
    import kbench

    kbench.record(res)
    build_usd = built["seconds"] * (BUILD_CORES * CORE_S_USD + BUILD_GIB * GIB_S_USD)
    gpu_usd = res["seconds"] * GPU_S_USD
    print(f"\nbuild {built['seconds']:.0f}s ~${build_usd:.4f}, GPU {res['seconds']:.0f}s ~${gpu_usd:.4f} "
          f"(+ container boot), total ~${build_usd + gpu_usd:.3f}")

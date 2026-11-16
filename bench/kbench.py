"""Kernel microbench harness: every case runs in a CUDA graph with its inputs rotated through enough copies to miss
the L2, and reports us/call, TFLOP/s, GB/s, the share of this card's measured roofline (memcpy bandwidth, cuBLAS
bf16 matmul), the error against a reference, and the change against the previous run of the same case.

A bench is a file bench/kernels/<name>.py whose `cases()` yields `Case`s:

    Case(name, setup, run, flops=..., bytes=..., ref=None)

`setup()` builds one set of inputs (a tuple), `run(*inputs)` launches the kernel under test (returning None means
"not supported here"), `ref(*inputs)` is what it must match. `bytes` is the DRAM traffic an ideal kernel needs and
`flops` its math: the roofline columns come from those two.

    uv run python bench/kbench.py glm_moe [name-filter]              # local GPU, free
    modal run scripts/modal_kbench.py --bench glm_moe                 # RTX PRO 6000 (sm_120), a few cents

Results: bench/results/kbench/<bench>-<gpu>.json (latest per case) and history.jsonl (every run).
This module imports torch only inside functions: `modal run` formats the table in a Python without torch.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

ERR_TOL = 2e-2
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "kbench")


@dataclass
class Case:
    name: str
    setup: Callable[[], tuple]
    run: Callable[..., Any]
    flops: float = 0.0
    bytes: float = 0.0
    ref: Optional[Callable[..., Any]] = None


def load_cases(bench: str, pattern: str = "") -> list:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", f"{bench}.py")
    spec = importlib.util.spec_from_file_location(f"kbench_{bench}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [c for c in mod.cases() if pattern in c.name]


def graph_time_us(call: Callable[[int], Any], copies: int, min_calls: int = 64, samples: int = 10):
    """Median us per `call(j)` (j cycles over the input copies) replayed from one CUDA graph, and the sample spread."""
    import torch

    calls = copies * max(1, math.ceil(min_calls / copies))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):          # Triton autotune / lazy allocations happen here, not in the capture
        for j in range(copies):
            call(j)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(calls):
            call(i % copies)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    graph.replay()
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    loops = max(1, math.ceil(5.0 / max(start.elapsed_time(end), 1e-3)))     # ~5 ms per sample
    times = []
    for _ in range(samples):
        start.record()
        for _ in range(loops):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (loops * calls))
    med = statistics.median(times)
    return med, (max(times) - min(times)) / med


def roofline():
    """(memcpy GB/s, cuBLAS bf16 TFLOP/s) measured on this card: the ceilings the % columns are relative to."""
    import torch

    src = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    t_copy, _ = graph_time_us(lambda j: dst.copy_(src), 1, min_calls=8, samples=5)
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    t_mm, _ = graph_time_us(lambda j: torch.matmul(a, b), 1, min_calls=4, samples=5)
    return 2 * src.numel() / t_copy / 1e3, 2 * 8192 ** 3 / t_mm / 1e6


def rel_error(out, ref) -> float:
    out, ref = out.float(), ref.float()
    return float((out - ref).abs().max() / ref.abs().max().clamp_min(1e-12))


def measure_case(case: Case, l2_bytes: int, bw: float, tflops: float) -> dict:
    import torch

    row = {"name": case.name}
    try:
        inputs = [case.setup()]
        probe = case.run(*inputs[0])
        if probe is None:
            return {**row, "us": None}
        if case.ref is not None:
            row["err"] = rel_error(probe, case.ref(*inputs[0]))
        free = torch.cuda.mem_get_info()[0]
        copies = 1
        if case.bytes:
            copies = max(1, min(32, math.ceil(3 * l2_bytes / case.bytes), int(0.3 * free // case.bytes)))
        inputs += [case.setup() for _ in range(copies - 1)]
        us, spread = graph_time_us(lambda j: case.run(*inputs[j]), copies)
    except Exception as exc:  # noqa: BLE001 - one broken case must not waste the paid run
        return {**row, "error": f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"[:200]}
    gbs, tfs = case.bytes / us / 1e3, case.flops / us / 1e6
    return {**row, "us": us, "spread": spread, "gbs": gbs, "tflops": tfs, "pct_bw": 100 * gbs / bw,
            "pct_flop": 100 * tfs / tflops, "copies": copies}


def run_bench(bench: str, pattern: str = "") -> dict:
    import torch

    props = torch.cuda.get_device_properties(0)
    bw, tflops = roofline()
    rows = []
    for case in load_cases(bench, pattern):
        rows.append(measure_case(case, props.L2_cache_size, bw, tflops))
        torch.cuda.empty_cache()
    return {"bench": bench, "gpu": props.name, "sm": f"{props.major}.{props.minor}", "bw_gbs": bw,
            "bf16_tflops": tflops, "cases": rows}


def format_table(res: dict, prev: Optional[dict] = None) -> str:
    old = {c["name"]: c for c in (prev or {}).get("cases", [])}
    width = max([len(c["name"]) for c in res["cases"]] + [4])
    lines = [f"{res['gpu']} (sm_{res['sm']}): memcpy {res['bw_gbs']:.0f} GB/s, cuBLAS bf16 {res['bf16_tflops']:.0f} TFLOP/s"
             f"   ('vs last': negative = faster)",
             f"{'case':{width}} {'us':>9} {'±%':>5} {'TFLOP/s':>8} {'GB/s':>7} {'%BW':>5} {'%FLOP':>6} {'rel err':>8} {'vs last':>8}"]
    for c in res["cases"]:
        if c.get("error"):
            lines.append(f"{c['name']:{width}} ERROR {c['error']}")
            continue
        if c.get("us") is None:
            lines.append(f"{c['name']:{width}} {'n/a':>9}")
            continue
        p = old.get(c["name"])
        delta = f"{100 * (c['us'] / p['us'] - 1):+7.1f}%" if p and p.get("us") else ""
        err = "-" if c.get("err") is None else f"{c['err']:.1e}" + ("!" if c["err"] > ERR_TOL else "")
        lines.append(f"{c['name']:{width}} {c['us']:9.2f} {100 * c['spread']:5.1f} {c['tflops']:8.2f} {c['gbs']:7.0f} "
                     f"{c['pct_bw']:5.0f} {c['pct_flop']:6.1f} {err:>8} {delta:>8}")
    return "\n".join(lines)


def record(res: dict) -> None:
    """Print the table against the previous run, then merge this run into the per-GPU latest file and the history."""
    os.makedirs(RESULTS, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", res["gpu"].lower()).strip("-")
    latest = os.path.join(RESULTS, f"{res['bench']}-{slug}.json")
    prev = json.load(open(latest)) if os.path.exists(latest) else None
    print(format_table(res, prev))
    merged = {c["name"]: c for c in (prev or {}).get("cases", [])}
    merged.update({c["name"]: c for c in res["cases"]})
    json.dump({**res, "cases": list(merged.values())}, open(latest, "w"), indent=1)
    with open(os.path.join(RESULTS, "history.jsonl"), "a") as f:
        f.write(json.dumps({**res, "time": time.time()}) + "\n")


if __name__ == "__main__":
    record(run_bench(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""))

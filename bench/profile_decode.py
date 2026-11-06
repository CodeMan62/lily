#!/usr/bin/env python
"""Per-kernel profile of TP decode: where one step's time goes, on every rank's real ModelRunner.

    python bench/profile_decode.py --model-path /models/glm53-flash-nvfp4 --tp 4 --bs 1

Prints the CUDA-graph step time, the kernels of a profiled window (rank 0) sorted by total
time with launches per step, and the latency of a hidden-sized all-reduce on its own.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import time

import torch


def _graph_ms(fn, iters: int = 200) -> float:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters // 10):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


def _table(title: str, prof, steps: int, top: int) -> None:
    rows = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    rows.sort(key=lambda e: e.self_device_time_total, reverse=True)
    busy = sum(e.self_device_time_total for e in rows) / 1e3 / steps
    print(f"\n=== {title}, kernel-busy {busy:.2f} ms, {sum(e.count for e in rows) / steps:.0f} launches ===", flush=True)
    print(f"{'ms/step':>8} {'n/step':>7}  kernel")
    for e in rows[:top]:
        print(f"{e.self_device_time_total / 1e3 / steps:8.3f} {e.count / steps:7.0f}  {e.key[:110]}", flush=True)


def _worker(rank: int, args, out) -> None:
    from lily.distributed import tensor_model_parallel_all_reduce
    from lily.engine import Engine, EngineConfig
    from lily.model_executor.forward_batch_info import ForwardBatch

    extra = {"quantize_lm_head": "nvfp4", "mamba_ssm_dtype": "bfloat16", "cuda_graph_max_bs": max(args.bs, 1)}
    extra.update(json.loads(args.server_args))
    eng = Engine(EngineConfig(model_path=args.model_path, tp_size=args.tp, rank=rank, context_length=8192,
                              max_running_requests=max(args.bs, 2) + 1, max_total_tokens=65536,
                              mem_fraction_static=0.8, server_args=extra))
    dev = eng.device
    reqs = [eng._new_seq() for _ in range(args.bs)]
    lens = []
    with torch.inference_mode():
        for i, r in enumerate(reqs):
            _, n = eng._extend(r, 0, [1000 + i + j for j in range(args.input_len)])
            lens.append(n)

        def step(tok: int = 1234):
            loc = torch.cat([eng._alloc(r, n, 1) for r, n in zip(reqs, lens)])
            for i in range(len(lens)):
                lens[i] += 1
            fb = ForwardBatch.for_decode([tok + i for i in range(args.bs)], list(lens), reqs, dev, out_cache_loc=loc)
            return eng.runner.forward(fb)

        for _ in range(5):
            step()
        torch.cuda.synchronize()
        t = time.perf_counter()
        n_steps = 50
        for i in range(n_steps):
            step(2000 + 7 * i)
        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t) / n_steps * 1e3

        from torch.profiler import ProfilerActivity, profile
        prof_steps = 5
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(prof_steps):
                step(3000 + 11 * i)
            torch.cuda.synchronize()

        if args.verify:
            from lily.speculative.graph_runner import VerifyGraphRunner

            model = eng.runner.model
            vg = VerifyGraphRunner(eng.runner, args.verify, len(model.recurrent_layers()), [5, 14, 24, 33, 42])
            n = args.verify + 1
            locs = torch.cat([eng._alloc(r, l, n) for r, l in zip(reqs, lens)])
            idx = torch.tensor(reqs, dtype=torch.int64, device=dev)
            # text-like ids: neighbouring tokens route to overlapping experts less than a repeat, more than noise
            ids = torch.randint(1000, 50000, (args.bs * n,), device=dev)
            ids = ids.to(torch.int64)
            torch.distributed.broadcast(ids, 0)

            def verify():
                return vg.replay(ids, idx, list(lens), locs)

            for _ in range(3):
                verify()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(20):
                verify()
            torch.cuda.synchronize()
            verify_ms = (time.perf_counter() - t) / 20 * 1e3
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as vprof:
                for _ in range(prof_steps):
                    verify()
                torch.cuda.synchronize()
            if rank == 0:
                _table(f"verify k={args.verify} bs={args.bs}: {verify_ms:.2f} ms", vprof, prof_steps, args.top)

        x = torch.randn(args.bs, eng.model_config.hidden_size, dtype=torch.bfloat16, device=dev)
        ar_ms = _graph_ms(lambda: tensor_model_parallel_all_reduce(x))

    from lily.distributed.distributed import broadcast_pyobj

    for name, payload in (("empty", []), ("accepted", [[1, 2, 3, 4]])):
        t = time.perf_counter()
        for _ in range(200):
            broadcast_pyobj(list(payload), rank, src=0)
        if rank == 0:
            print(f"broadcast_pyobj {name}: {(time.perf_counter() - t) / 200 * 1e6:.0f} us", flush=True)
    if rank == 0:
        _table(f"bs={args.bs} step {step_ms:.2f} ms ({1e3 / step_ms * args.bs:.1f} tok/s), all-reduce {ar_ms * 1e3:.1f} us",
               prof, prof_steps, args.top)
    out.put(rank)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--verify", type=int, default=0, help="also profile the verify graph with this many drafts")
    ap.add_argument("--server-args", default="{}", help="JSON of extra ServerArgs fields")
    args = ap.parse_args()
    print(subprocess.run("nvidia-smi topo -m", shell=True, capture_output=True, text=True).stdout, flush=True)
    n = torch.cuda.device_count()
    print("peer access:", [[torch.cuda.can_device_access_peer(i, j) for j in range(n) if j != i] for i in range(n)], flush=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, args, q)) for r in range(args.tp)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        if p.exitcode:
            raise SystemExit(f"rank failed: exit {p.exitcode}")


if __name__ == "__main__":
    main()

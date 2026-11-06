"""Decode profiler: `LILY_PROFILE_BS=1,8` (+ `LILY_PROFILE_STEPS`, default 20) profiles that many consecutive
decode steps the first time a batch of each listed size runs (TP rank 0, or every rank with
`LILY_PROFILE_ALL_RANKS=1`) and logs the kernels by GPU time per step: kernels inside CUDA-graph replays are
listed individually, NCCL kernel time includes the wait for the other ranks. `LILY_PROFILE_TRACE=<dir>` also
writes the window as a chrome trace (host phases `lily.*`, kernels, NCCL, graph launches) for Perfetto or
`bench/profile_report.py`. Never profile the measured sweep itself: CUPTI roughly halves throughput."""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


class DecodeProfiler:
    def __init__(self, sizes: List[int], steps: int, rank: int):
        self.pending, self.steps, self.rank = set(sizes), steps, rank
        self.prof = None
        self.bs = self.n = 0

    @classmethod
    def from_env(cls, tp_rank: int) -> Optional["DecodeProfiler"]:
        spec = os.environ.get("LILY_PROFILE_BS", "")
        if not spec or (tp_rank and not os.environ.get("LILY_PROFILE_ALL_RANKS")):
            return None
        return cls([int(x) for x in spec.split(",")], int(os.environ.get("LILY_PROFILE_STEPS", "20")), tp_rank)

    def step(self, bs: int) -> None:
        """Called before every decode step with its batch size."""
        if self.prof is None:
            if bs in self.pending:
                from torch.profiler import ProfilerActivity, profile

                self.pending.discard(bs)
                self.bs, self.n = bs, 0
                torch.cuda.synchronize()
                self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]).__enter__()
            return
        self.n += 1
        if bs == self.bs and self.n < self.steps:
            return
        torch.cuda.synchronize()
        self.prof.__exit__(None, None, None)
        logger.info("profile of %d decode steps at batch size %d (GPU time per step):\n%s", self.n, self.bs, self.table())
        if trace_dir := os.environ.get("LILY_PROFILE_TRACE"):
            os.makedirs(trace_dir, exist_ok=True)
            path = os.path.join(trace_dir, f"trace-bs{self.bs}-rank{self.rank}-{int(time.time())}.json")
            self.prof.export_chrome_trace(path)
            logger.info("profiler trace written to %s", path)
        self.prof = None

    def table(self, rows: int = 40) -> str:
        evs = self.prof.key_averages()
        items = sorted(((ev.self_device_time_total / self.n / 1000.0, ev.count / self.n, ev.key)
                        for ev in evs if ev.device_type.name != "CPU" and ev.self_device_time_total > 0), reverse=True)
        total = sum(t for t, _, _ in items)
        graphs = sum(ev.count for ev in evs if ev.key == "cudaGraphLaunch") / self.n
        lines = [f"{'ms/step':>8} {'%':>5} {'calls':>6}  kernel",
                 f"{total:8.3f} {100:5.1f} {'':>6}  total GPU time per step ({graphs:.1f} CUDA-graph launches per step)"]
        lines += [f"{t:8.3f} {100 * t / total:5.1f} {n:6.1f}  {name[:90]}" for t, n, name in items[:rows]]
        return "\n".join(lines)

"""Summarise a lily profiler trace (LILY_PROFILE_TRACE / `modal ... --profile-trace`): per step, wall vs GPU-busy
time and the gap the host leaves, the `lily.*` phase durations, and the top kernels. No GPU needed.

    python bench/profile_report.py results/trace-bs16-rank0-*.json
"""
import json
import sys
from collections import defaultdict


def _busy(intervals):
    """Total length of the union of (start, end) intervals: kernels on several streams overlap."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            total += (cur_e - cur_s) if cur_e is not None else 0.0
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + ((cur_e - cur_s) if cur_e is not None else 0.0)


def report(path: str, rows: int = 25) -> str:
    evs = [e for e in json.load(open(path))["traceEvents"] if e.get("ph") == "X"]
    gpu = [(e["ts"], e["ts"] + e["dur"], e["name"]) for e in evs if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    steps = sorted((e["ts"], e["ts"] + e["dur"]) for e in evs if e["name"] == "lily.forward")
    phases = defaultdict(float)
    for e in evs:
        if e["name"].startswith("lily."):
            phases[e["name"]] += e["dur"]
    n = max(len(steps), 1)
    first, last = (steps[0][0], steps[-1][1]) if steps else (min(s for s, _, _ in gpu), max(e for _, e, _ in gpu))
    wall = (last - first) / n / 1000
    busy = _busy([(s, e) for s, e, _ in gpu if first <= s <= last]) / n / 1000
    nccl = sum(e - s for s, e, name in gpu if "nccl" in name.lower()) / n / 1000
    launches = sum(1 for e in evs if e["name"] == "cudaGraphLaunch") / n
    out = [f"{path}: {n} steps, {wall:.2f} ms wall per step, GPU busy {busy:.2f} ms ({100 * busy / wall:.0f}%), "
           f"host gap {wall - busy:.2f} ms, NCCL kernels {nccl:.2f} ms, {launches:.1f} graph launches per step", "phases (ms per step):"]
    out += [f"  {v / n / 1000:8.3f}  {k}" for k, v in sorted(phases.items(), key=lambda kv: -kv[1])]
    per = defaultdict(lambda: [0.0, 0])
    for s, e, name in gpu:
        per[name][0] += e - s
        per[name][1] += 1
    out.append("top kernels (ms per step, calls per step):")
    out += [f"  {t / n / 1000:8.3f} {c / n:7.1f}  {name[:90]}" for name, (t, c) in sorted(per.items(), key=lambda kv: -kv[1][0])[:rows]]
    return "\n".join(out)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(report(p))

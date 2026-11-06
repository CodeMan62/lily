"""Fusion-candidate finder: count the GPU ops one speculative round launches, by the lily source line that
issues them (eager mode on a tiny model, so the counts are the launches a CUDA graph would replay).
This found the 487 -> 172 ops/round cut of small_fused.cu.

    python bench/op_attrib.py [batch_size] [k]      # tiny Qwen3.5 + its DFlash drafter from tests/
"""
import collections
import re
import sys
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, ".")
from tests.small_qwen35 import SmallQwen35  # noqa: E402
from tests.test_qwen35 import _scheduler  # noqa: E402
from lily.sampling.sampling_params import SamplingParams  # noqa: E402

VIEWS = {"detach", "alias", "view", "_unsafe_view", "t", "transpose", "permute", "expand", "unsqueeze", "squeeze", "slice",
         "select", "reshape", "empty", "empty_like", "empty_strided", "as_strided", "split", "unbind", "narrow", "lift_fresh",
         "_local_scalar_dense", "contiguous", "chunk", "split_with_sizes", "numel", "resize_", "set_"}


class Count(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.by_line, self.by_op = collections.Counter(), collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        op = str(func).split(".")[1]
        if op not in VIEWS and any(isinstance(a, torch.Tensor) and a.is_cuda for a in args):
            fr = next((f for f in reversed(traceback.extract_stack(limit=30)) if "/lily/lily/" in f.filename), None)
            self.by_line[f"{re.sub(r'.*/lily/lily/', '', fr.filename)}:{fr.lineno} {fr.name}" if fr else "?"] += 1
            self.by_op[op] += 1
        return out


if __name__ == "__main__":
    bs, k = (int(a) for a in (sys.argv[1:3] + ["1", "3"][len(sys.argv) - 1:]))
    tiny = SmallQwen35()
    sched = _scheduler(str(tiny.path), True, k=k, algo="DFLASH", max_reqs=bs, draft=tiny.build_dflash())
    sched.model_runner.server_args.disable_cuda_graph = True
    for i in range(bs):
        sched.add_request(f"r{i}", [5 + i, 9, 12, 7, 3, 40 + i], SamplingParams(max_new_tokens=40, ignore_eos=True))
    for _ in range(4):
        sched.step()
    with Count() as c:
        for _ in range(10):
            sched.step()
    print(f"bs={bs} k={k}: {sum(c.by_line.values()) / 10:.0f} GPU ops per round")
    for key, v in c.by_line.most_common(40):
        print(f"  {v / 10:6.1f}  {key}")
    print("by op:", ", ".join(f"{op} {v / 10:.0f}" for op, v in c.by_op.most_common(12)))

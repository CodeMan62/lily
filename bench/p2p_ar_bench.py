"""One-shot P2P all-reduce vs NCCL on the box's GPUs: identical result on every rank, close to NCCL, and the latency of
both inside a CUDA graph.    LILY_P2P_AR=1 torchrun --nproc-per-node 4 bench/p2p_ar_bench.py"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, ".")
from lily.distributed import distributed as D  # noqa: E402

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
D.init_distributed_environment(world, rank, f"{os.environ['MASTER_ADDR']}:{int(os.environ['MASTER_PORT']) + 7}", torch.device("cuda", rank))
assert D.init_p2p_allreduce(), "P2P all-reduce unavailable"
mod, bufs, _ = D.P2P
for rows in (8, 64, 256):                    # bs 1, 8, 32 verify rows
    for it in range(300):                    # parities and epochs wrap many times
        g = torch.Generator(device="cuda").manual_seed(it * 31 + rows * 7 + rank)
        x = torch.randn(rows, 4096, device="cuda", generator=g).bfloat16()
        ref = x.clone()
        dist.all_reduce(ref)
        mod.p2p_allreduce(x, bufs, rank, world)
        allx = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(allx, x)
        assert all(torch.equal(allx[0], a) for a in allx), "ranks disagree"
        assert (x.float() - ref.float()).abs().max() <= 0.05 * ref.float().abs().max(), "far from NCCL"


def us(fn, reps=97):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(5):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / 5 / reps * 1000


for rows in (8, 64):
    x = torch.randn(rows, 4096, device="cuda").bfloat16()
    t_nccl = us(lambda: dist.all_reduce(x))
    t_p2p = us(lambda: mod.p2p_allreduce(x, bufs, rank, world))
    if rank == 0:
        print(f"[{rows} x 4096 bf16] NCCL {t_nccl:.1f} us   one-shot P2P {t_p2p:.1f} us", flush=True)
if rank == 0:
    print("correctness: OK (identical on every rank, within bf16 of NCCL, 900 calls)", flush=True)
dist.barrier()
dist.destroy_process_group()

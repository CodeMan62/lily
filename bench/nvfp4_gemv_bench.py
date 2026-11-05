#!/usr/bin/env python
"""Decode GEMV microbench: lily CUDA NVFP4 GEMV vs the Triton dequant GEMM vs cuBLAS bf16.

    uv run python bench/nvfp4_gemv_bench.py [--m 1,2,4,8]

Shapes are the Qwen3.8-27B linears. Reports ms and achieved GB/s of *weight*
bytes (what a memory-bound decode step is limited by) next to a memcpy reference
for the card. Everything is timed inside a CUDA graph so launch overhead is out.
"""
import argparse, time, torch
from lily.layers.quantization import cuda_ops
from lily.layers.quantization.kernels import FAMILY_NVFP4, weight_only_matmul

SHAPES = [(5120, 5120, "o_proj / out_proj"), (10240, 5120, "in_proj_qkv"), (6144, 5120, "in_proj_z"),
          (14336, 5120, "attn qkv"), (34816, 5120, "gate_up"), (5120, 17408, "down_proj")]


def timeit(fn, iters=50):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): g.replay()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--m", default="1,2,4,8,16"); args = ap.parse_args()
    p = torch.cuda.get_device_properties(0); print(f"{p.name}  SMs={p.multi_processor_count}")
    buf = torch.empty(256 << 20, dtype=torch.uint8, device="cuda"); dst = torch.empty_like(buf)
    t = timeit(lambda: dst.copy_(buf)); print(f"memcpy reference: {2 * buf.numel() / t / 1e9:.0f} GB/s\n")
    assert cuda_ops.available(), "CUDA extension failed to build (set LILY_CUDA_VERBOSE=1)"
    for m in (int(v) for v in args.m.split(",")):
        print(f"== M={m} ==")
        print(f"{'layer':18s} {'N':>6s} {'K':>6s} | {'cuda gemv':>22s} | {'cuda mma':>22s} | {'triton':>22s} | {'bf16 cublas':>22s}")
        for n, k, name in SHAPES:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
            s = torch.randint(0x30, 0x40, (n, k // 16), dtype=torch.uint8, device="cuda")
            gs = torch.ones(n, dtype=torch.float32, device="cuda")
            wb = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
            qbytes, bbytes = w.numel() + s.numel(), wb.numel() * 2
            fmt = lambda t_, b_: f"{t_ * 1e3:7.3f} ms {b_ / t_ / 1e9:6.0f} GB/s" if t_ else f"{'n/a':>22s}"
            tg = timeit(lambda: cuda_ops._module().nvfp4_gemv(x, w, s, gs)) if m <= 8 else None
            tm = timeit(lambda: cuda_ops.nvfp4_gemm_tc(x, w, s, gs)) if m <= cuda_ops.GEMM_TC_MAX_M else None
            tt = timeit(lambda: weight_only_matmul(x, w, s, FAMILY_NVFP4, 0, gs))
            tb = timeit(lambda: torch.nn.functional.linear(x, wb))
            print(f"{name:18s} {n:6d} {k:6d} | {fmt(tg, qbytes)} | {fmt(tm, qbytes)} | {fmt(tt, qbytes)} | {fmt(tb, bbytes)}")
            del w, s, wb
        print()


if __name__ == "__main__":
    main()

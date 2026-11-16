"""Old vs fused paths at GLM-5.3-Flash per-card shapes (TP=4, 8 verify rows), each in a CUDA graph with enough weight
copies to miss the L2, so one card's numbers stand for the round.    python bench/fusion_bench.py"""
import sys

import torch

sys.path.insert(0, ".")
from lily.layers.quantization import cuda_ops  # noqa: E402

T, H, dev = 8, 4096, "cuda"
COPIES = 6 if torch.cuda.get_device_properties(0).L2_cache_size > 32 << 20 else 2


def us(fn, reps=40):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(5):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / 5 / reps * 1000


def rnd(*shape, scale=0.02, dtype=torch.bfloat16):
    return (torch.randn(*shape, device=dev) * scale).to(dtype)


i = [0]


def nxt(n):
    i[0] += 1
    return i[0] % n


rows = []
# 1. hyper-connection: expand + weights + collapse, then the next GEMM (KDA in_proj), vs the fused kernel + GEMM
streams, block = rnd(T, 4, H, scale=1), rnd(T, H, scale=1)
post, comb = torch.rand(T, 4, device=dev), torch.softmax(torch.randn(T, 4, 4, device=dev), -1)
fns, base, scale, nw = [rnd(24, 4 * H) for _ in range(COPIES)], rnd(24), rnd(3, scale=1), rnd(H, scale=1)
gemm = [rnd(6416, H) for _ in range(COPIES)]
x8 = rnd(T, H, scale=1)


def hc_old():
    j = nxt(COPIES)
    s2 = cuda_ops.hc_expand(block, post, comb, streams)
    _, _, _, h = cuda_ops.hc_collapse(s2, fns[j], base, scale, nw, 20, 1e-6, 1e-6, 1e-5)
    cuda_ops.bf16_linear(h, gemm[j])


def hc_new():
    j = nxt(COPIES)
    _, _, _, h = cuda_ops.hc_expand_collapse(block, post, comb, streams, fns[j], base, scale, nw, 20, 1e-6, 1e-6, 1e-5)
    cuda_ops.bf16_linear(h, gemm[j])


rows.append(("mHC expand+collapse, then a 6416x4096 GEMM", us(hc_old), us(hc_new), 84))
del gemm
# 2. router logits, 288 experts, fp32 weights
rw = [torch.randn(288, H, device=dev) * 0.02 for _ in range(32)]
rows.append(("router logits (cast + SGEMM + split-K vs one kernel)", us(lambda: torch.nn.functional.linear(x8.float(), rw[nxt(32)])),
             us(lambda: cuda_ops.router_logits(x8, rw[nxt(32)])), 42))
del rw
# 3. KDA projections: qkv + small + g_b + f_b vs in_proj + block-diagonal gate GEMM
qkv, small = [rnd(6144, H) for _ in range(COPIES)], [rnd(272, H) for _ in range(COPIES)]
gb, fb = rnd(2048, 128), rnd(2048, 128)
inp, gate = [torch.cat([q, s_]) for q, s_ in zip(qkv, small)], rnd(4096, 256)


def kda_old():
    j = nxt(COPIES)
    cuda_ops.bf16_linear(x8, qkv[j])
    f_a, g_a, _ = cuda_ops.bf16_linear(x8, small[j]).split([128, 128, 16], dim=-1)
    cuda_ops.bf16_linear(g_a.contiguous(), gb)
    cuda_ops.bf16_linear(f_a.contiguous(), fb)


def kda_new():
    y = cuda_ops.bf16_linear(x8, inp[nxt(COPIES)])
    cuda_ops.bf16_linear(y[:, 6144:6400], gate)


rows.append(("KDA projections (4 GEMMs vs 2)", us(kda_old), us(kda_new), 34))
print(f"{'path':55} {'old us':>8} {'new us':>8} {'x/round':>8} {'saved ms/round':>15}")
total = 0.0
for name, o, n, per in rows:
    total += (o - n) * per / 1000
    print(f"{name:55} {o:8.1f} {n:8.1f} {per:8d} {(o - n) * per / 1000:15.3f}")
print(f"{'total':55} {'':8} {'':8} {'':8} {total:15.3f}")

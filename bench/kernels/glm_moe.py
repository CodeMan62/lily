"""GLM-5.3-Flash MoE layer on one TP=4 rank, kernel by kernel: router logits, routing, the grouped NVFP4 gate_up
GEMM, clamped SwiGLU, the grouped down GEMM, combine. T = verify rows (8 = bs 1 x k+1 8, 128 = bs 16), routing from
random router logits (top-8 of 288). The GEMMs' `bytes` count only the active experts' weights, so %BW is how close
each comes to streaming them at memcpy speed.    modal run scripts/modal_kbench.py --bench glm_moe"""
import torch

from kbench import Case
from lily.layers.quantization import cuda_ops
from lily.layers.quantization.base import nvfp4_dequantize

H, I, E, TOPK, TP = 4096, 2048, 288, 8, 4
IR = I // TP            # per-rank intermediate slice
LIMIT, SCALING = 10.0, 2.5


def nvfp4_weights(n: int, k: int):
    w = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device="cuda")
    s = torch.randint(0x30, 0x40, (E, n, k // 16), dtype=torch.uint8, device="cuda")
    return w, s, torch.ones(E * n, dtype=torch.float32, device="cuda")


def expert_bytes(n: int, k: int) -> int:
    return n * k // 2 + n * k // 16 + 4 * n


def grouped_ref(x, w, s, gs, active, offsets, tokens, R):
    """Per-expert dequantized matmul over each active expert's rows."""
    n = w.shape[1]
    out = torch.zeros(R, n, dtype=torch.bfloat16, device="cuda")
    for e in active[active >= 0].tolist():
        lo, hi = int(offsets[e]), int(offsets[e + 1])
        wd = nvfp4_dequantize(w[e], s[e].view(torch.float8_e4m3fn), gs.view(E, -1)[e])
        out[lo:hi] = x[tokens[lo:hi].long()] @ wd.t()
    return out


def layer_cases(T: int, gen: torch.Generator):
    """The six kernels of one MoE layer at T rows (own scope: the setups must not see another T's tensors)."""
    R = T * TOPK
    x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", generator=gen)
    router_w = torch.randn(E, H, dtype=torch.float32, device="cuda", generator=gen) * 0.02
    bias = torch.zeros(E, dtype=torch.float32, device="cuda")
    logits = cuda_ops.router_logits(x, router_w)
    weights, active, offsets, tokens, inv = cuda_ops.moe_route(logits, bias, TOPK, SCALING, True, 0, E)
    active = active[:R]
    n_act = int((active >= 0).sum())
    rows = torch.arange(R, dtype=torch.int32, device="cuda")
    act = torch.randn(R, IR, dtype=torch.bfloat16, device="cuda", generator=gen)
    tag = f"T={T:<3} ({n_act} experts)"

    yield Case(f"router_logits  {tag}", lambda x=x: (x, torch.randn(E, H, device="cuda") * 0.02), cuda_ops.router_logits,
               2 * T * E * H, 4 * E * H + 2 * T * H, lambda x, w: x.float() @ w.t())
    yield Case(f"moe_route      {tag}", lambda l=logits: (l, bias),
               lambda l, b: cuda_ops.moe_route(l, b, TOPK, SCALING, True, 0, E)[0], 0, 4 * T * E)
    yield Case(f"moe_gemm gu    {tag}", lambda: (x, *nvfp4_weights(2 * IR, H), active, offsets, tokens, R),
               cuda_ops.nvfp4_moe_gemm, 2 * R * 2 * IR * H, n_act * expert_bytes(2 * IR, H) + 2 * R * (H + 2 * IR),
               grouped_ref)
    yield Case(f"moe_swiglu     {tag}", lambda: (torch.randn(R, 2 * IR, dtype=torch.bfloat16, device="cuda"),),
               lambda g: cuda_ops.moe_swiglu(g, LIMIT), 0, 2 * R * 3 * IR)
    yield Case(f"moe_gemm down  {tag}", lambda: (act, *nvfp4_weights(H, IR), active, offsets, rows, R),
               cuda_ops.nvfp4_moe_gemm, 2 * R * H * IR, n_act * expert_bytes(H, IR) + 2 * R * (IR + H), grouped_ref)
    yield Case(f"moe_combine    {tag}", lambda: (torch.randn(R, H, dtype=torch.bfloat16, device="cuda"), inv, weights,
                                                 torch.randn(T, H, dtype=torch.bfloat16, device="cuda")),
               cuda_ops.moe_combine, 0, 2 * (R + 2 * T) * H)


def cases():
    gen = torch.Generator(device="cuda").manual_seed(0)
    for T in (8, 128):
        yield from layer_cases(T, gen)

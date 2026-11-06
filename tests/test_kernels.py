"""lily's CUDA kernels against torch references where the model-level tests cannot reach: NVFP4
GEMV / tensor-core GEMM over odd shapes, and the MLA decode kernel over long contexts (dense and
with visible-position lists, several chunks). The fused GDN / KDA steps and the grouped MoE GEMM
are covered by the Qwen3.5 and GLM-5.3-Flash parity tests against transformers."""

import math

import pytest
import torch
import torch.nn.functional as F

from lily.layers.quantization import cuda_ops
from lily.layers.quantization.base import nvfp4_dequantize
from lily.mem_cache.memory_pool import ReqToTokenPool

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and cuda_ops.available()), reason="needs the CUDA extension")


def _random_nvfp4(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=g)
    scale = torch.randint(0x01, 0x7E, (n, k // 16), dtype=torch.uint8, device="cuda", generator=g)   # incl. e4m3 subnormals
    gs = torch.rand(n, device="cuda", generator=g) * 500 + 50
    return packed, scale.view(torch.float8_e4m3fn), gs


@pytest.mark.parametrize("m", [1, 4, 5, 16, 17])
@pytest.mark.parametrize("n,k", [(64, 1024), (40, 5120), (24, 1056)])
def test_nvfp4_matmul_matches_dequantized_reference(m, n, k):
    """M <= 4 takes the GEMV, 5..16 the tensor-core GEMM, above that Triton (None from the router)."""
    packed, scale, gs = _random_nvfp4(n, k, seed=m * 1000 + n + k)
    x = (torch.randn(m, k, device="cuda") * 0.5).bfloat16()
    ref = F.linear(x.float(), nvfp4_dequantize(packed, scale, gs).float())
    out = cuda_ops.nvfp4_gemv(x, packed, scale.view(torch.uint8), gs)
    if out is None:    # > 16 rows, or a K the tensor-core tiles cannot cover: Triton takes it
        assert m > cuda_ops.GEMM_TC_MAX_M or (m > cuda_ops.GEMV_MAX_M and not cuda_ops._module().nvfp4_gemm_tc_supported(m, k))
        return
    tol = ref.abs().max().item() * 2 ** -6 + 1e-2
    assert out.dtype == torch.bfloat16 and torch.allclose(out.float(), ref, atol=tol, rtol=2 ** -6), (out.float() - ref).abs().max()



@pytest.mark.parametrize("D,H", [(32, 3), (512, 16)])
@pytest.mark.parametrize("sparse", [False, True])
def test_mla_decode_matches_torch(D, H, sparse):
    gen = torch.Generator().manual_seed(D + H + int(sparse))
    lens = [1, 300, 1000]
    B, n_slots = len(lens), sum(lens) + 1
    pool = ReqToTokenPool(B + 1, max(lens) + 8, "cuda")
    perm, used = torch.randperm(n_slots - 1, generator=gen) + 1, 0
    for b, L in enumerate(lens):
        pool.write((b + 1, slice(0, L)), perm[used:used + L].to(torch.int32).cuda())
        used += L
    latent = (torch.randn(n_slots, D, generator=gen) * 0.7).bfloat16().cuda()
    q_lat = (torch.randn(B, H, D, generator=gen) * 0.3).bfloat16().cuda()
    visible = [list(range(L)) for L in lens]
    vis_idx = vis_cnt = None
    if sparse:
        visible = [sorted(torch.randperm(L, generator=gen)[:min(L, 700)].tolist()) for L in lens]
        vis_idx = torch.full((B, 703), -1, dtype=torch.int32)
        for b, v in enumerate(visible):
            vis_idx[b, :len(v)] = torch.tensor(v, dtype=torch.int32)
        vis_idx, vis_cnt = vis_idx.cuda(), torch.tensor([len(v) for v in visible], dtype=torch.int32, device="cuda")
    got = cuda_ops.mla_decode(q_lat, latent, pool.req_to_token, torch.arange(1, B + 1, device="cuda"),
                              torch.tensor(lens, device="cuda"), vis_idx, vis_cnt, D ** -0.5,
                              chunks=math.ceil((703 if sparse else max(lens)) / 64), chunk=64)
    for b in range(B):
        c = latent[pool.req_to_token[b + 1, visible[b]].long()].float()
        want = torch.softmax(q_lat[b].float() @ c.t() * D ** -0.5, dim=-1) @ c
        torch.testing.assert_close(got[b].float(), want, atol=2e-2, rtol=2e-2)


# ---- the small fused glue kernels (small_fused.cu) ---------------------------------------------------

def _eager_rmsnorm(x, w, eps):
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


@pytest.mark.parametrize("H", [128, 512, 2048, 4096])
def test_rmsnorm_matches_torch(H):
    gen = torch.Generator().manual_seed(H)
    x = (torch.randn(37, H, generator=gen) * 3).bfloat16().cuda()
    w = (torch.rand(H, generator=gen) + 0.5).bfloat16().cuda()
    torch.testing.assert_close(cuda_ops.rmsnorm(x, w, 1e-5), _eager_rmsnorm(x, w, 1e-5), atol=0, rtol=0)


def test_dyn_conv2_matches_torch():
    from lily.models.dflash import GroupedDynamicCausalConv

    gen = torch.Generator().manual_seed(3)
    H, group, B, L = 256, 16, 3, 8
    conv = GroupedDynamicCausalConv(H, 2, group).cuda()
    conv.base_kernel.copy_(torch.randn(2, 2, H, generator=gen) * 0.5)
    conv.kernel_projection.weight.copy_(torch.randn(4 * H // group, H, generator=gen) * 0.05)
    x = torch.randn(B, L, H, generator=gen).bfloat16().cuda()
    projected = conv.kernel_projection(x)
    for which in (0, 1):
        got = cuda_ops.dyn_conv2(x, projected, which * 2 * (H // group), conv.base_kernel[which], group)
        dyn = projected.view(B, L, 2, 2, H // group)[:, :, which]
        blocks = x.view(B, L, H // group, group)
        out = torch.zeros_like(blocks)
        for off in range(2):
            vals = blocks if off == 0 else F.pad(blocks[:, :-off], (0, 0, 0, 0, off, 0))
            out = out + conv.base_kernel[which, off].view(1, 1, H // group, group) * vals
            out = torch.addcmul(out, dyn[:, :, off].unsqueeze(-1), vals)
        torch.testing.assert_close(got, out.view(B, L, H), atol=0, rtol=0)


def test_rope_neox_matches_torch():
    from lily.layers.rotary import RotaryEmbedding

    gen = torch.Generator().manual_seed(5)
    rope = RotaryEmbedding(128, 128, 4096, 10000.0, is_neox_style=True, dtype=torch.bfloat16).cuda()
    pos = torch.randint(0, 4096, (24,), generator=gen).cuda()
    q = torch.randn(24, 8 * 128, generator=gen).bfloat16().cuda()
    k = torch.randn(24, 2 * 128, generator=gen).bfloat16().cuda()
    got = rope(pos, q, k)
    cs = rope.cos_sin_cache[pos]                      # the eager path, by hand
    cos, sin = (t.repeat(1, 2).unsqueeze(1) for t in cs.chunk(2, dim=-1))

    def rot(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    qh, kh = q.view(24, 8, 128), k.view(24, 2, 128)
    want = ((qh * cos + rot(qh) * sin).view(24, -1), (kh * cos + rot(kh) * sin).view(24, -1))
    torch.testing.assert_close(got[0], want[0], atol=0, rtol=0)
    torch.testing.assert_close(got[1], want[1], atol=0, rtol=0)


def test_select_greedy_matches_torch():
    gen = torch.Generator().manual_seed(7)
    B, S, K, R, V = 4, 8, 16, 256, 500
    unary = (torch.randn(B, S, K, generator=gen) * 4).cuda()
    cand = torch.stack([torch.stack([torch.randperm(V, generator=gen)[:K] for _ in range(S)]) for _ in range(B)]).cuda()
    proj = torch.randn(B, S, R, generator=gen).cuda()
    anchor = torch.randint(0, V, (B,), generator=gen).cuda()
    pred = (torch.randn(V, R, generator=gen) * 0.3).bfloat16().cuda()
    succ = (torch.randn(V, R, generator=gen) * 0.3).bfloat16().cuda()
    path, q = cuda_ops.select_greedy(unary, cand, proj, anchor, pred, succ)
    prev, want = anchor, []
    for pos in range(S):
        gate = F.embedding(prev, pred).float() * proj[:, pos]
        scores = unary[:, pos] + torch.einsum("br,bkr->bk", gate, F.embedding(cand[:, pos], succ).float())
        top = scores.topk(2, dim=-1).values
        assert (top[:, 0] - top[:, 1]).min() > 1e-3          # no near-ties in this seed: the paths must agree
        idx = scores.argmax(-1)
        prev = cand[:, pos].gather(1, idx.unsqueeze(1)).squeeze(1)
        want.append(prev)
        assert torch.equal(q[:, pos].argmax(-1), idx) and torch.equal(q[:, pos].sum(-1), torch.ones(B, device="cuda"))
    assert torch.equal(path, torch.stack(want, dim=1))

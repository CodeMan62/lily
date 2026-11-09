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


# ---- small_fused.cu: bit-exact against the torch formulations ------------------------------------------

def test_small_fused_matches_torch():
    from lily.layers.rotary import RotaryEmbedding

    gen = torch.Generator().manual_seed(3)
    x = (torch.randn(37, 512, generator=gen) * 3).bfloat16().cuda()
    w = (torch.rand(512, generator=gen) + 0.5).bfloat16().cuda()
    want = w * (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)).bfloat16()
    assert torch.equal(cuda_ops.rmsnorm(x, w, 1e-5), want)
    r = torch.randn(37, 512, generator=gen).bfloat16().cuda()
    s = x + r                                   # zero-centered (Qwen3.5) with the fused residual add
    want = (s.float() * torch.rsqrt(s.float().pow(2).mean(-1, keepdim=True) + 1e-5) * (1.0 + w.float())).bfloat16()
    got = cuda_ops.rmsnorm(x, w, 1e-5, r, zero_centered=True)
    assert torch.equal(got[0], want) and torch.equal(got[1], s)
    rope = RotaryEmbedding(128, 128, 4096, 10000.0, dtype=torch.bfloat16).cuda()
    pos = torch.randint(0, 4096, (24,), generator=gen).cuda()
    q, k = (torch.randn(24, h * 128, generator=gen).bfloat16().cuda() for h in (8, 2))
    got = rope(pos, q, k)
    cos, sin = (t.repeat(1, 2).unsqueeze(1) for t in rope.cos_sin_cache[pos].chunk(2, dim=-1))
    rot = lambda t: torch.cat((-t[..., 64:], t[..., :64]), dim=-1)  # noqa: E731
    for g, t in zip(got, (q.view(24, 8, 128), k.view(24, 2, 128))):
        assert torch.equal(g, (t * cos + rot(t) * sin).view(24, -1))

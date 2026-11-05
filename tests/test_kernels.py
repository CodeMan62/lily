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

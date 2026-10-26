"""Phase 2: Triton extend/decode kernels vs the torch-native reference over the same paged pool."""

from types import SimpleNamespace

import pytest
import torch

from lily.layers.attention.torch_native_backend import TorchNativeAttnBackend
from lily.layers.attention.triton_backend import TritonAttnBackend
from lily.layers.radix_attention import RadixAttention
from lily.mem_cache.allocator import TokenToKVPoolAllocator
from lily.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from lily.model_executor.forward_batch_info import ForwardBatch

DEVICE = "cuda"
POOL = 4096
MAX_CTX = 1024
LAYERS = 2

# (q heads, kv heads, head_dim) — MHA, GQA, odd GQA (SmolLM2), big head (Qwen3.8-27B attention)
SHAPES = [(8, 8, 64), (16, 8, 128), (9, 3, 64), (24, 4, 256)]
DTYPES = [torch.bfloat16, torch.float32]


def _runner(h, hkv, d, dtype):
    req_pool = ReqToTokenPool(8, MAX_CTX, DEVICE)
    kv_pool = MHATokenToKVPool(POOL, 1, dtype, hkv, d, LAYERS, DEVICE)
    alloc = TokenToKVPoolAllocator(POOL, dtype, DEVICE, kv_pool)
    cfg = SimpleNamespace(num_attention_heads=h, head_dim=d)
    return SimpleNamespace(device=torch.device(DEVICE), dtype=dtype, model_config=cfg,
                           req_to_token_pool=req_pool, token_to_kv_pool=kv_pool,
                           token_to_kv_pool_allocator=alloc)


def _tol(dtype):
    return dict(atol=1e-2, rtol=1e-2) if dtype == torch.bfloat16 else dict(atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("h,hkv,d", SHAPES)
def test_extend_matches_reference(h, hkv, d, dtype):
    torch.manual_seed(0)
    r = _runner(h, hkv, d, dtype)
    native, trit = TorchNativeAttnBackend(r), TritonAttnBackend(r)
    layer = RadixAttention(h, d, d**-0.5, hkv, layer_id=1)

    # 5 requests: mixed prefix lengths incl. 0, extend lengths incl. 1 and > BLOCK_M
    prefix = [0, 17, 130, 5, 300]
    ext = [1, 70, 33, 200, 129]
    bs = len(prefix)
    reqs = r.req_to_token_pool.alloc(bs)
    # prefix KV already in the pool
    for i in range(bs):
        if prefix[i]:
            loc = r.token_to_kv_pool_allocator.alloc(prefix[i])
            r.req_to_token_pool.write((reqs[i], slice(0, prefix[i])), loc.to(torch.int32))
            r.token_to_kv_pool.set_kv_buffer(1, loc, torch.randn(prefix[i], hkv, d, device=DEVICE, dtype=dtype),
                                             torch.randn(prefix[i], hkv, d, device=DEVICE, dtype=dtype))
    # new tokens
    total = sum(ext)
    out_loc = r.token_to_kv_pool_allocator.alloc(total)
    pt = 0
    for i in range(bs):
        r.req_to_token_pool.write((reqs[i], slice(prefix[i], prefix[i] + ext[i])), out_loc[pt:pt + ext[i]].to(torch.int32))
        pt += ext[i]
    ids = [[0] * e for e in ext]
    fb = ForwardBatch.for_extend(ids, prefix, reqs, DEVICE, out_cache_loc=out_loc)
    q = torch.randn(total, h * d, device=DEVICE, dtype=dtype)
    k = torch.randn(total, hkv * d, device=DEVICE, dtype=dtype)
    v = torch.randn(total, hkv * d, device=DEVICE, dtype=dtype)

    native.init_forward_metadata(fb)
    ref = native.forward_extend(q, k, v, layer, fb)
    trit.init_forward_metadata(fb)
    out = trit.forward_extend(q, k, v, layer, fb)
    torch.testing.assert_close(out, ref, **_tol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("h,hkv,d", SHAPES)
def test_decode_matches_reference(h, hkv, d, dtype):
    torch.manual_seed(0)
    r = _runner(h, hkv, d, dtype)
    native, trit = TorchNativeAttnBackend(r), TritonAttnBackend(r)
    layer = RadixAttention(h, d, d**-0.5, hkv, layer_id=0)

    seq_lens = [1, 8, 65, 500, 1000, 3]  # after appending the new token
    bs = len(seq_lens)
    reqs = r.req_to_token_pool.alloc(bs)
    for i in range(bs):
        n = seq_lens[i] - 1
        if n:
            loc = r.token_to_kv_pool_allocator.alloc(n)
            r.req_to_token_pool.write((reqs[i], slice(0, n)), loc.to(torch.int32))
            r.token_to_kv_pool.set_kv_buffer(0, loc, torch.randn(n, hkv, d, device=DEVICE, dtype=dtype),
                                             torch.randn(n, hkv, d, device=DEVICE, dtype=dtype))
    out_loc = r.token_to_kv_pool_allocator.alloc(bs)
    for i in range(bs):
        r.req_to_token_pool.write((reqs[i], seq_lens[i] - 1), out_loc[i].to(torch.int32))
    fb = ForwardBatch.for_decode([0] * bs, seq_lens, reqs, DEVICE, out_cache_loc=out_loc)
    q = torch.randn(bs, h * d, device=DEVICE, dtype=dtype)
    k = torch.randn(bs, hkv * d, device=DEVICE, dtype=dtype)
    v = torch.randn(bs, hkv * d, device=DEVICE, dtype=dtype)

    native.init_forward_metadata(fb)
    ref = native.forward_decode(q, k, v, layer, fb)
    trit.init_forward_metadata(fb)
    out = trit.forward_decode(q, k, v, layer, fb)
    torch.testing.assert_close(out, ref, **_tol(dtype))


def test_allocator_and_req_pool_roundtrip():
    torch.manual_seed(1)
    r = _runner(4, 4, 64, torch.bfloat16)
    alloc, req_pool = r.token_to_kv_pool_allocator, r.req_to_token_pool
    assert alloc.available_size() == POOL and req_pool.available_size() == 8
    live = []
    for _ in range(200):
        if live and torch.rand(()) < 0.4:
            alloc.free(live.pop(torch.randint(len(live), ()).item()))
        else:
            n = int(torch.randint(1, 64, ()))
            got = alloc.alloc(n)
            if got is None:
                assert alloc.available_size() < n
            else:
                assert got.numel() == n and (got > 0).all()
                live.append(got)
    seen = torch.cat(live) if live else torch.empty(0, dtype=torch.int64, device=DEVICE)
    assert seen.unique().numel() == seen.numel(), "double allocation"
    for t in live:
        alloc.free(t)
    assert alloc.available_size() == POOL
    assert alloc.free_pages.unique().numel() == POOL

    slots = req_pool.alloc(8)
    assert req_pool.alloc(1) is None
    req_pool.free(slots)
    assert req_pool.available_size() == 8

"""The serving engine on small public checkpoints (SmolLM2-135M, Qwen3-0.6B): model parity with
transformers, the Triton attention backend, CUDA graphs and overlap scheduling, continuous
batching, prefix caching, sampling, the OpenAI-compatible HTTP server, and config plumbing.
One test per subsystem."""

import random
import threading
import time
import uuid
from types import SimpleNamespace

import openai
import pytest
import torch
import torch.nn.functional as F
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

from lily.engine import Engine, EngineConfig
from lily.entrypoints.http_server import create_app
from lily.entrypoints.launch import make_endpoints
from lily.layers.attention.torch_native_backend import TorchNativeAttnBackend
from lily.layers.attention.triton_backend import TritonAttnBackend
from lily.layers.radix_attention import RadixAttention
from lily.layers.sampler import Sampler
from lily.mem_cache.allocator import TokenToKVPoolAllocator
from lily.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from lily.model_executor.forward_batch_info import ForwardBatch
from lily.models.model_config import ModelConfig
from lily.sampling.sampling_batch_info import SamplingBatchInfo
from lily.sampling.sampling_params import SamplingParams
from lily.scheduler.detokenizer import run_detokenizer_process
from lily.scheduler.scheduler import Scheduler, run_scheduler_process
from lily.scheduler.tokenizer import TokenizerFrontend
from lily.server_args import ServerArgs
from tests.models import SMALL_LLAMA, SMALL_QWEN3
from tests.util import free_cuda

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
PROMPT = ("The history of natural language processing generally started in the 1950s, although work can be found "
          "from earlier periods. In 1950, Alan Turing published an article titled Computing Machinery and Intelligence "
          "which proposed what is now called the Turing test as a criterion of intelligence.")


def _scheduler(model=SMALL_LLAMA, **kw) -> Scheduler:
    sa = dict(model_path=model, dtype="float32", attention_backend="triton", context_length=2048, max_running_requests=16,
              max_total_tokens=8192, disable_radix_cache=True)
    sa.update(kw)
    return Scheduler(ModelConfig.from_pretrained(model, dtype="float32", context_length=2048), ServerArgs(**sa))


def _run(sch, prompts, n, **params):
    for i, p in enumerate(prompts):
        sch.add_request(f"r{i}", p, SamplingParams(max_new_tokens=n, ignore_eos=True, **params))
    while sch.step():
        pass
    out = [list(sch._requests[f"r{i}"].result.output_ids) for i in range(len(prompts))]
    free_cuda()
    return out


def _prompts(n, lo, hi, seed, vocab=49152):
    rng = random.Random(seed)
    return [[rng.randrange(1000, vocab - 1000) for _ in range(rng.randint(lo, hi))] for _ in range(n)]


@pytest.mark.parametrize("model_path", [SMALL_LLAMA, SMALL_QWEN3])
def test_parity_with_transformers(model_path):
    """fp32: all prefill logits close and 60 greedy tokens identical; KV slots balance."""
    ids = AutoTokenizer.from_pretrained(model_path).encode(PROMPT)[:48]
    with torch.inference_mode():
        ref = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, attn_implementation="eager").cuda().eval()
        ref_logits = ref(torch.tensor([ids], device="cuda")).logits[0].float().cpu()
        ref_gen = ref.generate(torch.tensor([ids], device="cuda"), max_new_tokens=60, do_sample=False)[0, len(ids):].tolist()
        del ref
    free_cuda()
    eng = Engine(EngineConfig(model_path, dtype="float32", context_length=1024, max_running_requests=1, max_total_tokens=2048))
    torch.testing.assert_close(eng.all_prefill_logits(ids).cpu(), ref_logits, atol=1e-2, rtol=1e-3)
    kv = eng.kv_available()
    assert eng.generate_greedy(ids, 60) == ref_gen
    assert eng.kv_available() == kv, "KV slots leaked"
    del eng
    free_cuda()



def test_triton_attention_matches_torch_native():
    """Extend (mixed prefixes incl. 0, lengths incl. 1 and > BLOCK_M) and decode over one paged pool."""
    torch.manual_seed(0)
    h, hkv, d, dtype, dev = 16, 8, 128, torch.bfloat16, "cuda"
    req_pool, kv_pool = ReqToTokenPool(16, 1024, dev), MHATokenToKVPool(4096, 1, dtype, hkv, d, 2, dev)
    r = SimpleNamespace(device=torch.device(dev), dtype=dtype, model_config=SimpleNamespace(num_attention_heads=h, head_dim=d),
                        num_attention_heads=h, num_kv_heads=hkv, req_to_token_pool=req_pool, token_to_kv_pool=kv_pool,
                        token_to_kv_pool_allocator=TokenToKVPoolAllocator(4096, dtype, dev, kv_pool))
    native, trit = TorchNativeAttnBackend(r), TritonAttnBackend(r)
    for layer_id, prefix, ext in ((1, [0, 17, 130, 5, 300], [1, 70, 33, 200, 129]), (0, [0, 7, 64, 499, 999, 2], [1] * 6)):
        layer = RadixAttention(h, d, d ** -0.5, hkv, layer_id=layer_id)
        reqs = req_pool.alloc(len(prefix))
        for i, p in enumerate(prefix):
            if p:
                loc = r.token_to_kv_pool_allocator.alloc(p)
                req_pool.write((reqs[i], slice(0, p)), loc.to(torch.int32))
                kv_pool.set_kv_buffer(layer_id, loc, torch.randn(p, hkv, d, device=dev, dtype=dtype), torch.randn(p, hkv, d, device=dev, dtype=dtype))
        total = sum(ext)
        out_loc, pt = r.token_to_kv_pool_allocator.alloc(total), 0
        for i, e in enumerate(ext):
            req_pool.write((reqs[i], slice(prefix[i], prefix[i] + e)), out_loc[pt:pt + e].to(torch.int32))
            pt += e
        decode = all(e == 1 for e in ext)
        fb = ForwardBatch.for_decode([0] * len(ext), [p + 1 for p in prefix], reqs, dev, out_cache_loc=out_loc) if decode \
            else ForwardBatch.for_extend([[0] * e for e in ext], prefix, reqs, dev, out_cache_loc=out_loc)
        q, k, v = (torch.randn(total, n * d, device=dev, dtype=dtype) for n in (h, hkv, hkv))
        outs = []
        for backend in (native, trit):
            backend.init_forward_metadata(fb)
            outs.append(backend.forward_decode(q, k, v, layer, fb) if decode else backend.forward_extend(q, k, v, layer, fb))
        torch.testing.assert_close(outs[1], outs[0], atol=1e-2, rtol=1e-2)


def test_batching_graphs_overlap_and_chunking_do_not_change_output():
    """64 random prompts: batched == single, with CUDA graphs == eager, with overlap == without,
    chunked prefill == unchunked; KV slots balance (also after retraction in a tiny pool)."""
    prompts = _prompts(64, 4, 32, seed=3)
    single = []
    sch = _scheduler(max_running_requests=1, disable_cuda_graph=True)
    for p in prompts:
        single += _run(sch, [p], 16)
    sch = _scheduler(disable_cuda_graph=True, disable_overlap_schedule=True, max_running_requests=64)
    kv0 = sch.token_to_kv_pool_allocator.available_size()
    assert _run(sch, prompts, 16) == single and sch.token_to_kv_pool_allocator.available_size() == kv0
    sch = _scheduler(max_running_requests=64)      # graphs + overlap on
    assert sch.model_runner.cuda_graph_runner is not None
    assert _run(sch, prompts, 16) == single
    sch = _scheduler(max_total_tokens=160, max_running_requests=8, disable_cuda_graph=True)   # forces retraction
    kv0 = sch.token_to_kv_pool_allocator.available_size()
    assert _run(sch, prompts[:8], 8) == [s[:8] for s in single[:8]] and sch.token_to_kv_pool_allocator.available_size() == kv0
    long = _prompts(1, 150, 150, seed=5)
    chunked = _scheduler(max_running_requests=1, chunked_prefill_size=64)
    assert _run(chunked, long, 8) == _run(_scheduler(max_running_requests=1), long, 8) and chunked.prefill_tokens_computed == 150


def test_radix_cache_reuses_shared_prefix_and_survives_eviction():
    rng = random.Random(0)
    system = [rng.randrange(1000, 48000) for _ in range(48)]
    prompts = [system + [rng.randrange(1000, 48000) for _ in range(6)] for _ in range(32)]
    sch = _scheduler(max_running_requests=32, disable_radix_cache=False)
    _run(sch, prompts[:1], 6)
    warm, sch.prefill_tokens_computed = sch.prefill_tokens_computed, 0
    _run(sch, prompts, 6)
    assert sch.prefill_tokens_computed <= 32 * 6 + 16 and warm <= len(system) + 6 + 16     # only the unique tails
    prompts2 = _prompts(16, 10, 40, seed=11)
    evicted = _run(_scheduler(max_total_tokens=384, max_running_requests=8, disable_radix_cache=False, chunked_prefill_size=64), prompts2, 6)
    assert evicted == _run(_scheduler(max_running_requests=1), prompts2, 6)


def test_sampler_distribution_penalties_seeds_and_logprobs():
    torch.manual_seed(1)
    V, dev = 12, "cuda"
    info = lambda params, ids=None: SamplingBatchInfo(params, ids or [[] for _ in params], [torch.Generator(device=dev).manual_seed(1234 + i) for i in range(len(params))], device=dev)  # noqa: E731
    logits = torch.randn(V, device=dev) * 2.0
    sampler, counts = Sampler(), torch.zeros(V, device=dev)
    batch = logits.unsqueeze(0).expand(100, V).contiguous()
    for _ in range(400):
        counts += torch.bincount(sampler(batch, info([SamplingParams(temperature=1.0)] * 100)).next_token_ids, minlength=V).float()
    expected, observed = torch.softmax(logits, -1), counts / counts.sum()
    assert (((observed - expected) ** 2) / expected).sum().item() < 31.3      # chi-square, df 11, p 0.001
    greedy_logits = torch.randn(4, V, device=dev)
    assert torch.equal(sampler(greedy_logits, info([SamplingParams(temperature=0.0)] * 4)).next_token_ids, greedy_logits.argmax(-1))
    # penalties vs transformers' processor
    logits, ids = torch.randn(1, V, device=dev), torch.tensor([[1, 3, 3, 7, 11]], device=dev)
    ours = logits.clone()
    Sampler()._apply_penalties(ours, info([SamplingParams(temperature=0.0, repetition_penalty=1.7)], ids=[ids[0].tolist()]))
    torch.testing.assert_close(ours, RepetitionPenaltyLogitsProcessor(1.7)(ids, logits.clone()))
    # seeded reproducibility and logprobs through the scheduler
    prompt = _prompts(1, 6, 6, seed=21)[0]
    a, b = _run(_scheduler(max_running_requests=2), [prompt, prompt], 12, temperature=0.8, top_k=20, top_p=0.95, seed=1234)
    assert a == b
    sch = _scheduler(max_running_requests=1)
    sch.add_request("c", prompt, SamplingParams(max_new_tokens=4, temperature=0.0, logprobs=True, top_logprobs=3))
    while sch.step():
        pass
    res = sch._requests["c"].result
    assert len(res.logprobs) == 4 and all(x <= 0 for x in res.logprobs) and all(len(t) == 3 for t in res.top_logprobs)



def test_server_args_and_model_config():
    sa = ServerArgs.parse(["--model-path", "x", "--speculative-algorithm", "nextn"])
    assert (sa.speculative_algorithm, sa.speculative_num_steps, sa.speculative_num_draft_tokens) == ("MTP", 3, 4)
    for bad in (dict(speculative_algorithm="eagle3"), dict(quantize_lm_head="int8"), dict(mamba_ssm_dtype="fp16"), dict(quantization="awq"),
                dict(speculative_algorithm="dflash")):
        with pytest.raises(ValueError):
            ServerArgs(model_path="x", **bad)
    cfg = ModelConfig.from_pretrained(SMALL_QWEN3)
    assert "Qwen3ForCausalLM" in cfg.architectures and cfg.eos_token_ids and cfg.head_dim > 0


@pytest.mark.timeout(300)
def test_http_server_streaming_equals_non_streaming():
    """The OpenAI-compatible server over real scheduler / detokenizer threads."""
    tokenizer = AutoTokenizer.from_pretrained(SMALL_LLAMA)
    cfg = ModelConfig.from_pretrained(SMALL_LLAMA, dtype="float32", context_length=1024)
    sa = ServerArgs(model_path=SMALL_LLAMA, dtype="float32", attention_backend="triton", context_length=1024, max_running_requests=8,
                    max_total_tokens=4096, served_model_name="lily-test")
    endpoints, stop = make_endpoints(prefix=f"/tmp/lily_test_{uuid.uuid4().hex[:8]}"), threading.Event()
    workers = [threading.Thread(target=run_scheduler_process, args=(cfg, sa, endpoints.req, endpoints.token, stop), daemon=True),
               threading.Thread(target=run_detokenizer_process, args=(tokenizer, endpoints.token, endpoints.text, stop), daemon=True)]
    for w in workers:
        w.start()
    port = 38000 + (uuid.uuid4().int % 2000)
    server = uvicorn.Server(uvicorn.Config(create_app(TokenizerFrontend(tokenizer, sa, endpoints.req, endpoints.text), sa), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    deadline = time.time() + 60
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    try:
        client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="none", timeout=60.0)
        full = client.completions.create(model="lily-test", prompt="The quick brown fox", max_tokens=16, temperature=0.0)
        assert full.usage.completion_tokens == 16
        streamed = "".join(c.choices[0].text or "" for c in client.completions.create(model="lily-test", prompt="The quick brown fox", max_tokens=16, temperature=0.0, stream=True))
        assert streamed == full.choices[0].text
        msgs = [{"role": "user", "content": "Say hello."}]
        chat = client.chat.completions.create(model="lily-test", messages=msgs, max_tokens=16, temperature=0.0).choices[0].message
        assert chat.role == "assistant" and chat.content
        assert "".join(c.choices[0].delta.content or "" for c in client.chat.completions.create(model="lily-test", messages=msgs, max_tokens=16, temperature=0.0, stream=True)) == chat.content
    finally:
        server.should_exit = True
        th.join(timeout=10)
        stop.set()
        for w in workers:
            w.join(timeout=5)
        free_cuda()

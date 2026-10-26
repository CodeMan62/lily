"""Phase 7: CUDA graphs and overlap scheduling.

Three acceptance tests from the roadmap:
  1. bucketed decode graphs replay bit-exactly (greedy) vs eager;
  2. the backend advertises graph support correctly and ModelRunner honours
     `--disable-cuda-graph`;
  3. outputs are unchanged with graphs and overlap on (and overlap really runs).
"""

import random

import torch

from lily.models.model_config import ModelConfig
from lily.layers.attention.torch_native_backend import TorchNativeAttnBackend
from lily.layers.attention.triton_backend import TritonAttnBackend
from lily.scheduler.scheduler import Scheduler
from lily.model_executor.forward_batch_info import ForwardBatch
from lily.model_executor.model_runner import ModelRunner
from lily.sampling.sampling_params import SamplingParams
from lily.server_args import ServerArgs
from tests.models import SMALL_LLAMA
from tests.util import free_cuda

MODEL = SMALL_LLAMA


def _args(**kw) -> ServerArgs:
    base = dict(
        model_path=MODEL, dtype="float32", attention_backend="triton", context_length=1024,
        max_running_requests=16, max_total_tokens=8192, disable_radix_cache=True,
        chunked_prefill_size=1024, max_prefill_tokens=4096,
    )
    base.update(kw)
    return ServerArgs(**base)


def _seed_decode(model_runner, bs, seq_len=7):
    cfg = model_runner.model_config
    pool = model_runner.token_to_kv_pool
    alloc = model_runner.token_to_kv_pool_allocator
    rtp = model_runner.req_to_token_pool
    reqs = rtp.alloc(bs)
    for i in range(bs):
        loc = alloc.alloc(seq_len - 1)
        rtp.write((reqs[i], slice(0, seq_len - 1)), loc.to(torch.int32))
        pool.set_kv_buffer(
            0, loc,
            torch.randn(seq_len - 1, cfg.num_key_value_heads, cfg.head_dim, device="cuda"),
            torch.randn(seq_len - 1, cfg.num_key_value_heads, cfg.head_dim, device="cuda"),
        )
    out_loc = alloc.alloc(bs)
    for i in range(bs):
        rtp.write((reqs[i], seq_len - 1), out_loc[i].to(torch.int32))
    toks = [101 + 7 * i for i in range(bs)]
    return reqs, out_loc, toks, seq_len


def _release_decode(model_runner, reqs, seq_len):
    alloc = model_runner.token_to_kv_pool_allocator
    rtp = model_runner.req_to_token_pool
    for i in range(len(reqs)):
        alloc.free(rtp.req_to_token[reqs[i], :seq_len].long())
    rtp.free(list(reqs))


def test_cuda_graph_replay_matches_eager():
    cfg = ModelConfig.from_pretrained(MODEL, dtype="float32", context_length=1024)
    runner = ModelRunner(cfg, _args())
    assert runner.cuda_graph_runner is not None
    assert runner.cuda_graph_runner.capture_bs == [1, 2, 4, 8, 16]

    for bs in (1, 2, 3, 5, 8):
        reqs, out_loc, toks, seq_len = _seed_decode(runner, bs)
        eager = ForwardBatch.for_decode(toks, [seq_len] * bs, reqs, "cuda", out_cache_loc=out_loc)
        runner._attach(eager)
        logits_eager = runner.model(eager.input_ids, eager.positions, eager).next_token_logits.clone()

        graph_batch = ForwardBatch.for_decode(toks, [seq_len] * bs, reqs, "cuda", out_cache_loc=out_loc)
        logits_graph = runner.cuda_graph_runner.replay(graph_batch).next_token_logits.clone()
        torch.cuda.synchronize()

        assert logits_graph.shape == logits_eager.shape
        assert bool((logits_graph.argmax(-1) == logits_eager.argmax(-1)).all()), f"bs={bs} argmax differs"
        _release_decode(runner, reqs, seq_len)

    runner = None
    free_cuda()


def test_backend_support_and_disable_flag():
    # Class-level capability: the reference backend opts out, triton opts in.
    assert TritonAttnBackend.supports_cuda_graph(TritonAttnBackend.__new__(TritonAttnBackend)) is True
    assert TorchNativeAttnBackend.supports_cuda_graph(TorchNativeAttnBackend.__new__(TorchNativeAttnBackend)) is False

    cfg = ModelConfig.from_pretrained(MODEL, dtype="float32", context_length=1024)

    off = ModelRunner(cfg, _args(disable_cuda_graph=True))
    assert off.cuda_graph_runner is None
    off = None
    free_cuda()

    native = ModelRunner(cfg, _args(attention_backend="torch_native"))
    assert native.cuda_graph_runner is None, "torch-native backend must not be captured"
    native = None
    free_cuda()


def _drive(scheduler, prompts, max_new_tokens, overlap, tracker):
    done = {}
    for i, p in enumerate(prompts):
        scheduler.add_request(
            f"r{i}", p, SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens)
        )
    if overlap:
        prev = None
        while True:
            prev = scheduler.overlap_step(prev)
            for rid, item in scheduler._requests.items():
                if item.result and item.result.finished and rid not in done:
                    done[rid] = list(item.result.output_ids)
            if not scheduler.waiting_queue and (
                scheduler.running_batch is None or not scheduler.running_batch.reqs
            ) and prev is None:
                break
    else:
        while scheduler.waiting_queue or (
            scheduler.running_batch is not None and scheduler.running_batch.reqs
        ):
            if not scheduler.step():
                break
        for rid, item in scheduler._requests.items():
            if item.result is not None:
                done[rid] = list(item.result.output_ids)
    return [done[f"r{i}"] for i in range(len(prompts))]


def test_graphs_and_overlap_outputs_unchanged():
    rng = random.Random(7)
    prompts = [[rng.randrange(1000, 48000) for _ in range(rng.randint(6, 40))] for _ in range(12)]
    cfg = ModelConfig.from_pretrained(MODEL, dtype="float32", context_length=1024)

    ref_sch = Scheduler(cfg, _args(disable_cuda_graph=True, disable_overlap_schedule=True))
    reference = _drive(ref_sch, prompts, 16, overlap=False, tracker=None)
    ref_sch = None
    free_cuda()

    graph_sch = Scheduler(cfg, _args(disable_overlap_schedule=True))
    assert graph_sch.model_runner.cuda_graph_runner is not None
    with_graphs = _drive(graph_sch, prompts, 16, overlap=False, tracker=None)
    graph_sch = None
    free_cuda()

    # Overlap on: count how many batches actually took the overlap path.
    ov_sch = Scheduler(cfg, _args())
    calls = {"n": 0}
    original = ov_sch._stage_overlap_result
    ov_sch._stage_overlap_result = lambda batch, result: (calls.__setitem__("n", calls["n"] + 1), original(batch, result))[1]
    overlapped = _drive(ov_sch, prompts, 16, overlap=True, tracker=calls)
    ov_sch = None
    free_cuda()

    assert calls["n"] > 0, "overlap scheduling never engaged"
    assert with_graphs == reference, "CUDA graphs changed the output"
    assert overlapped == reference, "overlap scheduling changed the output"

"""Phase 3: scheduler and continuous batching.

Three acceptance tests from the roadmap:
  1. 64 concurrent prompts of random lengths match each prompt run alone;
  2. `available_size()` returns to its initial value after all requests finish,
     including after retraction;
  3. a long prompt with chunked prefill matches the unchunked run.

Pure unit checks of `SchedulePolicy` / `PrefillAdder` are folded into test 1's
module-level fixtures (they need no GPU).
"""

import random

import torch

from lily.models.model_config import ModelConfig
from lily.scheduler.schedule_batch import Req
from lily.scheduler.scheduler import PrefillAdder, SchedulePolicy, Scheduler
from lily.sampling.sampling_params import SamplingParams
from lily.server_args import ServerArgs
from tests.models import SMALL_LLAMA
from tests.util import free_cuda

MODEL = SMALL_LLAMA
VOCAB = 49152
MAX_TOTAL = 8192


def new_scheduler(max_total=MAX_TOTAL, max_reqs=1, disable_radix=True,
                  chunked_prefill_size=8192, schedule_policy="fcfs") -> Scheduler:
    cfg = ModelConfig.from_pretrained(MODEL, dtype="float32", context_length=2048)
    sa = ServerArgs(
        model_path=MODEL, dtype="float32", attention_backend="triton", context_length=2048,
        max_running_requests=max_reqs, max_total_tokens=max_total,
        chunked_prefill_size=chunked_prefill_size, disable_radix_cache=disable_radix,
        schedule_policy=schedule_policy,
    )
    return Scheduler(cfg, sa)


def run_batched(scheduler, prompts, max_new_tokens, ignore_eos=True):
    sp = SamplingParams(max_new_tokens=max_new_tokens, ignore_eos=ignore_eos)
    for i, p in enumerate(prompts):
        scheduler.add_request(f"r{i}", p, sp)
    while scheduler.step(): pass
    return [list(scheduler._requests[f"r{i}"].result.output_ids) for i in range(len(prompts))]


def run_single(scheduler, prompts, max_new_tokens, ignore_eos=True):
    sp = SamplingParams(max_new_tokens=max_new_tokens, ignore_eos=ignore_eos)
    out = []
    for i, p in enumerate(prompts):
        scheduler.add_request(f"s{i}", p, sp)
        while scheduler.step(): pass
        out.append(list(scheduler._requests[f"s{i}"].result.output_ids))
    return out


def random_prompts(n, lo, hi, seed):
    rng = random.Random(seed)
    return [[rng.randrange(1000, VOCAB - 1000) for _ in range(rng.randint(lo, hi))] for _ in range(n)]


def _req(rid, prefix_len, extend_len):
    r = Req(rid=rid, origin_input_ids=list(range(extend_len)),
            sampling_params=SamplingParams(), eos_token_ids=[])
    r.prefix_indices = list(range(prefix_len))
    r.extend_input_len = extend_len
    return r


def test_concurrent_batching_matches_single_run():
    # Policy/adder unit checks first (no GPU).
    reqs = [_req("a", 3, 2), _req("b", 10, 2), _req("c", 0, 2)]
    assert [r.rid for r in SchedulePolicy("fcfs").sort(reqs)] == ["a", "b", "c"]
    assert [r.rid for r in SchedulePolicy("lpm").sort(reqs)] == ["b", "a", "c"]
    selected = PrefillAdder(token_budget=10, max_running_requests=10, chunked_prefill_size=10).add(
        [_req(f"r{i}", 0, 4) for i in range(3)]
    )
    assert selected[0].extend_input_len == 4 and not selected[0].is_chunked
    assert selected[2].extend_input_len == 2 and selected[2].is_chunked

    prompts = random_prompts(64, 4, 32, seed=3)
    sch = new_scheduler(max_reqs=64, disable_radix=True)
    before = sch.token_to_kv_pool_allocator.available_size()
    batched = run_batched(sch, prompts, 16)
    assert sch.token_to_kv_pool_allocator.available_size() == before, "KV slots leaked"
    sch = None
    free_cuda()

    sch = new_scheduler(max_reqs=1, disable_radix=True)
    single = run_single(sch, prompts, 16)
    sch = None
    free_cuda()
    assert batched == single


def test_no_kv_leak_after_retraction():
    prompts = random_prompts(8, 12, 28, seed=11)
    sch = new_scheduler(max_total=160, max_reqs=8, disable_radix=True)
    before = sch.token_to_kv_pool_allocator.available_size()
    batched = run_batched(sch, prompts, 8)
    assert sch.token_to_kv_pool_allocator.available_size() == before, "KV slots leaked after retraction"
    sch = None
    free_cuda()

    sch = new_scheduler(max_reqs=1, disable_radix=True)
    single = run_single(sch, prompts, 8)
    sch = None
    free_cuda()
    assert batched == single


def test_chunked_prefill_matches_unchunked():
    prompts = random_prompts(1, 150, 150, seed=5)
    sch = new_scheduler(max_reqs=1, disable_radix=True, chunked_prefill_size=64)
    chunked = run_batched(sch, prompts, 8)
    assert sch.prefill_tokens_computed == 150
    sch = None
    free_cuda()

    sch = new_scheduler(max_reqs=1, disable_radix=True, chunked_prefill_size=8192)
    unchunked = run_batched(sch, prompts, 8)
    sch = None
    free_cuda()
    assert chunked == unchunked

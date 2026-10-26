"""Phase 5: sampler correctness.

Three acceptance tests:
  1. distribution matches the softmax of the logits (chi-square) and the greedy
     fast path is argmax;
  2. penalties match HF transformers' `LogitsProcessor` implementations;
  3. a seeded request is reproducible end-to-end through the scheduler and
     logprobs are returned when requested.
"""

import random

import torch
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

from lily.models.model_config import ModelConfig
from lily.layers.sampler import Sampler
from lily.scheduler.scheduler import Scheduler
from lily.sampling.sampling_batch_info import SamplingBatchInfo
from lily.sampling.sampling_params import SamplingParams
from lily.server_args import ServerArgs
from tests.models import SMALL_LLAMA
from tests.util import free_cuda

DEVICE = "cuda"
VOCAB = 12


def _info(params, token_ids=None):
    return SamplingBatchInfo(
        params,
        token_ids or [[] for _ in params],
        [torch.Generator(device=DEVICE).manual_seed(1234 + i) for i in range(len(params))],
        device=DEVICE,
    )


def test_distribution_matches_softmax_and_greedy_is_argmax():
    torch.manual_seed(1)
    logits = torch.randn(VOCAB, device=DEVICE) * 2.0
    expected = torch.softmax(logits, dim=-1)
    sampler = Sampler()

    batch, n = 100, 40000
    counts = torch.zeros(VOCAB, device=DEVICE)
    info = _info([SamplingParams(temperature=1.0) for _ in range(batch)])
    batch_logits = logits.unsqueeze(0).expand(batch, VOCAB).contiguous()
    for _ in range(n // batch):
        counts += torch.bincount(sampler(batch_logits, info).next_token_ids, minlength=VOCAB).float()

    observed = counts / counts.sum()
    chi2 = (((observed - expected) ** 2) / expected).sum().item()
    assert chi2 < 31.3, f"chi-square {chi2:.2f} too large"  # df=11, p=0.001 -> 31.3

    greedy_logits = torch.randn(4, VOCAB, device=DEVICE)
    out = sampler(greedy_logits, _info([SamplingParams(temperature=0.0) for _ in range(4)]))
    assert torch.equal(out.next_token_ids, torch.argmax(greedy_logits, dim=-1))
    assert out.next_token_logprobs is None


def _hf_frequency_presence_reference(logits, input_ids, freq, pres):
    """transformers 5.x dropped these processors; inline their formula."""
    counts = torch.zeros_like(logits).scatter_add_(
        1, input_ids, torch.ones_like(input_ids, dtype=logits.dtype)
    )
    ref = logits - counts * freq
    return ref - (counts > 0).to(logits.dtype) * pres


def test_penalties_match_hf():
    torch.manual_seed(4)
    logits = torch.randn(1, VOCAB, device=DEVICE)
    input_ids = torch.tensor([[1, 3, 3, 7, 11]], device=DEVICE)
    penalty = 1.7

    ref = RepetitionPenaltyLogitsProcessor(penalty)(input_ids, logits.clone())
    info = _info([SamplingParams(temperature=0.0, repetition_penalty=penalty)],
                 token_ids=[input_ids[0].tolist()])
    ours = logits.clone()
    Sampler()._apply_penalties(ours, info)
    torch.testing.assert_close(ours, ref)

    torch.manual_seed(5)
    logits = torch.randn(1, VOCAB, device=DEVICE)
    input_ids = torch.tensor([[1, 3, 3, 7, 7, 7]], device=DEVICE)
    freq, pres = 0.6, 0.9
    ref = _hf_frequency_presence_reference(logits.clone(), input_ids, freq, pres)
    info = _info([SamplingParams(temperature=0.0, frequency_penalty=freq, presence_penalty=pres)],
                 token_ids=[input_ids[0].tolist()])
    ours = logits.clone()
    Sampler()._apply_penalties(ours, info)
    torch.testing.assert_close(ours, ref)


def test_seeded_reproducible_and_logprobs():
    rng = random.Random(21)
    prompt = [rng.randrange(1000, 49152 - 1000) for _ in range(6)]

    cfg = ModelConfig.from_pretrained(SMALL_LLAMA, dtype="float32", context_length=2048)
    sa = ServerArgs(model_path=SMALL_LLAMA, dtype="float32", attention_backend="triton",
                    context_length=2048, max_running_requests=2, max_total_tokens=8192,
                    disable_radix_cache=True)
    sch = Scheduler(cfg, sa)
    sp = SamplingParams(max_new_tokens=12, temperature=0.8, top_k=20, top_p=0.95, seed=1234)
    sch.add_request("a", prompt, sp)
    sch.add_request("b", prompt, sp)
    while sch.step(): pass
    a = list(sch._requests["a"].result.output_ids)
    b = list(sch._requests["b"].result.output_ids)
    assert a == b, "same seed must reproduce the same tokens"
    sch = None
    free_cuda()

    sa2 = ServerArgs(model_path=SMALL_LLAMA, dtype="float32", attention_backend="triton",
                     context_length=2048, max_running_requests=1, max_total_tokens=8192,
                     disable_radix_cache=True)
    sch = Scheduler(cfg, sa2)
    sch.add_request("c", prompt, SamplingParams(max_new_tokens=4, temperature=0.0,
                                                logprobs=True, top_logprobs=3))
    while sch.step(): pass
    req = sch._requests["c"].result
    assert len(req.logprobs) == 4 and all(x <= 0 for x in req.logprobs)
    assert len(req.top_logprobs) == 4 and all(len(t) == 3 for t in req.top_logprobs)
    sch = None
    free_cuda()

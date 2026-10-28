"""End-to-end parity: the lily engine vs transformers' reference Qwen3_5ForCausalLM
on a tiny random-weights hybrid checkpoint (see tests/small_qwen35.py).

Covers: full-prefill logits, gated delta rule layers through conv + chunked scan,
full-attention layers, TP-independent greedy decode, and chunked continuation.
"""

import os

import pytest
import torch
import torch.nn.functional as F

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from tests.small_qwen35 import SmallQwen35

pytestmark = pytest.mark.gpu

CHECK = SmallQwen35()
IDS = list(range(10, 10 + 24))  # arbitrary token ids inside the vocab

PROMPT_TAIL_LEN = 4
DECODE_STEPS = 12


def _ref_logits(model, prompt_ids):
    with torch.inference_mode():
        out = model(torch.tensor([prompt_ids]))
    return out.logits[0].float()  # [L, vocab]


@pytest.fixture(scope="module")
def _free_after_module():
    yield
    CHECK.ref = None
    import gc

    from tests.util import free_cuda

    gc.collect()
    free_cuda()


def test_prefill_logits_match_reference():
    from lily.engine import Engine, EngineConfig

    ref = _ref_logits(CHECK.hf, IDS)

    eng = Engine(EngineConfig(str(CHECK.path), dtype="bfloat16", max_total_tokens=1024,
                              max_running_requests=1))
    try:
        got = eng.all_prefill_logits(IDS).float()
    finally:
        del eng

    sim = F.cosine_similarity(got, ref, dim=-1)
    assert sim.min() > 0.999, f"min cosine {sim.min():.5f}"
    assert (got - ref).abs().max() < 0.05


def test_greedy_decode_matches_reference():
    from lily.engine import Engine, EngineConfig

    eng = Engine(EngineConfig(str(CHECK.path), dtype="bfloat16", max_total_tokens=1024,
                              max_running_requests=1))
    try:
        got = eng.generate_greedy(IDS, DECODE_STEPS, ignore_eos=True)
    finally:
        del eng

    ref = CHECK.hf
    with torch.inference_mode():
        out = ref.generate(torch.tensor([IDS]), attention_mask=torch.ones(1, len(IDS), dtype=torch.long),
                           max_new_tokens=DECODE_STEPS, do_sample=False)

    ref_tokens = out[0][len(IDS):].tolist()
    assert got == ref_tokens, f"greedy tokens diverged:\nlily {got}\nHF   {ref_tokens}"


def test_chunked_prefill_continuation_matches():
    """Second chunk must pick up conv + recurrent state from the first."""
    from lily.engine import Engine, EngineConfig

    split = len(IDS) - PROMPT_TAIL_LEN

    # Continuation prefill: first chunk, then the tail.
    eng = Engine(EngineConfig(str(CHECK.path), dtype="bfloat16", max_total_tokens=1024,
                              max_running_requests=1))
    try:
        req = eng._new_seq()
        eng._extend(req, 0, IDS[:split])
        logits, _ = eng._extend(req, split, IDS[split:])
        tail = logits.next_token_logits[0].float()
    finally:
        del eng

    # Single-shot prefill on the same engine type.
    eng = Engine(EngineConfig(str(CHECK.path), dtype="bfloat16", max_total_tokens=1024,
                              max_running_requests=1))
    try:
        full = eng.prefill_logits(IDS).float()
    finally:
        del eng

    ref = _ref_logits(CHECK.hf, IDS)[-1]
    sim_full = F.cosine_similarity(full, ref, dim=0).item()
    sim_split = F.cosine_similarity(tail, ref, dim=0).item()
    assert sim_full > 0.999
    assert sim_split > 0.995, f"continuation chunk cosine {sim_split:.5f} (full {sim_full:.5f})"

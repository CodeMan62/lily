"""Phase 1 acceptance: lula's model matches HF transformers.

fp32: all-position logits close, and 100 greedy tokens identical.
bf16: argmax agreement over the prompt positions (bf16 numerics differ by kernel order,
so exact logit equality is not a meaningful oracle there).
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lula.engine import Engine, EngineConfig
from tests.models import SMALL_LLAMA, SMALL_QWEN3
from tests.util import free_cuda

PROMPT = (
    "The history of natural language processing generally started in the 1950s, although work can be "
    "found from earlier periods. In 1950, Alan Turing published an article titled Computing Machinery "
    "and Intelligence which proposed what is now called the Turing test as a criterion of intelligence. "
    "The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian "
    "sentences into English. The authors claimed that within three or five years, machine translation "
    "would be a solved problem."
)
N_PROMPT = 64
N_GEN = 100


@torch.inference_mode()
def hf_reference(model_path: str, dtype: torch.dtype, prompt_ids, n_gen: int):
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, attn_implementation="eager").cuda().eval()
    ids = torch.tensor([prompt_ids], device="cuda")
    out = model(ids, use_cache=True)
    all_logits = out.logits[0].float().cpu()
    past = out.past_key_values
    gen = []
    nxt = int(torch.argmax(out.logits[0, -1]))
    for _ in range(n_gen):
        gen.append(nxt)
        out = model(torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(torch.argmax(out.logits[0, -1]))
    del model, past, out
    free_cuda()
    return all_logits, gen


@pytest.mark.parametrize("backend", ["torch_native", "triton"])
@pytest.mark.parametrize("model_path", [SMALL_LLAMA, SMALL_QWEN3])
def test_fp32_parity_and_greedy(model_path, backend):
    tok = AutoTokenizer.from_pretrained(model_path)
    prompt_ids = tok.encode(PROMPT)[:N_PROMPT]
    assert len(prompt_ids) == N_PROMPT

    ref_logits, ref_gen = hf_reference(model_path, torch.float32, prompt_ids, N_GEN)

    eng = Engine(EngineConfig(model_path, dtype="float32", attention_backend=backend, context_length=1024, max_running_requests=1, max_total_tokens=2048))
    ours_logits = eng.all_prefill_logits(prompt_ids).cpu()
    torch.testing.assert_close(ours_logits, ref_logits, atol=1e-2, rtol=1e-3)

    kv_before = eng.kv_available()
    ours_gen = eng.generate_greedy(prompt_ids, N_GEN, ignore_eos=True)
    assert eng.kv_available() == kv_before, "KV slots leaked"
    assert ours_gen == ref_gen, f"first divergence at {next(i for i,(a,b) in enumerate(zip(ours_gen, ref_gen)) if a!=b)}"
    del eng
    free_cuda()


@pytest.mark.parametrize("backend", ["torch_native", "triton"])
@pytest.mark.parametrize("model_path", [SMALL_LLAMA, SMALL_QWEN3])
def test_bf16_argmax_agreement(model_path, backend):
    tok = AutoTokenizer.from_pretrained(model_path)
    prompt_ids = tok.encode(PROMPT)[:N_PROMPT]
    ref_logits, _ = hf_reference(model_path, torch.bfloat16, prompt_ids, 0)

    eng = Engine(EngineConfig(model_path, dtype="bfloat16", attention_backend=backend, context_length=1024, max_running_requests=1, max_total_tokens=2048))
    ours_logits = eng.all_prefill_logits(prompt_ids).cpu()
    agree = (ours_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
    max_diff = (ours_logits - ref_logits).abs().max().item()
    assert agree >= 0.95, f"argmax agreement {agree:.3f}, max |diff| {max_diff:.3f}"

    text = eng.generate_text("Write one sentence about the sea.", max_new_tokens=24)
    assert len(text.strip()) > 0
    del eng
    free_cuda()

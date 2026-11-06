"""Qwen3.5/3.8 hybrid (Gated DeltaNet + full attention) on the tiny random checkpoint of
tests/small_qwen35.py: parity with transformers, the NVFP4 checkpoint round trip through the
exporter and the merged `in_proj`, speculative decoding with the MTP head and the DFlash2 drafter
(exact up to near-ties, KV accounting, oracle drafter, sampling, penalties), and two-rank TP."""

import json
import os
import shutil
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from lily.engine import Engine, EngineConfig
from lily.layers.quantization.base import nvfp4_dequantize
from lily.models.model_config import ModelConfig
from lily.sampling.sampling_params import SamplingParams
from lily.scheduler.scheduler import Scheduler
from lily.server_args import ServerArgs
from tests.small_qwen35 import SmallQwen35
from tests.util import TIE_GAP, assert_matches_up_to_tie_flip, assert_tp2_matches_tp1, free_cuda, greedy_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDS = list(range(10, 34))
PROMPTS = [IDS, list(range(40, 52)), [5, 17, 29, 41, 53, 61, 3, 9]]
NEW = 20
ALGOS = ["MTP", "DFLASH"]


@pytest.fixture(scope="module")
def tiny():
    return SmallQwen35()


@pytest.fixture(scope="module")
def references(tiny):
    return [greedy_reference(str(tiny.path), p, NEW) for p in PROMPTS]


def _scheduler(path, spec, k=3, algo="MTP", max_reqs=4, draft=None):
    if spec and algo == "DFLASH" and draft is None:
        draft = SmallQwen35(build=False).build_dflash()
    sa = ServerArgs(model_path=path, dtype="bfloat16", attention_backend="triton", context_length=512, max_running_requests=max_reqs,
                    max_total_tokens=2048, disable_radix_cache=True, speculative_algorithm=algo if spec else None,
                    speculative_num_steps=k if spec else None, speculative_draft_model_path=draft)
    return Scheduler(ModelConfig.from_pretrained(path, dtype="bfloat16", context_length=512), sa)


def _run(sched, prompts, sp):
    for i, p in enumerate(prompts):
        sched.add_request(f"r{i}", p, sp)
    while sched.step():
        pass
    return [list(sched._requests[f"r{i}"].result.output_ids) for i in range(len(prompts))]


def test_engine_matches_transformers(tiny):
    with torch.inference_mode():
        want = tiny.hf(torch.tensor([IDS])).logits[0].float()
        ref_gen = tiny.hf.generate(torch.tensor([IDS]), attention_mask=torch.ones(1, len(IDS), dtype=torch.long),
                                   max_new_tokens=12, do_sample=False)[0][len(IDS):].tolist()
    eng = Engine(EngineConfig(str(tiny.path), dtype="bfloat16", max_total_tokens=1024, max_running_requests=1))
    got = eng.all_prefill_logits(IDS).float().cpu()
    assert F.cosine_similarity(got, want, dim=-1).min() > 0.999 and (got - want).abs().max() < 0.05
    assert eng.generate_greedy(IDS, 12, ignore_eos=True) == ref_gen
    del eng
    free_cuda()


def test_nvfp4_checkpoint_round_trip(tiny, tmp_path):
    """The exporter's compressed-tensors checkpoint (merged in_proj segments, scales, quantized lm_head)
    gives the same outputs as a bf16 checkpoint built from the dequantized weights; online lm_head
    quantization of that bf16 copy equals the checkpoint's head."""
    nv, deq = str(tmp_path / "nvfp4"), str(tmp_path / "dequant")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "quantize_nvfp4.py"), str(tiny.path), nv, "--include-lm-head"], check=True, cwd=ROOT)
    os.makedirs(deq)
    tensors = {}
    with safe_open(os.path.join(nv, "model.safetensors"), "pt") as sf:
        for name in sf.keys():
            if name.endswith(".weight_packed"):
                base = name[:-len(".weight_packed")]
                tensors[base + ".weight"] = nvfp4_dequantize(sf.get_tensor(name).cuda(), sf.get_tensor(base + ".weight_scale").cuda(),
                                                             sf.get_tensor(base + ".weight_global_scale").cuda()).cpu()
            elif not name.endswith((".weight_scale", ".weight_global_scale")):
                tensors[name] = sf.get_tensor(name)
    save_file(tensors, os.path.join(deq, "model.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(nv, "config.json")))
    cfg.pop("quantization_config")
    json.dump(cfg, open(os.path.join(deq, "config.json"), "w"))
    for fn in os.listdir(nv):
        if fn.endswith((".json", ".txt", ".model")) and fn != "config.json":
            shutil.copy(os.path.join(nv, fn), os.path.join(deq, fn))

    def run(path, **server_args):
        eng = Engine(EngineConfig(path, dtype="bfloat16", max_total_tokens=1024, max_running_requests=1, server_args=server_args or None))
        try:
            return eng.prefill_logits(IDS).float().cpu(), eng.generate_greedy(IDS, 12), eng.runner.model.lm_head.quant_method is not None
        finally:
            del eng
            free_cuda()

    ref_logits, ref_toks, _ = run(deq)
    q_logits, q_toks, q_head = run(nv)
    assert q_head and F.cosine_similarity(ref_logits, q_logits, dim=-1).item() > 0.999 and (ref_logits - q_logits).abs().max() < 0.05
    _, second, gaps = greedy_reference(deq, IDS, 12)
    assert_matches_up_to_tie_flip(q_toks, ref_toks, second, gaps, min_agree=4)
    o_logits, o_toks, o_head = run(deq, quantize_lm_head="nvfp4")
    assert o_head and torch.allclose(o_logits, q_logits, atol=0.05)
    for i, (a, b) in enumerate(zip(o_toks, q_toks)):
        if a != b:
            assert i >= 4 and gaps[i] < TIE_GAP, (i, gaps[i])
            break


@pytest.mark.parametrize("algo", ALGOS)
def test_speculative_decoding_matches_plain(tiny, references, algo):
    """Random drafter: nearly every round rolls back, output must equal plain greedy decoding
    token for token (verify + rollback + refresh); rounds run as CUDA graphs; KV slots balance.
    Then a greedy and a sampling request sharing rounds, and an oracle drafter accepting everything."""
    sp = SamplingParams(max_new_tokens=NEW, ignore_eos=True)
    spec = _scheduler(str(tiny.path), spec=True, algo=algo)
    kv0 = spec.token_to_kv_pool_allocator.available_size()
    got = _run(spec, PROMPTS, sp)
    verified = sum(assert_matches_up_to_tie_flip(g, *ref) for g, ref in zip(got, references))
    assert verified >= 2 * NEW and all(len(g) == NEW for g in got)
    assert spec.token_to_kv_pool_allocator.available_size() == kv0, "KV slots leaked"
    worker = spec.worker.spec
    assert worker.rounds > 0 and 1.0 <= worker.acceptance_length() <= 4.0 and worker.verify_graphs is not None
    spec.add_request("g", PROMPTS[0], sp)
    spec.add_request("s", PROMPTS[1], SamplingParams(max_new_tokens=NEW, ignore_eos=True, temperature=0.8, seed=3))
    while spec.step():
        pass
    assert_matches_up_to_tie_flip(list(spec._requests["g"].result.output_ids), *references[0])
    del spec
    free_cuda()

    first_tie = lambda i: next((j for j, g in enumerate(references[i][2]) if g < 0.012), NEW)   # noqa: E731
    idx = max(range(len(PROMPTS)), key=first_tie)
    ref, n, k = references[idx][0], min(NEW - 3, first_tie(idx)), 3
    assert n >= 6, "no tie-free prompt long enough"
    spec = _scheduler(str(tiny.path), spec=True, k=k, algo=algo)
    worker = spec.worker.spec
    if algo == "MTP":
        worker._draft_tokens = lambda logits, reqs, sampling, ahead: (
            torch.tensor([ref[len(r.output_ids) + a] if len(r.output_ids) + a < len(ref) else 0 for r, a in zip(reqs, ahead)],
                         dtype=torch.int64, device=logits.device), None)
    else:
        real_draft = worker.draft

        def oracle(hidden, logits, anchor, temps, gens):
            B, S, _ = hidden.shape
            path = torch.tensor([[ref[len(r.output_ids) + i] if len(r.output_ids) + i < len(ref) else 0 for i in range(S)]
                                 for r in worker._oracle_reqs], dtype=torch.int64, device=hidden.device)
            cands = path.unsqueeze(2).expand(B, S, worker.drafter.cfg.selector_top_k).contiguous()
            q = torch.zeros(B, S, worker.drafter.cfg.selector_top_k, device=hidden.device)
            q[..., 0] = 1.0
            return path, cands, q

        def draft(reqs, *a):
            worker._oracle_reqs = reqs
            return real_draft(reqs, *a)

        worker._select, worker.draft = oracle, draft
    assert _run(spec, [PROMPTS[idx]], SamplingParams(max_new_tokens=n, ignore_eos=True))[0] == ref[:n]
    assert worker.rounds == -(-(n - 1) // (k + 1)) and worker.accepted_drafts_total >= n - 1 - worker.rounds > 0
    del spec
    free_cuda()


def test_speculative_sampling_reproducible_and_penalties_fall_back(tiny):
    sp = SamplingParams(max_new_tokens=12, ignore_eos=True, temperature=0.8, top_p=0.95, seed=7)
    outs = []
    for _ in range(2):
        s = _scheduler(str(tiny.path), spec=True, k=2)
        outs.append(_run(s, PROMPTS[:2], sp))
        del s
        free_cuda()
    assert outs[0] == outs[1] and all(len(o) == 12 for o in outs[0])
    sp = SamplingParams(max_new_tokens=8, ignore_eos=True, repetition_penalty=1.3)
    ref = _run(_scheduler(str(tiny.path), spec=False), PROMPTS[:1], sp)[0]
    free_cuda()
    spec = _scheduler(str(tiny.path), spec=True)
    got = _run(spec, PROMPTS[:1], sp)[0]
    assert spec.worker.spec.draft_tokens_total == 0        # penalties: k = 0 rounds through the ordinary sampler
    _, second, gaps = greedy_reference(str(tiny.path), PROMPTS[0], 8, repetition_penalty=1.3)
    assert_matches_up_to_tie_flip(got, ref, second, gaps, min_agree=2)
    del spec
    free_cuda()


def test_tp2_matches_tp1(tiny):
    assert_tp2_matches_tp1(str(tiny.path), IDS[:20], 6, torch.ones(20, dtype=torch.bool), atol=0.05)


def test_tp2_matches_tp1_dflash(tiny):
    """The drafter under TP: sharded lm_head top-k for the candidates, sharded drafter MLP."""
    draft = SmallQwen35(build=False).build_dflash()
    assert_tp2_matches_tp1(str(tiny.path), IDS[:20], 8, torch.ones(20, dtype=torch.bool), atol=0.05,
                           extra={"speculative_algorithm": "DFLASH", "speculative_draft_model_path": draft, "speculative_num_steps": 3})

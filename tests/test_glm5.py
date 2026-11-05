"""GLM-5.3-Flash on tiny random models against transformers 5.19 (`glm5_next`): the MLA + DSA
indexer block (torch prefill path, CUDA decode kernel, decode-time k-pool selection), the whole
model through the ModelOpt-layout loader (prefill, teacher-forced decode with the sparse selection
active, CUDA-graph decode; this also covers the KDA and grouped NVFP4 MoE kernels against
transformers' modules), MTP speculative decoding, and two-rank TP.

Tiny random MoE models put many tokens on a routing near-tie (two experts within bf16 noise of the
top-k cut), where lily and transformers legitimately pick different experts; such positions are
masked with the margins `SmallGlm5.logits` returns.
"""

import pytest
import torch
import torch.nn.functional as F

from lily.layers.quantization import cuda_ops
from lily.mem_cache.memory_pool import ReqToTokenPool
from lily.model_executor.forward_batch_info import ForwardBatch
from lily.models.glm5_flash import DSAIndexer, MLAAttention
from tests.util import ROUTING_TIE, assert_matches_up_to_tie_flip, assert_tp2_matches_tp1, free_cuda, greedy_reference

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and cuda_ops.available()), reason="needs the CUDA extension")
transformers = pytest.importorskip("transformers")
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention  # noqa: E402
from tests.small_glm5 import SmallGlm5  # noqa: E402

IDS = list(range(10, 34))
HID, QL, KVL, NOPE, IH, IHD, TOPK, KP = 64, 32, 32, 32, 2, 32, 8, 4


def _cfg(**kw):
    base = dict(hidden_size=HID, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2, q_lora_rank=QL, kv_lora_rank=KVL,
                qk_nope_head_dim=NOPE, qk_rope_head_dim=0, v_head_dim=NOPE, index_n_heads=IH, index_head_dim=IHD, index_topk=TOPK,
                index_kpool=KP, index_kpool_always_select_tail=True, rms_norm_eps=1e-5, intermediate_size=128, moe_intermediate_size=128,
                n_routed_experts=8, num_experts_per_tok=3, n_shared_experts=1, routed_scaling_factor=2.5, norm_topk_prob=True,
                swiglu_limit=10.0, vocab_size=64, linear_num_heads=2, linear_head_dim=32, layer_types=["indexed_attention"] * 4,
                indexer_types=["full"] * 4)
    base.update(kw)
    cfg = Glm5NextTextConfig(**base)
    cfg.num_local_experts = cfg.n_routed_experts
    return cfg


def test_mla_dsa_matches_transformers():
    """20 tokens with index_topk 8 / pools of 4: dense up to 8 tokens, sparse beyond. Lily runs the
    reference weights as one prefill (torch path), token by token (decode kernel + pool-score kernel)
    and as prefill + 3-row continuation (kernel verify path); the decode-time selection must pick
    exactly the positions the prefill selection does."""
    torch.manual_seed(0)
    cfg = _cfg()
    ref = Glm5NextTextAttention(cfg, layer_idx=0)
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.1)
        for p in (ref.q_a_layernorm.weight, ref.kv_a_layernorm.weight, ref.indexer.k_norm.weight):
            p.normal_(1, 0.1)
        ref.indexer.index_kpool_compress_ape.normal_(0, 0.3)
    ref = ref.cuda().to(torch.bfloat16).eval()
    idx = DSAIndexer(HID, QL, IH, IHD, TOPK, KP, True)
    ly = MLAAttention(HID, 2, QL, KVL, NOPE, NOPE, cfg.rms_norm_eps, idx, layer_id=0, kv_layer_idx=0, max_visible=TOPK + KP - 1).cuda().to(torch.bfloat16)
    with torch.no_grad():
        for name in ("q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj", "o_proj"):
            getattr(ly, name).weight.copy_(getattr(ref, name).weight)
        for name in ("wq_b", "wk", "weights_proj"):
            getattr(idx, name).weight.copy_(getattr(ref.indexer, name).weight)
        idx.k_norm_weight.copy_(ref.indexer.k_norm.weight); idx.k_norm_bias.copy_(ref.indexer.k_norm.bias)
        idx.index_kpool_compress_ape.copy_(ref.indexer.index_kpool_compress_ape); idx.index_kpool_compress_gate.copy_(ref.indexer.index_kpool_compress_gate)
    L = 20
    x = (torch.randn(L, HID, device="cuda") * 0.5).bfloat16()
    req_pool = ReqToTokenPool(4, 64, "cuda")
    latent = torch.zeros(64, KVL, device="cuda", dtype=torch.bfloat16)
    index = torch.zeros(64, 2 * IHD, device="cuda", dtype=torch.bfloat16)
    nxt = [1]

    def run(rows, req, start):
        n = rows.shape[0]
        loc = torch.arange(nxt[0], nxt[0] + n, device="cuda")
        nxt[0] += n
        req_pool.write((req, slice(start, start + n)), loc.to(torch.int32))
        fb = ForwardBatch.for_extend([[0] * n], [start], [req], "cuda", out_cache_loc=loc) if (start == 0 or n > 1) \
            else ForwardBatch.for_decode([0], [start + 1], [req], "cuda", out_cache_loc=loc)
        fb.req_to_token_pool = req_pool
        return ly(rows, fb, latent, index)[0]

    tol = dict(atol=3e-2, rtol=3e-2)
    with torch.inference_mode():
        want = ref(x.unsqueeze(0), attention_mask=torch.ones(1, L, dtype=torch.bool, device="cuda"))[0][0]
        torch.testing.assert_close(run(x, 1, 0).float(), want.float(), **tol)
        torch.testing.assert_close(torch.cat([run(x[t:t + 1], 2, t) for t in range(L)]).float(), want.float(), **tol)
        a, b = run(x[:5], 3, 0), run(x[5:8], 3, 5)          # 3-row continuation within the dense regime: kernel path
        torch.testing.assert_close(torch.cat([a, b]).float(), want[:8].float(), **tol)
        # decode-time selection == prefill selection for the newest token of request 2
        cache = index[req_pool.req_to_token[2, :L].long()]
        q_resid = ly.q_a_layernorm(ly.q_a_proj(x))
        mask = idx.select(x[L - 1:], q_resid[L - 1:], torch.tensor([L - 1], device="cuda"), cache)[0]
        fb = ForwardBatch.for_decode([0], [L], [2], "cuda")
        fb.req_to_token_pool = req_pool
        vis, cnt = idx.decode_visible(x[L - 1:], q_resid[L - 1:], fb, index)
        assert set(vis[0, :cnt[0]].tolist()) == set(mask.nonzero().flatten().tolist()) and TOPK <= mask.sum() < L



@pytest.fixture(scope="module")
def check():
    c = SmallGlm5()
    yield c
    c.ref = None
    free_cuda()


@pytest.fixture(scope="module")
def engine(check):
    from lily.engine import Engine, EngineConfig

    eng = Engine(EngineConfig(str(check.path), dtype="bfloat16", max_total_tokens=512, max_running_requests=2))
    assert eng.runner.cuda_graph_runner is not None and eng.runner.attn_backend is None
    yield eng
    del eng
    free_cuda()


def test_model_matches_transformers(check, engine):
    """ModelOpt-layout loader + the whole stack: prefill logits, chunked prefill, 12 teacher-forced
    decode steps with the sparse selection active (24 + 12 tokens > index_topk), and greedy decode
    on a short prompt that stays in the dense regime, i.e. replays CUDA graphs."""
    want, margin = check.logits(IDS)
    strict = margin >= SmallGlm5.ROUTING_TIE
    assert strict.sum() >= len(IDS) // 2
    got = engine.all_prefill_logits(IDS).float().cpu()
    assert F.cosine_similarity(got, want, dim=-1)[strict].min() > 0.995 and (got - want)[strict].abs().max() < 0.3
    req = engine._new_seq()
    _, n = engine._extend(req, 0, IDS[:11])
    out, n = engine._extend(req, n, IDS[11:])
    engine._release(req, n)
    assert (out.next_token_logits[0].float().cpu() - want[-1]).abs().max() < 0.3
    toks, want, margin = check.teacher_forced(IDS, 12)
    strict = margin >= SmallGlm5.ROUTING_TIE
    assert strict.sum() >= 6
    req, rows = engine._new_seq(), []
    out, n = engine._extend(req, 0, IDS)
    rows.append(out.next_token_logits[0])
    for t in toks[:-1]:
        rows.append(engine._decode(req, n, t).next_token_logits[0])
        n += 1
    engine._release(req, n)
    got = torch.stack(rows).float().cpu()
    assert F.cosine_similarity(got, want, dim=-1)[strict].min() > 0.995 and (got - want)[strict].abs().max() < 0.3
    short = IDS[:3]
    toks, rows, margin = check.teacher_forced(short, 4)
    top2 = rows.topk(2, dim=-1)
    fb = ForwardBatch.for_decode([0], [7], [1], "cuda")
    assert engine.runner.cuda_graph_runner.can_run(fb)
    got = engine.generate_greedy(short, 4, ignore_eos=True)
    cut = next((i for i, m in enumerate(margin.tolist()) if m < SmallGlm5.ROUTING_TIE), 4)
    assert_matches_up_to_tie_flip(got[:cut], toks[:cut], top2.indices[:, 1].tolist(), (top2.values[:, 0] - top2.values[:, 1]).tolist(), min_agree=1, tie_gap=0.1)


def test_mtp_speculative_decoding_matches_plain():
    """index_topk 64 keeps every sequence dense so verify and draft steps replay CUDA graphs. The
    random MTP layer rarely drafts right, so rounds roll back KDA states and latent slots; output
    must equal plain decoding until the first near-tie. Then an oracle drafter accepts everything."""
    from tests.test_qwen35 import PROMPTS, _run, _scheduler

    path = str(SmallGlm5(path="/tmp/opencode/small-glm5-mtp", index_topk=64).path)
    free_cuda()
    refs = [greedy_reference(path, p, 16, with_routing_margins=True) for p in PROMPTS]
    spec = _scheduler(path, spec=True, k=3)
    from lily.sampling.sampling_params import SamplingParams

    kv0 = spec.token_to_kv_pool_allocator.available_size()
    got = _run(spec, PROMPTS, SamplingParams(max_new_tokens=16, ignore_eos=True))
    verified = 0
    for g, (ref, second, gaps, margins) in zip(got, refs):
        cut = next((i for i, m in enumerate(margins) if m < ROUTING_TIE), len(ref))
        verified += assert_matches_up_to_tie_flip(g[:cut], ref[:cut], second, gaps, min_agree=1) if cut else 0
    worker = spec.worker.spec
    assert verified >= 12 and spec.token_to_kv_pool_allocator.available_size() == kv0
    assert worker.rounds > 0 and worker.verify_graphs is not None and worker.draft_graphs is not None
    del spec
    free_cuda()
    first_tie = lambda i: next((j for j in range(16) if refs[i][2][j] < 0.012 or refs[i][3][j] < ROUTING_TIE), 16)   # noqa: E731
    idx = max(range(len(PROMPTS)), key=first_tie)
    ref, n = refs[idx][0], min(13, first_tie(idx))
    if n < 5:
        pytest.skip(f"no tie-free prompt long enough ({n})")
    spec = _scheduler(path, spec=True, k=3)
    worker = spec.worker.spec
    worker._draft_tokens = lambda logits, reqs, sampling, ahead: (
        torch.tensor([ref[len(r.output_ids) + a] if len(r.output_ids) + a < len(ref) else 0 for r, a in zip(reqs, ahead)],
                     dtype=torch.int64, device=logits.device), None)
    assert _run(spec, [PROMPTS[idx]], SamplingParams(max_new_tokens=n, ignore_eos=True))[0] == ref[:n]
    assert worker.rounds == -(-(n - 1) // 4) and worker.accepted_drafts_total >= n - 1 - worker.rounds > 0
    del spec
    free_cuda()


def test_tp2_matches_tp1(check):
    _, margin = check.logits(IDS[:20])
    assert_tp2_matches_tp1(str(check.path), IDS[:20], 6, margin >= SmallGlm5.ROUTING_TIE, atol=0.3)

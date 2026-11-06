"""Tiny random GLM-5.3-Flash (`Glm5NextForConditionalGeneration`, text only) for the test suite.

Two artifacts from one set of random weights:
  * `ref`: transformers' Glm5NextTextModel + an lm_head, bf16, with its MoE experts and dense
    MLPs replaced by the *dequantized* NVFP4 tensors, so it computes exactly what lily sees;
  * `path`: a checkpoint directory in the layout of `nvidia/GLM-5.3-Flash-NVFP4`
    (ModelOpt naming: experts and dense MLPs as `weight` u8 / `weight_scale` e4m3 /
    `weight_scale_2` fp32 per tensor; everything else bf16; `model.language_model.*` prefix;
    `q_conv1d/k_conv1d/v_conv1d`; `hc_attn_*` / `hc_ffn_*`; `self_attn.f_a_proj` etc.).

Dims respect the kernels' constraints: hidden and MoE intermediate multiples of 128, KDA
head_dim 32, kernel 4.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from lily.layers.quantization.base import nvfp4_quantize

VOCAB, HIDDEN, LAYERS = 64, 128, 5
LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "indexed_attention", "linear_attention"]
MLP_TYPES = ["dense", "dense", "sparse", "sparse", "sparse"]
EXPERTS, TOPK, MOE_I, DENSE_I = 8, 2, 128, 256
KDA_H, KDA_D = 2, 32
MLA = dict(num_attention_heads=2, num_key_value_heads=2, q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=32,
           qk_rope_head_dim=0, v_head_dim=32, index_n_heads=2, index_head_dim=32, index_topk=8, index_kpool=4)


def text_config(moe_i: int = MOE_I):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

    cfg = Glm5NextTextConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=LAYERS, intermediate_size=DENSE_I,
        moe_intermediate_size=moe_i, n_routed_experts=EXPERTS, num_experts_per_tok=TOPK, n_shared_experts=1,
        routed_scaling_factor=2.5, norm_topk_prob=True, swiglu_limit=10.0, rms_norm_eps=1e-5,
        linear_num_heads=KDA_H, linear_head_dim=KDA_D, linear_conv_kernel_dim=4, linear_lower_bound=-5.0,
        hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, layer_types=list(LAYER_TYPES), mlp_layer_types=list(MLP_TYPES),
        indexer_types=["full"] * LAYERS, index_kpool_always_select_tail=True, max_position_embeddings=4096,
        pad_token_id=0, eos_token_id=1, tie_word_embeddings=False, **MLA)
    cfg.num_local_experts = EXPERTS
    return cfg


class SmallGlm5:
    """`index_topk` 8 (default) makes the sparse selection kick in after 8 tokens; a larger value
    keeps every test sequence in the dense regime (CUDA-graph decode, verify graphs)."""

    def __init__(self, path: str = "/tmp/opencode/small-glm5", seed: int = 5, index_topk: int = 8, moe_i: int = MOE_I):
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

        self.path = Path(path)
        self.moe_i = moe_i           # expert slices need moe_i / tp to stay a multiple of 128
        self.cfg = text_config(moe_i)
        self.cfg.index_topk = index_topk
        self.seed = seed
        torch.manual_seed(seed)
        self.ref = Glm5NextTextModel(self.cfg)
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
        with torch.no_grad():
            for p in list(self.ref.parameters()) + list(self.lm_head.parameters()):
                p.normal_(0, 0.08)
            for layer in self.ref.layers:
                for hc in (layer.attn_hc, layer.ffn_hc):
                    hc.scale.fill_(1.0)
                    hc.base.zero_()
                for n in (layer.input_layernorm, layer.post_attention_layernorm):
                    n.weight.normal_(1.0, 0.05)
                if layer.block_type == "linear_attention":
                    sa = layer.self_attn
                    sa.forget_gate.A_log.normal_(0, 0.5)
                    sa.forget_gate.dt_bias.normal_(0, 0.5)
                    sa.o_norm.weight.normal_(1.0, 0.05)
                else:
                    sa = layer.self_attn
                    sa.q_a_layernorm.weight.normal_(1, 0.05)
                    sa.kv_a_layernorm.weight.normal_(1, 0.05)
                    sa.indexer.k_norm.weight.normal_(1, 0.05)
                    sa.indexer.k_norm.bias.normal_(0, 0.05)
                    sa.indexer.index_kpool_compress_gate.normal_(0, 0.1)
                if hasattr(layer.mlp, "gate"):
                    layer.mlp.gate.e_score_correction_bias.normal_(0, 0.3)
            self.ref.norm.weight.normal_(1.0, 0.05)
        self.ref = self.ref.to(torch.bfloat16).eval()
        self.lm_head = self.lm_head.to(torch.bfloat16)
        self._write_checkpoint()
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer").save_pretrained(self.path)

    # ---- checkpoint in NVIDIA's layout ----------------------------------------------

    def _write_checkpoint(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        out = {}
        P = "model.language_model."

        def quant(name: str, w: torch.Tensor) -> torch.Tensor:
            """Store NVFP4 (ModelOpt names) and return the dequantized bf16 weight for the reference."""
            packed, scale, gs = nvfp4_quantize(w.cuda() if torch.cuda.is_available() else w)
            out[name + ".weight"] = packed.cpu()
            out[name + ".weight_scale"] = scale.cpu()
            # ModelOpt convention: weight_scale_2 multiplies (lily's global scale divides)
            out[name + ".weight_scale_2"] = (1.0 / gs.float()).reshape(()).cpu()
            out[name + ".input_scale"] = torch.ones((), dtype=torch.float32)
            from lily.layers.quantization.base import NVFP4LinearMethod
            layer = torch.nn.Module()
            layer.weight_packed = torch.nn.Parameter(packed, requires_grad=False)
            layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
            layer.weight_global_scale = torch.nn.Parameter(gs.expand(packed.shape[0]).contiguous(), requires_grad=False)
            return NVFP4LinearMethod().dequantized_weight(layer).to(w.device)

        with torch.no_grad():
            out["lm_head.weight"] = self.lm_head.weight.data.clone()
            out[P + "embed_tokens.weight"] = self.ref.embed_tokens.weight.data.clone()
            out[P + "norm.weight"] = self.ref.norm.weight.data.clone()
            for i, layer in enumerate(self.ref.layers):
                L = f"{P}layers.{i}."
                for tag, hc in (("attn", layer.attn_hc), ("ffn", layer.ffn_hc)):
                    out[L + f"hc_{tag}_fn"] = hc.fn.data.clone()
                    out[L + f"hc_{tag}_base"] = hc.base.data.clone()
                    out[L + f"hc_{tag}_scale"] = hc.scale.data.clone()
                out[L + "input_layernorm.weight"] = layer.input_layernorm.weight.data.clone()
                out[L + "post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight.data.clone()
                sa = layer.self_attn
                if layer.block_type == "linear_attention":
                    qkv = KDA_H * KDA_D
                    conv = sa.conv1d.weight.data.squeeze(1)       # [3*qkv, 4]
                    for name, w in (("q_conv1d", conv[:qkv]), ("k_conv1d", conv[qkv:2 * qkv]), ("v_conv1d", conv[2 * qkv:])):
                        out[L + f"self_attn.{name}.weight"] = w.unsqueeze(1).clone().to(torch.bfloat16)
                    for name in ("q_proj", "k_proj", "v_proj", "b_proj", "g_a_proj", "g_b_proj", "o_proj"):
                        out[L + f"self_attn.{name}.weight"] = getattr(sa, name).weight.data.clone()
                    out[L + "self_attn.f_a_proj.weight"] = sa.forget_gate.f_a_proj.weight.data.clone()
                    out[L + "self_attn.f_b_proj.weight"] = sa.forget_gate.f_b_proj.weight.data.clone()
                    out[L + "self_attn.dt_bias"] = sa.forget_gate.dt_bias.data.float().clone()
                    out[L + "self_attn.A_log"] = sa.forget_gate.A_log.data.float().clone()
                    out[L + "self_attn.o_norm.weight"] = sa.o_norm.weight.data.clone()
                else:
                    for name in ("q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj", "o_proj"):
                        out[L + f"self_attn.{name}.weight"] = getattr(sa, name).weight.data.clone()
                    ix = sa.indexer
                    for name in ("wq_b", "wk", "weights_proj"):
                        out[L + f"self_attn.indexer.{name}.weight"] = getattr(ix, name).weight.data.clone()
                    out[L + "self_attn.indexer.k_norm.weight"] = ix.k_norm.weight.data.clone()
                    out[L + "self_attn.indexer.k_norm.bias"] = ix.k_norm.bias.data.clone()
                    out[L + "self_attn.indexer.index_kpool_compress_ape"] = ix.index_kpool_compress_ape.data.clone()
                    out[L + "self_attn.indexer.index_kpool_compress_gate"] = ix.index_kpool_compress_gate.data.clone()
                mlp = layer.mlp
                if hasattr(mlp, "gate"):   # MoE
                    out[L + "mlp.gate.weight"] = mlp.gate.weight.data.clone().to(torch.bfloat16)
                    out[L + "mlp.gate.e_score_correction_bias"] = mlp.gate.e_score_correction_bias.data.float().clone()
                    for name in ("gate_proj", "up_proj", "down_proj"):
                        out[L + f"mlp.shared_experts.{name}.weight"] = getattr(mlp.shared_experts, name).weight.data.clone()
                    for e in range(EXPERTS):
                        gu = mlp.experts.gate_up_proj[e]
                        mlp.experts.gate_up_proj[e, :self.moe_i] = quant(L + f"mlp.experts.{e}.gate_proj", gu[:self.moe_i])
                        mlp.experts.gate_up_proj[e, self.moe_i:] = quant(L + f"mlp.experts.{e}.up_proj", gu[self.moe_i:])
                        mlp.experts.down_proj[e] = quant(L + f"mlp.experts.{e}.down_proj", mlp.experts.down_proj[e])
                else:
                    for name in ("gate_proj", "up_proj", "down_proj"):
                        lin = getattr(mlp, name)
                        lin.weight.data.copy_(quant(L + f"mlp.{name}", lin.weight.data))
        self._add_mtp_layer(out, quant)
        save_file(out, str(self.path / "model.safetensors"), metadata={"format": "pt"})
        hf = {
            "architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next",
            "text_config": {**{k: v for k, v in self.cfg.to_dict().items() if not k.startswith("_")},
                            "architectures": ["Glm5NextTextModel"], "model_type": "glm5_next_text", "dtype": "bfloat16",
                            "num_nextn_predict_layers": 1},
            "quantization_config": {
                "config_groups": {"group_0": {"input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                                              "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                                              "targets": ["Linear"]}},
                "ignore": ["lm_head", "model.language_model.embed_tokens"]
                          + [f"model.language_model.layers.{i}.self_attn*" for i in range(LAYERS + 1)]
                          + [f"model.language_model.layers.{i}.mlp.gate" for i in range(LAYERS + 1)]
                          + [f"model.language_model.layers.{i}.mlp.shared_experts*" for i in range(LAYERS + 1)],
                "quant_algo": "NVFP4", "quant_method": "modelopt", "producer": {"name": "modelopt"},
            },
        }
        (self.path / "config.json").write_text(json.dumps(hf, indent=2))

    def _add_mtp_layer(self, out: dict, quant) -> None:
        """Random MTP layer `layers.{LAYERS}` with the real checkpoint's tensor names (no HF
        reference exists for it; the speculative tests only need a drafter that loads and runs)."""
        gen = torch.Generator().manual_seed(self.seed + 1)
        L = f"model.language_model.layers.{LAYERS}."
        H, QL, KVL, NOPE, VD = HIDDEN, MLA["q_lora_rank"], MLA["kv_lora_rank"], MLA["qk_nope_head_dim"], MLA["v_head_dim"]
        NH, IH, IHD, KP = MLA["num_attention_heads"], MLA["index_n_heads"], MLA["index_head_dim"], MLA["index_kpool"]

        def rnd(*shape, std=0.08, mean=0.0):
            return (torch.randn(*shape, generator=gen) * std + mean).to(torch.bfloat16)

        for n in ("enorm", "hnorm", "input_layernorm", "post_attention_layernorm", "shared_head.norm"):
            out[L + n + ".weight"] = rnd(H, std=0.05, mean=1.0)
        out[L + "eh_proj.weight"] = rnd(H, 2 * H)
        sa = L + "self_attn."
        out[sa + "q_a_proj.weight"] = rnd(QL, H); out[sa + "q_a_layernorm.weight"] = rnd(QL, std=0.05, mean=1.0)
        out[sa + "q_b_proj.weight"] = rnd(NH * NOPE, QL); out[sa + "kv_a_proj_with_mqa.weight"] = rnd(KVL, H)
        out[sa + "kv_a_layernorm.weight"] = rnd(KVL, std=0.05, mean=1.0); out[sa + "kv_b_proj.weight"] = rnd(NH * (NOPE + VD), KVL)
        out[sa + "o_proj.weight"] = rnd(H, NH * VD)
        ix = sa + "indexer."
        out[ix + "wq_b.weight"] = rnd(IH * IHD, QL); out[ix + "wk.weight"] = rnd(IHD, H)
        out[ix + "k_norm.weight"] = rnd(IHD, std=0.05, mean=1.0); out[ix + "k_norm.bias"] = rnd(IHD, std=0.05)
        out[ix + "weights_proj.weight"] = rnd(IH, H); out[ix + "index_kpool_compress_ape"] = rnd(KP, IHD, std=0.3)
        out[ix + "index_kpool_compress_gate"] = rnd(IHD, H, std=0.1)
        out[L + "mlp.gate.weight"] = rnd(EXPERTS, H); out[L + "mlp.gate.e_score_correction_bias"] = (torch.randn(EXPERTS, generator=gen) * 0.3)
        for n, shape in (("gate_proj", (self.moe_i, H)), ("up_proj", (self.moe_i, H)), ("down_proj", (H, self.moe_i))):
            out[L + f"mlp.shared_experts.{n}.weight"] = rnd(*shape)
            for e in range(EXPERTS):
                quant(L + f"mlp.experts.{e}.{n}", rnd(*shape))

    # ---- reference -----------------------------------------------------------------------

    @torch.inference_mode()
    def logits(self, ids):
        """Full-sequence reference logits [L, V] fp32 (CPU model), and per position the smallest
        routing margin over the MoE layers: the gap between the last selected expert's corrected
        score and the best unselected one. A tiny margin means another implementation's bf16
        noise can legitimately pick a different expert there (then the logits differ a lot)."""
        margins = []

        def hook(mod, inp, out):
            scores = torch.sigmoid(out[0]) + mod.e_score_correction_bias      # [T, E] fp32
            top = scores.topk(mod.top_k + 1, dim=-1).values
            margins.append(top[:, -2] - top[:, -1])

        handles = [l.mlp.gate.register_forward_hook(hook) for l in self.ref.layers if hasattr(l.mlp, "gate")]
        try:
            x = torch.tensor([ids])
            h = self.ref(input_ids=x, attention_mask=torch.ones_like(x, dtype=torch.bool)).last_hidden_state
        finally:
            for hd in handles:
                hd.remove()
        return self.lm_head(h)[0].float(), torch.stack(margins).min(dim=0).values

    ROUTING_TIE = 0.01   # corrected-score margin below which a routing flip is bf16 noise

    def teacher_forced(self, ids, n):
        """n greedy steps of the reference: (tokens, next-token logits per step [n, V], routing
        margin of the predicting position per step [n]). Feeding these tokens to another engine
        and comparing its per-step logits (where the margin is clear) tests its decode path
        step by step without a near-tie ever ending the comparison."""
        seq = list(ids)
        toks, rows, margins = [], [], []
        for _ in range(n):
            logits, margin = self.logits(seq)
            rows.append(logits[-1])
            margins.append(margin[-1])
            toks.append(int(logits[-1].argmax()))
            seq.append(toks[-1])
        return toks, torch.stack(rows), torch.stack(margins)


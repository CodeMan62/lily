"""Tiny random-weights Qwen3.5 hybrid checkpoint used by test_qwen35_engine.py.

Weights come from transformers' own Qwen3_5ForCausalLM, so the reference and
the lily engine read identical tensors. Saved in HF format (model_type
`qwen3_5_text`, arch `Qwen3_5ForCausalLM`, which lily registers).
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from transformers.models.qwen3_5 import configuration_qwen3_5 as cfgm

VOCAB = 64
HIDDEN = 64
LAYERS = 8  # hybrid: 3 linear-attention, then 1 full-attention, repeating

LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2


class SmallQwen35:
    """Builds (once) and wraps the tiny hybrid checkpoint directory."""

    def __init__(self, path: str = "/tmp/opencode/small-qwen35", seed: int = 7, build: bool = True):
        self.path = Path(path)
        self.ref = None
        if build:
            self._build(seed)
        self.hf = self._reference()

    def _build(self, seed):
        cfg_path = self.path / "config.json"
        if cfg_path.exists() and "mtp_num_hidden_layers" in cfg_path.read_text():
            return
        self.path.mkdir(parents=True, exist_ok=True)
        text = cfgm.Qwen3_5TextConfig(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
            intermediate_size=128,
            num_hidden_layers=LAYERS,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            layer_types=LAYER_TYPES,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
        )
        model = Qwen3_5ForCausalLM(text)
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for param in model.parameters():
                param.normal_(0.0, 0.05)
        model.config.architectures = ["Qwen3_5ForCausalLM"]
        model.save_pretrained(self.path, safe_serialization=True)
        self._add_mtp_head(text, gen)

        tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        tok.save_pretrained(self.path)

    def _add_mtp_head(self, text, gen) -> None:
        """Append a random MTP block (same tensor names/shapes as Qwen3.8-27B's `mtp.*`:
        fc + one gated full-attention decoder layer + three norms) and flag it in config."""
        from safetensors.torch import load_file, save_file

        H, I = text.hidden_size, text.intermediate_size
        nh, nkv, hd = text.num_attention_heads, text.num_key_value_heads, text.head_dim

        def rnd(*shape, std=0.05):
            return torch.randn(*shape, generator=gen).mul_(std).to(torch.bfloat16)

        mtp = {
            "mtp.fc.weight": rnd(H, 2 * H),
            "mtp.pre_fc_norm_embedding.weight": rnd(H),
            "mtp.pre_fc_norm_hidden.weight": rnd(H),
            "mtp.norm.weight": rnd(H),
            "mtp.layers.0.input_layernorm.weight": rnd(H),
            "mtp.layers.0.post_attention_layernorm.weight": rnd(H),
            "mtp.layers.0.self_attn.q_proj.weight": rnd(nh * hd * 2, H),   # [q; gate] per head
            "mtp.layers.0.self_attn.k_proj.weight": rnd(nkv * hd, H),
            "mtp.layers.0.self_attn.v_proj.weight": rnd(nkv * hd, H),
            "mtp.layers.0.self_attn.o_proj.weight": rnd(H, nh * hd),
            "mtp.layers.0.self_attn.q_norm.weight": rnd(hd),
            "mtp.layers.0.self_attn.k_norm.weight": rnd(hd),
            "mtp.layers.0.mlp.gate_proj.weight": rnd(I, H),
            "mtp.layers.0.mlp.up_proj.weight": rnd(I, H),
            "mtp.layers.0.mlp.down_proj.weight": rnd(H, I),
        }
        st_path = self.path / "model.safetensors"
        tensors = load_file(str(st_path))
        tensors.update(mtp)
        save_file(tensors, str(st_path), metadata={"format": "pt"})
        cfg_path = self.path / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["mtp_num_hidden_layers"] = 1
        cfg_path.write_text(json.dumps(cfg, indent=2))

    def build_dflash(self, seed: int = 11) -> str:
        """Random-weights DFlash2 drafter for this tiny target (same tensor names/shapes as
        z-lab/Qwen3.8-27B-DFlash2, scaled down). Returns its directory."""
        from safetensors.torch import save_file

        out = self.path.parent / "small-qwen35-dflash2"
        cfg_path = out / "config.json"
        if cfg_path.exists():
            return str(out)
        out.mkdir(parents=True, exist_ok=True)
        gen = torch.Generator().manual_seed(seed)
        H, I, nh, nkv, hd = HIDDEN, 128, 2, 1, 32
        layers, group, ksize, rank, topk = 2, 16, 2, 8, 4
        target_layer_ids = [1, 3, 5]

        def rnd(*shape, std=0.05):
            return torch.randn(*shape, generator=gen).mul_(std).to(torch.bfloat16)

        t = {
            "fc.weight": rnd(H, len(target_layer_ids) * H),
            "hidden_norm.weight": (1 + rnd(H)),
            "norm.weight": (1 + rnd(H)),
            "candidate_selector.hidden_projection.weight": rnd(rank, H),
            "candidate_selector.predecessor_codebook": rnd(VOCAB, rank, std=0.3),
            "candidate_selector.successor_codebook": rnd(VOCAB, rank, std=0.3),
        }
        groups = H // group
        for i in range(layers):
            pre = f"layers.{i}."
            t.update({
                pre + "input_layernorm.weight": 1 + rnd(H), pre + "post_attention_layernorm.weight": 1 + rnd(H),
                pre + "self_attn.q_proj.weight": rnd(nh * hd, H), pre + "self_attn.k_proj.weight": rnd(nkv * hd, H),
                pre + "self_attn.v_proj.weight": rnd(nkv * hd, H), pre + "self_attn.o_proj.weight": rnd(H, nh * hd),
                pre + "self_attn.q_norm.weight": 1 + rnd(hd), pre + "self_attn.k_norm.weight": 1 + rnd(hd),
                pre + "mlp.gate_proj.weight": rnd(I, H), pre + "mlp.up_proj.weight": rnd(I, H), pre + "mlp.down_proj.weight": rnd(H, I),
                pre + "attention_conv.base_kernel": rnd(2, ksize, H, std=0.2), pre + "attention_conv.kernel_projection.weight": rnd(2 * ksize * groups, H),
                pre + "mlp_conv.base_kernel": rnd(2, ksize, H, std=0.2), pre + "mlp_conv.kernel_projection.weight": rnd(2 * ksize * groups, H),
            })
        save_file(t, str(out / "model.safetensors"), metadata={"format": "pt"})
        cfg = {
            "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "attention_bias": False, "is_causal": False,
            "dflash_config": {"block_size": 4, "conv_group_size": group, "conv_kernel_size": ksize,
                              "mask_token_id": VOCAB - 1, "selector_rank": rank, "selector_top_k": topk,
                              "target_layer_ids": target_layer_ids},
            "dtype": "bfloat16", "head_dim": hd, "hidden_act": "silu", "hidden_size": H, "intermediate_size": I,
            "layer_types": ["sliding_attention"] * layers, "max_position_embeddings": 4096, "num_attention_heads": nh,
            "num_hidden_layers": layers, "num_key_value_heads": nkv, "num_target_layers": LAYERS, "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}, "sliding_window": 2048,
            "tie_word_embeddings": False, "use_sliding_window": True, "vocab_size": VOCAB,
        }
        cfg_path.write_text(json.dumps(cfg, indent=2))
        return str(out)

    def _reference(self):
        if self.ref is not None:
            return self.ref
        from transformers import AutoModelForCausalLM

        self.ref = AutoModelForCausalLM.from_pretrained(self.path, dtype=torch.bfloat16).eval()
        return self.ref

    @property
    def config(self) -> dict:
        return json.loads((self.path / "config.json").read_text())

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
        if (self.path / "config.json").exists():
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

        tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        tok.save_pretrained(self.path)

    def _reference(self):
        if self.ref is not None:
            return self.ref
        from transformers import AutoModelForCausalLM

        self.ref = AutoModelForCausalLM.from_pretrained(self.path, dtype=torch.bfloat16).eval()
        return self.ref

    @property
    def config(self) -> dict:
        return json.loads((self.path / "config.json").read_text())

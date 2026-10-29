"""Model configuration parsed from an HF checkpoint. Mirrors sglang/srt/configs/model_config.py."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from huggingface_hub import snapshot_download

_STR_DTYPE = {
    "float32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_model_path(model_path: str, revision: Optional[str] = None) -> str:
    """Return a local directory for `model_path`, downloading configs/weights if needed."""
    if os.path.isdir(model_path):
        return model_path
    return snapshot_download(
        model_path,
        revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"],
    )


@dataclass
class ModelConfig:
    model_path: str
    hf_config: Dict[str, Any]
    hf_text_config: Dict[str, Any]
    architectures: List[str]
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    tie_word_embeddings: bool
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Optional[Dict[str, Any]]
    partial_rotary_factor: float
    context_len: int
    dtype: torch.dtype
    hidden_act: str
    attention_bias: bool
    layer_types: Optional[List[str]]  # e.g. ["linear_attention", ..., "full_attention"]
    quantization_config: Optional[Dict[str, Any]]
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_size: int = 4
    mamba_ssm_dtype: Optional[str] = None
    eos_token_ids: List[int] = field(default_factory=list)
    bos_token_id: Optional[int] = None
    generation_config: Dict[str, Any] = field(default_factory=dict)

    # ---- construction ------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: str = "auto",
        context_length: Optional[int] = None,
        revision: Optional[str] = None,
    ) -> "ModelConfig":
        local = resolve_model_path(model_path, revision)
        with open(os.path.join(local, "config.json")) as f:
            hf_config = json.load(f)
        text = hf_config.get("text_config", hf_config)

        gen_cfg_path = os.path.join(local, "generation_config.json")
        generation_config: Dict[str, Any] = {}
        if os.path.exists(gen_cfg_path):
            with open(gen_cfg_path) as f:
                generation_config = json.load(f)

        n_heads = text["num_attention_heads"]
        hidden = text["hidden_size"]
        head_dim = text.get("head_dim") or hidden // n_heads
        n_kv = text.get("num_key_value_heads") or n_heads

        rope_scaling = text.get("rope_scaling")
        rope_params = text.get("rope_parameters")  # newer HF configs
        rope_theta = text.get("rope_theta", 10000.0)
        if rope_params:
            rope_theta = rope_params.get("rope_theta", rope_theta)
            if rope_params.get("rope_type", "default") != "default":
                rope_scaling = dict(rope_params)
        if rope_scaling is not None and "rope_type" not in rope_scaling and "type" in rope_scaling:
            rope_scaling = {**rope_scaling, "rope_type": rope_scaling["type"]}

        eos = generation_config.get("eos_token_id", text.get("eos_token_id"))
        if eos is None:
            eos_ids: List[int] = []
        elif isinstance(eos, int):
            eos_ids = [eos]
        else:
            eos_ids = list(eos)

        hf_dtype_str = text.get("torch_dtype") or text.get("dtype") or hf_config.get("torch_dtype")
        model_dtype = _resolve_dtype(dtype, hf_dtype_str)

        max_pos = text.get("max_position_embeddings", 4096)
        if rope_scaling and rope_scaling.get("rope_type") == "yarn":
            factor = rope_scaling.get("factor", 1.0)
            orig = rope_scaling.get("original_max_position_embeddings", max_pos)
            max_pos = max(max_pos, int(orig * factor))

        return cls(
            model_path=local,
            hf_config=hf_config,
            hf_text_config=text,
            architectures=hf_config.get("architectures", []),
            model_type=hf_config.get("model_type", ""),
            hidden_size=hidden,
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv,
            head_dim=head_dim,
            intermediate_size=text["intermediate_size"],
            vocab_size=text["vocab_size"],
            tie_word_embeddings=bool(text.get("tie_word_embeddings", hf_config.get("tie_word_embeddings", False))),
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            rope_theta=float(rope_theta),
            rope_scaling=rope_scaling,
            partial_rotary_factor=float(text.get("partial_rotary_factor",
                                                 (rope_params or {}).get("partial_rotary_factor", 1.0))),
            context_len=context_length or max_pos,
            dtype=model_dtype,
            hidden_act=text.get("hidden_act", "silu"),
            attention_bias=bool(text.get("attention_bias", False)),
            layer_types=text.get("layer_types"),
            quantization_config=hf_config.get("quantization_config") or text.get("quantization_config"),
            linear_num_key_heads=text.get("linear_num_key_heads", 0),
            linear_num_value_heads=text.get("linear_num_value_heads", 0),
            linear_key_head_dim=text.get("linear_key_head_dim", 0),
            linear_value_head_dim=text.get("linear_value_head_dim", 0),
            linear_conv_kernel_size=text.get("linear_conv_kernel_dim", 4),
            mamba_ssm_dtype=text.get("mamba_ssm_dtype"),
            eos_token_ids=eos_ids,
            bos_token_id=text.get("bos_token_id"),
            generation_config=generation_config,
        )

    # ---- derived -----------------------------------------------------------

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def is_hybrid(self) -> bool:
        return bool(self.layer_types) and any(t != "full_attention" for t in self.layer_types)

    @property
    def full_attention_layer_ids(self) -> List[int]:
        if not self.layer_types:
            return list(range(self.num_hidden_layers))
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @property
    def linear_attention_layer_ids(self) -> List[int]:
        if not self.layer_types:
            return []
        return [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]

    def get_num_attention_heads(self, tp_size: int) -> int:
        return self.num_attention_heads // tp_size

    def get_num_kv_heads(self, tp_size: int) -> int:
        # With fewer KV heads than TP ranks, each rank holds one replicated head.
        return max(1, self.num_key_value_heads // tp_size)

    def get_linear_key_heads(self, tp_size: int) -> int:
        return self.linear_num_key_heads // tp_size

    def get_linear_value_heads(self, tp_size: int) -> int:
        return self.linear_num_value_heads // tp_size

    @property
    def mamba_state_dtype(self) -> torch.dtype:
        return torch.float32 if (self.mamba_ssm_dtype or "float32") == "float32" else torch.bfloat16

    def summary(self) -> str:
        return (
            f"{self.architectures} type={self.model_type} layers={self.num_hidden_layers} "
            f"hidden={self.hidden_size} heads={self.num_attention_heads}/{self.num_key_value_heads} "
            f"head_dim={self.head_dim} vocab={self.vocab_size} tied={self.tie_word_embeddings} "
            f"rope_theta={self.rope_theta} rope_scaling={self.rope_scaling} "
            f"rotary_dim={self.rotary_dim} ctx={self.context_len} dtype={self.dtype} "
            f"hybrid={self.is_hybrid} quant={(self.quantization_config or {}).get('quant_method')}"
        )


def _resolve_dtype(requested: str, hf_dtype: Optional[str]) -> torch.dtype:
    if requested != "auto":
        return _STR_DTYPE[requested]
    if hf_dtype is None:
        return torch.bfloat16
    dt = _STR_DTYPE.get(hf_dtype, torch.bfloat16)
    # Serve fp32 checkpoints in bf16 like SGLang does.
    return torch.bfloat16 if dt == torch.float32 else dt

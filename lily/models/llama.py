"""Llama-family model. Mirrors sglang/srt/models/llama.py (reduced)."""

from __future__ import annotations

import logging

from typing import Iterable, List, Optional, Set, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn

from lily.models.model_config import ModelConfig

from lily.layers.layernorm import RMSNorm
from lily.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear, default_weight_loader
from lily.layers.logits_processor import LogitsProcessorOutput
from lily.layers.quantization.base import (
    OnlineNVFP4LinearMethod, QuantizationConfig, UnquantizedLinearMethod, _IgnoredPrefixConfig,
)
from lily.layers.radix_attention import RadixAttention
from lily.layers.rotary import get_rope
from lily.layers.vocab_embedding import ParallelLMHead, VocabParallelEmbedding
from lily.distributed import get_tp_info
from lily.model_executor.forward_batch_info import ForwardBatch
from lily.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str,
                 quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation {hidden_act!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        d = gate_up.shape[-1] // 2
        x = F.silu(gate_up[..., :d]) * gate_up[..., d:]
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):
    qkv_cls: Type[nn.Module] = QKVParallelLinear
    q_size_multiplier: int = 1
    use_rope_scaling: bool = True

    def __init__(self, config: ModelConfig, layer_id: int,
                 quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size, _ = get_tp_info()
        self.num_heads = config.get_num_attention_heads(tp_size)
        self.num_kv_heads = config.get_num_kv_heads(tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim * self.q_size_multiplier
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.layer_id = layer_id

        self.qkv_proj = self.qkv_cls(
            self.hidden_size, self.head_dim, self.num_heads, self.num_kv_heads,
            bias=config.attention_bias, quant_config=quant_config, prefix=f"{prefix}.qkv_proj")
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias,
            quant_config=quant_config, prefix=f"{prefix}.o_proj")
        self.rotary_emb = get_rope(
            self.head_dim, rotary_dim=config.rotary_dim, max_position=config.context_len,
            base=config.rope_theta, is_neox_style=True,
            rope_scaling=config.rope_scaling if self.use_rope_scaling else None)
        self.attn = RadixAttention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads, layer_id,
                                   prefix=f"{prefix}.attn")

    def _qkv(self, hidden_states: torch.Tensor):
        qkv, _ = self.qkv_proj(hidden_states)
        return qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        q, k, v = self._qkv(hidden_states)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int, attn_cls: Type[nn.Module] = LlamaAttention,
                 quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__()
        self.self_attn = attn_cls(config, layer_id, quant_config=quant_config, prefix=f"{prefix}.self_attn")
        self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size, config.hidden_act,
                            quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @property
    def attn_layer(self) -> nn.Module:
        """Attention block for this layer; hybrid models override it."""
        return self.self_attn

    def forward(self, positions, hidden_states, forward_batch, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.attn_layer(positions, hidden_states, forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(self, config: ModelConfig, attn_cls: Type[nn.Module] = LlamaAttention,
                 quant_config: Optional[QuantizationConfig] = None, prefix: str = "",
                 layer_cls: Type[nn.Module] = LlamaDecoderLayer, zero_centered: bool = False):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens")
        self.layers = nn.ModuleList([
            layer_cls(config, i, attn_cls, quant_config=quant_config, prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, zero_centered=zero_centered)

    def _prepare_forward(self, input_ids, positions, forward_batch, input_embeds) -> None:
        """Per-model state setup run once before the layer stack."""

    def forward(self, input_ids, positions, forward_batch, input_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        self._prepare_forward(input_ids, positions, forward_batch, input_embeds)
        residual = None
        aux_layers = forward_batch.capture_aux_layers
        aux: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
            if aux_layers is not None and i in aux_layers:
                # the full residual stream after layer i (HF's hidden_states[i + 1])
                aux.append(hidden_states + residual if residual is not None else hidden_states)
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_layers is not None:
            forward_batch.aux_hidden_states = torch.cat(aux, dim=-1)
        return hidden_states


class LlamaForCausalLMBase(nn.Module):
    """Shared head/loader for Llama-shaped models; subclasses pick the attention class."""

    attn_cls: Type[nn.Module] = LlamaAttention
    model_cls: Type[nn.Module] = LlamaModel

    stacked_params_mapping = [
        # (param_name, weight_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    def __init__(self, config: ModelConfig, quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = self.model_cls(config, self.attn_cls, quant_config=quant_config, prefix="model")
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix="lm_head",
                                      quant_method=self._lm_head_quant_method(config, quant_config))
        if config.tie_word_embeddings:
            self.lm_head.tie_weights(self.model.embed_tokens)

    @staticmethod
    def _lm_head_quant_method(config: ModelConfig, quant_config: Optional[QuantizationConfig]):
        """Checkpoint-quantized lm_head wins; else the online request (`--quantize-lm-head`)."""
        if config.tie_word_embeddings:
            if config.lm_head_quantization:
                logger.warning("lm_head is tied to the embeddings; --quantize-lm-head ignored")
            return None
        # Only pre-quantized checkpoints carry a quantized lm_head; the online
        # `--quantization` methods leave it alone unless asked via --quantize-lm-head.
        if isinstance(quant_config, _IgnoredPrefixConfig):
            method = quant_config.get_quant_method(None, "lm_head")
            if method is not None and not isinstance(method, UnquantizedLinearMethod):
                return method
        if config.lm_head_quantization == "nvfp4":
            return OnlineNVFP4LinearMethod()
        return None

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch,
                input_embeds: Optional[torch.Tensor] = None) -> LogitsProcessorOutput:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        mode = forward_batch.forward_mode
        if mode.is_decode() or mode.is_idle() or forward_batch.return_all_logits:
            last_hidden = hidden_states
        else:
            last_index = forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
            last_hidden = hidden_states[last_index]
        logits = self.lm_head.compute_logits(last_hidden)
        return LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden_states if forward_batch.capture_hidden_states else None,
            aux_hidden_states=getattr(forward_batch, "aux_hidden_states", None))

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Tuple[Set[str], Set[str]]:
        """Returns (loaded_param_names, unexpected_weight_names)."""
        params_dict = dict(self.named_parameters())
        loaded: Set[str] = set()
        unexpected: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.startswith("lm_head.") and self.config.tie_word_embeddings:
                continue
            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    unexpected.add(name)
                    break
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(name)
                break
            else:
                if name not in params_dict:
                    unexpected.add(name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)
        return loaded, unexpected


@ModelRegistry.register("LlamaForCausalLM")
class LlamaForCausalLM(LlamaForCausalLMBase):
    attn_cls = LlamaAttention

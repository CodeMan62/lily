"""Rotary embeddings: default, Llama-3 scaled, YaRN. Mirrors sglang/srt/layers/rotary_embedding.py (reduced).

Supports partial rotary (rotary_dim < head_size) and NeoX-style rotate-half,
which is what Llama/Qwen checkpoints use.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype or torch.get_default_dtype()
        cache = self._compute_cos_sin_cache()
        self.register_buffer("cos_sin_cache", cache.to(self.dtype), persistent=False)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        return 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        return torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [max_pos, rotary_dim]

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """positions: [N]; query: [N, heads*head_size] or [N, heads, head_size]; key likewise."""
        q_shape, k_shape = query.shape, key.shape
        n = positions.shape[0]
        query = query.view(n, -1, self.head_size)
        key = key.view(n, -1, self.head_size)
        cos_sin = self.cos_sin_cache[positions]  # [N, rotary_dim]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            cos = cos.repeat(1, 2).unsqueeze(1)  # [N, 1, rotary_dim]
            sin = sin.repeat(1, 2).unsqueeze(1)
            rotate = _rotate_neox
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1)
            rotate = _rotate_gptj

        q_rot, q_pass = query[..., : self.rotary_dim], query[..., self.rotary_dim :]
        k_rot, k_pass = key[..., : self.rotary_dim], key[..., self.rotary_dim :]
        q_rot = q_rot * cos + rotate(q_rot) * sin
        k_rot = k_rot * cos + rotate(k_rot) * sin
        query = torch.cat((q_rot, q_pass), dim=-1).view(q_shape)
        key = torch.cat((k_rot, k_pass), dim=-1).view(k_shape)
        return query, key


class Llama3RotaryEmbedding(RotaryEmbedding):
    def __init__(self, *args, scaling_factor: float, low_freq_factor: float, high_freq_factor: float,
                 orig_max_position: int, **kw):
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position
        super().__init__(*args, **kw)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        inv_freqs = super()._compute_inv_freq(base)
        low_freq_wavelen = self.orig_max_position / self.low_freq_factor
        high_freq_wavelen = self.orig_max_position / self.high_freq_factor
        wave_len = 2 * math.pi / inv_freqs
        if self.low_freq_factor != self.high_freq_factor:
            smooth = (self.orig_max_position / wave_len - self.low_freq_factor) / (
                self.high_freq_factor - self.low_freq_factor
            )
        else:
            smooth = 0
        new_freqs = torch.where(
            wave_len < high_freq_wavelen,
            inv_freqs,
            torch.where(
                wave_len > low_freq_wavelen,
                inv_freqs / self.scaling_factor,
                (1 - smooth) * inv_freqs / self.scaling_factor + smooth * inv_freqs,
            ),
        )
        return new_freqs


def _yarn_find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low, high, dim, dtype=torch.float32):
    if low == high:
        high += 0.001
    linear = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    return torch.clamp(linear, 0, 1)


def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class YaRNScalingRotaryEmbedding(RotaryEmbedding):
    def __init__(self, head_size, rotary_dim, orig_max_position, base, is_neox_style, scaling_factor,
                 dtype=None, *, extrapolation_factor=1.0, attn_factor=1.0, beta_fast=32, beta_slow=1,
                 mscale=1.0, mscale_all_dim=0.0):
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = float(
            _yarn_get_mscale(scaling_factor, mscale) / _yarn_get_mscale(scaling_factor, mscale_all_dim) * attn_factor
        )
        super().__init__(head_size, rotary_dim, int(orig_max_position * scaling_factor), base, is_neox_style, dtype)
        self.orig_max_position = orig_max_position

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:  # type: ignore[override]
        pos_freqs = self.base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scaling_factor * pos_freqs)
        low, high = _yarn_find_correction_range(
            self.beta_fast, self.beta_slow, self.rotary_dim, self.base,
            self.max_position_embeddings // int(self.scaling_factor) if self.scaling_factor >= 1 else self.max_position_embeddings,
        )
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2)) * self.extrapolation_factor
        return inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        return torch.cat((freqs.cos() * self.mscale, freqs.sin() * self.mscale), dim=-1)


_ROPE_CACHE: Dict[Any, RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
) -> RotaryEmbedding:
    dtype = dtype or torch.get_default_dtype()
    key = (head_size, rotary_dim, max_position, base, is_neox_style,
           tuple(sorted(rope_scaling.items())) if rope_scaling else None, dtype)
    if key in _ROPE_CACHE:
        return _ROPE_CACHE[key]

    if rope_scaling is None or rope_scaling.get("rope_type", rope_scaling.get("type", "default")) == "default":
        emb = RotaryEmbedding(head_size, rotary_dim, max_position, base, is_neox_style, dtype)
    else:
        rtype = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if rtype == "llama3":
            emb = Llama3RotaryEmbedding(
                head_size, rotary_dim, max_position, base, is_neox_style, dtype,
                scaling_factor=rope_scaling["factor"],
                low_freq_factor=rope_scaling["low_freq_factor"],
                high_freq_factor=rope_scaling["high_freq_factor"],
                orig_max_position=rope_scaling["original_max_position_embeddings"],
            )
        elif rtype == "yarn":
            orig = rope_scaling["original_max_position_embeddings"]
            extra = {k: v for k, v in rope_scaling.items()
                     if k in ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim")}
            emb = YaRNScalingRotaryEmbedding(head_size, rotary_dim, orig, base, is_neox_style,
                                             rope_scaling["factor"], dtype, **extra)
        else:
            raise ValueError(f"Unsupported rope scaling type {rtype!r}")
    _ROPE_CACHE[key] = emb
    return emb

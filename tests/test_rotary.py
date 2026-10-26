"""Rotary inv_freq / scaling parity with transformers' rope init functions (default, llama3, yarn)."""

import torch
from transformers import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from lily.layers.rotary import get_rope

HEAD, BASE, MAXPOS = 128, 500000.0, 8192


def _hf_inv_freq(rope_type, rope_params):
    cfg = PretrainedConfig()
    params = {"rope_type": rope_type, "rope_theta": BASE, **rope_params}
    cfg.rope_parameters = params
    cfg.rope_scaling = params
    cfg.rope_theta = BASE
    cfg.head_dim = HEAD
    cfg.hidden_size = HEAD * 8
    cfg.num_attention_heads = 8
    cfg.max_position_embeddings = MAXPOS
    cfg.partial_rotary_factor = 1.0
    return ROPE_INIT_FUNCTIONS[rope_type](cfg, "cpu")


def _ours_inv_freq_and_scale(rope_scaling):
    emb = get_rope(HEAD, HEAD, MAXPOS, BASE, True, rope_scaling, dtype=torch.float32)
    cache = emb.cos_sin_cache  # [maxpos, HEAD] = [cos | sin]
    cos1 = cache[1, : HEAD // 2]
    sin1 = cache[1, HEAD // 2 :]
    scale = torch.sqrt(cos1**2 + sin1**2)[0].item()
    angle = torch.atan2(sin1, cos1)  # inv_freq * 1 (mod 2pi; fine for small freqs)
    return angle, scale


def test_default():
    inv = 1.0 / (BASE ** (torch.arange(0, HEAD, 2, dtype=torch.float32) / HEAD))
    factor = 1.0
    angle, scale = _ours_inv_freq_and_scale(None)
    torch.testing.assert_close(angle, inv, atol=1e-5, rtol=1e-5)
    assert abs(scale - factor) < 1e-5


def test_llama3():
    p = {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
    inv, factor = _hf_inv_freq("llama3", p)
    angle, scale = _ours_inv_freq_and_scale({"rope_type": "llama3", **p})
    torch.testing.assert_close(angle, inv, atol=1e-5, rtol=1e-5)
    assert abs(scale - factor) < 1e-5


def test_yarn():
    p = {"factor": 4.0, "original_max_position_embeddings": 8192}
    inv, factor = _hf_inv_freq("yarn", p)
    angle, scale = _ours_inv_freq_and_scale({"rope_type": "yarn", **p})
    torch.testing.assert_close(angle, inv, atol=1e-5, rtol=1e-5)
    assert abs(scale - factor) < 1e-4, (scale, factor)

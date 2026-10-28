"""Parity tests for the gated delta-rule ops in lily/layers/attention/gated_delta_rule.py.

- chunked scan == token-by-token recurrence (same math, discretized differently)
- conv1d state bookkeeping: fresh prefill, continuation, and decode step agree
"""

import pytest
import torch

from lily.layers.attention.gated_delta_rule import (
    causal_conv1d_step,
    causal_conv1d_stream,
    chunked_gated_delta_rule,
    recurrent_gated_delta_rule,
)

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def _inputs(L, h=4, k_dim=32, v_dim=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(1, L, h, k_dim, generator=g),
        torch.randn(1, L, h, k_dim, generator=g),
        torch.randn(1, L, h, v_dim, generator=g),
        -torch.rand(1, L, h, generator=g).exp(),   # g < 0 (log decay)
        torch.rand(1, L, h, generator=g),          # beta in [0, 1]
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("chunk_size", [16, 64])
def test_chunked_matches_recurrence(device, chunk_size):
    q, k, v, g, beta = (x.to(device) for x in _inputs(100))
    o_chunk = chunked_gated_delta_rule(q, k, v, g, beta, None, chunk_size=chunk_size)

    state = torch.zeros(1, q.shape[-2], q.shape[-1], v.shape[-1], device=device)
    steps = []
    for t in range(q.shape[1]):
        steps.append(recurrent_gated_delta_rule(q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], state))
    o_step = torch.stack(steps, dim=1)  # [1, L, h, v_dim]

    torch.testing.assert_close(o_chunk, o_step[0], rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_chunked_matches_recurrence_with_initial_state(device):
    q, k, v, g, beta = (x.to(device) for x in _inputs(70, seed=1))
    s0 = torch.randn(q.shape[-2], q.shape[-1], v.shape[-1], device=device) * 0.5

    o_chunk = chunked_gated_delta_rule(q, k, v, g, beta, s0, chunk_size=64)
    state = s0.unsqueeze(0).clone()
    steps = []
    for t in range(q.shape[1]):
        steps.append(recurrent_gated_delta_rule(q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], state))
    o_step = torch.stack(steps, dim=1)

    torch.testing.assert_close(o_chunk, o_step[0], rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("L", [1, 2, 3, 8])
def test_fresh_stream_state_is_left_padded(device, L):
    """Conv state after a fresh L-token prefill = last 4 of [pad | tokens]."""
    x = torch.randn(64, L, device=device)          # [conv_dim, L]
    w = torch.randn(64, 4, device=device)
    out, state = causal_conv1d_stream(x, None, w, 4)

    import torch.nn.functional as F
    padded = F.pad(x, (3, 0))
    ref = F.conv1d(padded, w.unsqueeze(1), padding=0, groups=64)
    torch.testing.assert_close(out, F.silu(ref))  # `out` already has silu applied
    torch.testing.assert_close(state, padded[:, -4:] if padded.shape[-1] >= 4 else padded)


@pytest.mark.parametrize("device", DEVICES)
def test_continuation_stream_matches_token_loop(device):
    """[prev | new] stream == fresh conv over the concatenated tokens."""
    x1 = torch.randn(64, 9, device=device)
    x2 = torch.randn(64, 5, device=device)
    w = torch.randn(64, 4, device=device)

    _, s1 = causal_conv1d_stream(x1, None, w, 4)
    out2, s2 = causal_conv1d_stream(x2, s1, w, 4)

    all_x = torch.cat([x1, x2], dim=-1)
    ref_out, ref_state = causal_conv1d_stream(all_x, None, w, 4)

    torch.testing.assert_close(out2, ref_out[:, 9:], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(s2, ref_state)


@pytest.mark.parametrize("device", DEVICES)
def test_decode_step_matches_stream(device):
    """One-token continuation via causal_conv1d_step == stream of 1 token."""
    x1 = torch.randn(5, 64, 9, device=device)  # [N, conv_dim, L]
    x2 = torch.randn(5, 64, device=device)     # [N, conv_dim] (decode: 1 token)
    w = torch.randn(64, 4, device=device)

    conv_state = torch.zeros(5, 64, 4, device=device)
    for i in range(5):
        _, s = causal_conv1d_stream(x1[i], None, w, 4)
        conv_state[i] = s

    stepped_state = conv_state.clone()
    out_step = causal_conv1d_step(x2, stepped_state, w)

    outs, states = [], []
    for i in range(5):
        o, s = causal_conv1d_stream(x2[i].unsqueeze(-1), conv_state[i].clone(), w, 4)
        outs.append(o[:, 0])
        states.append(s)
    torch.testing.assert_close(out_step, torch.stack(outs, dim=0), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(stepped_state, torch.stack(states, dim=0))

"""Gated delta-rule linear attention ops (Qwen3-Next / Qwen3.5 hybrid layers).

Eager PyTorch implementations of the four building blocks the transformers
reference uses, arranged for serving with a per-request state pool:

  - `causal_conv1d_step`    : decode-time conv update (one token per request)
  - `causal_conv1d_stream`  : extend-time conv over [prev_state | new tokens]
  - `recurrent_gated_delta_rule` : one recurrence step (decode) / 1-token seqs
  - `chunked_gated_delta_rule`   : chunked scan for multi-token prefill
  - `gated_rmsnorm`         : RMSNormGated output norm (norm before gate, silu)

Shapes: everything is [batch, ...] where batch = the set of requests in the
step. State layouts match MambaStatePool exactly.

The math mirrors transformers (5.x) `torch_chunk_gated_delta_rule` /
`torch_recurrent_gated_delta_rule` / `causal_conv1d_*`, recomputed here
independently; `tests/test_hybrid_parity.py` checks chunked == recurrent.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


# ---------------------------------------------------------------------------
# Causal conv1d with a shared state pool
# ---------------------------------------------------------------------------

def causal_conv1d_step(hidden: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Decode update: one new token per request.

    hidden: [N, conv_dim] (batch-first); conv_state: [N, conv_dim, kernel];
    weight: [conv_dim, kernel].
    Updates conv_state in place and returns the silu-activated outputs [N, conv_dim].
    """
    kernel = conv_state.shape[-1]
    if weight.dim() == 2:  # [conv_dim, kernel] -> grouped conv layout [conv_dim, 1, kernel]
        weight = weight.unsqueeze(1)
    # The state pool may be fp32 (mamba_ssm_dtype) while weights are bf16:
    # compute in the activation dtype, write the state back in the pool dtype.
    stream = torch.cat([conv_state.to(hidden.dtype), hidden.unsqueeze(-1)], dim=-1)  # [N, conv_dim, kernel + 1]
    out = F.conv1d(stream, weight.to(hidden.dtype), padding=0, groups=stream.shape[1])
    conv_state.copy_(stream[:, :, -kernel:])
    return F.silu(out[:, :, -1])


def causal_conv1d_stream(
    hidden: torch.Tensor, prev_state: Optional[torch.Tensor], weight: torch.Tensor, kernel: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extend update for one request: new tokens' conv outputs with context.

    hidden: [conv_dim, L]; prev_state: [conv_dim, kernel] (continuation) or None
    (fresh prefill, where causal_conv1d pads the left context with zeros itself).
    Does not mutate `prev_state`; returns `(outputs [conv_dim, L], new_state
    [conv_dim, kernel])` — the caller writes `new_state` back to the pool.
    """
    L = hidden.shape[-1]
    if weight.dim() == 2:  # [conv_dim, kernel] -> grouped conv layout [conv_dim, 1, kernel]
        weight = weight.unsqueeze(1)
    if prev_state is not None:
        stream = torch.cat([prev_state.to(hidden.dtype), hidden], dim=-1)  # [conv_dim, kernel + L]
        new_state = stream[:, -kernel:]
    else:
        # Fresh prefill: causal_conv1d pads the left with kernel-1 zeros and
        # yields L outputs; the trailing kernel columns are the new state.
        stream = F.pad(hidden, (kernel - 1, 0))
        new_state = stream[:, -kernel:]
    out = F.conv1d(stream, weight.to(stream.dtype), padding=0, groups=stream.shape[0])
    out = out[..., -L:]   # continuation yields kernel+L windows; keep the last L
    return F.silu(out), new_state


# ---------------------------------------------------------------------------
# Recurrent (single-token) gated delta rule
# ---------------------------------------------------------------------------

def recurrent_gated_delta_rule(
    q: torch.Tensor,          # [N, num_v_heads, k_head_dim] bf16
    k: torch.Tensor,          # [N, num_v_heads, k_head_dim]
    v: torch.Tensor,          # [N, num_v_heads, v_head_dim]
    g: torch.Tensor,          # [N, num_v_heads] log decay (<= 0)
    beta: torch.Tensor,       # [N, num_v_heads] in [0, 1]
    state: torch.Tensor,      # [N, num_v_heads, k_head_dim, v_head_dim]
) -> torch.Tensor:
    """One recurrence step. Updates `state` in place; returns o [N, v_head_dim *? per-token output rows].

    o must be reshaped by the caller (returns [N, num_v_heads, v_head_dim]).
    """
    q, k, v = q.to(state.dtype), k.to(state.dtype), v.to(state.dtype)
    q = l2norm(q)
    k = l2norm(k)
    q = q * (q.shape[-1] ** -0.5)

    decay = g.exp()[..., None, None]
    state.mul_(decay)
    kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)          # [N, h, v_dim]
    delta = (v - kv_mem) * beta.unsqueeze(-1)                # [N, h, v_dim]
    state.add_(k.unsqueeze(-1) * delta.unsqueeze(-2))
    o = (state * q.unsqueeze(-1)).sum(dim=-2)                # [N, h, v_dim]
    return o


# ---------------------------------------------------------------------------
# Chunked (prefill) gated delta rule
# ---------------------------------------------------------------------------

def chunked_gated_delta_rule(
    q: torch.Tensor,         # [1, L, num_v_heads, k_head_dim] bf16
    k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],          # [num_v_heads, k_head_dim, v_head_dim] or None
    final_state_out: Optional[torch.Tensor] = None, # write target for the new state
    chunk_size: int = 64,
) -> torch.Tensor:
    """Multi-token prefill of the gated delta rule for a single request.

    Returns the per-token outputs [L, num_v_heads, v_head_dim] (bf16).
    """
    initial_dtype = q.dtype
    h, k_dim = q.shape[-2], q.shape[-1]
    v_dim = v.shape[-1]
    L = q.shape[1]

    q, k, v, beta, g = [x.transpose(1, 2).float().contiguous() for x in (q, k, v, beta, g)]
    q, k = l2norm(q), l2norm(k)
    q = q * (k_dim ** -0.5)

    pad = (chunk_size - L % chunk_size) % chunk_size
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    L_pad = L + pad

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)

    q, k, k_beta, v_beta = [x.view(1, h, -1, chunk_size, x.shape[-1]) for x in (q, k, k_beta, v_beta)]
    g = g.view(1, h, -1, chunk_size)

    tri = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).triu(1)
    cum_decay = g.cumsum(dim=3)
    pairwise = (cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)).masked_fill(tri, float("-inf")).exp()

    ut = (k_beta @ k.transpose(-1, -2)) * pairwise
    attn_intra = (q @ k.transpose(-1, -2)) * pairwise
    decayed_kb = k_beta * cum_decay.exp().unsqueeze(-1)

    v_new = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut, decayed_kb, upper=False, unitriangular=True)

    state = (
        torch.zeros((1, h, k_dim, v_dim), dtype=torch.float32, device=q.device)
        if initial_state is None else initial_state.unsqueeze(0).to(torch.float32)
    )

    q = q * cum_decay.exp().unsqueeze(-1)
    k = k * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    o = torch.empty(1, h, L_pad // chunk_size, chunk_size, v_dim, dtype=torch.float32, device=q.device)
    for i in range(L_pad // chunk_size):
        v_new_i = v_new[:, :, i] - k_cumdecay[:, :, i] @ state
        o[:, :, i] = q[:, :, i] @ state + attn_intra[:, :, i] @ v_new_i
        state = state * chunk_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ v_new_i
    state = state.squeeze(0)
    if final_state_out is not None:
        final_state_out.copy_(
            state.unsqueeze(0) if final_state_out.dim() == 4 else state
        )
    o = o.view(1, h, L_pad, v_dim)[0, :, :L].transpose(0, 1)
    return o.contiguous().to(initial_dtype)


def gated_rmsnorm(x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNormGated: norm in fp32 before the weight, then silu(gate)."""
    dtype = x.dtype
    x32 = x.float()
    out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    out = weight * out.to(dtype)
    return out * F.silu(gate.float()).to(dtype)

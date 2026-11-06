"""Mixture-of-experts FFN (GLM-5.3-Flash / DeepSeek-V3 style routing) with NVFP4 experts.

Routing (mirrors transformers' Glm5NextTextTopkRouter with n_group = 1):
    scores = sigmoid(x @ W_router^T)                      (fp32)
    topk   = top-k of (scores + e_score_correction_bias)
    weights = scores[topk] / sum (if norm_topk_prob) * routed_scaling_factor
Experts: SwiGLU with the gate clamped at +limit and up at +-limit (`swiglu_limit`), then
down_proj; plus one shared expert (an ordinary clamped-SwiGLU MLP) added to the sum.

Expert weights are kept stacked, [E, 2I, K/2] (gate|up) and [E, H, I/2] (down) in NVFP4,
and run through one grouped tensor-core GEMM per projection per layer: tokens are sorted by
expert on the device (static shapes, CUDA-graph safe), every active expert streams its
weights once for all the tokens that chose it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from lily.distributed import get_tp_info, tensor_model_parallel_all_reduce
from lily.layers.quantization import cuda_ops
from lily.utils import run_beside
from lily.layers.quantization.base import NVFP4_GROUP_SIZE, nvfp4_dequantize


def clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: Optional[float]) -> torch.Tensor:
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


class TopkRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int, routed_scaling_factor: float,
                 norm_topk_prob: bool = True):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        # kept in fp32 like the reference (router logits are computed in fp32)
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size, dtype=torch.float32), requires_grad=False)
        self.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts, dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x [T, H] -> (weights fp32 [T, k], expert ids int64 [T, k])."""
        logits = F.linear(x.float(), self.weight)
        scores = torch.sigmoid(logits)
        idx = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, idx)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * self.routed_scaling_factor, idx


class NVFP4Experts(nn.Module):
    """Stacked NVFP4 expert weights. Loaded per expert (`experts.{e}.{gate,up,down}_proj.*`)."""

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        E, H, I = num_experts, hidden_size, intermediate_size
        assert H % 128 == 0 and I % 128 == 0, "grouped NVFP4 GEMM needs K % 128 == 0"
        self.num_experts, self.hidden_size, self.intermediate_size = E, H, I
        g = NVFP4_GROUP_SIZE
        self.gate_up_packed = nn.Parameter(torch.empty(E, 2 * I, H // 2, dtype=torch.uint8), requires_grad=False)
        self.gate_up_scale = nn.Parameter(torch.empty(E, 2 * I, H // g, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.gate_up_gscale = nn.Parameter(torch.ones(E * 2 * I, dtype=torch.float32), requires_grad=False)
        self.down_packed = nn.Parameter(torch.empty(E, H, I // 2, dtype=torch.uint8), requires_grad=False)
        self.down_scale = nn.Parameter(torch.empty(E, H, I // g, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.down_gscale = nn.Parameter(torch.ones(E * H, dtype=torch.float32), requires_grad=False)

    def load_expert(self, e: int, proj: str, kind: str, tensor: torch.Tensor) -> None:
        """proj in gate/up/down; kind in weight_packed/weight_scale/weight_global_scale."""
        I = self.intermediate_size
        if proj == "down":
            dst = {"weight_packed": self.down_packed[e], "weight_scale": self.down_scale[e],
                   "weight_global_scale": self.down_gscale.view(self.num_experts, -1)[e]}[kind]
        else:
            off = 0 if proj == "gate" else I
            dst = {"weight_packed": self.gate_up_packed[e, off:off + I], "weight_scale": self.gate_up_scale[e, off:off + I],
                   "weight_global_scale": self.gate_up_gscale.view(self.num_experts, -1)[e, off:off + I]}[kind]
        if kind == "weight_global_scale":
            dst.fill_(tensor.reshape(()).item() if tensor.numel() == 1 else tensor)
        else:
            dst.copy_(tensor.view(dst.shape) if tensor.dtype == dst.dtype else tensor.view(dst.dtype).view(dst.shape))

    def dequantize(self, e: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """bf16 (gate_up [2I, H], down [H, I]) of one expert; reference / fallback path."""
        E = self.num_experts
        return (nvfp4_dequantize(self.gate_up_packed[e], self.gate_up_scale[e], self.gate_up_gscale.view(E, -1)[e]),
                nvfp4_dequantize(self.down_packed[e], self.down_scale[e], self.down_gscale.view(E, -1)[e]))


def sort_by_expert(expert_ids: torch.Tensor, num_experts: int):
    """[T, k] expert ids -> (active [E] int32 with -1 padding, offsets [E+1] int32, tokens [T*k] int32,
    order [T*k] int64): the (token, slot) pairs sorted by expert, CSR offsets per expert, and the
    permutation mapping sorted rows back to (token, slot). Static shapes: graph-safe."""
    T, k = expert_ids.shape
    flat = expert_ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    sorted_e = flat[order]
    # (bincount syncs with the host to size its output; scatter_add is CUDA-graph safe)
    counts = torch.zeros(num_experts, dtype=torch.int64, device=flat.device).scatter_add_(0, sorted_e, torch.ones_like(sorted_e))
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
    offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)
    tokens = (order // k).to(torch.int32)
    # experts with work, in a fixed-size list: expert ids sorted by (count == 0) so active ones come first
    has = counts > 0
    active = torch.where(has, torch.arange(num_experts, device=flat.device), torch.full_like(counts, num_experts))
    active = torch.sort(active).values.to(torch.int32)
    active = torch.where(active >= num_experts, torch.full_like(active, -1), active)
    return active, offsets, tokens, order


class NVFP4MoE(nn.Module):
    """Under tensor parallelism every rank holds a 1/tp slice of every expert's intermediate dim
    (gate/up rows, down columns) and one all-reduce sums the ranks. Expert parallelism would read
    less per rank on average, but a decode token's few experts land unevenly and every rank then
    waits for the busiest one; with slices all ranks read the same bytes. The router is replicated
    and the shared expert, itself row-parallel, is folded into the same all-reduce (build it with
    `reduce_results=False`)."""

    SMALL_BATCH = 64     # tokens the fused routing path takes; the route kernel is one block

    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int, top_k: int,
                 routed_scaling_factor: float, norm_topk_prob: bool, swiglu_limit: Optional[float],
                 shared_expert: Optional[nn.Module]):
        super().__init__()
        tp_size, _ = get_tp_info()
        assert intermediate_size % tp_size == 0, (intermediate_size, tp_size)
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size // tp_size
        self.num_experts, self.top_k = num_experts, top_k
        self.swiglu_limit = swiglu_limit
        self.gate = TopkRouter(hidden_size, num_experts, top_k, routed_scaling_factor, norm_topk_prob)
        self.experts = NVFP4Experts(num_experts, hidden_size, self.intermediate_size)
        self.shared_experts = shared_expert
        self._arange: dict = {}

    def _rows(self, n: int, device) -> torch.Tensor:
        """arange(n), kept so a CUDA graph reads one stable tensor."""
        rows = self._arange.get(n)
        if rows is None:
            rows = self._arange[n] = torch.arange(n, dtype=torch.int32, device=device)
        return rows

    def _forward_small_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Decode and verify batches: same maths as `forward` through the fused routing kernels."""
        g, ex = self.gate, self.experts

        def routed():
            logits = F.linear(x.float(), g.weight)
            weights, active, offsets, tokens, inv = cuda_ops.moe_route(
                logits, g.e_score_correction_bias, self.top_k, g.routed_scaling_factor, g.norm_topk_prob, 0, self.num_experts)
            R = x.shape[0] * self.top_k
            active = active[:R]             # at most one expert per row: the GEMM's grid needs no more
            gate_up = cuda_ops.nvfp4_moe_gemm(x, ex.gate_up_packed, ex.gate_up_scale.view(torch.uint8), ex.gate_up_gscale,
                                              active, offsets, tokens, R)
            act = cuda_ops.moe_swiglu(gate_up, self.swiglu_limit)
            return cuda_ops.nvfp4_moe_gemm(act, ex.down_packed, ex.down_scale.view(torch.uint8), ex.down_gscale,
                                           active, offsets, self._rows(R, x.device), R), inv, weights

        # the shared expert hides routing, whose kernels leave most of the GPU idle
        shared, (down, inv, weights) = run_beside(lambda: self.shared_experts(x) if self.shared_experts is not None else None, routed)
        return tensor_model_parallel_all_reduce(cuda_ops.moe_combine(down, inv, weights, shared))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        if x.is_cuda and cuda_ops.nvfp4_moe_supported(x.shape[1]) and T <= self.SMALL_BATCH \
                and cuda_ops.moe_route_supported(T, self.top_k, self.num_experts):
            return self._forward_small_batch(x)
        weights, ids = self.gate(x)                                     # [T, k]
        active, offsets, tokens, order = sort_by_expert(ids, self.num_experts)
        R = T * self.top_k
        ex = self.experts
        if cuda_ops.available() and cuda_ops.nvfp4_moe_supported(x.shape[1]):
            gate_up = cuda_ops.nvfp4_moe_gemm(x, ex.gate_up_packed, ex.gate_up_scale.view(torch.uint8), ex.gate_up_gscale,
                                              active, offsets, tokens, R)                     # [R, 2I]
            act = clamped_swiglu(gate_up[:, :self.intermediate_size], gate_up[:, self.intermediate_size:], self.swiglu_limit)
            # the down projection's "tokens" are the sorted rows themselves
            down = cuda_ops.nvfp4_moe_gemm(act.contiguous(), ex.down_packed, ex.down_scale.view(torch.uint8), ex.down_gscale,
                                           active, offsets, self._rows(R, x.device), R)         # [R, H]
        else:
            down = self._eager(x, ids, order)
        # back to token order: sorted row r is (token, slot) = divmod(order[r], k). A gather and a
        # fixed-order sum over the k slots (not index_add_'s atomics) keeps the result deterministic.
        inv = torch.empty_like(order)
        inv[order] = torch.arange(R, device=x.device)
        per_slot = down[inv].view(T, self.top_k, self.hidden_size)
        out = (per_slot * weights.to(down.dtype).unsqueeze(-1)).sum(dim=1)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return tensor_model_parallel_all_reduce(out)

    def _eager(self, x, ids, order):
        """Dequantized per-expert path (tests, CPU, odd shapes)."""
        T, k = ids.shape
        flat = ids.reshape(-1)[order]
        tok = order // k
        down = torch.zeros(T * k, self.hidden_size, dtype=x.dtype, device=x.device)
        for e in flat.unique().tolist():
            rows = (flat == e).nonzero(as_tuple=True)[0]
            gu, dn = self.experts.dequantize(e)
            h = F.linear(x[tok[rows]], gu)
            a = clamped_swiglu(h[:, :self.intermediate_size], h[:, self.intermediate_size:], self.swiglu_limit)
            down[rows] = F.linear(a, dn)
        return down

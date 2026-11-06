"""CUDA kernels (lily/csrc), JIT-compiled on first use: NVFP4 decode GEMV and the fused
Gated DeltaNet layer step.

`torch.utils.cpp_extension.load` builds the extension with the system nvcc for
the GPUs visible in this process (or `TORCH_CUDA_ARCH_LIST`) and caches the
binary under `LILY_CUDA_BUILD_DIR` (default ~/.cache/lily/cuda). A failed build
logs a warning and every wrapper returns None so callers fall back to Triton.

Environment:
  LILY_DISABLE_CUDA_KERNELS=1  never build or use the CUDA kernels
  LILY_CUDA_BUILD_DIR=<dir>    where the compiled extension is cached
  LILY_CUDA_VERBOSE=1          print the nvcc command lines
  LILY_CUDA_BUILD_ONLY=1       build without a GPU (needs TORCH_CUDA_ARCH_LIST), e.g. in an image build
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_CSRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "csrc")
_SOURCES = [os.path.join(_CSRC, f) for f in ("bindings.cpp", "nvfp4_gemv.cu", "nvfp4_gemm_tc.cu", "nvfp4_moe.cu", "gdn_fused.cu",
                                                "kda_fused.cu", "mla_decode.cu", "hc_fused.cu", "moe_fused.cu", "bf16_skinny.cu",
                                                "small_fused.cu")]

# Row-count routing for NVFP4 linears: the CUDA-core GEMV streams weights at HBM
# bandwidth up to 4 rows; 5..16 rows go to the tensor-core GEMM (one dequant per
# weight whatever M); larger M falls back to the Triton tiled GEMM.
GEMV_MAX_M = 4
GEMM_TC_MAX_M = 16


@functools.lru_cache(maxsize=None)
def _module():
    if os.environ.get("LILY_DISABLE_CUDA_KERNELS"):
        return None
    # LILY_CUDA_BUILD_ONLY: compile on a GPU-less machine (container image build) for the
    # archs in TORCH_CUDA_ARCH_LIST; set the same TORCH_CUDA_ARCH_LIST at run time so ninja
    # finds the objects up to date.
    if not torch.cuda.is_available() and not os.environ.get("LILY_CUDA_BUILD_ONLY"):
        return None
    from torch.utils.cpp_extension import load

    build_dir = os.environ.get("LILY_CUDA_BUILD_DIR", os.path.expanduser("~/.cache/lily/cuda"))
    os.makedirs(build_dir, exist_ok=True)
    try:
        mod = load(
            name="lily_cuda",
            sources=_SOURCES,
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo", "-allow-unsupported-compiler"],
            extra_cflags=["-O3", "-std=c++20"],
            build_directory=build_dir,
            verbose=bool(os.environ.get("LILY_CUDA_VERBOSE")),
        )
    except Exception as exc:  # nvcc missing, unsupported host compiler, ...
        logger.warning("lily CUDA kernels unavailable, using Triton fallbacks: %s", str(exc).splitlines()[-1] if str(exc) else exc)
        return None
    logger.info("lily CUDA kernels loaded from %s", build_dir)
    return mod


def available() -> bool:
    return _module() is not None


def nvfp4_gemv(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor,
               global_scale: torch.Tensor) -> Optional[torch.Tensor]:
    """`x[..., K] @ dequant(w)^T` for up to 16 rows (GEMV then tensor-core GEMM), or None when no CUDA path applies.

    w: uint8 [N, K/2] packed E2M1; scale: uint8 view of the e4m3 [N, K/16] block
    scales; global_scale: fp32 [N] or [1]. Returns bf16 [..., N].
    """
    mod = _module()
    if mod is None or x.dtype is not torch.bfloat16:
        return None
    x2 = x.reshape(-1, x.shape[-1])
    m, k = x2.shape
    if not x2.is_contiguous() or x2.data_ptr() % 16 or w.data_ptr() % 16:
        return None
    if m <= GEMV_MAX_M and k % 32 == 0:
        out = mod.nvfp4_gemv(x2, w, scale, global_scale)
    elif m <= GEMM_TC_MAX_M and mod.nvfp4_gemm_tc_supported(m, k):
        out = mod.nvfp4_gemm_tc(x2, w, scale, global_scale)
    else:
        return None
    return out.view(*x.shape[:-1], w.shape[0])


def nvfp4_gemm_tc(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor) -> Optional[torch.Tensor]:
    """The tensor-core path alone (tests / benchmarks); None if unsupported."""
    mod = _module()
    x2 = x.reshape(-1, x.shape[-1])
    if mod is None or not mod.nvfp4_gemm_tc_supported(x2.shape[0], x2.shape[1]):
        return None
    return mod.nvfp4_gemm_tc(x2.contiguous(), w, scale, global_scale).view(*x.shape[:-1], w.shape[0])


def gdn_fused_supported(k_head_dim: int, v_head_dim: int, kernel_size: int, state_dtype: torch.dtype) -> bool:
    mod = _module()
    return mod is not None and mod.gdn_fused_supported(k_head_dim, v_head_dim, kernel_size, state_dtype)


def gdn_fused_forward(proj: torch.Tensor, conv_w: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor,
                      norm_w: torch.Tensor, eps: float, tok_starts: torch.Tensor, tok_lens: torch.Tensor,
                      slots: torch.Tensor, fresh: Optional[torch.Tensor], conv_in: torch.Tensor, conv_out: torch.Tensor,
                      rec_in: torch.Tensor, rec_out: torch.Tensor, conv_snap: Optional[torch.Tensor],
                      rec_snap: Optional[torch.Tensor], num_k_heads: int) -> torch.Tensor:
    """One fused Gated DeltaNet layer step over a varlen batch; see csrc/gdn_fused.cu.

    proj: bf16 [T, >= conv_dim + nv*VD + 2*nv] merged in_proj output laid out as
    [q | k | v | z | b | a]; request b owns rows [tok_starts[b], tok_starts[b] + tok_lens[b]).
    Returns bf16 [T, nv*VD] (normed, gated; feed to out_proj).
    States are updated in place (`*_out`), optionally from a different `*_in`
    buffer; `*_snap` receives the pre-update state when given.
    """
    return _module().gdn_fused_forward(proj, conv_w, A_log, dt_bias, norm_w, float(eps), tok_starts, tok_lens, slots, fresh,
                                       conv_in, conv_out, rec_in, rec_out, conv_snap, rec_snap, int(num_k_heads))


GDN_REPLAY_COLUMNS = ("proj", "conv_w", "A_log", "dt_bias", "conv_in", "conv_out", "rec_in", "rec_out")


def gdn_fused_replay(table: torch.Tensor, proj_stride: int, tok_starts: torch.Tensor, tok_lens: torch.Tensor, slots: torch.Tensor,
                     num_k_heads: int, num_v_heads: int, head_dim: int, kernel_size: int, state_dtype: torch.dtype) -> None:
    """Re-run every layer's recurrence over the accepted prefix in one launch.

    `table` is int64 [layers, 8] of data pointers in GDN_REPLAY_COLUMNS order
    (proj rows are the accepted tokens' merged projections, request-major).
    """
    _module().gdn_fused_replay(table, int(proj_stride), tok_starts, tok_lens, slots, int(num_k_heads), int(num_v_heads),
                               int(head_dim), int(kernel_size), state_dtype)


# ---- Kimi Delta Attention (GLM-5.3-Flash linear attention) ----------------------------

KDA_REPLAY_COLUMNS = ("proj", "beta", "fgate", "conv_w", "A_log", "dt_bias", "conv_in", "conv_out", "rec_in", "rec_out")


def kda_fused_supported(head_dim: int, kernel_size: int, state_dtype: torch.dtype) -> bool:
    mod = _module()
    return mod is not None and mod.kda_fused_supported(head_dim, kernel_size, state_dtype)


def kda_fused_forward(proj, gate, beta_raw, fgate, conv_w, A_log, dt_bias, lower_bound: float, norm_w, eps: float,
                      tok_starts, tok_lens, slots, fresh, conv_in, conv_out, rec_in, rec_out, conv_snap, rec_snap):
    """One fused KDA layer step; see csrc/kda_fused.cu. proj = [q | k | v] (bf16), gate = g_b(g_a(x)),
    beta_raw = b_proj(x), fgate = f_b(f_a(x)); A_log/dt_bias/norm_w fp32. Returns bf16 [T, H*D]."""
    return _module().kda_fused_forward(proj, gate, beta_raw, fgate, conv_w, A_log, dt_bias, float(lower_bound), norm_w,
                                       float(eps), tok_starts, tok_lens, slots, fresh, conv_in, conv_out, rec_in, rec_out,
                                       conv_snap, rec_snap)


def kda_fused_replay(table, proj_stride: int, beta_stride: int, fgate_stride: int, lower_bound: float,
                     tok_starts, tok_lens, slots, num_heads: int, head_dim: int, kernel_size: int, state_dtype: torch.dtype) -> None:
    _module().kda_fused_replay(table, int(proj_stride), int(beta_stride), int(fgate_stride), float(lower_bound),
                               tok_starts, tok_lens, slots, int(num_heads), int(head_dim), int(kernel_size), state_dtype)


# ---- bf16 linears of decode and verify batches ---------------------------------------------

def l2_prefetch(w: torch.Tensor) -> None:
    """Pull `w` into L2; a few blocks are enough and more would slow what runs beside them."""
    _module().l2_prefetch(w, 32)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> Optional[torch.Tensor]:
    """w * bf16(x * rsqrt(mean(x^2) + eps)) over the last dim of a bf16 tensor; None when unavailable."""
    mod = _module()
    if mod is None or not x.is_cuda or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or x.shape[-1] % 8:
        return None
    rows = x.numel() // x.shape[-1]
    return mod.rmsnorm_fused(x.reshape(rows, -1).contiguous(), w, float(eps)).view(x.shape)


def dyn_conv2(x: torch.Tensor, dyn: torch.Tensor, dyn_off: int, base: torch.Tensor, group: int) -> Optional[torch.Tensor]:
    """The DFlash drafter's two-tap conv: x [B, L, H] bf16, dyn [B, L, D] (the taps' per-group kernels at
    columns dyn_off.. laid out [2, H/group]), base [2, H]. None when unavailable."""
    mod = _module()
    if mod is None or not x.is_cuda or x.dtype != torch.bfloat16 or dyn.dtype != torch.bfloat16 or x.shape[-1] % 8 or group % 8:
        return None
    return mod.dyn_conv2(x.contiguous(), dyn, int(dyn_off), base.contiguous(), int(group))


def rope_neox(q: torch.Tensor, k: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor):
    """Rotate q [N, Hq, D] and k [N, Hk, D] (bf16, rotary over the whole head) -> (q, k) copies; None when unavailable."""
    mod = _module()
    if (mod is None or not q.is_cuda or q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or cos_sin.dtype != torch.bfloat16
            or positions.dtype != torch.int64 or q.shape[-1] % 2):
        return None
    return mod.rope_neox(q.contiguous(), k.contiguous(), cos_sin, positions.contiguous())


def select_greedy(unary: torch.Tensor, cand: torch.Tensor, proj: torch.Tensor, anchor: torch.Tensor,
                  pred_cb: torch.Tensor, succ_cb: torch.Tensor):
    """The DFlash candidate selector's greedy path -> (path [B, S] int64, one-hot q [B, S, K] fp32); None when unavailable."""
    mod = _module()
    if mod is None or not unary.is_cuda or not mod.select_greedy_supported(unary.shape[-1], proj.shape[-1]):
        return None
    return mod.select_greedy(unary.contiguous(), cand.contiguous(), proj.contiguous(), anchor.contiguous(), pred_cb, succ_cb)


def bf16_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x @ w^T: the weight-streaming kernel for small batches, cuBLAS otherwise."""
    mod = _module()
    rows = x.numel() // x.shape[-1] if x.shape[-1] else 0
    strided = x.dim() == 2 and x.stride(1) == 1 and x.stride(0) % 8 == 0       # a column slice of a wider output
    if (mod is not None and x.is_cuda and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and (strided or x.is_contiguous())
            and w.is_contiguous() and mod.bf16_skinny_supported(rows, w.shape[0], x.shape[-1])):
        return mod.bf16_skinny(x if strided else x.view(rows, -1), w).view(*x.shape[:-1], w.shape[0])
    return torch.nn.functional.linear(x, w)


# ---- MoE experts ------------------------------------------------------------------------

def nvfp4_moe_supported(k: int) -> bool:
    mod = _module()
    return mod is not None and mod.nvfp4_moe_supported(int(k))


def nvfp4_moe_gemm(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor, gscale: torch.Tensor,
                   active_experts: torch.Tensor, offsets: torch.Tensor, tokens: torch.Tensor, num_rows: int) -> torch.Tensor:
    """out[r] = x[tokens[r]] @ dequant(w[e])^T for the rows r of expert e (CSR `offsets`), every
    active expert's weights streamed once. w [E, N, K/2] uint8, scale [E, N, K/16] uint8 view of
    e4m3, gscale fp32 [E] or [E*N]. Returns bf16 [num_rows, N]."""
    return _module().nvfp4_moe_gemm(x, w, scale, gscale, active_experts, offsets, tokens, int(num_rows))


def moe_route_supported(tokens: int, top_k: int, num_experts: int) -> bool:
    mod = _module()
    return mod is not None and mod.moe_route_supported(int(tokens), int(top_k), int(num_experts))


def moe_route(logits: torch.Tensor, bias: torch.Tensor, top_k: int, scaling: float, norm: bool, offset: int, local_experts: int):
    """Router logits fp32 [T, E] -> (weights fp32 [T, k], active, offsets, tokens, inv) int32: this
    rank's (token, slot) pairs sorted by expert as `nvfp4_moe_gemm` takes them; inv[t * k + s] is the
    pair's sorted row, -1 for an expert of another rank."""
    return _module().moe_route(logits, bias, int(top_k), float(scaling), bool(norm), int(offset), int(local_experts))


def moe_swiglu(gate_up: torch.Tensor, limit: Optional[float]) -> torch.Tensor:
    """bf16 [rows, 2I] (gate | up) -> [rows, I]: silu(min(gate, limit)) * clamp(up, +-limit)."""
    return _module().moe_swiglu(gate_up, float("inf") if limit is None else float(limit))


def moe_combine(down: torch.Tensor, inv: torch.Tensor, weights: torch.Tensor, shared: Optional[torch.Tensor]) -> torch.Tensor:
    """out[t] = sum_s weights[t, s] * down[inv[t, s]] (+ shared[t]); bf16 [T, H]."""
    return _module().moe_combine(down, inv, weights, shared)


# ---- mHC hyper-connections (GLM-5.3-Flash) ------------------------------------------------

def hc_fused_supported(hc_mult: int, hidden_size: int) -> bool:
    mod = _module()
    return mod is not None and mod.hc_fused_supported(int(hc_mult), int(hidden_size))


def hc_collapse(streams: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, norm_w: torch.Tensor,
                iters: int, hc_eps: float, rms_eps: float, norm_eps: float):
    """streams bf16 [T, 4, H] -> (post fp32 [T, 4], comb fp32 [T, 4, 4], collapsed bf16 [T, H],
    RMSNorm(collapsed) bf16 [T, H] with weight `norm_w`, epsilon `norm_eps`)."""
    return _module().hc_collapse(streams, fn, base, scale, norm_w, int(iters), float(hc_eps), float(rms_eps), float(norm_eps))


def hc_expand(block_out: torch.Tensor, post: torch.Tensor, comb: torch.Tensor, streams: torch.Tensor) -> torch.Tensor:
    """-> bf16 [T, 4, H]: post[:, :, None] * block_out[:, None, :] + comb^T @ streams."""
    return _module().hc_expand(block_out.contiguous(), post, comb, streams)


# ---- MLA decode + DSA selection (GLM-5.3-Flash) -------------------------------------------

MLA_DECODE_CHUNK = 32     # one warp walks a chunk serially: short enough that short contexts spread over SMs


def mla_decode_supported(latent_dim: int) -> bool:
    mod = _module()
    return mod is not None and mod.mla_decode_supported(int(latent_dim))


def mla_decode(q_lat: torch.Tensor, latent: torch.Tensor, req_to_token: torch.Tensor, req_pool_indices: torch.Tensor,
               seq_lens: torch.Tensor, vis_idx: Optional[torch.Tensor], vis_cnt: Optional[torch.Tensor], scale: float,
               chunks: int, chunk: int = MLA_DECODE_CHUNK) -> torch.Tensor:
    """Attention of one decode token per request over its cached latents (dense: positions
    0..seq_len-1; sparse: the `vis_cnt[b]` positions in `vis_idx[b]`). q_lat bf16 [B, H, D]
    (queries already absorbed into latent space); returns the softmax-weighted latent sum
    bf16 [B, H, D]. `chunks` fixes the launch shape (ceil(max visible / chunk)): static for graphs."""
    mod = _module()
    m, l, o = mod.mla_decode_partials(q_lat, latent, req_to_token, req_pool_indices, seq_lens, vis_idx, vis_cnt,
                                      float(scale), int(chunk), int(chunks))
    # combine the per-chunk online-softmax partials [B, chunks, H] / [B, chunks, H, D]: one kernel that reads
    # only each row's ceil(n / chunk) chunks (the launch shape is static for graphs, the context usually short)
    return mod.mla_decode_combine(m, l, o, seq_lens, vis_cnt, int(chunk))


def dsa_pool_scores_supported(head_dim: int, kpool: int) -> bool:
    mod = _module()
    return mod is not None and mod.dsa_pool_scores_supported(int(head_dim), int(kpool))


def dsa_pool_scores(index_cache: torch.Tensor, ape: torch.Tensor, req_to_token: torch.Tensor, req_pool_indices: torch.Tensor,
                    seq_lens: torch.Tensor, q_idx: torch.Tensor, w: torch.Tensor, scale: float, kpool: int, p_cap: int) -> torch.Tensor:
    """DSA indexer scores of every complete k-pool of each request for its newest token:
    fp32 [B, p_cap], -inf where the pool is incomplete or past the request's end."""
    return _module().dsa_pool_scores(index_cache, ape, req_to_token, req_pool_indices, seq_lens, q_idx.contiguous(),
                                     w.contiguous(), float(scale), int(kpool), int(p_cap))

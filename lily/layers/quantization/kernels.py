"""Triton fused dequant GEMM for weight-only quantized layers (see weight_only.py).

The kernel reads the quantized weight directly, dequantizes tiles in-register
and does the bf16 matmul, so no [N, K] dequantized matrix is ever materialized.
Callers fall back to a torch dequant path when shapes/dtypes don't fit the
kernel's assumptions.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK_M = 16  # tl.dot minimum; padded with masks
_BLOCK_N = 64
FAMILY_INT8 = tl.constexpr(0)
FAMILY_INT4 = tl.constexpr(1)
FAMILY_FP8 = tl.constexpr(2)
FAMILY_NVFP4 = tl.constexpr(3)
# fp8 e4m3fn subnormal base 2^-6; NaN (0x7f/0xff) never occurs because torch's
# cast saturates at 448, which is byte 0x7e.
FP8_SUBNORMAL = tl.constexpr(2.0 ** -6)


@triton.jit
def _decode_e2m1(nibble):
    """E2M1 code -> signed value on {0, 0.5, 1, 1.5, 2, 3, 4, 6} (bit 3 = sign).

    Codes 0-7 map to 0/0.5/1/1.5/2/3/4/6: e = a // 2, m = a % 2.
    e == 0 is the subnormal (implicit mantissa 0), e > 0 is (1 + m/2) * 2^(e-1).
    """
    a = nibble & 0x07
    m = (a & 0x01).to(tl.float32)
    e = (a >> 1).to(tl.float32)
    val = tl.where(a < 2, m * 0.5, (1.0 + 0.5 * m) * tl.exp2(e - 1.0))
    # sign lives in bit 3 of the *unmasked* nibble
    return val * tl.where((nibble & 0x08) != 0, -1.0, 1.0)


@triton.jit
def _decode_e4m3(b):
    """e4m3fn byte -> fp32 value (same math as the FAMILY_FP8 inline decode)."""
    b = b.to(tl.int32)
    sign = tl.where((b & 0x80) != 0, -1.0, 1.0)
    e = ((b >> 3) & 0x0F).to(tl.float32)
    m = (b & 0x07).to(tl.float32) / 8.0
    return sign * tl.where(e == 0, m * FP8_SUBNORMAL, tl.exp2(e - 7.0) * (1.0 + m))


@triton.jit
def _weight_only_matmul_kernel(x_ptr, w_ptr, scale_ptr, out_ptr,
                               M, N, K_REAL, K_STORED, SCALE_STRIDE, GLOBAL_SCALE, GS_STRIDE,
                               FAMILY: tl.constexpr,
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    bytes_per_row = K_STORED // 2

    for ki in range(0, tl.cdiv(K_STORED, BLOCK_K)):
        offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_REAL
        if FAMILY == FAMILY_INT4:
            offs_kb = ki * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
            b = tl.load(w_ptr + offs_n[:, None] * bytes_per_row + offs_kb[None, :], mask=n_mask[:, None], other=0)
            lo = ((b & 0x0F).to(tl.int8) - 8).to(tl.bfloat16)
            hi = ((b >> 4).to(tl.int8) - 8).to(tl.bfloat16)
            w_tile = tl.interleave(lo, hi)  # [BLOCK_N, BLOCK_K] bf16
            s = tl.load(scale_ptr + offs_n * SCALE_STRIDE + ki, mask=n_mask, other=0.0)
            w_tile = (w_tile.to(tl.float32) * s[:, None]).to(tl.bfloat16)
        elif FAMILY == FAMILY_NVFP4:
            # NVFP4: E2M1 codes packed two per byte (low nibble first), one e4m3
            # byte per 16 elements, fp32 global scale applied to `acc` at the end.
            # BLOCK_K must be a multiple of 16 so scale groups align.
            offs_kb = ki * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
            b = tl.load(w_ptr + offs_n[:, None] * bytes_per_row + offs_kb[None, :], mask=n_mask[:, None], other=0)
            w_tile = tl.interleave(_decode_e2m1(b & 0x0F), _decode_e2m1(b >> 4))  # [BLOCK_N, BLOCK_K] fp32
            w_tile = w_tile.reshape(BLOCK_N, BLOCK_K // 16, 16)
            offs_s = ki * (BLOCK_K // 16) + tl.arange(0, BLOCK_K // 16)
            sb = tl.load(scale_ptr + offs_n[:, None] * SCALE_STRIDE + offs_s[None, :],
                         mask=n_mask[:, None], other=0)
            s = _decode_e4m3(sb)
            w_tile = (w_tile * s[:, :, None]).reshape(BLOCK_N, BLOCK_K).to(tl.bfloat16)
        elif FAMILY == FAMILY_FP8:
            b = tl.load(w_ptr + offs_n[:, None] * K_STORED + offs_k[None, :],
                        mask=n_mask[:, None] & k_mask[None, :], other=0)
            w_tile = _decode_e4m3(b).to(tl.bfloat16)
            s = tl.load(scale_ptr + offs_n, mask=n_mask, other=1.0)
            w_tile = (w_tile.to(tl.float32) * s[:, None]).to(tl.bfloat16)
        else:
            w_tile = tl.load(w_ptr + offs_n[:, None] * K_STORED + offs_k[None, :],
                             mask=n_mask[:, None] & k_mask[None, :], other=0)
            w_tile = w_tile.to(tl.bfloat16)
        x_tile = tl.load(x_ptr + offs_m[:, None] * K_REAL + offs_k[None, :],
                         mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        acc = tl.dot(x_tile, tl.trans(w_tile), acc)

    if FAMILY == FAMILY_INT8:
        s = tl.load(scale_ptr + offs_n, mask=n_mask, other=1.0)
        acc = acc * s[None, :]
    elif FAMILY == FAMILY_NVFP4:
        # global_scale is an fp32 tensor read per program (no host sync): one
        # value per output row (GS_STRIDE=1) or a single [1] scalar (GS_STRIDE=0).
        # compressed-tensors convention: dequant = e2m1 * block_scale / global_scale
        gs = tl.load(GLOBAL_SCALE + offs_n * GS_STRIDE, mask=n_mask, other=1.0)
        acc = acc / gs[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def weight_only_matmul(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor,
                       family: int, group_size: int = 0,
                       global_scale: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """Fused dequant matmul, or None when the inputs don't fit the kernel's constraints.

    `family`: FAMILY_INT8 (int8 weight, per-channel scale), FAMILY_INT4 (uint8-packed
    int4, per-group scale, BLOCK_K == group_size), FAMILY_FP8 (e4m3 bytes, per-channel
    scale), FAMILY_NVFP4 (e2m1 nibbles, e4m3 per-16 scale, divided by fp32 global scale).
    `global_scale` (NVFP4) is a GPU fp32 tensor read inside the kernel: either
    a [1] scalar or one value per output row ([N]).
    """
    if x.dtype is not torch.bfloat16 or not x.is_contiguous() or not w.is_contiguous():
        return None
    n, dim1 = w.shape
    if n != scale.shape[0]:
        return None
    if family == FAMILY_NVFP4:
        if global_scale is None:
            global_scale = torch.ones(1, dtype=torch.float32, device=x.device)
        if global_scale.numel() not in (1, n) or not global_scale.is_contiguous():
            return None
        gs_stride = 0 if global_scale.numel() == 1 else 1
    else:
        gs_stride = 0
    # packed families (INT4, NVFP4) store K/2 bytes per row: stored column count is 2*dim1
    k_stored = 2 * dim1 if family in (FAMILY_INT4, FAMILY_NVFP4) else dim1
    k_real = x.shape[-1]
    if family == FAMILY_INT4:
        if k_real > k_stored:  # unpadded columns must fit inside the padded weight
            return None
    elif k_real != k_stored:
        return None
    if family == FAMILY_INT4:
        block_k = group_size
        if block_k % 2 or block_k < 16 or k_stored % block_k or (block_k & (block_k - 1)):
            return None
    elif family == FAMILY_NVFP4:
        block_k = max(16, min(64, triton.next_power_of_2(k_stored)))
        if k_stored % block_k:
            return None
    else:
        # Per-channel families (INT8 / FP8): K masking in the kernel handles
        # remainders; BLOCK_K must be a power of two for tl.arange.
        block_k = min(64, triton.next_power_of_2(k_stored))
        if block_k < 16:
            return None
    block_m, block_n = _BLOCK_M, _BLOCK_N
    x2 = x.view(-1, x.shape[-1])
    m = x2.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _weight_only_matmul_kernel[grid](
        x2, w, scale, out, m, n, x2.shape[1], k_stored,
        scale.stride(0) if scale.dim() > 1 else 0, global_scale if global_scale is not None else 1.0, gs_stride,
        FAMILY=family, BLOCK_M=block_m, BLOCK_N=block_n,
        BLOCK_K=block_k, num_warps=4,
    )
    return out.view(*x.shape[:-1], n)

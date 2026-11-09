// NVFP4 weight-only GEMM for a few activation rows (M <= 16) on tensor cores.
//
//   out[M, N] = x[M, K] (bf16) @ dequant(W)[N, K]^T,  W packed E2M1 + e4m3 block scales
//
// The decode GEMV (nvfp4_gemv.cu) is bandwidth-bound up to 4 rows; beyond that its
// CUDA-core FMAs and activation re-reads dominate. Here each 16-byte weight load feeds
// `mma.sync.m16n8k16` (bf16 in, fp32 accumulate), so the per-weight work is one
// dequant regardless of M, and the weights are still streamed exactly once.
//
// Layout trick: a warp owns 8 weight rows (one n8 tile). Lane (g = lane/4, t = lane%4)
// loads 16 contiguous bytes of row g covering K offsets [32t, 32t+32) of a 128-wide
// K super-step. The mma's logical k index is a fixed permutation of the physical one:
// for the c-th k16 chunk of a super-step, lane t's fragment slots {2t, 2t+1, 2t+8, 2t+9}
// hold physical K {32t+4c, .., 32t+4c+3}: four consecutive nibbles of its own bytes,
// so the B fragments need no shuffles and the matching A fragment is one 8-byte load
// of x[row, 32t+4c .. +3]. One super-step = 8 mmas per warp per 512 B of weights.
//
// Scales: each lane's 32 nibbles span 2 blocks of 16 (groups 2t, 2t+1 of the
// super-step), chunk c lies in group 2t + c/4. The weight tile is multiplied by its
// block scale and rounded to bf16 before the mma (same rounding as the Triton GEMM);
// the per-row global scale is applied to the fp32 accumulators at the end.
//
// Split-K across blockIdx.y keeps enough blocks in flight for small N; partial sums
// go to an fp32 workspace and a tiny reduce kernel writes the bf16 result
// (deterministic, no atomics).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace {

constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS = 32 * WARPS_PER_BLOCK;
constexpr int ROWS_PER_WARP = 8;           // one n8 tile
constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;  // 64
constexpr int SUPER_K = 128;               // K per warp-load (32 lanes * 32 nibbles / 8 rows)
constexpr int MAX_M = 16;

__device__ __forceinline__ void mma_16816_bf16(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ float e4m3_to_float(uint32_t b) {
    const uint32_t e = (b >> 3) & 0xFu;
    const uint32_t m = b & 0x7u;
    const float mag = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
    return (b & 0x80u) ? -mag : mag;
}

__device__ __forceinline__ uint32_t e2m1_to_bf16_bits(uint32_t nib) {
    // Exact bf16 of {0, .5, 1, 1.5, 2, 3, 4, 6}: codes >= 2 are 1.0 (0x3F80) with exponent
    // (code >> 1) - 1 and mantissa bit (code & 1); codes 0/1 are 0 and 0.5 (0x3F00).
    const uint32_t a = nib & 7u;
    const uint32_t normal = 0x3F00u + ((a >> 1) << 7) + ((a & 1u) << 6);
    const uint32_t v = a >= 2u ? normal : a * 0x3F00u;
    return v | ((nib & 8u) << 12);
}

__device__ __forceinline__ uint32_t e2m1x2_to_bf16x2_bits(uint32_t byte) {
    // low nibble -> low half = even element
    return e2m1_to_bf16_bits(byte & 0xFu) | (e2m1_to_bf16_bits(byte >> 4) << 16);
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}

// Four E2M1 nibbles (16 bits, nibble i = element i) -> two bf16x2 words {e0, e1}, {e2, e3}, bit-identical
// to e2m1_to_bf16_bits. Every magnitude's bf16 is a (hi, lo) byte pair from an 8-entry table, so two
// byte-permutes look up all four at once (~11 ops instead of ~36); the sign bits go to bits 15/31 with
// one multiply (the stray product bits are masked off).
__device__ __forceinline__ void e2m1x4_to_bf16x2x2(uint32_t v, uint32_t& w01, uint32_t& w23) {
    const uint32_t idx = v & 0x7777u;
    const uint32_t hi = prmt(0x3F3F3F00u, 0x40404040u, idx);   // codes 0..7: 00 3F 3F 3F 40 40 40 40
    const uint32_t lo = prmt(0xC0800000u, 0xC0804000u, idx);   //             00 00 80 C0 00 40 80 C0
    w01 = prmt(lo, hi, 0x5140u) | (((v & 0x0088u) * 0x01001000u) & 0x80008000u);
    w23 = prmt(lo, hi, 0x7362u) | (((v & 0x8800u) * 0x00010010u) & 0x80008000u);
}

__device__ __forceinline__ uint32_t bf16x2_mul(uint32_t a_bits, __nv_bfloat162 s) {
    // E2M1 (<= 2 significant bits) times an e4m3 scale (<= 4) is exact in bf16 (8 bits):
    // this multiply introduces no rounding, so the path matches the fp32 GEMV exactly.
    __nv_bfloat162 a = *reinterpret_cast<__nv_bfloat162*>(&a_bits);
    __nv_bfloat162 r = __hmul2(a, s);
    return *reinterpret_cast<uint32_t*>(&r);
}

template <bool SPLIT>
__global__ void __launch_bounds__(THREADS)
nvfp4_gemm_tc_kernel(const __nv_bfloat16* __restrict__ x,   // [M, K]
                     const uint8_t* __restrict__ w,          // [N, K/2]
                     const uint8_t* __restrict__ scale,      // [N, K/16]
                     const float* __restrict__ gscale,       // [N] or [1]
                     int gs_stride,
                     __nv_bfloat16* __restrict__ out,        // [M, N]       (SPLIT == false)
                     float* __restrict__ partial,            // [S, M, N]    (SPLIT == true)
                     int M, int N, int K, int k_per_split) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int n_base = blockIdx.x * ROWS_PER_BLOCK + warp * ROWS_PER_WARP;
    if (n_base >= N) return;
    const int n_row = min(n_base + g, N - 1);          // this lane's weight row (clamped for ragged N)
    const int k0 = blockIdx.y * k_per_split, k1 = min(k0 + k_per_split, K);

    const uint8_t* __restrict__ wrow = w + (size_t)n_row * (K / 2);
    const uint8_t* __restrict__ srow = scale + (size_t)n_row * (K / 16);
    // activation rows g and g+8 (rows >= M read row 0 and are discarded)
    const __nv_bfloat16* __restrict__ xa = x + (size_t)min(g, M - 1) * K;
    const __nv_bfloat16* __restrict__ xb = x + (size_t)min(g + 8, M - 1) * K;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    // this lane: 32 nibbles of row n_row at K offsets [ks + 32t, ks + 32t + 32); the next super-step's
    // weights and scales are loaded before this one's math so a load is always in flight
    uint4 packed_next = *reinterpret_cast<const uint4*>(wrow + (k0 + 32 * t) / 2);
    uint32_t s2_next = *reinterpret_cast<const uint16_t*>(srow + (k0 + 32 * t) / 16);
    for (int ks = k0; ks < k1; ks += SUPER_K) {
        const uint4 packed = packed_next;
        const uint32_t s2 = s2_next;
        const int kn = ks + SUPER_K < k1 ? ks + SUPER_K : ks;
        packed_next = *reinterpret_cast<const uint4*>(wrow + (kn + 32 * t) / 2);
        s2_next = *reinterpret_cast<const uint16_t*>(srow + (kn + 32 * t) / 16);
        const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
        // block scales as bf16x2 (exact: an e4m3 value fits in bf16)
        const __nv_bfloat162 sc[2] = {__float2bfloat162_rn(e4m3_to_float(s2 & 0xFFu)),
                                      __float2bfloat162_rn(e4m3_to_float((s2 >> 8) & 0xFFu))};
        // A: this lane's x[g][32t .. 32t+31] (and row g+8) as 4 x 16 B instead of 8 x 8 B per row: half
        // the L1 sectors. With M <= 8 the g+8 half of the mma is padding, so it is not loaded at all.
        uint32_t xa_w[16], xb_w[16];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint4 v = *reinterpret_cast<const uint4*>(xa + ks + 32 * t + 8 * i);
            xa_w[4 * i] = v.x; xa_w[4 * i + 1] = v.y; xa_w[4 * i + 2] = v.z; xa_w[4 * i + 3] = v.w;
        }
        if (M > 8) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint4 v = *reinterpret_cast<const uint4*>(xb + ks + 32 * t + 8 * i);
                xb_w[4 * i] = v.x; xb_w[4 * i + 1] = v.y; xb_w[4 * i + 2] = v.z; xb_w[4 * i + 3] = v.w;
            }
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) xb_w[i] = 0u;
        }

#pragma unroll
        for (int c = 0; c < 8; ++c) {
            // nibbles 4c..4c+3 = bytes 2c, 2c+1 = half c%2 of word c/2
            const uint32_t nib4 = (words[c >> 1] >> (16 * (c & 1))) & 0xFFFFu;
            const __nv_bfloat162 s = sc[c >> 2];
            uint32_t w01, w23;
            e2m1x4_to_bf16x2x2(nib4, w01, w23);
            uint32_t bfrag[2];
            bfrag[0] = bf16x2_mul(w01, s);
            bfrag[1] = bf16x2_mul(w23, s);
            // A: x[g][32t+4c .. +3] -> a0 (first pair), a2 (second pair); same for row g+8 -> a1, a3
            const uint32_t afrag[4] = {xa_w[2 * c], xb_w[2 * c], xa_w[2 * c + 1], xb_w[2 * c + 1]};
            mma_16816_bf16(acc, afrag, bfrag);
        }
    }

    // C fragment: acc[0], acc[1] -> (m = g, n = n_base + 2t, +1); acc[2], acc[3] -> (m = g + 8, same n)
    const int n0 = n_base + 2 * t, n1 = n0 + 1;
    const float inv0 = n0 < N ? 1.f / gscale[(size_t)n0 * gs_stride] : 0.f;
    const float inv1 = n1 < N ? 1.f / gscale[(size_t)n1 * gs_stride] : 0.f;
    if (SPLIT) {
        float* dst = partial + (size_t)blockIdx.y * M * N;
        if (g < M) {
            if (n0 < N) dst[(size_t)g * N + n0] = acc[0] * inv0;
            if (n1 < N) dst[(size_t)g * N + n1] = acc[1] * inv1;
        }
        if (g + 8 < M) {
            if (n0 < N) dst[(size_t)(g + 8) * N + n0] = acc[2] * inv0;
            if (n1 < N) dst[(size_t)(g + 8) * N + n1] = acc[3] * inv1;
        }
    } else {
        if (g < M) {
            if (n0 < N) out[(size_t)g * N + n0] = __float2bfloat16(acc[0] * inv0);
            if (n1 < N) out[(size_t)g * N + n1] = __float2bfloat16(acc[1] * inv1);
        }
        if (g + 8 < M) {
            if (n0 < N) out[(size_t)(g + 8) * N + n0] = __float2bfloat16(acc[2] * inv0);
            if (n1 < N) out[(size_t)(g + 8) * N + n1] = __float2bfloat16(acc[3] * inv1);
        }
    }
}

__global__ void reduce_splits_kernel(const float* __restrict__ partial, __nv_bfloat16* __restrict__ out,
                                     int S, int MN) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MN) return;
    float v = 0.f;
    for (int s = 0; s < S; ++s) v += partial[(size_t)s * MN + i];
    out[i] = __float2bfloat16(v);
}

}  // namespace

bool nvfp4_gemm_tc_supported(int64_t M, int64_t K) {
    return M >= 1 && M <= MAX_M && K % SUPER_K == 0;
}

torch::Tensor nvfp4_gemm_tc(torch::Tensor x, torch::Tensor w, torch::Tensor scale, torch::Tensor gscale) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(),
                "nvfp4_gemm_tc: x must be contiguous bf16 [M, K]");
    TORCH_CHECK(w.scalar_type() == torch::kUInt8 && w.dim() == 2 && w.is_contiguous(), "nvfp4_gemm_tc: w must be uint8 [N, K/2]");
    TORCH_CHECK(scale.scalar_type() == torch::kUInt8 && scale.is_contiguous(), "nvfp4_gemm_tc: scale must be uint8 [N, K/16]");
    TORCH_CHECK(gscale.scalar_type() == torch::kFloat32 && gscale.is_contiguous(), "nvfp4_gemm_tc: global scale must be fp32");
    const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
    TORCH_CHECK(nvfp4_gemm_tc_supported(M, K), "nvfp4_gemm_tc: needs 1 <= M <= 16 and K % 128 == 0 (M=", M, ", K=", K, ")");
    TORCH_CHECK(w.size(1) * 2 == K && scale.size(0) == N && scale.size(1) * 16 == K, "nvfp4_gemm_tc: shape mismatch");
    TORCH_CHECK(gscale.numel() == 1 || gscale.numel() == N, "nvfp4_gemm_tc: global scale must have 1 or N elements");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
                "nvfp4_gemm_tc: alignment");

    const c10::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({M, N}, x.options());
    const int gs_stride = gscale.numel() == 1 ? 0 : 1;
    const int n_blocks = (int)((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    const int num_super = (int)(K / SUPER_K);
    // enough blocks for the whole GPU (>= 2 per SM), without splitting K finer than needed
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int S = 1;
    while (S < 8 && n_blocks * S < 2 * sms && num_super % (S * 2) == 0) S *= 2;
    const int k_per_split = (int)(K / S);
    const dim3 grid(n_blocks, S);
    if (S == 1) {
        nvfp4_gemm_tc_kernel<false><<<grid, THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), w.data_ptr<uint8_t>(),
            scale.data_ptr<uint8_t>(), gscale.data_ptr<float>(), gs_stride,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), nullptr, (int)M, (int)N, (int)K, k_per_split);
    } else {
        auto partial = torch::empty({S, M, N}, x.options().dtype(torch::kFloat32));
        nvfp4_gemm_tc_kernel<true><<<grid, THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), w.data_ptr<uint8_t>(),
            scale.data_ptr<uint8_t>(), gscale.data_ptr<float>(), gs_stride,
            nullptr, partial.data_ptr<float>(), (int)M, (int)N, (int)K, k_per_split);
        const int MN = (int)(M * N);
        reduce_splits_kernel<<<(MN + 255) / 256, 256, 0, stream>>>(
            partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), S, MN);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// NVFP4 weight-only decode GEMV: out[M, N] = x[M, K] @ dequant(W)[N, K]^T
//
// Layout (compressed-tensors `nvfp4-pack-quantized`, see layers/quantization/base.py):
//   w      uint8 [N, K/2]   E2M1 codes, two per byte, low nibble = even element
//   scale  uint8 [N, K/16]  one float8_e4m3fn block scale per 16 elements
//   gscale fp32  [N] or [1] dequant = e2m1 * block_scale / global_scale
//
// Decode is memory bound: the whole kernel is organised around streaming the
// packed weights once at full bandwidth. One warp owns one output row, each
// lane reads 16 contiguous bytes (32 elements, 2 scale groups) per step so a
// warp issues one fully coalesced 512 B load per step. The activation rows
// (tiny, shared by every warp) come from L1/L2. No shared memory, no atomics,
// no tensor cores: nothing here needs anything newer than sm_80.
//
// E2M1 decode is a bit trick rather than a LUT or a branch: an E2M1 nibble
// placed in the fp16 exponent/mantissa fields reads back as value * 2^-14 for
// both normal and subnormal codes, so the 2^14 is folded into the block scale.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace {

constexpr int WARP = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS = WARP * WARPS_PER_BLOCK;
constexpr int ELEMS_PER_LANE = 32;                        // 16 bytes of codes
constexpr int ELEMS_PER_STEP = WARP * ELEMS_PER_LANE;     // 1024 elements per warp step
constexpr int MAX_MT = 4;                                 // activation rows per block

__device__ __forceinline__ float e4m3_to_float(uint32_t b) {
    // float8_e4m3fn: s eeee mmm, bias 7, no inf; NaN never appears in scales.
    const uint32_t e = (b >> 3) & 0xFu;
    const uint32_t m = b & 0x7u;
    // normal: 2^(e-7) * (1 + m/8) as fp32 bits; subnormal (e == 0): m * 2^-9
    const float mag = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
    return (b & 0x80u) ? -mag : mag;
}

__device__ __forceinline__ float2 e2m1x2_to_float2_scaled(uint32_t byte) {
    // Returns (lo, hi) * 2^-14. Caller folds the 2^14 into the block scale.
    const uint32_t lo = byte & 0xFu, hi = (byte >> 4) & 0xFu;
    const uint32_t hlo = ((lo & 7u) << 9) | ((lo & 8u) << 12);
    const uint32_t hhi = ((hi & 7u) << 9) | ((hi & 8u) << 12);
    const uint32_t bits = hlo | (hhi << 16);
    __half2 h = *reinterpret_cast<const __half2*>(&bits);
    return __half22float2(h);
}

template <int MT, int R>
__global__ void __launch_bounds__(THREADS)
nvfp4_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                  const uint8_t* __restrict__ w,
                  const uint8_t* __restrict__ scale,
                  const float* __restrict__ gscale,
                  int gs_stride,
                  __nv_bfloat16* __restrict__ out,
                  int M, int N, int K) {
    const int warp = threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    const int n0 = (blockIdx.x * WARPS_PER_BLOCK + warp) * R;
    if (n0 >= N) return;  // whole warp leaves together
    const int m0 = blockIdx.y * MT;

    // Rows past N (ragged tail) read row n0 again and are never stored.
    const uint8_t* __restrict__ wrow[R];
    const uint8_t* __restrict__ srow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int n = min(n0 + r, N - 1);
        wrow[r] = w + (size_t)n * (K / 2);
        srow[r] = scale + (size_t)n * (K / 16);
    }
    const __nv_bfloat16* xm[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) xm[m] = x + (size_t)min(m0 + m, M - 1) * K;

    float acc[R][MT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

#pragma unroll 2
    for (int k0 = lane * ELEMS_PER_LANE; k0 < K; k0 += ELEMS_PER_STEP) {
        // Issue every global load for this step up front: R weight vectors,
        // R scale pairs, MT activation rows (the latter are L1/L2 hits).
        uint4 packed[R];
        uint32_t s2[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            packed[r] = *reinterpret_cast<const uint4*>(wrow[r] + k0 / 2);
            s2[r] = *reinterpret_cast<const uint16_t*>(srow[r] + k0 / 16);
        }
        uint4 xq[MT][4];
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                xq[m][j] = *reinterpret_cast<const uint4*>(xm[m] + k0 + j * 8);

#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float f0 = e4m3_to_float(s2[r] & 0xFFu) * 16384.f;
            const float f1 = e4m3_to_float((s2[r] >> 8) & 0xFFu) * 16384.f;
            const uint32_t words[4] = {packed[r].x, packed[r].y, packed[r].z, packed[r].w};
            float g[MT][2];
#pragma unroll
            for (int m = 0; m < MT; ++m) { g[m][0] = 0.f; g[m][1] = 0.f; }
#pragma unroll
            for (int wi = 0; wi < 4; ++wi) {
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const float2 wv = e2m1x2_to_float2_scaled((words[wi] >> (8 * b)) & 0xFFu);
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        const uint32_t xbits = b == 0 ? xq[m][wi].x : b == 1 ? xq[m][wi].y : b == 2 ? xq[m][wi].z : xq[m][wi].w;
                        // bf16 -> fp32 is a 16-bit shift; no conversion instruction needed
                        const float xlo = __uint_as_float(xbits << 16);
                        const float xhi = __uint_as_float(xbits & 0xFFFF0000u);
                        g[m][wi >> 1] = fmaf(xlo, wv.x, g[m][wi >> 1]);
                        g[m][wi >> 1] = fmaf(xhi, wv.y, g[m][wi >> 1]);
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < MT; ++m) acc[r][m] = fmaf(g[m][0], f0, fmaf(g[m][1], f1, acc[r][m]));
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float v = acc[r][m];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            acc[r][m] = v;
        }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int n = n0 + r;
            if (n >= N) break;
            const float inv = 1.f / gscale[(size_t)n * gs_stride];
#pragma unroll
            for (int m = 0; m < MT; ++m)
                if (m0 + m < M) out[(size_t)(m0 + m) * N + n] = __float2bfloat16(acc[r][m] * inv);
        }
    }
}

template <int MT, int R>
void launch(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& scale,
            const torch::Tensor& gscale, int gs_stride, torch::Tensor& out, int M, int N, int K,
            cudaStream_t stream) {
    const int rows_per_block = WARPS_PER_BLOCK * R;
    const dim3 grid((N + rows_per_block - 1) / rows_per_block, (M + MT - 1) / MT);
    nvfp4_gemv_kernel<MT, R><<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        w.data_ptr<uint8_t>(), scale.data_ptr<uint8_t>(), gscale.data_ptr<float>(), gs_stride,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), M, N, K);
}

}  // namespace

torch::Tensor nvfp4_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor scale, torch::Tensor gscale) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && scale.is_cuda() && gscale.is_cuda(), "nvfp4_gemv: all inputs must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "nvfp4_gemv: x must be contiguous bf16 [M, K]");
    TORCH_CHECK(w.scalar_type() == torch::kUInt8 && w.dim() == 2 && w.is_contiguous(), "nvfp4_gemv: w must be contiguous uint8 [N, K/2]");
    TORCH_CHECK(scale.scalar_type() == torch::kUInt8 && scale.dim() == 2 && scale.is_contiguous(), "nvfp4_gemv: scale must be contiguous uint8 [N, K/16]");
    TORCH_CHECK(gscale.scalar_type() == torch::kFloat32 && gscale.is_contiguous(), "nvfp4_gemv: global scale must be contiguous fp32");
    const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
    TORCH_CHECK(K % ELEMS_PER_LANE == 0, "nvfp4_gemv: K must be a multiple of 32, got ", K);
    TORCH_CHECK(w.size(1) * 2 == K, "nvfp4_gemv: w has ", w.size(1) * 2, " columns, x has K=", K);
    TORCH_CHECK(scale.size(0) == N && scale.size(1) * 16 == K, "nvfp4_gemv: scale must be [N, K/16]");
    TORCH_CHECK(gscale.numel() == 1 || gscale.numel() == N, "nvfp4_gemv: global scale must have 1 or N elements");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
                "nvfp4_gemv: x and w must be 16-byte aligned");
    TORCH_CHECK(M >= 1 && N >= 1, "nvfp4_gemv: empty input");

    const c10::cuda::OptionalCUDAGuard guard(x.device());
    auto out = torch::empty({M, N}, x.options());
    const int gs_stride = gscale.numel() == 1 ? 0 : 1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // M == 1 is pure weight streaming: one row per warp keeps the most warps in
    // flight. For M > 1 each lane reuses its activation loads across two rows,
    // which halves the per-weight-byte L1 traffic and conversion work.
    if (M == 1) launch<1, 1>(x, w, scale, gscale, gs_stride, out, M, N, K, stream);
    else if (M == 2) launch<2, 2>(x, w, scale, gscale, gs_stride, out, M, N, K, stream);
    else launch<MAX_MT, 2>(x, w, scale, gscale, gs_stride, out, M, N, K, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

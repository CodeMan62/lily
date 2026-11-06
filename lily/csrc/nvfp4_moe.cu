// Grouped NVFP4 GEMM for MoE experts: one launch per layer, every expert that received
// tokens is processed once, its weights streamed once, on tensor cores (mma.sync bf16).
//
//   for each expert e with rows R_e = {tokens routed to e} (CSR: offsets[e]..offsets[e+1]):
//       out[r, :] = x[token[r], :] @ dequant(W[e])^T       for r in R_e
//
// W is [E, N, K/2] packed E2M1 with [E, N, K/16] e4m3 block scales and an fp32 global scale
// per expert (or per expert row). Rows are handled 16 at a time with the same lane/fragment
// layout as nvfp4_gemm_tc.cu (lane (g, t) loads 16 B of weight row g at K offsets
// [32t, 32t+32) of a 128-wide super-step; chunk c uses its nibbles 4c..4c+3 and the matching
// 8-byte activation load). Experts with more than 16 rows re-stream their weights per 16 rows.
//
// Grid: (row tiles of the output = (N / 64) per expert, summed over experts) x (K splits);
// block = 8 warps = 64 output rows of one expert. A small host-side prefix over
// ceil(rows_e / 16) * (N / 64) would be needed to index tiles exactly; instead each block
// loops over the row-16 groups of its expert, so the grid is simply (n_tiles, E_active).
// Expert ids and row ranges are read from device tensors, so the launch is CUDA-graph safe
// (the grid is sized for the maximum number of active experts).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace {

constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS = 32 * WARPS_PER_BLOCK;
constexpr int ROWS_PER_WARP = 8;
constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;  // 64 weight rows (output features)
constexpr int SUPER_K = 128;
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
    const uint32_t a = nib & 7u;
    const uint32_t normal = 0x3F00u + ((a >> 1) << 7) + ((a & 1u) << 6);
    const uint32_t v = a >= 2u ? normal : a * 0x3F00u;
    return v | ((nib & 8u) << 12);
}

__device__ __forceinline__ uint32_t e2m1x2_to_bf16x2_bits(uint32_t byte) {
    return e2m1_to_bf16_bits(byte & 0xFu) | (e2m1_to_bf16_bits(byte >> 4) << 16);
}

__device__ __forceinline__ uint32_t bf16x2_mul(uint32_t a_bits, __nv_bfloat162 s) {
    __nv_bfloat162 a = *reinterpret_cast<__nv_bfloat162*>(&a_bits);
    __nv_bfloat162 r = __hmul2(a, s);
    return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ uint32_t load_x_pair(const __nv_bfloat16* __restrict__ x, int row_token, int k, int K,
                                                int valid, uint2* out) {
    // 4 consecutive bf16 of x[row_token][k .. k+3]; zero for padding rows
    if (!valid) { out->x = 0u; out->y = 0u; return 0u; }
    *out = *reinterpret_cast<const uint2*>(x + (size_t)row_token * K + k);
    return 0u;
}

__global__ void __launch_bounds__(THREADS)
nvfp4_moe_gemm_kernel(const __nv_bfloat16* __restrict__ x,    // [T, K]
                      const uint8_t* __restrict__ w,           // [E, N, K/2]
                      const uint8_t* __restrict__ scale,       // [E, N, K/16]
                      const float* __restrict__ gscale,        // [E * gs_rows]
                      int gs_rows,                             // 1 (per expert) or N (per expert row)
                      const int32_t* __restrict__ active_experts,  // [E_active_max] expert ids, -1 = none
                      const int32_t* __restrict__ offsets,     // [E + 1] CSR over sorted (token, slot) pairs
                      const int32_t* __restrict__ tokens,      // [R] token index of each sorted pair
                      __nv_bfloat16* __restrict__ out,         // [R, N] one row per sorted pair
                      int N, int K) {
    const int e_slot = blockIdx.y;
    const int e = active_experts[e_slot];
    if (e < 0) return;
    const int r0 = offsets[e], r1 = offsets[e + 1];
    if (r0 == r1) return;

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int n_base = blockIdx.x * ROWS_PER_BLOCK + warp * ROWS_PER_WARP;
    if (n_base >= N) return;
    const int n_row = min(n_base + g, N - 1);

    const uint8_t* __restrict__ wrow = w + ((size_t)e * N + n_row) * (K / 2);
    const uint8_t* __restrict__ srow = scale + ((size_t)e * N + n_row) * (K / 16);
    const int n0 = n_base + 2 * t, n1 = n0 + 1;
    const float inv0 = n0 < N ? 1.f / gscale[(size_t)e * gs_rows + (gs_rows == 1 ? 0 : n0)] : 0.f;
    const float inv1 = n1 < N ? 1.f / gscale[(size_t)e * gs_rows + (gs_rows == 1 ? 0 : n1)] : 0.f;

    // 16 rows of this expert at a time
    for (int rbase = r0; rbase < r1; rbase += MAX_M) {
        const int ra = rbase + g, rb = rbase + g + 8;                 // the two pair rows this lane feeds
        const int tok_a = ra < r1 ? tokens[ra] : 0;
        const int tok_b = rb < r1 ? tokens[rb] : 0;
        const int va = ra < r1, vb = rb < r1;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};

        uint4 packed = *reinterpret_cast<const uint4*>(wrow + (32 * t) / 2);
        uint32_t s2 = *reinterpret_cast<const uint16_t*>(srow + (32 * t) / 16);
        for (int ks = 0; ks < K; ks += SUPER_K) {
            // the next K step's weights stay in flight during this step's maths
            const int kn = ks + SUPER_K < K ? ks + SUPER_K : ks;
            const uint4 packed_next = *reinterpret_cast<const uint4*>(wrow + (kn + 32 * t) / 2);
            const uint32_t s2_next = *reinterpret_cast<const uint16_t*>(srow + (kn + 32 * t) / 16);
            const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
            const __nv_bfloat162 sc[2] = {__float2bfloat162_rn(e4m3_to_float(s2 & 0xFFu)),
                                          __float2bfloat162_rn(e4m3_to_float((s2 >> 8) & 0xFFu))};
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const uint32_t word = words[c >> 1];
                const uint32_t b0 = (word >> (8 * ((2 * c) & 3))) & 0xFFu;
                const uint32_t b1 = (word >> (8 * ((2 * c + 1) & 3))) & 0xFFu;
                const __nv_bfloat162 s = sc[c >> 2];
                uint32_t bfrag[2];
                bfrag[0] = bf16x2_mul(e2m1x2_to_bf16x2_bits(b0), s);
                bfrag[1] = bf16x2_mul(e2m1x2_to_bf16x2_bits(b1), s);
                uint2 xa, xb;
                load_x_pair(x, tok_a, ks + 32 * t + 4 * c, K, va, &xa);
                load_x_pair(x, tok_b, ks + 32 * t + 4 * c, K, vb, &xb);
                const uint32_t afrag[4] = {xa.x, xb.x, xa.y, xb.y};
                mma_16816_bf16(acc, afrag, bfrag);
            }
            packed = packed_next;
            s2 = s2_next;
        }
        // C fragment -> pair rows ra (acc[0], acc[1]) and rb (acc[2], acc[3]) at columns n0, n1
        if (va) {
            if (n0 < N) out[(size_t)ra * N + n0] = __float2bfloat16(acc[0] * inv0);
            if (n1 < N) out[(size_t)ra * N + n1] = __float2bfloat16(acc[1] * inv1);
        }
        if (vb) {
            if (n0 < N) out[(size_t)rb * N + n0] = __float2bfloat16(acc[2] * inv0);
            if (n1 < N) out[(size_t)rb * N + n1] = __float2bfloat16(acc[3] * inv1);
        }
    }
}

}  // namespace

bool nvfp4_moe_supported(int64_t K) { return K % SUPER_K == 0; }

torch::Tensor nvfp4_moe_gemm(torch::Tensor x, torch::Tensor w, torch::Tensor scale, torch::Tensor gscale,
                             torch::Tensor active_experts, torch::Tensor offsets, torch::Tensor tokens, int64_t num_rows) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "moe_gemm: x bf16 [T, K]");
    TORCH_CHECK(w.scalar_type() == torch::kUInt8 && w.dim() == 3 && w.is_contiguous(), "moe_gemm: w uint8 [E, N, K/2]");
    TORCH_CHECK(scale.scalar_type() == torch::kUInt8 && scale.dim() == 3 && scale.is_contiguous(), "moe_gemm: scale uint8 [E, N, K/16]");
    TORCH_CHECK(gscale.scalar_type() == torch::kFloat32 && gscale.is_contiguous(), "moe_gemm: gscale fp32");
    TORCH_CHECK(active_experts.scalar_type() == torch::kInt32 && offsets.scalar_type() == torch::kInt32
                && tokens.scalar_type() == torch::kInt32, "moe_gemm: routing tensors must be int32");
    const int64_t K = x.size(1), E = w.size(0), N = w.size(1);
    TORCH_CHECK(nvfp4_moe_supported(K) && w.size(2) * 2 == K && scale.size(0) == E && scale.size(1) == N && scale.size(2) * 16 == K, "moe_gemm: shapes");
    TORCH_CHECK(gscale.numel() == E || gscale.numel() == E * N, "moe_gemm: gscale must be [E] or [E*N]");
    TORCH_CHECK(offsets.numel() == E + 1, "moe_gemm: offsets must be [E+1]");
    const int gs_rows = gscale.numel() == E ? 1 : (int)N;

    const c10::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({num_rows, N}, x.options());
    if (num_rows == 0 || active_experts.numel() == 0) return out;
    const dim3 grid((unsigned)((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK), (unsigned)active_experts.numel());
    nvfp4_moe_gemm_kernel<<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), w.data_ptr<uint8_t>(), scale.data_ptr<uint8_t>(),
        gscale.data_ptr<float>(), gs_rows, active_experts.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(),
        tokens.data_ptr<int32_t>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), (int)N, (int)K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

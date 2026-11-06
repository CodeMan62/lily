// bf16 linear for decode and verify batches (M <= 8 rows): out[m, n] = sum_k x[m, k] * w[n, k].
// These GEMMs only stream the weights, and cuBLAS' small-tile kernels stop near half of the card's
// bandwidth. Here the weights are the MMA's A operand (16 rows) and the batch its 8-wide B operand,
// so M <= 8 costs one tile. K is walked in a permuted order, the same for A and B: a lane loads 16
// contiguous bytes per row and they feed two k16 steps, no shared memory on the way to the MMA.
// Block = 16 weight rows; its 8 warps split K in 128-wide pieces and reduce through shared memory.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace {

constexpr int WARPS = 8;
constexpr int THREADS = 32 * WARPS;
constexpr int ROWS = 16;
constexpr int MAX_M = 8;
constexpr int K_PIECE = 128;           // per warp and iteration: 4 loads of 32 k
constexpr int UNROLL = K_PIECE / 32;

__device__ __forceinline__ void mma_16816(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 load_stream(const void* p) {     // weights: read once, skip L1
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__global__ void __launch_bounds__(THREADS)
bf16_skinny_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
                   int M, int N, int K, int ldx) {
    __shared__ float red[WARPS][ROWS][MAX_M];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * ROWS;
    const int ra = min(n0 + g, N - 1), rb = min(n0 + g + 8, N - 1);
    const __nv_bfloat16* wa = w + (size_t)ra * K;
    const __nv_bfloat16* wb = w + (size_t)rb * K;
    const __nv_bfloat16* xr = x + (size_t)min(g, M - 1) * ldx;
    const bool valid = g < M;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k0 = warp * K_PIECE; k0 < K; k0 += WARPS * K_PIECE) {
        uint4 a[UNROLL], b[UNROLL], v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int k = k0 + 32 * u + 8 * t;
            a[u] = load_stream(wa + k);
            b[u] = load_stream(wb + k);
            v[u] = valid ? *reinterpret_cast<const uint4*>(xr + k) : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            mma_16816(c, a[u].x, b[u].x, a[u].y, b[u].y, v[u].x, v[u].y);
            mma_16816(c, a[u].z, b[u].z, a[u].w, b[u].w, v[u].z, v[u].w);
        }
    }
    // c[0], c[1]: weight row g, batch rows 2t, 2t+1; c[2], c[3]: weight row g + 8
    red[warp][g][2 * t] = c[0];
    red[warp][g][2 * t + 1] = c[1];
    red[warp][g + 8][2 * t] = c[2];
    red[warp][g + 8][2 * t + 1] = c[3];
    __syncthreads();
    if (threadIdx.x < ROWS * MAX_M) {
        const int row = threadIdx.x % ROWS, m = threadIdx.x / ROWS;     // consecutive threads, consecutive n
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < WARPS; ++i) s += red[i][row][m];
        if (m < M && n0 + row < N) out[(size_t)m * N + n0 + row] = __float2bfloat16(s);
    }
}

// Router logits, fp32 like the reference: one block per expert, the fp32 row is streamed once for all
// M rows of bf16 activations.
constexpr int ROUTER_THREADS = 256;
constexpr int ROUTER_K = ROUTER_THREADS * 4;

__global__ void __launch_bounds__(ROUTER_THREADS)
router_logits_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, int M, int N, int K) {
    __shared__ float red[ROUTER_THREADS / 32][MAX_M];
    const float* wr = w + (size_t)blockIdx.x * K;
    float acc[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.f;
#pragma unroll 4
    for (int k = threadIdx.x * 4; k < K; k += ROUTER_K) {
        const float4 v = *reinterpret_cast<const float4*>(wr + k);
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m >= M) break;
            const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(x + (size_t)m * K + k);
            const float2 a = __bfloat1622float2(xp[0]), b = __bfloat1622float2(xp[1]);
            acc[m] += a.x * v.x + a.y * v.y + b.x * v.z + b.y * v.w;
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        float s = acc[m];
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) red[warp][m] = s;
    }
    __syncthreads();
    if (threadIdx.x < M) {
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < ROUTER_THREADS / 32; ++i) s += red[i][threadIdx.x];
        out[(size_t)threadIdx.x * N + blockIdx.x] = s;
    }
}

}  // namespace

bool router_logits_supported(int64_t M, int64_t K) { return M >= 1 && M <= MAX_M && K % ROUTER_K == 0; }

torch::Tensor router_logits(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kFloat);
    const at::cuda::CUDAGuard guard(x.device());
    const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
    auto out = torch::empty({M, N}, x.options().dtype(at::kFloat));
    router_logits_kernel<<<N, ROUTER_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), w.data_ptr<float>(), out.data_ptr<float>(), M, N, K);
    return out;
}

bool bf16_skinny_supported(int64_t M, int64_t N, int64_t K) { return M >= 1 && M <= MAX_M && N >= 256 && K % K_PIECE == 0; }

torch::Tensor bf16_skinny(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 8 == 0 && w.is_contiguous() && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16);
    const at::cuda::CUDAGuard guard(x.device());
    const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
    auto out = torch::empty({M, N}, x.options());
    bf16_skinny_kernel<<<(N + ROWS - 1) / ROWS, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K, (int)x.stride(0));
    return out;
}

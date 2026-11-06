// Small fused kernels for the per-token glue that torch runs as 5-12 launches each: RMSNorm, the DFlash
// drafter's two-tap dynamic conv, neox rotary, and the drafter's greedy candidate selection. Every op
// keeps the torch path's bf16 rounding points, so the outputs match the eager implementation bit for bit
// (the selector's fp32 dot products may differ in summation order).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cfloat>

namespace {

constexpr int VEC = 8;

__device__ __forceinline__ float rb(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ void unpack8(const uint4& v, float f[VEC]) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) { const float2 t = __bfloat1622float2(p[i]); f[2 * i] = t.x; f[2 * i + 1] = t.y; }
}
__device__ __forceinline__ uint4 pack8(const float f[VEC]) {
    uint4 v;
    __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ---- RMSNorm: one warp per row --------------------------------------------------------------------------
// out = w * bf16(x * rsqrt(mean(x^2) + eps)): the variance in fp32, then the HF rounding order.
constexpr int NORM_WARPS = 8;

__global__ void __launch_bounds__(NORM_WARPS * 32)
rmsnorm_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
               int N, int H, float eps) {
    const int row = blockIdx.x * NORM_WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= N) return;
    const int nv = H / VEC;
    const uint4* xv = reinterpret_cast<const uint4*>(x + (size_t)row * H);
    const uint4* wv = reinterpret_cast<const uint4*>(w);
    uint4* ov = reinterpret_cast<uint4*>(out + (size_t)row * H);
    float ss = 0.f;
    for (int i = lane; i < nv; i += 32) {
        float f[VEC];
        unpack8(xv[i], f);
#pragma unroll
        for (int j = 0; j < VEC; ++j) ss += f[j] * f[j];
    }
    const float inv = rsqrtf(warp_sum(ss) / (float)H + eps);
    for (int i = lane; i < nv; i += 32) {
        float f[VEC], g[VEC];
        unpack8(xv[i], f);
        unpack8(wv[i], g);
#pragma unroll
        for (int j = 0; j < VEC; ++j) f[j] = g[j] * rb(f[j] * inv);
        ov[i] = pack8(f);
    }
}

// ---- two-tap grouped dynamic conv (DFlash drafter) -----------------------------------------------------
// out[b,l,h] = ((base0[h] * x[b,l,h]) + dyn[b,l,0,g] * x[b,l,h]) + (base1[h] * x[b,l-1,h]) + dyn[b,l,1,g] * x[b,l-1,h]
// with the torch rounding after every op; x[b,-1] = 0. dyn rows are at `dyn + (b*L+l)*dyn_stride + dyn_off`,
// laid out [2 taps][G groups].
__global__ void dyn_conv2_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ dyn,
                                 const __nv_bfloat16* __restrict__ base, __nv_bfloat16* __restrict__ out,
                                 int B, int L, int H, int group, int dyn_stride, int dyn_off) {
    const int nv = H / VEC;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * L * nv) return;
    const int i = idx % nv, bl = idx / nv, l = bl % L, b = bl / L;
    const int G = H / group, g = (i * VEC) / group;
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    const __nv_bfloat16* d = dyn + (size_t)bl * dyn_stride + dyn_off;
    const float d0 = bf(d[g]), d1 = bf(d[G + g]);
    float cur[VEC], prev[VEC], b0[VEC], b1[VEC], o[VEC];
    unpack8(xv[(size_t)bl * nv + i], cur);
    if (l > 0) unpack8(xv[(size_t)(bl - 1) * nv + i], prev);
    else {
#pragma unroll
        for (int j = 0; j < VEC; ++j) prev[j] = 0.f;
    }
    unpack8(reinterpret_cast<const uint4*>(base)[i], b0);
    unpack8(reinterpret_cast<const uint4*>(base + H)[i], b1);
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
        float v = rb(b0[j] * cur[j]);            // zeros + k*x
        v = rb(v + d0 * cur[j]);                 // addcmul
        v = rb(v + rb(b1[j] * prev[j]));
        v = rb(v + d1 * prev[j]);
        o[j] = v;
    }
    reinterpret_cast<uint4*>(out)[idx] = pack8(o);
}

// ---- neox rotary over whole heads --------------------------------------------------------------------------
// q [N, Hq, D], k [N, Hk, D] bf16 -> rotated copies; cos_sin [max_pos, D] bf16 (cos | sin halves).
__global__ void rope_neox_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                                 __nv_bfloat16* __restrict__ qo, __nv_bfloat16* __restrict__ ko,
                                 const __nv_bfloat16* __restrict__ cos_sin, const int64_t* __restrict__ pos,
                                 int N, int Hq, int Hk, int D) {
    const int half = D / 2;
    const int total = N * (Hq + Hk) * half;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int j = idx % half, nh = idx / half, h = nh % (Hq + Hk), n = nh / (Hq + Hk);
    const __nv_bfloat16* cs = cos_sin + (size_t)pos[n] * D;
    const float c = bf(cs[j]), s = bf(cs[half + j]);
    const __nv_bfloat16* src;
    __nv_bfloat16* dst;
    if (h < Hq) { src = q + ((size_t)n * Hq + h) * D; dst = qo + ((size_t)n * Hq + h) * D; }
    else { src = k + ((size_t)n * Hk + (h - Hq)) * D; dst = ko + ((size_t)n * Hk + (h - Hq)) * D; }
    const float x1 = bf(src[j]), x2 = bf(src[half + j]);
    dst[j] = __float2bfloat16(rb(x1 * c) + rb(-x2 * s));
    dst[half + j] = __float2bfloat16(rb(x2 * c) + rb(x1 * s));
}

// ---- greedy candidate selection (DFlash drafter) -----------------------------------------------------------
// One block per batch row, walking the S positions: score[k] = unary[k] + (pred[prev] * proj[pos]) . succ[cand[k]].
constexpr int SEL_THREADS = 256;
constexpr int SEL_MAX_R = 1024;
constexpr int SEL_MAX_K = 64;

__global__ void __launch_bounds__(SEL_THREADS)
select_greedy_kernel(const float* __restrict__ unary, const int64_t* __restrict__ cand, const float* __restrict__ proj,
                     const int64_t* __restrict__ anchor, const __nv_bfloat16* __restrict__ pred_cb,
                     const __nv_bfloat16* __restrict__ succ_cb, int64_t* __restrict__ path, float* __restrict__ q,
                     int S, int K, int R) {
    __shared__ float gate[SEL_MAX_R];
    __shared__ float score[SEL_MAX_K];
    __shared__ int64_t prev_s;
    const int b = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (threadIdx.x == 0) prev_s = anchor[b];
    __syncthreads();
    for (int pos = 0; pos < S; ++pos) {
        const int64_t prev = prev_s;
        const float* pr = proj + ((size_t)b * S + pos) * R;
        const __nv_bfloat16* pc = pred_cb + (size_t)prev * R;
        for (int r = threadIdx.x; r < R; r += SEL_THREADS) gate[r] = bf(pc[r]) * pr[r];
        __syncthreads();
        const int64_t* cd = cand + ((size_t)b * S + pos) * K;
        const float* un = unary + ((size_t)b * S + pos) * K;
        for (int k = warp; k < K; k += SEL_THREADS / 32) {
            const __nv_bfloat16* sc = succ_cb + (size_t)cd[k] * R;
            float acc = 0.f;
            for (int r = lane; r < R; r += 32) acc += gate[r] * bf(sc[r]);
            acc = warp_sum(acc);
            if (lane == 0) score[k] = un[k] + acc;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int best = 0;
            for (int k = 1; k < K; ++k) if (score[k] > score[best]) best = k;
            prev_s = cd[best];
            path[(size_t)b * S + pos] = prev_s;
            for (int k = 0; k < K; ++k) q[((size_t)b * S + pos) * K + k] = (k == best) ? 1.f : 0.f;
        }
        __syncthreads();
    }
}

}  // namespace

torch::Tensor rmsnorm_fused(torch::Tensor x, torch::Tensor w, double eps) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16);
    TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && w.is_contiguous() && x.size(1) % VEC == 0 && w.numel() == x.size(1));
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x);
    const int N = x.size(0), H = x.size(1);
    if (N == 0) return out;
    auto stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_kernel<<<(N + NORM_WARPS - 1) / NORM_WARPS, NORM_WARPS * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), N, H, (float)eps);
    return out;
}

torch::Tensor dyn_conv2(torch::Tensor x, torch::Tensor dyn, int64_t dyn_off, torch::Tensor base, int64_t group) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 3 && x.is_contiguous() && x.scalar_type() == at::kBFloat16);
    TORCH_CHECK(dyn.dim() == 3 && dyn.stride(2) == 1 && dyn.scalar_type() == at::kBFloat16 && dyn.size(0) == x.size(0) && dyn.size(1) == x.size(1));
    TORCH_CHECK(base.is_contiguous() && base.dim() == 2 && base.size(0) == 2 && base.size(1) == x.size(2) && base.scalar_type() == at::kBFloat16);
    const int B = x.size(0), L = x.size(1), H = x.size(2);
    TORCH_CHECK(H % VEC == 0 && group % VEC == 0 && H % group == 0);
    TORCH_CHECK(dyn_off + 2 * (H / group) <= dyn.size(2) && dyn.stride(1) == dyn.size(2) && dyn.stride(0) == dyn.size(1) * dyn.size(2));
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x);
    const int total = B * L * (H / VEC);
    if (total == 0) return out;
    auto stream = at::cuda::getCurrentCUDAStream();
    dyn_conv2_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(dyn.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(base.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        B, L, H, (int)group, (int)dyn.size(2), (int)dyn_off);
    return out;
}

std::vector<torch::Tensor> rope_neox(torch::Tensor q, torch::Tensor k, torch::Tensor cos_sin, torch::Tensor pos) {
    TORCH_CHECK(q.is_cuda() && q.dim() == 3 && k.dim() == 3 && q.is_contiguous() && k.is_contiguous());
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 && cos_sin.scalar_type() == at::kBFloat16);
    TORCH_CHECK(pos.scalar_type() == at::kLong && pos.is_contiguous() && cos_sin.is_contiguous());
    const int N = q.size(0), Hq = q.size(1), Hk = k.size(1), D = q.size(2);
    TORCH_CHECK(k.size(0) == N && k.size(2) == D && D % 2 == 0 && cos_sin.size(1) == D && pos.numel() == N);
    const c10::cuda::CUDAGuard guard(q.device());
    auto qo = torch::empty_like(q);
    auto ko = torch::empty_like(k);
    const int total = N * (Hq + Hk) * (D / 2);
    if (total == 0) return {qo, ko};
    auto stream = at::cuda::getCurrentCUDAStream();
    rope_neox_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(qo.data_ptr()), reinterpret_cast<__nv_bfloat16*>(ko.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(cos_sin.data_ptr()), pos.data_ptr<int64_t>(), N, Hq, Hk, D);
    return {qo, ko};
}

bool select_greedy_supported(int64_t K, int64_t R) { return K <= SEL_MAX_K && R <= SEL_MAX_R; }

std::vector<torch::Tensor> select_greedy(torch::Tensor unary, torch::Tensor cand, torch::Tensor proj, torch::Tensor anchor,
                                         torch::Tensor pred_cb, torch::Tensor succ_cb) {
    TORCH_CHECK(unary.is_cuda() && unary.dim() == 3 && unary.scalar_type() == at::kFloat && unary.is_contiguous());
    TORCH_CHECK(cand.sizes() == unary.sizes() && cand.scalar_type() == at::kLong && cand.is_contiguous());
    TORCH_CHECK(proj.dim() == 3 && proj.scalar_type() == at::kFloat && proj.is_contiguous());
    TORCH_CHECK(anchor.scalar_type() == at::kLong && anchor.is_contiguous() && anchor.numel() == unary.size(0));
    TORCH_CHECK(pred_cb.scalar_type() == at::kBFloat16 && succ_cb.scalar_type() == at::kBFloat16 && pred_cb.is_contiguous() && succ_cb.is_contiguous());
    const int B = unary.size(0), S = unary.size(1), K = unary.size(2), R = proj.size(2);
    TORCH_CHECK(select_greedy_supported(K, R) && pred_cb.size(1) == R && succ_cb.size(1) == R && proj.size(0) == B && proj.size(1) == S);
    const c10::cuda::CUDAGuard guard(unary.device());
    auto path = torch::empty({B, S}, cand.options());
    auto q = torch::empty({B, S, K}, unary.options());
    if (B == 0) return {path, q};
    auto stream = at::cuda::getCurrentCUDAStream();
    select_greedy_kernel<<<B, SEL_THREADS, 0, stream>>>(
        unary.data_ptr<float>(), cand.data_ptr<int64_t>(), proj.data_ptr<float>(), anchor.data_ptr<int64_t>(),
        reinterpret_cast<const __nv_bfloat16*>(pred_cb.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(succ_cb.data_ptr()),
        path.data_ptr<int64_t>(), q.data_ptr<float>(), S, K, R);
    return {path, q};
}

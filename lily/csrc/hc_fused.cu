// Fused mHC hyper-connections (GLM-5.3-Flash), N = 4 residual streams. The torch formulation is
// ~130 tiny kernels per collapse (20 Sinkhorn iterations), 90 collapses per token; these three
// kernels replace all of them and round where the torch path rounds.
//
//   hc_weights : w[t, r] = streams[t] . fn[r] for the W rows of fn, and sum(streams[t]^2) in w[t, W]
//   hc_collapse: per token, pre / post / comb from w (one 16-lane group runs Sinkhorn, lane = (i, j)),
//                collapsed = sum_i pre[i] * streams[i], then RMSNorm of the collapsed row
//   hc_expand  : streams'[j] = post[j] * block_out + sum_i comb[i][j] * streams[i]
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace {

constexpr int N = 4;
constexpr int W = (2 + N) * N;         // pre | post | comb
constexpr int THREADS = 256;
constexpr int VEC = 8;                 // bf16 per 16-byte load

__device__ __forceinline__ void unpack8(const uint4& v, float f[VEC]) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 x = __bfloat1622float2(p[i]);
        f[2 * i] = x.x;
        f[2 * i + 1] = x.y;
    }
}

__device__ __forceinline__ uint4 pack8(const float f[VEC]) {
    uint4 v;
    __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    return v;
}

__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ float block_sum(float v, float* red) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < THREADS / 32; ++i) s += red[i];
    __syncthreads();
    return s;
}

// grid (W + 1, T): block r < W is one dot product, block W the sum of squares
__global__ void __launch_bounds__(THREADS)
hc_weights_kernel(const __nv_bfloat16* __restrict__ streams, const __nv_bfloat16* __restrict__ fn,
                  float* __restrict__ w, int K) {
    __shared__ float red[THREADS / 32];
    const int r = blockIdx.x, t = blockIdx.y;
    const uint4* x = reinterpret_cast<const uint4*>(streams + (size_t)t * K);
    const uint4* f = reinterpret_cast<const uint4*>(fn + (size_t)(r < W ? r : 0) * K);
    float acc = 0.f;
    for (int i = threadIdx.x; i < K / VEC; i += THREADS) {
        float a[VEC], b[VEC];
        unpack8(x[i], a);
        if (r < W) {
            unpack8(f[i], b);
#pragma unroll
            for (int j = 0; j < VEC; ++j) acc += a[j] * b[j];
        } else {
#pragma unroll
            for (int j = 0; j < VEC; ++j) acc += a[j] * a[j];
        }
    }
    const float s = block_sum(acc, red);
    if (threadIdx.x == 0) w[(size_t)t * (W + 1) + r] = s;
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + __expf(-x)); }

// grid (T): lanes 0..15 of warp 0 hold comb[i][j], i = lane >> 2, j = lane & 3
__global__ void __launch_bounds__(THREADS)
hc_collapse_kernel(const __nv_bfloat16* __restrict__ streams, const float* __restrict__ w,
                   const __nv_bfloat16* __restrict__ base, const __nv_bfloat16* __restrict__ scale,
                   const __nv_bfloat16* __restrict__ norm_w, float* __restrict__ post_out,
                   float* __restrict__ comb_out, __nv_bfloat16* __restrict__ collapsed,
                   __nv_bfloat16* __restrict__ normed, int H, int iters, float hc_eps, float rms_eps, float norm_eps) {
    __shared__ float pre[N];
    __shared__ float red[THREADS / 32];
    asm volatile("griddepcontrol.launch_dependents;");  // the next GEMM may start streaming its weights
    const int t = blockIdx.x;
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        const float* wt = w + (size_t)t * (W + 1);
        const float rs = rsqrtf(wt[W] / (float)(N * H) + rms_eps);
        const float s_pre = __bfloat162float(scale[0]), s_post = __bfloat162float(scale[1]), s_comb = __bfloat162float(scale[2]);
        if (lane < N) {
            pre[lane] = sigmoidf(wt[lane] * rs * s_pre + __bfloat162float(base[lane])) + hc_eps;
            post_out[(size_t)t * N + lane] = 2.f * sigmoidf(wt[N + lane] * rs * s_post + __bfloat162float(base[N + lane]));
        }
        const int l = lane & 15;
        float c = wt[2 * N + l] * rs * s_comb + __bfloat162float(base[2 * N + l]);
        // softmax along j, then the column / row normalisations; xor 1,2 walk j, xor 4,8 walk i
        float m = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        c = __expf(c - m);
        float s = c + __shfl_xor_sync(0xffffffffu, c, 1);
        s += __shfl_xor_sync(0xffffffffu, s, 2);
        c = c / s + hc_eps;
        s = c + __shfl_xor_sync(0xffffffffu, c, 4);
        s += __shfl_xor_sync(0xffffffffu, s, 8);
        c = c / (s + hc_eps);
        for (int it = 0; it < iters - 1; ++it) {
            s = c + __shfl_xor_sync(0xffffffffu, c, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            c = c / (s + hc_eps);
            s = c + __shfl_xor_sync(0xffffffffu, c, 4);
            s += __shfl_xor_sync(0xffffffffu, s, 8);
            c = c / (s + hc_eps);
        }
        if (lane < N * N) comb_out[(size_t)t * N * N + lane] = c;
    }
    __syncthreads();
    const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
    const uint4* x = reinterpret_cast<const uint4*>(streams + (size_t)t * N * H);
    uint4* out = reinterpret_cast<uint4*>(collapsed + (size_t)t * H);
    const int nv = H / VEC;
    float ss = 0.f;
    for (int i = threadIdx.x; i < nv; i += THREADS) {
        float a[VEC], b[VEC], c[VEC], d[VEC], o[VEC];
        unpack8(x[i], a);
        unpack8(x[nv + i], b);
        unpack8(x[2 * nv + i], c);
        unpack8(x[3 * nv + i], d);
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            o[j] = round_bf16(p0 * a[j] + p1 * b[j] + p2 * c[j] + p3 * d[j]);
            ss += o[j] * o[j];
        }
        out[i] = pack8(o);
    }
    const float inv = rsqrtf(block_sum(ss, red) / (float)H + norm_eps);
    const uint4* wv = reinterpret_cast<const uint4*>(norm_w);
    uint4* nout = reinterpret_cast<uint4*>(normed + (size_t)t * H);
    for (int i = threadIdx.x; i < nv; i += THREADS) {     // each thread re-reads what it wrote
        float o[VEC], g[VEC];
        unpack8(out[i], o);
        unpack8(wv[i], g);
#pragma unroll
        for (int j = 0; j < VEC; ++j) o[j] = g[j] * round_bf16(o[j] * inv);
        nout[i] = pack8(o);
    }
}

// grid (ceil(H / VEC / THREADS), T)
__global__ void __launch_bounds__(THREADS)
hc_expand_kernel(const __nv_bfloat16* __restrict__ block_out, const float* __restrict__ post,
                 const float* __restrict__ comb, const __nv_bfloat16* __restrict__ streams,
                 __nv_bfloat16* __restrict__ out, int H) {
    const int t = blockIdx.y, nv = H / VEC;
    const int i = blockIdx.x * THREADS + threadIdx.x;
    if (i >= nv) return;
    float s[N][VEC], y[VEC];
    const uint4* x = reinterpret_cast<const uint4*>(streams + (size_t)t * N * H);
#pragma unroll
    for (int n = 0; n < N; ++n) unpack8(x[n * nv + i], s[n]);
    unpack8(reinterpret_cast<const uint4*>(block_out + (size_t)t * H)[i], y);
    uint4* o = reinterpret_cast<uint4*>(out + (size_t)t * N * H);
#pragma unroll
    for (int n = 0; n < N; ++n) {
        const float pj = round_bf16(post[(size_t)t * N + n]);
        float c[N], r[VEC];
#pragma unroll
        for (int m = 0; m < N; ++m) c[m] = round_bf16(comb[(size_t)t * N * N + m * N + n]);
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float mix = round_bf16(c[0] * s[0][j] + c[1] * s[1][j] + c[2] * s[2][j] + c[3] * s[3][j]);
            r[j] = round_bf16(pj * y[j]) + mix;
        }
        o[n * nv + i] = pack8(r);
    }
}

}  // namespace

bool hc_fused_supported(int64_t hc_mult, int64_t H) { return hc_mult == N && H % VEC == 0; }

std::vector<torch::Tensor> hc_collapse(torch::Tensor streams, torch::Tensor fn, torch::Tensor base, torch::Tensor scale,
                                       torch::Tensor norm_w, int64_t iters, double hc_eps, double rms_eps, double norm_eps) {
    TORCH_CHECK(streams.is_contiguous() && fn.is_contiguous() && streams.dim() == 3 && streams.size(1) == N);
    TORCH_CHECK(streams.scalar_type() == at::kBFloat16 && fn.scalar_type() == at::kBFloat16 && norm_w.scalar_type() == at::kBFloat16);
    const at::cuda::CUDAGuard guard(streams.device());
    const int T = (int)streams.size(0), H = (int)streams.size(2);
    auto f32 = streams.options().dtype(at::kFloat);
    auto w = torch::empty({T, W + 1}, f32);
    auto post = torch::empty({T, N}, f32);
    auto comb = torch::empty({T, N, N}, f32);
    auto collapsed = torch::empty({T, H}, streams.options());
    auto normed = torch::empty({T, H}, streams.options());
    if (T == 0) return {post, comb, collapsed, normed};
    auto stream = at::cuda::getCurrentCUDAStream();
    auto bf = [](const torch::Tensor& x) { return reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); };
    hc_weights_kernel<<<dim3(W + 1, T), THREADS, 0, stream>>>(bf(streams), bf(fn), w.data_ptr<float>(), N * H);
    hc_collapse_kernel<<<T, THREADS, 0, stream>>>(bf(streams), w.data_ptr<float>(), bf(base), bf(scale), bf(norm_w),
                                                  post.data_ptr<float>(), comb.data_ptr<float>(),
                                                  reinterpret_cast<__nv_bfloat16*>(collapsed.data_ptr()),
                                                  reinterpret_cast<__nv_bfloat16*>(normed.data_ptr()), H, (int)iters,
                                                  (float)hc_eps, (float)rms_eps, (float)norm_eps);
    return {post, comb, collapsed, normed};
}

torch::Tensor hc_expand(torch::Tensor block_out, torch::Tensor post, torch::Tensor comb, torch::Tensor streams) {
    TORCH_CHECK(block_out.is_contiguous() && post.is_contiguous() && comb.is_contiguous() && streams.is_contiguous());
    const at::cuda::CUDAGuard guard(streams.device());
    const int T = (int)streams.size(0), H = (int)streams.size(2);
    auto out = torch::empty_like(streams);
    if (T == 0) return out;
    const dim3 grid((unsigned)((H / VEC + THREADS - 1) / THREADS), (unsigned)T);
    hc_expand_kernel<<<grid, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(block_out.data_ptr()), post.data_ptr<float>(), comb.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(streams.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), H);
    return out;
}

// Fused Kimi Delta Attention (KDA) layer step: GLM-5.3-Flash linear attention.
//
// Same structure as gdn_fused.cu (one block per (head, request), thread j owns column j of
// the [KD, VD] state, varlen requests, in-place state pools with optional snapshot) with
// the KDA differences, mirroring transformers' Glm5NextTextLinearAttention:
//   * three causal convs (q, k, v; the checkpoint's q/k/v_conv1d concatenated) in fp32,
//     silu in fp32, rounded to bf16 (the module's activation dtype);
//   * l2norm as x / sqrt(sum x^2 + 1e-6) in fp32, q scaled by KD^-0.5;
//   * the forget gate is a *vector* per head: g[i] = lb * sigmoid(exp(A_log[h]) * (f[i] + dt_bias[h, i]))
//     with f = f_b(f_a(x)) (bf16, computed outside), lb = -5 by default; the state row i
//     decays by exp(g[i]);
//   * beta = sigmoid(b_proj(x)) rounded to bf16 like the reference;
//   * o is rounded to bf16, then RMSNorm in fp32 with an fp32 weight, times sigmoid(gate)
//     in fp32 (gate = g_b(g_a(x)) bf16), one final cast.
// Heads: q, k, v all have H heads (no grouping).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace {

template <int N>
__device__ __forceinline__ float2 block_sum2(float2 v, float2* red) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v.x += __shfl_xor_sync(0xffffffffu, v.x, o);
        v.y += __shfl_xor_sync(0xffffffffu, v.y, o);
    }
    constexpr int WARPS = N / 32;
    if (WARPS == 1) return v;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float2 total = make_float2(0.f, 0.f);
#pragma unroll
    for (int w = 0; w < WARPS; ++w) { total.x += red[w].x; total.y += red[w].y; }
    return total;
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16(x)); }

template <typename T> __device__ __forceinline__ float to_f32(T v);
template <> __device__ __forceinline__ float to_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <int KD, int VD, int KERNEL, typename StateT, bool WRITE_OUT>
__device__ __forceinline__ void
kda_block(const int h, const int b,
          const __nv_bfloat16* __restrict__ proj, int proj_stride,     // [T, >= 3*H*KD]: q | k | v
          const __nv_bfloat16* __restrict__ gate, int gate_stride,     // [T, H*VD]  g_b(g_a(x))
          const __nv_bfloat16* __restrict__ beta_raw, int beta_stride, // [T, H]     b_proj(x)
          const __nv_bfloat16* __restrict__ fgate, int fgate_stride,   // [T, H*KD]  f_b(f_a(x))
          const __nv_bfloat16* __restrict__ conv_w,                    // [3*H*KD, KERNEL]
          const float* __restrict__ A_log,                             // [H]
          const float* __restrict__ dt_bias,                           // [H*KD]
          float lower_bound,
          const float* __restrict__ norm_w,                            // [VD] fp32
          float eps,
          const int32_t* __restrict__ tok_starts, const int32_t* __restrict__ tok_lens,
          const int64_t* __restrict__ slots, const uint8_t* __restrict__ fresh,
          const StateT* __restrict__ conv_in, StateT* __restrict__ conv_out,
          const StateT* __restrict__ rec_in, StateT* __restrict__ rec_out,
          StateT* __restrict__ conv_snap, StateT* __restrict__ rec_snap,
          __nv_bfloat16* __restrict__ out, int H) {
    static_assert(KD == VD, "thread j owns q[j], k[j], v[j] and state column j");
    const int j = threadIdx.x;
    const int t0 = tok_starts[b], t1 = t0 + tok_lens[b];
    if (t0 == t1) return;

    const int HD = H * KD;
    const int conv_dim = 3 * HD;
    const int cq = h * KD + j, ck = HD + h * KD + j, cv = 2 * HD + h * VD + j;
    const int64_t slot = slots[b];
    const bool is_fresh = fresh != nullptr && fresh[b] != 0;

    __shared__ float sq[KD];
    __shared__ float sk[KD];
    __shared__ float sg[KD];
    __shared__ float2 red2[32];

    float S[KD];
    const size_t rec_base = (size_t)slot * H * KD * VD + (size_t)h * KD * VD + j;
    if (is_fresh) {
#pragma unroll
        for (int i = 0; i < KD; ++i) S[i] = 0.f;
    } else {
#pragma unroll
        for (int i = 0; i < KD; ++i) S[i] = to_f32<StateT>(rec_in[rec_base + (size_t)i * VD]);
    }
    if (rec_snap != nullptr) {
#pragma unroll
        for (int i = 0; i < KD; ++i) rec_snap[rec_base + (size_t)i * VD] = from_f32<StateT>(S[i]);
    }
    float wq[KERNEL], wk[KERNEL], wv[KERNEL];
    const size_t conv_base = (size_t)slot * conv_dim * KERNEL;
#pragma unroll
    for (int t = 0; t < KERNEL; ++t) {
        if (is_fresh) { wq[t] = wk[t] = wv[t] = 0.f; }
        else {
            wq[t] = to_f32<StateT>(conv_in[conv_base + (size_t)cq * KERNEL + t]);
            wk[t] = to_f32<StateT>(conv_in[conv_base + (size_t)ck * KERNEL + t]);
            wv[t] = to_f32<StateT>(conv_in[conv_base + (size_t)cv * KERNEL + t]);
        }
    }
    if (conv_snap != nullptr) {
#pragma unroll
        for (int t = 0; t < KERNEL; ++t) {
            conv_snap[conv_base + (size_t)cq * KERNEL + t] = from_f32<StateT>(wq[t]);
            conv_snap[conv_base + (size_t)ck * KERNEL + t] = from_f32<StateT>(wk[t]);
            conv_snap[conv_base + (size_t)cv * KERNEL + t] = from_f32<StateT>(wv[t]);
        }
    }
    float fq[KERNEL], fk[KERNEL], fv[KERNEL];
#pragma unroll
    for (int t = 0; t < KERNEL; ++t) {
        fq[t] = __bfloat162float(conv_w[(size_t)cq * KERNEL + t]);
        fk[t] = __bfloat162float(conv_w[(size_t)ck * KERNEL + t]);
        fv[t] = __bfloat162float(conv_w[(size_t)cv * KERNEL + t]);
    }
    const float expA = __expf(A_log[h]);
    const float dtb = dt_bias[h * KD + j];
    const float q_scale = rsqrtf((float)KD);
    __nv_bfloat16* orow0 = WRITE_OUT ? out + (size_t)h * VD + j : nullptr;
    const size_t ostride = (size_t)H * VD;

    for (int t = t0; t < t1; ++t) {
        const __nv_bfloat16* row = proj + (size_t)t * proj_stride;
        const float xq = __bfloat162float(row[cq]);
        const float xk = __bfloat162float(row[ck]);
        const float xv = __bfloat162float(row[cv]);
        float q = fq[KERNEL - 1] * xq, k = fk[KERNEL - 1] * xk, v = fv[KERNEL - 1] * xv;
#pragma unroll
        for (int i = 0; i < KERNEL - 1; ++i) {
            q = fmaf(fq[i], wq[i + 1], q);
            k = fmaf(fk[i], wk[i + 1], k);
            v = fmaf(fv[i], wv[i + 1], v);
        }
#pragma unroll
        for (int i = 0; i < KERNEL - 1; ++i) { wq[i] = wq[i + 1]; wk[i] = wk[i + 1]; wv[i] = wv[i + 1]; }
        wq[KERNEL - 1] = xq; wk[KERNEL - 1] = xk; wv[KERNEL - 1] = xv;
        // conv and silu in fp32, output in the activation dtype (bf16)
        q = bf16r(silu_f(q));
        k = bf16r(silu_f(k));
        v = bf16r(silu_f(v));

        const float2 norms = block_sum2<VD>(make_float2(q * q, k * k), red2);
        q = q / sqrtf(norms.x + 1e-6f) * q_scale;
        k = k / sqrtf(norms.y + 1e-6f);
        // per-channel forget gate for this head: row i of the state decays by exp(g[i])
        const float f = __bfloat162float(fgate[(size_t)t * fgate_stride + h * KD + j]);
        const float g = lower_bound * sigmoid_f(expA * (f + dtb));
        sq[j] = q;
        sk[j] = k;
        sg[j] = __expf(g);
        __syncthreads();

        const float beta = bf16r(sigmoid_f(__bfloat162float(beta_raw[(size_t)t * beta_stride + h])));
        float kv0 = 0.f, kv1 = 0.f, kv2 = 0.f, kv3 = 0.f;
#pragma unroll
        for (int i = 0; i < KD; i += 4) {
            S[i] *= sg[i]; S[i + 1] *= sg[i + 1]; S[i + 2] *= sg[i + 2]; S[i + 3] *= sg[i + 3];
            kv0 = fmaf(S[i], sk[i], kv0);
            kv1 = fmaf(S[i + 1], sk[i + 1], kv1);
            kv2 = fmaf(S[i + 2], sk[i + 2], kv2);
            kv3 = fmaf(S[i + 3], sk[i + 3], kv3);
        }
        const float delta = (v - ((kv0 + kv1) + (kv2 + kv3))) * beta;
        float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
#pragma unroll
        for (int i = 0; i < KD; i += 4) {
            S[i] = fmaf(sk[i], delta, S[i]);
            S[i + 1] = fmaf(sk[i + 1], delta, S[i + 1]);
            S[i + 2] = fmaf(sk[i + 2], delta, S[i + 2]);
            S[i + 3] = fmaf(sk[i + 3], delta, S[i + 3]);
            o0 = fmaf(S[i], sq[i], o0);
            o1 = fmaf(S[i + 1], sq[i + 1], o1);
            o2 = fmaf(S[i + 2], sq[i + 2], o2);
            o3 = fmaf(S[i + 3], sq[i + 3], o3);
        }
        if (WRITE_OUT) orow0[(size_t)t * ostride] = __float2bfloat16((o0 + o1) + (o2 + o3));
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < KD; ++i) rec_out[rec_base + (size_t)i * VD] = from_f32<StateT>(S[i]);
#pragma unroll
    for (int t = 0; t < KERNEL; ++t) {
        conv_out[conv_base + (size_t)cq * KERNEL + t] = from_f32<StateT>(wq[t]);
        conv_out[conv_base + (size_t)ck * KERNEL + t] = from_f32<StateT>(wk[t]);
        conv_out[conv_base + (size_t)cv * KERNEL + t] = from_f32<StateT>(wv[t]);
    }

    if (!WRITE_OUT) return;
    __syncthreads();
    // gated RMSNorm pass, one warp per token: fp32 norm, fp32 weight, sigmoid(gate) in fp32
    constexpr int PER_LANE = VD / 32;
    constexpr int WARPS = VD / 32;
    const int lane = j & 31, warp = j >> 5;
    for (int t = t0 + warp; t < t1; t += WARPS) {
        __nv_bfloat16* orow = out + (size_t)t * ostride + (size_t)h * VD;
        const __nv_bfloat16* grow = gate + (size_t)t * gate_stride + (size_t)h * VD;
        float o[PER_LANE];
        float ss = 0.f;
#pragma unroll
        for (int c = 0; c < PER_LANE; ++c) { o[c] = __bfloat162float(orow[lane * PER_LANE + c]); ss = fmaf(o[c], o[c], ss); }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
        const float inv = rsqrtf(ss * (1.f / VD) + eps);
#pragma unroll
        for (int c = 0; c < PER_LANE; ++c) {
            const int col = lane * PER_LANE + c;
            const float r = norm_w[col] * (o[c] * inv) * sigmoid_f(__bfloat162float(grow[col]));
            orow[col] = __float2bfloat16(r);
        }
    }
}

template <int KD, int VD, int KERNEL, typename StateT>
__global__ void __launch_bounds__(VD)
kda_fused_kernel(const __nv_bfloat16* __restrict__ proj, int proj_stride,
                 const __nv_bfloat16* __restrict__ gate, int gate_stride,
                 const __nv_bfloat16* __restrict__ beta_raw, int beta_stride,
                 const __nv_bfloat16* __restrict__ fgate, int fgate_stride,
                 const __nv_bfloat16* __restrict__ conv_w, const float* __restrict__ A_log,
                 const float* __restrict__ dt_bias, float lower_bound, const float* __restrict__ norm_w, float eps,
                 const int32_t* __restrict__ tok_starts, const int32_t* __restrict__ tok_lens,
                 const int64_t* __restrict__ slots, const uint8_t* __restrict__ fresh,
                 const StateT* __restrict__ conv_in, StateT* __restrict__ conv_out,
                 const StateT* __restrict__ rec_in, StateT* __restrict__ rec_out,
                 StateT* __restrict__ conv_snap, StateT* __restrict__ rec_snap,
                 __nv_bfloat16* __restrict__ out, int H) {
    kda_block<KD, VD, KERNEL, StateT, true>(blockIdx.x, blockIdx.y, proj, proj_stride, gate, gate_stride, beta_raw, beta_stride,
                                            fgate, fgate_stride, conv_w, A_log, dt_bias, lower_bound, norm_w, eps,
                                            tok_starts, tok_lens, slots, fresh, conv_in, conv_out, rec_in, rec_out,
                                            conv_snap, rec_snap, out, H);
}

// Pointer-table columns for the multi-layer replay (one row per layer).
enum KdaReplayPtr { KP_PROJ = 0, KP_BETA, KP_FGATE, KP_CONV_W, KP_A_LOG, KP_DT_BIAS, KP_CONV_IN, KP_CONV_OUT, KP_REC_IN, KP_REC_OUT, KP_COUNT };

template <int KD, int VD, int KERNEL, typename StateT>
__global__ void __launch_bounds__(VD)
kda_replay_kernel(const int64_t* __restrict__ table, int proj_stride, int beta_stride, int fgate_stride, float lower_bound,
                  const int32_t* __restrict__ tok_starts, const int32_t* __restrict__ tok_lens,
                  const int64_t* __restrict__ slots, int H) {
    const int64_t* row = table + (size_t)blockIdx.z * KP_COUNT;
    kda_block<KD, VD, KERNEL, StateT, false>(
        blockIdx.x, blockIdx.y,
        reinterpret_cast<const __nv_bfloat16*>(row[KP_PROJ]), proj_stride, nullptr, 0,
        reinterpret_cast<const __nv_bfloat16*>(row[KP_BETA]), beta_stride,
        reinterpret_cast<const __nv_bfloat16*>(row[KP_FGATE]), fgate_stride,
        reinterpret_cast<const __nv_bfloat16*>(row[KP_CONV_W]),
        reinterpret_cast<const float*>(row[KP_A_LOG]), reinterpret_cast<const float*>(row[KP_DT_BIAS]), lower_bound,
        nullptr, 0.f, tok_starts, tok_lens, slots, nullptr,
        reinterpret_cast<const StateT*>(row[KP_CONV_IN]), reinterpret_cast<StateT*>(row[KP_CONV_OUT]),
        reinterpret_cast<const StateT*>(row[KP_REC_IN]), reinterpret_cast<StateT*>(row[KP_REC_OUT]),
        nullptr, nullptr, nullptr, H);
}

}  // namespace

bool kda_fused_supported(int64_t head_dim, int64_t kernel_size, c10::ScalarType state_dtype) {
    return (head_dim == 128 || head_dim == 64 || head_dim == 32) && kernel_size == 4
        && (state_dtype == torch::kFloat32 || state_dtype == torch::kBFloat16);
}

torch::Tensor kda_fused_forward(torch::Tensor proj, torch::Tensor gate, torch::Tensor beta_raw, torch::Tensor fgate,
                                torch::Tensor conv_w, torch::Tensor A_log, torch::Tensor dt_bias, double lower_bound,
                                torch::Tensor norm_w, double eps, torch::Tensor tok_starts, torch::Tensor tok_lens,
                                torch::Tensor slots, c10::optional<torch::Tensor> fresh,
                                torch::Tensor conv_in, torch::Tensor conv_out, torch::Tensor rec_in, torch::Tensor rec_out,
                                c10::optional<torch::Tensor> conv_snap, c10::optional<torch::Tensor> rec_snap) {
    TORCH_CHECK(proj.is_cuda() && proj.dim() == 2 && proj.scalar_type() == torch::kBFloat16 && proj.stride(1) == 1, "kda: proj");
    TORCH_CHECK(gate.dim() == 2 && gate.scalar_type() == torch::kBFloat16 && gate.stride(1) == 1, "kda: gate");
    TORCH_CHECK(beta_raw.dim() == 2 && beta_raw.scalar_type() == torch::kBFloat16 && beta_raw.stride(1) == 1, "kda: beta");
    TORCH_CHECK(fgate.dim() == 2 && fgate.scalar_type() == torch::kBFloat16 && fgate.stride(1) == 1, "kda: fgate");
    TORCH_CHECK(rec_in.dim() == 4 && rec_out.sizes() == rec_in.sizes() && rec_in.is_contiguous() && rec_out.is_contiguous(), "kda: rec states");
    TORCH_CHECK(conv_in.dim() == 3 && conv_out.sizes() == conv_in.sizes() && conv_in.is_contiguous() && conv_out.is_contiguous(), "kda: conv states");
    const int64_t H = rec_in.size(1), KD = rec_in.size(2), VD = rec_in.size(3), KERNEL = conv_in.size(2);
    TORCH_CHECK(KD == VD && conv_in.size(1) == 3 * H * KD, "kda: dims");
    TORCH_CHECK(conv_w.is_contiguous() && conv_w.scalar_type() == torch::kBFloat16 && conv_w.size(0) == 3 * H * KD && conv_w.size(1) == KERNEL, "kda: conv weight");
    TORCH_CHECK(A_log.scalar_type() == torch::kFloat32 && A_log.numel() == H && A_log.is_contiguous(), "kda: A_log");
    TORCH_CHECK(dt_bias.scalar_type() == torch::kFloat32 && dt_bias.numel() == H * KD && dt_bias.is_contiguous(), "kda: dt_bias");
    TORCH_CHECK(norm_w.scalar_type() == torch::kFloat32 && norm_w.numel() == VD && norm_w.is_contiguous(), "kda: norm weight (fp32)");
    TORCH_CHECK(gate.size(1) == H * VD && fgate.size(1) == H * KD && beta_raw.size(1) == H, "kda: gate/fgate/beta widths");
    TORCH_CHECK(tok_starts.scalar_type() == torch::kInt32 && tok_lens.scalar_type() == torch::kInt32 && slots.scalar_type() == torch::kInt64, "kda: meta");
    const auto st = rec_in.scalar_type();
    TORCH_CHECK(kda_fused_supported(KD, KERNEL, st), "kda: unsupported head dim / kernel / state dtype");

    const c10::cuda::OptionalCUDAGuard guard(proj.device());
    const int B = (int)slots.numel();
    auto out = torch::empty({proj.size(0), H * VD}, proj.options());
    if (B == 0 || proj.size(0) == 0) return out;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(H, B);
#define LILY_KDA_LAUNCH(KD_, T_)                                                                                      \
    kda_fused_kernel<KD_, KD_, 4, T_><<<grid, KD_, 0, stream>>>(                                                       \
        reinterpret_cast<const __nv_bfloat16*>(proj.data_ptr<at::BFloat16>()), (int)proj.stride(0),                  \
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr<at::BFloat16>()), (int)gate.stride(0),                  \
        reinterpret_cast<const __nv_bfloat16*>(beta_raw.data_ptr<at::BFloat16>()), (int)beta_raw.stride(0),          \
        reinterpret_cast<const __nv_bfloat16*>(fgate.data_ptr<at::BFloat16>()), (int)fgate.stride(0),                \
        reinterpret_cast<const __nv_bfloat16*>(conv_w.data_ptr<at::BFloat16>()), A_log.data_ptr<float>(),            \
        dt_bias.data_ptr<float>(), (float)lower_bound, norm_w.data_ptr<float>(), (float)eps,                         \
        tok_starts.data_ptr<int32_t>(), tok_lens.data_ptr<int32_t>(), slots.data_ptr<int64_t>(),                     \
        fresh.has_value() ? fresh->data_ptr<uint8_t>() : nullptr,                                                     \
        reinterpret_cast<const T_*>(conv_in.data_ptr()), reinterpret_cast<T_*>(conv_out.data_ptr()),                 \
        reinterpret_cast<const T_*>(rec_in.data_ptr()), reinterpret_cast<T_*>(rec_out.data_ptr()),                   \
        conv_snap.has_value() ? reinterpret_cast<T_*>(conv_snap->data_ptr()) : nullptr,                              \
        rec_snap.has_value() ? reinterpret_cast<T_*>(rec_snap->data_ptr()) : nullptr,                                \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), (int)H)
    if (st == torch::kFloat32) {
        if (KD == 128) LILY_KDA_LAUNCH(128, float); else if (KD == 64) LILY_KDA_LAUNCH(64, float); else LILY_KDA_LAUNCH(32, float);
    } else {
        if (KD == 128) LILY_KDA_LAUNCH(128, __nv_bfloat16); else if (KD == 64) LILY_KDA_LAUNCH(64, __nv_bfloat16); else LILY_KDA_LAUNCH(32, __nv_bfloat16);
    }
#undef LILY_KDA_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void kda_fused_replay(torch::Tensor table, int64_t proj_stride, int64_t beta_stride, int64_t fgate_stride, double lower_bound,
                      torch::Tensor tok_starts, torch::Tensor tok_lens, torch::Tensor slots,
                      int64_t num_heads, int64_t head_dim, int64_t kernel_size, c10::ScalarType state_dtype) {
    TORCH_CHECK(table.is_cuda() && table.scalar_type() == torch::kInt64 && table.dim() == 2 && table.size(1) == KP_COUNT
                && table.is_contiguous(), "kda_replay: table must be int64 [layers, ", (int)KP_COUNT, "]");
    TORCH_CHECK(kda_fused_supported(head_dim, kernel_size, state_dtype), "kda_replay: unsupported dims");
    const c10::cuda::OptionalCUDAGuard guard(table.device());
    const int B = (int)slots.numel(), L = (int)table.size(0);
    if (B == 0 || L == 0) return;
    const dim3 grid(num_heads, B, L);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define LILY_KDA_REPLAY(KD_, T_) \
    kda_replay_kernel<KD_, KD_, 4, T_><<<grid, KD_, 0, stream>>>(table.data_ptr<int64_t>(), (int)proj_stride, (int)beta_stride, \
        (int)fgate_stride, (float)lower_bound, tok_starts.data_ptr<int32_t>(), tok_lens.data_ptr<int32_t>(), slots.data_ptr<int64_t>(), (int)num_heads)
    if (state_dtype == torch::kFloat32) {
        if (head_dim == 128) LILY_KDA_REPLAY(128, float); else if (head_dim == 64) LILY_KDA_REPLAY(64, float); else LILY_KDA_REPLAY(32, float);
    } else {
        if (head_dim == 128) LILY_KDA_REPLAY(128, __nv_bfloat16); else if (head_dim == 64) LILY_KDA_REPLAY(64, __nv_bfloat16); else LILY_KDA_REPLAY(32, __nv_bfloat16);
    }
#undef LILY_KDA_REPLAY
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Fused Gated DeltaNet layer step (Qwen3-Next / Qwen3.5 linear attention).
//
// One launch runs, for every token of every request in the batch:
//   causal conv1d (+ state update)  ->  silu  ->  l2-norm q/k  ->  gated delta
//   rule recurrence (+ state update) ->  per-head RMSNorm * silu(z)
// and leaves a [T, nv*VD] bf16 tensor ready for out_proj.
//
// Grid: one block per (v-head, request). Thread j owns column j of the
// [KD, VD] recurrent state, kept in registers for the whole token loop, so the
// state is read from HBM once and written once per call however many tokens
// the request extends by. q and k for the current token are exchanged through
// shared memory. Requests are variable length (tok_starts/tok_lens), so prefill,
// chunked extend, speculative verify and single-token decode all share this
// kernel; decode is simply T == B.
//
// State pools are [slots, ...] and addressed by `slots[b]`. `*_in` may be
// `*_out` (in-place update) or a different buffer (replay from a snapshot).
// `*_snap`, when given, receives a copy of the *input* state so a speculative
// verify can be rolled back by replaying the accepted prefix from the snapshot.
// `fresh[b] != 0` starts request b from the zero state without reading the pool.
//
// Math mirrors lily/layers/attention/gated_delta_rule.py (the eager reference):
//   q, k, v = silu(conv(x));  q = l2n(q) * KD^-0.5;  k = l2n(k)
//   beta = sigmoid(b);  g = -exp(A_log) * softplus(a + dt_bias)
//   S = S * exp(g);  o = S^T q after  S += k (v - S^T k)^T beta
//   out = rmsnorm(o) * w * silu(z)

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace {

template <int N>
__device__ __forceinline__ float2 block_sum2(float2 v, float2* red) {
    // Two sums at once over the N threads of the block.
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
__device__ __forceinline__ float softplus_f(float x) {
    // torch.nn.functional.softplus default (beta=1, threshold=20)
    return x > 20.f ? x : log1pf(__expf(x));
}

template <typename T> __device__ __forceinline__ float to_f32(T v);
template <> __device__ __forceinline__ float to_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <int KD, int VD, int KERNEL, typename StateT, bool WRITE_OUT>
__device__ __forceinline__ void
gdn_block(const int h, const int b,
          const __nv_bfloat16* __restrict__ proj, int proj_stride,
          const __nv_bfloat16* __restrict__ conv_w,     // [conv_dim, KERNEL]
          const float* __restrict__ A_log,              // [nv]
          const float* __restrict__ dt_bias,            // [nv]
          const __nv_bfloat16* __restrict__ norm_w,     // [VD]
          float eps,
          const int32_t* __restrict__ tok_starts,       // [B] first proj row of each request
          const int32_t* __restrict__ tok_lens,         // [B] tokens of each request
          const int64_t* __restrict__ slots,            // [B]
          const uint8_t* __restrict__ fresh,            // [B] or nullptr
          const StateT* __restrict__ conv_in, StateT* __restrict__ conv_out,
          const StateT* __restrict__ rec_in, StateT* __restrict__ rec_out,
          StateT* __restrict__ conv_snap, StateT* __restrict__ rec_snap,
          __nv_bfloat16* __restrict__ out,
          int nk, int nv) {
    static_assert(KD == VD, "thread j owns q[j], k[j], v[j] and state column j");
    const int j = threadIdx.x;
    const int t0 = tok_starts[b], t1 = t0 + tok_lens[b];
    if (t0 == t1) return;

    const int v_per_k = nv / nk;
    const int kh = h / v_per_k;
    const int conv_dim = 2 * nk * KD + nv * VD;
    const int cq = kh * KD + j;                   // conv channel of q[j]
    const int ck = nk * KD + kh * KD + j;         // conv channel of k[j]
    const int cv = 2 * nk * KD + h * VD + j;      // conv channel of v[j]
    const int cz = conv_dim + h * VD + j;         // z column
    const int cb = conv_dim + nv * VD + h;        // b column
    const int ca = conv_dim + nv * VD + nv + h;   // a column

    const int64_t slot = slots[b];
    const bool is_fresh = fresh != nullptr && fresh[b] != 0;

    __shared__ float sq[KD];
    __shared__ float sk[KD];
    __shared__ float2 red2[32];

    // ---- load state ------------------------------------------------------
    float S[KD];
    const size_t rec_base = (size_t)slot * nv * KD * VD + (size_t)h * KD * VD + j;
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

    float wq[KERNEL], wk[KERNEL], wv[KERNEL];   // conv windows (oldest first)
    const size_t conv_base = (size_t)slot * conv_dim * KERNEL;
#pragma unroll
    for (int t = 0; t < KERNEL; ++t) {
        if (is_fresh) {
            wq[t] = wk[t] = wv[t] = 0.f;
        } else {
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
    float fq[KERNEL], fk[KERNEL], fv[KERNEL];   // conv taps
#pragma unroll
    for (int t = 0; t < KERNEL; ++t) {
        fq[t] = __bfloat162float(conv_w[(size_t)cq * KERNEL + t]);
        fk[t] = __bfloat162float(conv_w[(size_t)ck * KERNEL + t]);
        fv[t] = __bfloat162float(conv_w[(size_t)cv * KERNEL + t]);
    }
    const float neg_expA = -__expf(A_log[h]);
    const float dtb = dt_bias[h];
    const float q_scale = rsqrtf((float)KD);

    // ---- token loop: recurrence ---------------------------------------------
    __nv_bfloat16* orow0 = WRITE_OUT ? out + (size_t)h * VD + j : nullptr;
    const size_t ostride = (size_t)nv * VD;
    for (int t = t0; t < t1; ++t) {
        const __nv_bfloat16* row = proj + (size_t)t * proj_stride;
        const float xq = __bfloat162float(row[cq]);
        const float xk = __bfloat162float(row[ck]);
        const float xv = __bfloat162float(row[cv]);

        // causal conv over [window[1..], x]; then shift the window
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
        // The reference runs the conv and silu in bf16 (activation dtype): round
        // the conv output and the silu output like it does.
        q = __bfloat162float(__float2bfloat16(silu_f(__bfloat162float(__float2bfloat16(q)))));
        k = __bfloat162float(__float2bfloat16(silu_f(__bfloat162float(__float2bfloat16(k)))));
        v = __bfloat162float(__float2bfloat16(silu_f(__bfloat162float(__float2bfloat16(v)))));

        const float2 norms = block_sum2<VD>(make_float2(q * q, k * k), red2);
        q = q * rsqrtf(norms.x + 1e-6f) * q_scale;
        k = k * rsqrtf(norms.y + 1e-6f);
        sq[j] = q;
        sk[j] = k;
        __syncthreads();

        const float beta = sigmoid_f(__bfloat162float(row[cb]));
        const float g = neg_expA * softplus_f(__bfloat162float(row[ca]) + dtb);
        const float decay = __expf(g);

        // four independent partial sums keep the FMA chains short
        float kv0 = 0.f, kv1 = 0.f, kv2 = 0.f, kv3 = 0.f;
#pragma unroll
        for (int i = 0; i < KD; i += 4) {
            S[i] *= decay; S[i + 1] *= decay; S[i + 2] *= decay; S[i + 3] *= decay;
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
        // raw o in bf16, like the reference's `o.to(output.dtype)`; normed in the pass below
        if (WRITE_OUT) orow0[(size_t)t * ostride] = __float2bfloat16((o0 + o1) + (o2 + o3));
        __syncthreads();  // sq/sk reused next token
    }

    // ---- write state ----------------------------------------------------------
#pragma unroll
    for (int i = 0; i < KD; ++i) rec_out[rec_base + (size_t)i * VD] = from_f32<StateT>(S[i]);
#pragma unroll
    for (int t = 0; t < KERNEL; ++t) {
        conv_out[conv_base + (size_t)cq * KERNEL + t] = from_f32<StateT>(wq[t]);
        conv_out[conv_base + (size_t)ck * KERNEL + t] = from_f32<StateT>(wk[t]);
        conv_out[conv_base + (size_t)cv * KERNEL + t] = from_f32<StateT>(wv[t]);
    }

    // ---- gated RMSNorm pass: one warp per token, VD/32 columns per lane ---------
    // (independent across tokens, so no block-wide sync on the sequential path)
    if (!WRITE_OUT) return;
    __syncthreads();  // all raw o rows of this block are written
    constexpr int PER_LANE = VD / 32;
    const int lane = j & 31, warp = j >> 5;
    constexpr int WARPS = VD / 32;
    for (int t = t0 + warp; t < t1; t += WARPS) {
        __nv_bfloat16* orow = out + (size_t)t * ostride + (size_t)h * VD;
        const __nv_bfloat16* row = proj + (size_t)t * proj_stride + conv_dim + (size_t)h * VD;
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
            // reference: norm in fp32 -> bf16, * bf16 weight -> bf16, * bf16 silu(z) -> bf16
            float r = __bfloat162float(__float2bfloat16(o[c] * inv));
            r = __bfloat162float(__float2bfloat16(__bfloat162float(norm_w[col]) * r));
            r = r * __bfloat162float(__float2bfloat16(silu_f(__bfloat162float(row[col]))));
            orow[col] = __float2bfloat16(r);
        }
    }
}

template <int KD, int VD, int KERNEL, typename StateT>
__global__ void __launch_bounds__(VD)
gdn_fused_kernel(const __nv_bfloat16* __restrict__ proj, int proj_stride,
                 const __nv_bfloat16* __restrict__ conv_w, const float* __restrict__ A_log,
                 const float* __restrict__ dt_bias, const __nv_bfloat16* __restrict__ norm_w, float eps,
                 const int32_t* __restrict__ tok_starts, const int32_t* __restrict__ tok_lens,
                 const int64_t* __restrict__ slots, const uint8_t* __restrict__ fresh,
                 const StateT* __restrict__ conv_in, StateT* __restrict__ conv_out,
                 const StateT* __restrict__ rec_in, StateT* __restrict__ rec_out,
                 StateT* __restrict__ conv_snap, StateT* __restrict__ rec_snap,
                 __nv_bfloat16* __restrict__ out, int nk, int nv) {
    gdn_block<KD, VD, KERNEL, StateT, true>(blockIdx.x, blockIdx.y, proj, proj_stride, conv_w, A_log, dt_bias, norm_w,
                                            eps, tok_starts, tok_lens, slots, fresh, conv_in, conv_out, rec_in, rec_out,
                                            conv_snap, rec_snap, out, nk, nv);
}

// Pointer-table columns for the multi-layer replay (one row per layer).
enum ReplayPtr { RP_PROJ = 0, RP_CONV_W, RP_A_LOG, RP_DT_BIAS, RP_CONV_IN, RP_CONV_OUT, RP_REC_IN, RP_REC_OUT, RP_COUNT };

template <int KD, int VD, int KERNEL, typename StateT>
__global__ void __launch_bounds__(VD)
gdn_replay_kernel(const int64_t* __restrict__ table, int proj_stride,
                  const int32_t* __restrict__ tok_starts, const int32_t* __restrict__ tok_lens,
                  const int64_t* __restrict__ slots, int nk, int nv) {
    // States only (no outputs): re-run the recurrence of every layer over the
    // accepted prefix, from the snapshot into the live pool. blockIdx.z = layer.
    const int64_t* row = table + (size_t)blockIdx.z * RP_COUNT;
    gdn_block<KD, VD, KERNEL, StateT, false>(
        blockIdx.x, blockIdx.y,
        reinterpret_cast<const __nv_bfloat16*>(row[RP_PROJ]), proj_stride,
        reinterpret_cast<const __nv_bfloat16*>(row[RP_CONV_W]),
        reinterpret_cast<const float*>(row[RP_A_LOG]), reinterpret_cast<const float*>(row[RP_DT_BIAS]),
        nullptr, 0.f, tok_starts, tok_lens, slots, nullptr,
        reinterpret_cast<const StateT*>(row[RP_CONV_IN]), reinterpret_cast<StateT*>(row[RP_CONV_OUT]),
        reinterpret_cast<const StateT*>(row[RP_REC_IN]), reinterpret_cast<StateT*>(row[RP_REC_OUT]),
        nullptr, nullptr, nullptr, nk, nv);
}

template <int KD, int VD, int KERNEL, typename StateT>
void launch_gdn(const torch::Tensor& proj, const torch::Tensor& conv_w, const torch::Tensor& A_log,
                const torch::Tensor& dt_bias, const torch::Tensor& norm_w, double eps,
                const torch::Tensor& tok_starts, const torch::Tensor& tok_lens, const torch::Tensor& slots,
                const c10::optional<torch::Tensor>& fresh,
                const torch::Tensor& conv_in, torch::Tensor& conv_out, const torch::Tensor& rec_in, torch::Tensor& rec_out,
                const c10::optional<torch::Tensor>& conv_snap, const c10::optional<torch::Tensor>& rec_snap,
                torch::Tensor& out, int nk, int nv, int B, cudaStream_t stream) {
    const dim3 grid(nv, B);
    gdn_fused_kernel<KD, VD, KERNEL, StateT><<<grid, VD, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(proj.data_ptr<at::BFloat16>()), (int)proj.stride(0),
        reinterpret_cast<const __nv_bfloat16*>(conv_w.data_ptr<at::BFloat16>()),
        A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr<at::BFloat16>()), (float)eps,
        tok_starts.data_ptr<int32_t>(), tok_lens.data_ptr<int32_t>(), slots.data_ptr<int64_t>(),
        fresh.has_value() ? fresh->data_ptr<uint8_t>() : nullptr,
        reinterpret_cast<const StateT*>(conv_in.data_ptr()), reinterpret_cast<StateT*>(conv_out.data_ptr()),
        reinterpret_cast<const StateT*>(rec_in.data_ptr()), reinterpret_cast<StateT*>(rec_out.data_ptr()),
        conv_snap.has_value() ? reinterpret_cast<StateT*>(conv_snap->data_ptr()) : nullptr,
        rec_snap.has_value() ? reinterpret_cast<StateT*>(rec_snap->data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), nk, nv);
}

}  // namespace

bool gdn_fused_supported(int64_t k_head_dim, int64_t v_head_dim, int64_t kernel_size, c10::ScalarType state_dtype) {
    const bool dims = (k_head_dim == v_head_dim) && (k_head_dim == 128 || k_head_dim == 64 || k_head_dim == 32);
    const bool dt = state_dtype == torch::kFloat32 || state_dtype == torch::kBFloat16;
    return dims && kernel_size == 4 && dt;
}

torch::Tensor gdn_fused_forward(torch::Tensor proj, torch::Tensor conv_w, torch::Tensor A_log, torch::Tensor dt_bias,
                                torch::Tensor norm_w, double eps, torch::Tensor tok_starts, torch::Tensor tok_lens,
                                torch::Tensor slots, c10::optional<torch::Tensor> fresh,
                                torch::Tensor conv_in, torch::Tensor conv_out, torch::Tensor rec_in, torch::Tensor rec_out,
                                c10::optional<torch::Tensor> conv_snap, c10::optional<torch::Tensor> rec_snap,
                                int64_t num_k_heads) {
    TORCH_CHECK(proj.is_cuda() && proj.dim() == 2 && proj.scalar_type() == torch::kBFloat16 && proj.stride(1) == 1,
                "gdn_fused: proj must be a bf16 [T, N] tensor with unit column stride");
    TORCH_CHECK(rec_in.dim() == 4 && rec_out.sizes() == rec_in.sizes(), "gdn_fused: recurrent states must be [slots, nv, KD, VD]");
    TORCH_CHECK(conv_in.dim() == 3 && conv_out.sizes() == conv_in.sizes(), "gdn_fused: conv states must be [slots, conv_dim, KERNEL]");
    TORCH_CHECK(rec_in.is_contiguous() && rec_out.is_contiguous() && conv_in.is_contiguous() && conv_out.is_contiguous());
    const int64_t nv = rec_in.size(1), KD = rec_in.size(2), VD = rec_in.size(3), KERNEL = conv_in.size(2);
    const int64_t nk = num_k_heads;
    const int64_t conv_dim = 2 * nk * KD + nv * VD;
    TORCH_CHECK(nv % nk == 0, "gdn_fused: value heads must be a multiple of key heads");
    TORCH_CHECK(conv_in.size(1) == conv_dim, "gdn_fused: conv_dim mismatch");
    TORCH_CHECK(conv_w.is_contiguous() && conv_w.dim() == 2 && conv_w.size(0) == conv_dim && conv_w.size(1) == KERNEL
                && conv_w.scalar_type() == torch::kBFloat16, "gdn_fused: conv weight must be bf16 [conv_dim, KERNEL]");
    TORCH_CHECK(proj.size(1) >= conv_dim + nv * VD + 2 * nv, "gdn_fused: proj has too few columns");
    TORCH_CHECK(A_log.scalar_type() == torch::kFloat32 && dt_bias.scalar_type() == torch::kFloat32 && A_log.numel() == nv
                && dt_bias.numel() == nv && A_log.is_contiguous() && dt_bias.is_contiguous(), "gdn_fused: A_log/dt_bias must be fp32 [nv]");
    TORCH_CHECK(norm_w.scalar_type() == torch::kBFloat16 && norm_w.numel() == VD && norm_w.is_contiguous(), "gdn_fused: norm weight must be bf16 [VD]");
    TORCH_CHECK(tok_starts.scalar_type() == torch::kInt32 && tok_starts.is_contiguous() && tok_lens.scalar_type() == torch::kInt32
                && tok_lens.is_contiguous() && tok_starts.numel() == tok_lens.numel(), "gdn_fused: tok_starts/tok_lens must be int32 [B]");
    TORCH_CHECK(slots.scalar_type() == torch::kInt64 && slots.is_contiguous() && slots.numel() == tok_starts.numel(), "gdn_fused: slots must be int64 [B]");
    if (fresh.has_value()) TORCH_CHECK(fresh->scalar_type() == torch::kUInt8 && fresh->numel() == slots.numel(), "gdn_fused: fresh must be uint8 [B]");
    const auto st = rec_in.scalar_type();
    TORCH_CHECK(rec_out.scalar_type() == st && conv_in.scalar_type() == st && conv_out.scalar_type() == st, "gdn_fused: state dtypes differ");
    if (rec_snap.has_value()) TORCH_CHECK(rec_snap->sizes() == rec_in.sizes() && rec_snap->scalar_type() == st && rec_snap->is_contiguous());
    if (conv_snap.has_value()) TORCH_CHECK(conv_snap->sizes() == conv_in.sizes() && conv_snap->scalar_type() == st && conv_snap->is_contiguous());
    TORCH_CHECK(gdn_fused_supported(KD, VD, KERNEL, st), "gdn_fused: unsupported head dims / kernel / state dtype");

    const c10::cuda::OptionalCUDAGuard guard(proj.device());
    const int B = (int)slots.numel();
    auto out = torch::empty({proj.size(0), nv * VD}, proj.options());
    if (B == 0 || proj.size(0) == 0) return out;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define LILY_GDN_LAUNCH(KD_, T_)                                                                                \
    launch_gdn<KD_, KD_, 4, T_>(proj, conv_w, A_log, dt_bias, norm_w, eps, tok_starts, tok_lens, slots, fresh, conv_in, \
                                conv_out, rec_in, rec_out, conv_snap, rec_snap, out, (int)nk, (int)nv, B, stream)
    if (st == torch::kFloat32) {
        if (KD == 128) LILY_GDN_LAUNCH(128, float);
        else if (KD == 64) LILY_GDN_LAUNCH(64, float);
        else LILY_GDN_LAUNCH(32, float);
    } else {
        if (KD == 128) LILY_GDN_LAUNCH(128, __nv_bfloat16);
        else if (KD == 64) LILY_GDN_LAUNCH(64, __nv_bfloat16);
        else LILY_GDN_LAUNCH(32, __nv_bfloat16);
    }
#undef LILY_GDN_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


void gdn_fused_replay(torch::Tensor table, int64_t proj_stride, torch::Tensor tok_starts, torch::Tensor tok_lens,
                      torch::Tensor slots, int64_t num_k_heads, int64_t num_v_heads, int64_t head_dim, int64_t kernel_size,
                      c10::ScalarType state_dtype) {
    TORCH_CHECK(table.is_cuda() && table.scalar_type() == torch::kInt64 && table.dim() == 2 && table.size(1) == RP_COUNT
                && table.is_contiguous(), "gdn_fused_replay: table must be int64 [layers, ", (int)RP_COUNT, "]");
    TORCH_CHECK(tok_starts.scalar_type() == torch::kInt32 && tok_lens.scalar_type() == torch::kInt32 && slots.scalar_type() == torch::kInt64);
    TORCH_CHECK(gdn_fused_supported(head_dim, head_dim, kernel_size, state_dtype), "gdn_fused_replay: unsupported dims");
    const c10::cuda::OptionalCUDAGuard guard(table.device());
    const int B = (int)slots.numel(), L = (int)table.size(0);
    if (B == 0 || L == 0) return;
    const dim3 grid(num_v_heads, B, L);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t* tp = table.data_ptr<int64_t>();
    const int32_t* ts = tok_starts.data_ptr<int32_t>();
    const int32_t* tl = tok_lens.data_ptr<int32_t>();
    const int64_t* sl = slots.data_ptr<int64_t>();
#define LILY_GDN_REPLAY(KD_, T_) \
    gdn_replay_kernel<KD_, KD_, 4, T_><<<grid, KD_, 0, stream>>>(tp, (int)proj_stride, ts, tl, sl, (int)num_k_heads, (int)num_v_heads)
    if (state_dtype == torch::kFloat32) {
        if (head_dim == 128) LILY_GDN_REPLAY(128, float);
        else if (head_dim == 64) LILY_GDN_REPLAY(64, float);
        else LILY_GDN_REPLAY(32, float);
    } else {
        if (head_dim == 128) LILY_GDN_REPLAY(128, __nv_bfloat16);
        else if (head_dim == 64) LILY_GDN_REPLAY(64, __nv_bfloat16);
        else LILY_GDN_REPLAY(32, __nv_bfloat16);
    }
#undef LILY_GDN_REPLAY
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// MLA decode over the latent cache (GLM-5.3-Flash full-attention layers), absorbed form, plus
// the DSA k-pool scoring used for the decode-time top-k selection.
//
// Attention: with c_t the cached latent of token t (kv_lora_rank values, shared by all heads)
// and q_lat[h] = W_k[h]^T q[h] the query mapped into latent space,
//     scores[h, t] = scale * q_lat[h] . c_t,    out_lat[h] = softmax_t(scores[h, :]) . c_t
// (the caller applies W_v[h] afterwards). The tokens of a request are its req_to_token row:
// positions 0..L-1 (dense) or an explicit visible-position list (sparse, from the indexer).
// Grid (chunks, ceil(H/16), B), block 16 warps: warp = one head, walks every token of the chunk
// with an online softmax (the 16 warps of a block share the token loads through L1), and writes
// (m, l, acc[D]) partials; the combine over chunks is a few torch ops in cuda_ops.mla_decode.
// `chunks` is fixed by the caller (ceil(max visible tokens / chunk)) so the launch shape is
// static and the decode step can be captured in a CUDA graph; chunks past the end write empty
// partials (m = -inf, l = 0).
//
// Pool scoring (Glm5NextTextIndexer semantics): pool p = tokens [p*KP, (p+1)*KP); per channel
// a softmax over the pool's gate logits (+ learned ape) weights the keys, giving pool_key;
// score[p] = sum_h w[h] * relu(scale * q_idx[h] . pool_key). Only complete pools score;
// the others get -inf. One warp per pool, q_idx of the request in shared memory.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <cfloat>
#include <cstdint>
#include <vector>

namespace {

constexpr int WARPS = 16, THREADS = WARPS * 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16(x)); }

// Lane `lane` holds the PER contiguous elements [lane*PER, (lane+1)*PER) of a D = 32*PER row.
template <int PER>
__device__ __forceinline__ void load_row(const __nv_bfloat16* __restrict__ p, int lane, float* out) {
    p += lane * PER;
    if constexpr (PER % 8 == 0) {
#pragma unroll
        for (int v = 0; v < PER / 8; ++v) {
            const uint4 raw = *reinterpret_cast<const uint4*>(p + 8 * v);
            const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 f = __bfloat1622float2(h2[k]);
                out[8 * v + 2 * k] = f.x;
                out[8 * v + 2 * k + 1] = f.y;
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < PER; ++j) out[j] = __bfloat162float(p[j]);
    }
}

// ---- attention over the latent cache ---------------------------------------------------------

template <int D>
__global__ void __launch_bounds__(THREADS)
mla_decode_kernel(const __nv_bfloat16* __restrict__ q_lat,        // [B, H, D]
                  const __nv_bfloat16* __restrict__ latent,       // [slots, D]
                  const int32_t* __restrict__ req_to_token,       // [R, rtt_stride]
                  int64_t rtt_stride,
                  const int64_t* __restrict__ req_pool_indices,   // [B]
                  const int64_t* __restrict__ seq_lens,           // [B]
                  const int32_t* __restrict__ vis_idx,            // [B, S_cap] visible positions, or null (dense)
                  int S_cap,
                  const int32_t* __restrict__ vis_cnt,            // [B], or null
                  float scale, int H, int chunk,
                  float* __restrict__ part_m,                     // [B, chunks, H]
                  float* __restrict__ part_l,                     // [B, chunks, H]
                  float* __restrict__ part_o) {                   // [B, chunks, H, D]
    constexpr int PER = D / 32;
    const int b = blockIdx.z, ck = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = blockIdx.y * WARPS + warp;
    if (h >= H) return;
    const int n = vis_idx ? vis_cnt[b] : (int)seq_lens[b];
    const int t0 = ck * chunk, t1 = min(t0 + chunk, n);
    const int chunks = gridDim.x;
    if (t0 >= n) {                       // past the row's end: an empty partial, which the combine skips
        if (lane == 0) {
            const size_t base = ((size_t)b * chunks + ck) * H + h;
            part_m[base] = -FLT_MAX;
            part_l[base] = 0.f;
        }
        return;
    }

    float q[PER], acc[PER];
    load_row<PER>(q_lat + ((size_t)b * H + h) * D, lane, q);
#pragma unroll
    for (int j = 0; j < PER; ++j) acc[j] = 0.f;
    float m = -FLT_MAX, l = 0.f;

    const int32_t* row = req_to_token + (size_t)req_pool_indices[b] * rtt_stride;
    const int32_t* vrow = vis_idx ? vis_idx + (size_t)b * S_cap : nullptr;
    float cv[PER];
    if (t0 < t1) load_row<PER>(latent + (size_t)row[vrow ? vrow[t0] : t0] * D, lane, cv);
    for (int t = t0; t < t1; ++t) {
        float nxt[PER];
        const bool more = t + 1 < t1;
        if (more) load_row<PER>(latent + (size_t)row[vrow ? vrow[t + 1] : t + 1] * D, lane, nxt);   // prefetch
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < PER; ++j) s += q[j] * cv[j];
        s = warp_sum(s) * scale;
        const float m_new = fmaxf(m, s);
        const float corr = __expf(m - m_new), p = __expf(s - m_new);
        l = l * corr + p;
#pragma unroll
        for (int j = 0; j < PER; ++j) acc[j] = acc[j] * corr + p * cv[j];
        m = m_new;
        if (more) {
#pragma unroll
            for (int j = 0; j < PER; ++j) cv[j] = nxt[j];
        }
    }
    const size_t base = ((size_t)b * chunks + ck) * H + h;
    if (lane == 0) { part_m[base] = m; part_l[base] = l; }
    float* o = part_o + base * D + lane * PER;
#pragma unroll
    for (int j = 0; j < PER; ++j) o[j] = acc[j];
}

// Combine of the per-chunk partials: one warp per (b, h), only the row's ceil(n / chunk) chunks are read.
template <int D>
__global__ void __launch_bounds__(THREADS)
mla_combine_kernel(const float* __restrict__ part_m, const float* __restrict__ part_l, const float* __restrict__ part_o,
                   const int64_t* __restrict__ seq_lens, const int32_t* __restrict__ vis_cnt, int chunk, int chunks,
                   int B, int H, __nv_bfloat16* __restrict__ out) {
    constexpr int PER = D / 32;
    const int idx = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (idx >= B * H) return;
    const int b = idx / H, h = idx % H;
    const int n = vis_cnt ? vis_cnt[b] : (int)seq_lens[b];
    const int nc = min(chunks, (n + chunk - 1) / chunk);
    float M = -FLT_MAX, L = 0.f, acc[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) acc[j] = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t base = ((size_t)b * chunks + c) * H + h;
        const float l_c = part_l[base];
        if (l_c == 0.f) continue;
        const float m_c = part_m[base];
        const float M_new = fmaxf(M, m_c);
        const float corr = __expf(M - M_new), w = __expf(m_c - M_new);
        L = L * corr + w * l_c;
        const float* o = part_o + base * D + lane * PER;
#pragma unroll
        for (int j = 0; j < PER; ++j) acc[j] = acc[j] * corr + w * o[j];
        M = M_new;
    }
    const float inv = L > 0.f ? 1.f / L : 0.f;
    __nv_bfloat16* dst = out + ((size_t)b * H + h) * D + lane * PER;
#pragma unroll
    for (int j = 0; j < PER; ++j) dst[j] = __float2bfloat16(acc[j] * inv);
}

// ---- DSA k-pool scores -----------------------------------------------------------------------

constexpr int MAX_KPOOL = 8;

template <int HD>
__global__ void __launch_bounds__(THREADS)
dsa_pool_scores_kernel(const __nv_bfloat16* __restrict__ index_cache,   // [slots, 2*HD]: key | gate logits
                       const __nv_bfloat16* __restrict__ ape,           // [KP, HD]
                       const int32_t* __restrict__ req_to_token, int64_t rtt_stride,
                       const int64_t* __restrict__ req_pool_indices,    // [B]
                       const int64_t* __restrict__ seq_lens,            // [B]
                       const __nv_bfloat16* __restrict__ q_idx,         // [B, IH, HD]
                       const float* __restrict__ w,                     // [B, IH] head weights (already * IH^-0.5)
                       float scale, int IH, int KP, int P_cap,
                       float* __restrict__ scores) {                    // [B, P_cap]
    constexpr int PER = HD / 32;
    extern __shared__ float smem[];           // q_idx[b] as fp32 [IH, HD], then w[b] [IH]
    float* qs = smem;
    float* ws = smem + IH * HD;
    const int b = blockIdx.y;
    for (int i = threadIdx.x; i < IH * HD; i += THREADS) qs[i] = __bfloat162float(q_idx[(size_t)b * IH * HD + i]);
    for (int i = threadIdx.x; i < IH; i += THREADS) ws[i] = w[(size_t)b * IH + i];
    __syncthreads();

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int p = blockIdx.x * WARPS + warp;
    if (p >= P_cap) return;
    const int L = (int)seq_lens[b];
    if ((p + 1) * KP > L) {                   // incomplete (or absent) pool: never selectable
        if (lane == 0) scores[(size_t)b * P_cap + p] = -INFINITY;
        return;
    }
    const int32_t* row = req_to_token + (size_t)req_pool_indices[b] * rtt_stride;
    float key[MAX_KPOOL][PER], logit[MAX_KPOOL][PER];
    for (int j = 0; j < KP; ++j) {
        const __nv_bfloat16* e = index_cache + (size_t)row[p * KP + j] * (2 * HD);
        float a[PER];
        load_row<PER>(e, lane, key[j]);
        load_row<PER>(e + HD, lane, logit[j]);
        load_row<PER>(ape + (size_t)j * HD, lane, a);
#pragma unroll
        for (int c = 0; c < PER; ++c) logit[j][c] += a[c];
    }
    // per channel: softmax over the pool's tokens, bf16-rounded weights and products like the torch path
    float pool_key[PER];
#pragma unroll
    for (int c = 0; c < PER; ++c) {
        float mx = -FLT_MAX;
        for (int j = 0; j < KP; ++j) mx = fmaxf(mx, logit[j][c]);
        float sum = 0.f;
        for (int j = 0; j < KP; ++j) { logit[j][c] = __expf(logit[j][c] - mx); sum += logit[j][c]; }
        float acc = 0.f;
        for (int j = 0; j < KP; ++j) acc += bf16r(bf16r(logit[j][c] / sum) * key[j][c]);
        pool_key[c] = bf16r(acc);
    }
    float total = 0.f;
    for (int hh = 0; hh < IH; ++hh) {
        const float* qh = qs + hh * HD + lane * PER;
        float d = 0.f;
#pragma unroll
        for (int c = 0; c < PER; ++c) d += qh[c] * pool_key[c];
        d = warp_sum(d) * scale;
        total += ws[hh] * fmaxf(d, 0.f);
    }
    if (lane == 0) scores[(size_t)b * P_cap + p] = total;
}

}  // namespace

bool mla_decode_supported(int64_t D) {
    return D == 32 || D == 64 || D == 128 || D == 256 || D == 512;
}

std::vector<torch::Tensor> mla_decode_partials(torch::Tensor q_lat, torch::Tensor latent, torch::Tensor req_to_token,
                                               torch::Tensor req_pool_indices, torch::Tensor seq_lens,
                                               c10::optional<torch::Tensor> vis_idx, c10::optional<torch::Tensor> vis_cnt,
                                               double scale, int64_t chunk, int64_t chunks) {
    TORCH_CHECK(q_lat.is_cuda() && q_lat.dim() == 3 && q_lat.scalar_type() == torch::kBFloat16 && q_lat.is_contiguous(),
                "mla_decode: q_lat must be a contiguous bf16 [B, H, D] CUDA tensor");
    TORCH_CHECK(latent.scalar_type() == torch::kBFloat16 && latent.dim() == 2 && latent.is_contiguous(), "mla_decode: latent [slots, D] bf16");
    TORCH_CHECK(req_to_token.scalar_type() == torch::kInt32 && req_pool_indices.scalar_type() == torch::kInt64
                && seq_lens.scalar_type() == torch::kInt64, "mla_decode: index dtypes");
    const int64_t B = q_lat.size(0), H = q_lat.size(1), D = q_lat.size(2);
    TORCH_CHECK(latent.size(1) == D && mla_decode_supported(D), "mla_decode: unsupported latent dim ", D);
    TORCH_CHECK(chunks >= 1 && chunk >= 1, "mla_decode: chunk sizes");
    const int32_t* vi = nullptr;
    const int32_t* vc = nullptr;
    int S_cap = 0;
    if (vis_idx.has_value()) {
        TORCH_CHECK(vis_cnt.has_value() && vis_idx->scalar_type() == torch::kInt32 && vis_cnt->scalar_type() == torch::kInt32
                    && vis_idx->is_contiguous() && vis_idx->size(0) == B, "mla_decode: vis_idx [B, S_cap] int32 with vis_cnt [B] int32");
        vi = vis_idx->data_ptr<int32_t>();
        vc = vis_cnt->data_ptr<int32_t>();
        S_cap = (int)vis_idx->size(1);
    }
    const c10::cuda::CUDAGuard guard(q_lat.device());
    auto fopt = q_lat.options().dtype(torch::kFloat32);
    auto part_m = torch::empty({B, chunks, H}, fopt);
    auto part_l = torch::empty({B, chunks, H}, fopt);
    auto part_o = torch::empty({B, chunks, H, D}, fopt);
    const dim3 grid((unsigned)chunks, (unsigned)((H + WARPS - 1) / WARPS), (unsigned)B);
    auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(DD)                                                                                                   \
    mla_decode_kernel<DD><<<grid, THREADS, 0, stream>>>(                                                             \
        reinterpret_cast<const __nv_bfloat16*>(q_lat.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(latent.data_ptr()), \
        req_to_token.data_ptr<int32_t>(), req_to_token.stride(0), req_pool_indices.data_ptr<int64_t>(),              \
        seq_lens.data_ptr<int64_t>(), vi, S_cap, vc, (float)scale, (int)H, (int)chunk,                               \
        part_m.data_ptr<float>(), part_l.data_ptr<float>(), part_o.data_ptr<float>())
    switch (D) {
        case 32: LAUNCH(32); break;
        case 64: LAUNCH(64); break;
        case 128: LAUNCH(128); break;
        case 256: LAUNCH(256); break;
        default: LAUNCH(512); break;
    }
#undef LAUNCH
    return {part_m, part_l, part_o};
}

torch::Tensor mla_decode_combine(torch::Tensor part_m, torch::Tensor part_l, torch::Tensor part_o, torch::Tensor seq_lens,
                                 c10::optional<torch::Tensor> vis_cnt, int64_t chunk) {
    TORCH_CHECK(part_o.dim() == 4 && part_o.is_contiguous() && part_o.scalar_type() == torch::kFloat32, "mla_combine: part_o [B, chunks, H, D] fp32");
    const int64_t B = part_o.size(0), chunks = part_o.size(1), H = part_o.size(2), D = part_o.size(3);
    TORCH_CHECK(part_m.is_contiguous() && part_l.is_contiguous() && part_m.numel() == B * chunks * H && part_l.numel() == part_m.numel());
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt64 && seq_lens.is_contiguous() && mla_decode_supported(D));
    const int32_t* vc = nullptr;
    if (vis_cnt.has_value()) {
        TORCH_CHECK(vis_cnt->scalar_type() == torch::kInt32 && vis_cnt->is_contiguous() && vis_cnt->numel() == B);
        vc = vis_cnt->data_ptr<int32_t>();
    }
    const c10::cuda::CUDAGuard guard(part_o.device());
    auto out = torch::empty({B, H, D}, part_o.options().dtype(torch::kBFloat16));
    if (B * H == 0) return out;
    const int blocks = (int)((B * H + WARPS - 1) / WARPS);
    auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_C(DD)                                                                                                 \
    mla_combine_kernel<DD><<<blocks, THREADS, 0, stream>>>(part_m.data_ptr<float>(), part_l.data_ptr<float>(),       \
        part_o.data_ptr<float>(), seq_lens.data_ptr<int64_t>(), vc, (int)chunk, (int)chunks, (int)B, (int)H,         \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()))
    switch (D) {
        case 32: LAUNCH_C(32); break;
        case 64: LAUNCH_C(64); break;
        case 128: LAUNCH_C(128); break;
        case 256: LAUNCH_C(256); break;
        default: LAUNCH_C(512); break;
    }
#undef LAUNCH_C
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

bool dsa_pool_scores_supported(int64_t head_dim, int64_t kpool) {
    return (head_dim == 32 || head_dim == 64 || head_dim == 128) && kpool >= 1 && kpool <= MAX_KPOOL;
}

torch::Tensor dsa_pool_scores(torch::Tensor index_cache, torch::Tensor ape, torch::Tensor req_to_token,
                              torch::Tensor req_pool_indices, torch::Tensor seq_lens, torch::Tensor q_idx,
                              torch::Tensor w, double scale, int64_t kpool, int64_t P_cap) {
    TORCH_CHECK(q_idx.is_cuda() && q_idx.dim() == 3 && q_idx.scalar_type() == torch::kBFloat16 && q_idx.is_contiguous(),
                "dsa_pool_scores: q_idx bf16 [B, IH, HD] contiguous");
    const int64_t B = q_idx.size(0), IH = q_idx.size(1), HD = q_idx.size(2);
    TORCH_CHECK(dsa_pool_scores_supported(HD, kpool), "dsa_pool_scores: unsupported head_dim/kpool ", HD, " ", kpool);
    TORCH_CHECK(index_cache.scalar_type() == torch::kBFloat16 && index_cache.size(1) == 2 * HD && index_cache.is_contiguous(),
                "dsa_pool_scores: index_cache [slots, 2*HD] bf16");
    TORCH_CHECK(ape.scalar_type() == torch::kBFloat16 && ape.size(0) == kpool && ape.size(1) == HD && ape.is_contiguous(), "dsa_pool_scores: ape [KP, HD]");
    TORCH_CHECK(w.scalar_type() == torch::kFloat32 && w.is_contiguous() && w.size(0) == B && w.size(1) == IH, "dsa_pool_scores: w fp32 [B, IH]");
    TORCH_CHECK(req_to_token.scalar_type() == torch::kInt32 && req_pool_indices.scalar_type() == torch::kInt64
                && seq_lens.scalar_type() == torch::kInt64, "dsa_pool_scores: index dtypes");
    TORCH_CHECK(P_cap >= 1, "dsa_pool_scores: P_cap");
    const c10::cuda::CUDAGuard guard(q_idx.device());
    auto scores = torch::empty({B, P_cap}, q_idx.options().dtype(torch::kFloat32));
    const dim3 grid((unsigned)((P_cap + WARPS - 1) / WARPS), (unsigned)B);
    const size_t smem = (size_t)(IH * HD + IH) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(HH)                                                                                                   \
    dsa_pool_scores_kernel<HH><<<grid, THREADS, smem, stream>>>(                                                     \
        reinterpret_cast<const __nv_bfloat16*>(index_cache.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(ape.data_ptr()), \
        req_to_token.data_ptr<int32_t>(), req_to_token.stride(0), req_pool_indices.data_ptr<int64_t>(),              \
        seq_lens.data_ptr<int64_t>(), reinterpret_cast<const __nv_bfloat16*>(q_idx.data_ptr()), w.data_ptr<float>(), \
        (float)scale, (int)IH, (int)kpool, (int)P_cap, scores.data_ptr<float>())
    switch (HD) {
        case 32: LAUNCH(32); break;
        case 64: LAUNCH(64); break;
        default: LAUNCH(128); break;
    }
#undef LAUNCH
    return scores;
}

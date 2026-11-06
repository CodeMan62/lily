// Small-batch MoE glue around the grouped NVFP4 GEMM: what torch does here in ~50 tiny kernels per
// layer (top-k, sort, CSR, gathers, masks) is three kernels. Rounding follows the torch path.
//
//   moe_route   : one block. Warp per token: sigmoid scores, biased top-k, normalised weights. Then
//                 the (token, slot) pairs sorted by expert, stable, as the CSR the GEMM wants.
//   moe_swiglu  : clamped SwiGLU over [rows, 2I] -> [rows, I]
//   moe_combine : out[t] = sum_slot w[t, s] * down[inv[t, s]] (+ shared[t]); inv < 0 = another rank's expert
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cfloat>

namespace {

constexpr int ROUTE_THREADS = 1024;
constexpr int MAX_ROWS = 1024;         // T * top_k
constexpr int MAX_PER_LANE = 16;       // experts per lane: E <= 512
constexpr int THREADS = 256;
constexpr int VEC = 8;

__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16(x)); }

__global__ void __launch_bounds__(ROUTE_THREADS)
moe_route_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int T, int E, int k, float scaling,
                 bool norm, int offset, int E_local, float* __restrict__ weights, int32_t* __restrict__ active,
                 int32_t* __restrict__ offsets, int32_t* __restrict__ tokens, int32_t* __restrict__ inv) {
    __shared__ int ids[MAX_ROWS];
    __shared__ int counts[513];
    __shared__ int starts[513];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = ROUTE_THREADS / 32;
    for (int i = threadIdx.x; i <= E_local; i += ROUTE_THREADS) counts[i] = 0;
    for (int t = warp; t < T; t += nwarps) {
        float score[MAX_PER_LANE], biased[MAX_PER_LANE];
#pragma unroll
        for (int i = 0; i < MAX_PER_LANE; ++i) {
            const int e = lane + 32 * i;
            if (e < E) {
                score[i] = 1.f / (1.f + expf(-logits[(size_t)t * E + e]));
                biased[i] = score[i] + bias[e];
            } else {
                score[i] = 0.f;
                biased[i] = -FLT_MAX;
            }
        }
        float sum = 0.f, mine = 0.f;
        for (int s = 0; s < k; ++s) {
            float best = -FLT_MAX, chosen = 0.f;
            int who = INT32_MAX;
#pragma unroll
            for (int i = 0; i < MAX_PER_LANE; ++i)
                if (biased[i] > best) { best = biased[i]; who = lane + 32 * i; chosen = score[i]; }
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                const float ob = __shfl_xor_sync(0xffffffffu, best, o);
                const int ow = __shfl_xor_sync(0xffffffffu, who, o);
                const float oc = __shfl_xor_sync(0xffffffffu, chosen, o);
                if (ob > best || (ob == best && ow < who)) { best = ob; who = ow; chosen = oc; }
            }
#pragma unroll
            for (int i = 0; i < MAX_PER_LANE; ++i)
                if (lane + 32 * i == who) biased[i] = -FLT_MAX;
            sum += chosen;
            if (lane == s) mine = chosen;
            if (lane == 0) {
                const int local = who - offset;
                ids[t * k + s] = (local >= 0 && local < E_local) ? local : E_local;
            }
        }
        if (lane < k) weights[(size_t)t * k + lane] = (norm ? mine / (sum + 1e-20f) : mine) * scaling;
    }
    __syncthreads();
    const int R = T * k;
    for (int f = threadIdx.x; f < R; f += ROUTE_THREADS) atomicAdd(&counts[ids[f]], 1);
    __syncthreads();
    if (warp == 0) {                   // lane owns a contiguous run of experts: one warp scan
        const int per = (E_local + 31) / 32, e0 = min(lane * per, E_local), e1 = min(e0 + per, E_local);
        int rows = 0, used = 0;
        for (int e = e0; e < e1; ++e) {
            rows += counts[e];
            used += counts[e] > 0;
        }
        int rows_incl = rows, used_incl = used;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int r = __shfl_up_sync(0xffffffffu, rows_incl, o), u = __shfl_up_sync(0xffffffffu, used_incl, o);
            if (lane >= o) { rows_incl += r; used_incl += u; }
        }
        int run = rows_incl - rows, n = used_incl - used;
        for (int e = e0; e < e1; ++e) {
            starts[e] = run;
            offsets[e] = run;
            if (counts[e]) active[n++] = e;
            run += counts[e];
        }
        const int total = __shfl_sync(0xffffffffu, rows_incl, 31), total_used = __shfl_sync(0xffffffffu, used_incl, 31);
        if (lane == 0) {
            starts[E_local] = total;
            offsets[E_local] = total;
        }
        for (int i = total_used + lane; i < E_local; i += 32) active[i] = -1;
    }
    __syncthreads();
    for (int f = threadIdx.x; f < R; f += ROUTE_THREADS) {
        const int e = ids[f];
        if (e == E_local) {
            inv[f] = -1;
            continue;
        }
        int rank = 0;
        for (int g = 0; g < f; ++g) rank += ids[g] == e;
        const int row = starts[e] + rank;
        tokens[row] = f / k;
        inv[f] = row;
    }
}

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

// grid (ceil(I / VEC / THREADS), rows)
__global__ void __launch_bounds__(THREADS)
moe_swiglu_kernel(const __nv_bfloat16* __restrict__ gate_up, __nv_bfloat16* __restrict__ out, int I, float limit) {
    const int i = blockIdx.x * THREADS + threadIdx.x, r = blockIdx.y;
    if (i >= I / VEC) return;
    const uint4* row = reinterpret_cast<const uint4*>(gate_up + (size_t)r * 2 * I);
    float g[VEC], u[VEC], o[VEC];
    unpack8(row[i], g);
    unpack8(row[I / VEC + i], u);
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
        const float x = fminf(g[j], limit), y = fmaxf(fminf(u[j], limit), -limit);
        o[j] = round_bf16(x / (1.f + expf(-x))) * y;
    }
    reinterpret_cast<uint4*>(out + (size_t)r * I)[i] = pack8(o);
}

// grid (ceil(H / VEC / THREADS), T)
__global__ void __launch_bounds__(THREADS)
moe_combine_kernel(const __nv_bfloat16* __restrict__ down, const int32_t* __restrict__ inv, const float* __restrict__ weights,
                   const __nv_bfloat16* __restrict__ shared, __nv_bfloat16* __restrict__ out, int H, int k) {
    const int i = blockIdx.x * THREADS + threadIdx.x, t = blockIdx.y;
    if (i >= H / VEC) return;
    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[j] = 0.f;
    for (int s = 0; s < k; ++s) {
        const int row = inv[t * k + s];
        if (row < 0) continue;
        const float w = round_bf16(weights[t * k + s]);
        float d[VEC];
        unpack8(reinterpret_cast<const uint4*>(down + (size_t)row * H)[i], d);
#pragma unroll
        for (int j = 0; j < VEC; ++j) acc[j] += round_bf16(d[j] * w);
    }
    if (shared) {
        float h[VEC];
        unpack8(reinterpret_cast<const uint4*>(shared + (size_t)t * H)[i], h);
#pragma unroll
        for (int j = 0; j < VEC; ++j) acc[j] = round_bf16(acc[j]) + h[j];
    }
    reinterpret_cast<uint4*>(out + (size_t)t * H)[i] = pack8(acc);
}

const __nv_bfloat16* bf(const torch::Tensor& x) { return reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); }
__nv_bfloat16* bf_mut(torch::Tensor& x) { return reinterpret_cast<__nv_bfloat16*>(x.data_ptr()); }

}  // namespace

bool moe_route_supported(int64_t T, int64_t k, int64_t E) { return T * k <= MAX_ROWS && k <= 32 && E <= 32 * MAX_PER_LANE; }

// -> weights fp32 [T, k], active int32 [E_local], offsets int32 [E_local + 1], tokens int32 [T * k], inv int32 [T * k]
std::vector<torch::Tensor> moe_route(torch::Tensor logits, torch::Tensor bias, int64_t k, double scaling, bool norm,
                                     int64_t offset, int64_t E_local) {
    TORCH_CHECK(logits.is_contiguous() && logits.scalar_type() == at::kFloat && bias.scalar_type() == at::kFloat);
    const at::cuda::CUDAGuard guard(logits.device());
    const int T = (int)logits.size(0), E = (int)logits.size(1);
    auto i32 = logits.options().dtype(at::kInt);
    auto weights = torch::empty({T, k}, logits.options());
    auto active = torch::empty({E_local}, i32);
    auto offsets = torch::empty({E_local + 1}, i32);
    auto tokens = torch::zeros({T * k}, i32);
    auto inv = torch::empty({T * k}, i32);
    moe_route_kernel<<<1, ROUTE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<float>(), bias.data_ptr<float>(), T, E, (int)k, (float)scaling, norm, (int)offset, (int)E_local,
        weights.data_ptr<float>(), active.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(), tokens.data_ptr<int32_t>(),
        inv.data_ptr<int32_t>());
    return {weights, active, offsets, tokens, inv};
}

torch::Tensor moe_swiglu(torch::Tensor gate_up, double limit) {
    TORCH_CHECK(gate_up.is_contiguous() && gate_up.scalar_type() == at::kBFloat16 && gate_up.size(1) % (2 * VEC) == 0);
    const at::cuda::CUDAGuard guard(gate_up.device());
    const int rows = (int)gate_up.size(0), I = (int)gate_up.size(1) / 2;
    auto out = torch::empty({rows, I}, gate_up.options());
    if (rows == 0) return out;
    const dim3 grid((unsigned)((I / VEC + THREADS - 1) / THREADS), (unsigned)rows);
    moe_swiglu_kernel<<<grid, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(bf(gate_up), bf_mut(out), I, (float)limit);
    return out;
}

torch::Tensor moe_combine(torch::Tensor down, torch::Tensor inv, torch::Tensor weights, c10::optional<torch::Tensor> shared) {
    TORCH_CHECK(down.is_contiguous() && inv.is_contiguous() && weights.is_contiguous());
    const at::cuda::CUDAGuard guard(down.device());
    const int T = (int)weights.size(0), k = (int)weights.size(1), H = (int)down.size(1);
    auto out = torch::empty({T, H}, down.options());
    const dim3 grid((unsigned)((H / VEC + THREADS - 1) / THREADS), (unsigned)T);
    moe_combine_kernel<<<grid, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        bf(down), inv.data_ptr<int32_t>(), weights.data_ptr<float>(), shared ? bf(*shared) : nullptr, bf_mut(out), H, k);
    return out;
}

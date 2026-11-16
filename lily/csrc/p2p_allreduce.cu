// One-shot all-reduce of small bf16 tensors between the TP ranks of one box over PCIe peer-to-peer (no NVLink):
// NCCL's ring needs 2 * (world - 1) dependent hops, this needs one. Graph-safe: no host arguments change per call.
//
// Every rank owns one cudaMalloc'd buffer, mapped into every other rank with CUDA IPC:
//   [flags: NBLK x WORLD ints, written by the peers][epochs: NBLK ints, local][data: 2 parities x WORLD x MAXB]
// Call e, block b: (1) write x's slice b into data[e & 1][rank] of every rank (posted P2P writes), (2) system fence,
// then flags[b][rank] = e on every rank, (3) wait until flags[b][p] >= e for all p, (4) sum data[e & 1][0..world-1]
// slice b in rank order (fp32, the same bits on every rank) into x. A rank only rewrites parity e & 1 at call e + 2,
// after call e + 1's wait proved every peer finished call e (calls on a stream do not overlap).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int NBLK = 32, THREADS = 512, MAXW = 8;
constexpr int64_t MAXB = 2 << 20;                       // bytes per rank per parity
constexpr int64_t HEADER = 4096;                         // flags + epochs, aligned

__device__ __forceinline__ int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release(int* p, int v) {
    asm volatile("st.release.sys.global.b32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__global__ void __launch_bounds__(THREADS)
p2p_allreduce_kernel(__nv_bfloat16* __restrict__ x, int64_t n8, const int64_t* __restrict__ bufs, int rank, int world) {
    __shared__ int epoch_s;
    const int b = blockIdx.x;
    char* own = reinterpret_cast<char*>(bufs[rank]);
    int* epochs = reinterpret_cast<int*>(own) + NBLK * MAXW;
    if (threadIdx.x == 0) epoch_s = epochs[b] + 1;
    __syncthreads();
    const int epoch = epoch_s;
    const int64_t par = (int64_t)(epoch & 1) * world * MAXB;
    const int64_t per = (n8 + NBLK - 1) / NBLK, lo = b * per, hi = min(n8, lo + per);
    uint4* xv = reinterpret_cast<uint4*>(x);
    // (1) push this slice to every rank
    for (int64_t i = lo + threadIdx.x; i < hi; i += THREADS) {
        const uint4 v = xv[i];
        for (int p = 0; p < world; ++p)
            reinterpret_cast<uint4*>(reinterpret_cast<char*>(bufs[p]) + HEADER + par + (int64_t)rank * MAXB)[i] = v;
    }
    __syncthreads();
    // (2) announce, (3) wait for everyone
    if (threadIdx.x == 0) {
        __threadfence_system();
        for (int p = 0; p < world; ++p) st_release(reinterpret_cast<int*>(bufs[p]) + b * MAXW + rank, epoch);
        const int* flags = reinterpret_cast<const int*>(own) + b * MAXW;
        for (int p = 0; p < world; ++p)
            while (ld_acquire(flags + p) < epoch) {}
        __threadfence_system();
        epochs[b] = epoch;
    }
    __syncthreads();
    // (4) sum in rank order
    const char* data = own + HEADER + par;
    for (int64_t i = lo + threadIdx.x; i < hi; i += THREADS) {
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int p = 0; p < world; ++p) {
            const uint4 v = __ldcv(reinterpret_cast<const uint4*>(data + (int64_t)p * MAXB) + i);
            const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __bfloat1622float2(h[j]);
                acc[2 * j] += f.x;
                acc[2 * j + 1] += f.y;
            }
        }
        uint4 o;
        __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
        for (int j = 0; j < 4; ++j) oh[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
        xv[i] = o;
    }
}

}  // namespace

int64_t p2p_buffer_bytes(int64_t world) { return HEADER + 2 * world * MAXB; }
int64_t p2p_max_bytes() { return MAXB; }

std::vector<int64_t> p2p_alloc(int64_t world) {
    void* p = nullptr;
    C10_CUDA_CHECK(cudaMalloc(&p, p2p_buffer_bytes(world)));
    C10_CUDA_CHECK(cudaMemset(p, 0, p2p_buffer_bytes(world)));
    return {reinterpret_cast<int64_t>(p)};
}

torch::Tensor p2p_handle(int64_t ptr) {
    cudaIpcMemHandle_t h;
    C10_CUDA_CHECK(cudaIpcGetMemHandle(&h, reinterpret_cast<void*>(ptr)));
    auto t = torch::empty({(int64_t)sizeof(h)}, torch::kUInt8);
    memcpy(t.data_ptr(), &h, sizeof(h));
    return t;
}

int64_t p2p_open(torch::Tensor handle) {
    TORCH_CHECK(handle.numel() == (int64_t)sizeof(cudaIpcMemHandle_t) && handle.scalar_type() == torch::kUInt8);
    cudaIpcMemHandle_t h;
    memcpy(&h, handle.data_ptr(), sizeof(h));
    void* p = nullptr;
    C10_CUDA_CHECK(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess));
    return reinterpret_cast<int64_t>(p);
}

void p2p_allreduce(torch::Tensor x, torch::Tensor bufs, int64_t rank, int64_t world) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.numel() % 8 == 0
                && x.numel() * 2 <= MAXB, "p2p_allreduce: contiguous bf16, numel % 8 == 0, <= 2 MiB");
    TORCH_CHECK(bufs.is_cuda() && bufs.scalar_type() == at::kLong && bufs.numel() == world && world <= MAXW);
    const c10::cuda::CUDAGuard guard(x.device());
    if (x.numel() == 0) return;
    p2p_allreduce_kernel<<<NBLK, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr()), x.numel() / 8, bufs.data_ptr<int64_t>(), (int)rank, (int)world);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

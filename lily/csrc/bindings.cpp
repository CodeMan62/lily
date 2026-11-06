// Python bindings for lily's CUDA kernels. One extension module, one entry per kernel file.
#include <torch/extension.h>

#include <vector>

torch::Tensor nvfp4_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor scale, torch::Tensor gscale);
bool nvfp4_gemm_tc_supported(int64_t M, int64_t K);
torch::Tensor nvfp4_gemm_tc(torch::Tensor x, torch::Tensor w, torch::Tensor scale, torch::Tensor gscale);

bool gdn_fused_supported(int64_t k_head_dim, int64_t v_head_dim, int64_t kernel_size, c10::ScalarType state_dtype);
torch::Tensor gdn_fused_forward(torch::Tensor proj, torch::Tensor conv_w, torch::Tensor A_log, torch::Tensor dt_bias,
                                torch::Tensor norm_w, double eps, torch::Tensor tok_starts, torch::Tensor tok_lens,
                                torch::Tensor slots, c10::optional<torch::Tensor> fresh,
                                torch::Tensor conv_in, torch::Tensor conv_out, torch::Tensor rec_in, torch::Tensor rec_out,
                                c10::optional<torch::Tensor> conv_snap, c10::optional<torch::Tensor> rec_snap,
                                int64_t num_k_heads);
void gdn_fused_replay(torch::Tensor table, int64_t proj_stride, torch::Tensor tok_starts, torch::Tensor tok_lens,
                      torch::Tensor slots,
                      int64_t num_k_heads, int64_t num_v_heads, int64_t head_dim, int64_t kernel_size,
                      c10::ScalarType state_dtype);

bool kda_fused_supported(int64_t head_dim, int64_t kernel_size, c10::ScalarType state_dtype);
torch::Tensor kda_fused_forward(torch::Tensor proj, torch::Tensor gate, torch::Tensor beta_raw, torch::Tensor fgate,
                                torch::Tensor conv_w, torch::Tensor A_log, torch::Tensor dt_bias, double lower_bound,
                                torch::Tensor norm_w, double eps, torch::Tensor tok_starts, torch::Tensor tok_lens,
                                torch::Tensor slots, c10::optional<torch::Tensor> fresh,
                                torch::Tensor conv_in, torch::Tensor conv_out, torch::Tensor rec_in, torch::Tensor rec_out,
                                c10::optional<torch::Tensor> conv_snap, c10::optional<torch::Tensor> rec_snap);
void kda_fused_replay(torch::Tensor table, int64_t proj_stride, int64_t beta_stride, int64_t fgate_stride, double lower_bound,
                      torch::Tensor tok_starts, torch::Tensor tok_lens, torch::Tensor slots,
                      int64_t num_heads, int64_t head_dim, int64_t kernel_size, c10::ScalarType state_dtype);

bool nvfp4_moe_supported(int64_t K);
torch::Tensor nvfp4_moe_gemm(torch::Tensor x, torch::Tensor w, torch::Tensor scale, torch::Tensor gscale,
                             torch::Tensor active_experts, torch::Tensor offsets, torch::Tensor tokens, int64_t num_rows);

bool mla_decode_supported(int64_t D);
std::vector<torch::Tensor> mla_decode_partials(torch::Tensor q_lat, torch::Tensor latent, torch::Tensor req_to_token,
                                               torch::Tensor req_pool_indices, torch::Tensor seq_lens,
                                               c10::optional<torch::Tensor> vis_idx, c10::optional<torch::Tensor> vis_cnt,
                                               double scale, int64_t chunk, int64_t chunks);
torch::Tensor mla_decode_combine(torch::Tensor part_m, torch::Tensor part_l, torch::Tensor part_o, torch::Tensor seq_lens,
                                 c10::optional<torch::Tensor> vis_cnt, int64_t chunk);
bool dsa_pool_scores_supported(int64_t head_dim, int64_t kpool);
torch::Tensor dsa_pool_scores(torch::Tensor index_cache, torch::Tensor ape, torch::Tensor req_to_token,
                              torch::Tensor req_pool_indices, torch::Tensor seq_lens, torch::Tensor q_idx,
                              torch::Tensor w, double scale, int64_t kpool, int64_t P_cap);

bool hc_fused_supported(int64_t hc_mult, int64_t H);
std::vector<torch::Tensor> hc_collapse(torch::Tensor streams, torch::Tensor fn, torch::Tensor base, torch::Tensor scale,
                                       torch::Tensor norm_w, int64_t iters, double hc_eps, double rms_eps, double norm_eps);
torch::Tensor hc_expand(torch::Tensor block_out, torch::Tensor post, torch::Tensor comb, torch::Tensor streams);

bool moe_route_supported(int64_t T, int64_t k, int64_t E);
std::vector<torch::Tensor> moe_route(torch::Tensor logits, torch::Tensor bias, int64_t k, double scaling, bool norm,
                                     int64_t offset, int64_t E_local);
torch::Tensor moe_swiglu(torch::Tensor gate_up, double limit);
torch::Tensor moe_combine(torch::Tensor down, torch::Tensor inv, torch::Tensor weights, c10::optional<torch::Tensor> shared);

bool bf16_skinny_supported(int64_t M, int64_t N, int64_t K);
torch::Tensor bf16_skinny(torch::Tensor x, torch::Tensor w);
void l2_prefetch(torch::Tensor w, int64_t blocks);

torch::Tensor rmsnorm_fused(torch::Tensor x, torch::Tensor w, double eps);
torch::Tensor dyn_conv2(torch::Tensor x, torch::Tensor dyn, int64_t dyn_off, torch::Tensor base, int64_t group);
std::vector<torch::Tensor> rope_neox(torch::Tensor q, torch::Tensor k, torch::Tensor cos_sin, torch::Tensor pos);
bool select_greedy_supported(int64_t K, int64_t R);
std::vector<torch::Tensor> select_greedy(torch::Tensor unary, torch::Tensor cand, torch::Tensor proj, torch::Tensor anchor,
                                         torch::Tensor pred_cb, torch::Tensor succ_cb);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bf16_skinny_supported", &bf16_skinny_supported);
    m.def("l2_prefetch", &l2_prefetch);
    m.def("rmsnorm_fused", &rmsnorm_fused, "RMSNorm of bf16 rows (fp32 variance, HF rounding)");
    m.def("dyn_conv2", &dyn_conv2, "DFlash two-tap grouped dynamic causal conv over a block");
    m.def("rope_neox", &rope_neox, "Neox rotary over whole heads of q and k (bf16 cache)");
    m.def("select_greedy_supported", &select_greedy_supported);
    m.def("select_greedy", &select_greedy, "DFlash greedy candidate path through a block");
    m.def("bf16_skinny", &bf16_skinny, "bf16 linear for M <= 8 rows, weight streaming on tensor cores");
    m.def("moe_route_supported", &moe_route_supported);
    m.def("moe_route", &moe_route, "Router top-k and the sorted CSR of (token, slot) pairs for a small batch");
    m.def("moe_swiglu", &moe_swiglu, "Clamped SwiGLU over [rows, 2I]");
    m.def("moe_combine", &moe_combine, "Weighted sum of the expert rows of each token, plus the shared expert");
    m.def("hc_fused_supported", &hc_fused_supported);
    m.def("hc_collapse", &hc_collapse, "mHC collapse: stream weights, Sinkhorn, weighted sum, RMSNorm");
    m.def("hc_expand", &hc_expand, "mHC expand: block output into the streams, streams mixed");
    m.def("mla_decode_supported", &mla_decode_supported);
    m.def("mla_decode_partials", &mla_decode_partials, "MLA decode over the latent cache: per-chunk online-softmax partials");
    m.def("mla_decode_combine", &mla_decode_combine, "Combine the per-chunk MLA partials (only each row's chunks are read)");
    m.def("dsa_pool_scores_supported", &dsa_pool_scores_supported);
    m.def("dsa_pool_scores", &dsa_pool_scores, "DSA indexer k-pool scores for one decode step");
    m.def("nvfp4_moe_supported", &nvfp4_moe_supported);
    m.def("nvfp4_moe_gemm", &nvfp4_moe_gemm, "Grouped NVFP4 GEMM over experts (CSR of tokens per expert), tensor cores");
    m.def("kda_fused_supported", &kda_fused_supported);
    m.def("kda_fused_forward", &kda_fused_forward, "Fused Kimi Delta Attention layer step (GLM-5.3-Flash) over a varlen batch");
    m.def("kda_fused_replay", &kda_fused_replay, "Replay every KDA layer's recurrence over an accepted prefix (one launch)");
    m.def("nvfp4_gemv", &nvfp4_gemv, "NVFP4 weight-only decode GEMV (bf16 activations)");
    m.def("nvfp4_gemm_tc_supported", &nvfp4_gemm_tc_supported);
    m.def("nvfp4_gemm_tc", &nvfp4_gemm_tc, "NVFP4 weight-only GEMM for M <= 16 rows on tensor cores");
    m.def("gdn_fused_supported", &gdn_fused_supported, "Whether the fused GDN kernel covers these dims");
    m.def("gdn_fused_forward", &gdn_fused_forward, "Fused conv1d + gated delta rule + gated RMSNorm over a varlen batch",
          py::arg("proj"), py::arg("conv_w"), py::arg("A_log"), py::arg("dt_bias"), py::arg("norm_w"), py::arg("eps"),
          py::arg("tok_starts"), py::arg("tok_lens"), py::arg("slots"), py::arg("fresh"), py::arg("conv_in"), py::arg("conv_out"),
          py::arg("rec_in"), py::arg("rec_out"), py::arg("conv_snap"), py::arg("rec_snap"), py::arg("num_k_heads"));
    m.def("gdn_fused_replay", &gdn_fused_replay, "Replay every GDN layer's recurrence over an accepted prefix (one launch)",
          py::arg("table"), py::arg("proj_stride"), py::arg("tok_starts"), py::arg("tok_lens"), py::arg("slots"), py::arg("num_k_heads"),
          py::arg("num_v_heads"), py::arg("head_dim"), py::arg("kernel_size"), py::arg("state_dtype"));
}

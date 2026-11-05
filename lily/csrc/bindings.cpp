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
bool dsa_pool_scores_supported(int64_t head_dim, int64_t kpool);
torch::Tensor dsa_pool_scores(torch::Tensor index_cache, torch::Tensor ape, torch::Tensor req_to_token,
                              torch::Tensor req_pool_indices, torch::Tensor seq_lens, torch::Tensor q_idx,
                              torch::Tensor w, double scale, int64_t kpool, int64_t P_cap);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mla_decode_supported", &mla_decode_supported);
    m.def("mla_decode_partials", &mla_decode_partials, "MLA decode over the latent cache: per-chunk online-softmax partials");
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

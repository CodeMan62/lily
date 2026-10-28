TODO -> Date: 20/10/26
- see cuda graphs -> low priority
- add spec decoding -> med priority
- multi-gpu inference -> very high priority
- server q3.8-27B on 5090 -> v1 launch
- figure out a better way to work on kernels maybe read deepinfer launched today 


Multi-gpu progres
-> add backend communication -> Done
-> weight sharding layers done -> DOne 
-> add sharding in loader -> Done (per-rank narrowing in weight_loaders)
-> scheduler synchronization -> Done (untested, needs 2 GPUs)

Quantization progress (26/10/26)
-> online weight-only quant at load time: fp8 (e4m3 + per-channel), w8a16 (int8 + per-channel), w4a16 (int4 + group-128) -> Done
-> fused triton dequant GEMM for all three (no [N,K] dequant matrix per forward) -> Done
-> cli: --quantization fp8|w8a16|w4a16 (engine/serving); pre-quantized (gptq/awq-style packed) checkpoints rejected with a clear error
-> tested on RTX 3050 6GB: qwen3-0.6B bf16 1137MiB -> w8a16 718MiB, fp8 718MiB, w4a16 520MiB params
-> decode tok/s on 3050 (qwen3-0.6B): bf16 91, fp8 96, w8a16 118, w4a16 127 -> Done

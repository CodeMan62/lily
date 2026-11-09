mode - CodeMan62/qwen3.8-27B-NVFP4A16
first pass
hybrid linear-attention model: disabling radix cache (states are not prefix-cacheable)
  bs=1   prefill    602.1 ms (      850 tok/s)   decode step   65.36 ms median /   65.42 p99 (   15.3 tok/s)
  bs=2   prefill   1160.6 ms (      882 tok/s)   decode step   68.56 ms median /   68.63 p99 (   29.2 tok/s)
  bs=4   prefill   2281.7 ms (      898 tok/s)   decode step   70.84 ms median /   70.89 p99 (   56.5 tok/s)
  bs=8   prefill   4631.7 ms (      884 tok/s)   decode step   73.71 ms median /   73.77 p99 (  108.5 tok/s)


| Prompt tokens | TTFT (s) | Implied prefill (tok/s) | Decode (tok/s) |
|---|---|---|---|
| 128 | 0.156 | 820 | 271 |
| 1024 | 1.125 | 910 | 304 |
| 8192 | 9.280 | 883 | 303 |
| 16384 | 18.952 | 865 | 270 |


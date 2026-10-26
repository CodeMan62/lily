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

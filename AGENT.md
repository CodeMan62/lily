# Agent rules for lily

## No duplicate code

- Before writing a helper, search for an existing one (`grep -rn`) and reuse it.
- If the same logic is needed in a second place, move it into one function and call it from both. Don't copy it.
- Shared helpers live where they belong:
  - `lily/utils.py`: small generic helpers (`ceil_div`, ...).
  - `lily/distributed/distributed.py`: process groups, collectives, and TP info (`get_tp_info`).
  - `ModelConfig`: anything derived from the model config, e.g. per-rank head counts (`get_num_attention_heads(tp_size)`, `get_num_kv_heads(tp_size)`).
  - `lily/models/loader.py` (`read_tp_shard`): how checkpoint tensors are split across TP ranks.
- Delete code that nothing uses instead of leaving it in.

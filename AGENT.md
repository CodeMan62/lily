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

- Comments should explain constraints or non-obvious decisions. Keep task history and conversations out of code and user documentation.
- Keep documentation concise and factual. Do not use punchy slogans or promotional language.
- Update documentation where users need it to understand changed usage. Small features do not automatically need a README headline, new section, or feature announcement. Prefer a short update in the existing relevant section.
- Do not commit superfluous documentation or scatter unnecessary Markdown files through the repository. New documents need a clear purpose beyond recording the current task. Link to authoritative code and configuration instead of duplicating details.
- Do not commit build outputs (dist/), extension .zip archives, model weights, or profiling dumps. Intentional tracked assets are limited to extension icons, static web assets, and explicitly requested docs media.
- Run every test, build, or measurement needed to validate the change instead of skipping it and adding a caveat that it was not run. Only report a browser, GPU, or environment gap when the required runtime or hardware is genuinely unavailable. Report the checks actually run and their results, and do not present inferred correctness or speedups as measured.
- Do not write the PR description yourself. Ask the user to write it out after giving them sufficient information about the changes. If the user still asks you to write the description yourself, give them a warning but proceed with noting this override in the PR description.
- Do not reply to the reviewer's comment yourself. Same rule as above.

"""Download a checkpoint into the HF cache and print its parsed ModelConfig."""

import argparse

from lula.models.model_config import ModelConfig

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="+")
    args = p.parse_args()
    for m in args.model:
        cfg = ModelConfig.from_pretrained(m)
        print(f"{m}\n  path={cfg.model_path}\n  {cfg.summary()}")

#!/usr/bin/env bash
# Thin wrapper: engine-level + HTTP-level benchmark in one run. See bench/bench.py --help.
#   bench/bench.sh --model-path CodeMan62/qwen3.8-27B-NVFP4A16
cd "$(dirname "$0")/.."
exec uv run python bench/bench.py "$@"

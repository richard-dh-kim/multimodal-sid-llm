#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints/tokenizer
python -m sid_llm.training.train_tokenizer "$@"

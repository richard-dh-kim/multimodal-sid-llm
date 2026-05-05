#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
python -m sid_llm.training.train_cpt "$@"

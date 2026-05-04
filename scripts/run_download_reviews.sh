#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m sid_llm.data.download_reviews --items-in "${1:-data/catalog/items.parquet}" --out "${2:-data/catalog/interactions.parquet}"

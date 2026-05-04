#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m sid_llm.data.download_metadata --max-per-category "${1:-50000}" --out "${2:-data/catalog/items.parquet}"

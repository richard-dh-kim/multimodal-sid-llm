#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m sid_llm.data.download_images --items-in "${1:-data/catalog/items.parquet}" --images-dir "${2:-data/images}"

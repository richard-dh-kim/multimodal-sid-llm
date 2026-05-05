#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m sid_llm.data.embed_items \
  --catalog-in data/catalog/catalog.parquet \
  --images-dir data/images \
  --out "${1:-data/catalog/embeddings_b1.parquet}" \
  --model-checkpoint "${2:-openai/clip-vit-base-patch32}"

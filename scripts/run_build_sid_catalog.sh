#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m sid_llm.data.build_sid_catalog "$@"

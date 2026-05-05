#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run streamlit run dashboard/app.py "$@"

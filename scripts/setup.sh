#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
command -v uv >/dev/null || { echo 'Install uv from https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }
GIT_LFS_SKIP_SMUDGE=1 uv sync --project "$PI05_ROOT/openpi" --frozen
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps -e "$PI05_ROOT/openpi"
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/collect_environment.py
echo 'Training environment ready. GPU acceptance is a separate smoke test.'

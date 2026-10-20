#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
PRO_DIR="$PI05_ROOT/runs/cache/LIBERO-PRO"
PRO_REV="$("$PYTHON" -c 'import json; print(json.load(open("configs/evaluation.json"))["libero_pro"]["commit"])')"
if [[ ! -d "$PRO_DIR/.git" ]]; then
  git clone --no-checkout https://github.com/Zxy-MLlab/LIBERO-PRO.git "$PRO_DIR"
fi
# checkout refuses conflicting local edits; do not reset/clean this directory.
git -C "$PRO_DIR" checkout --detach "$PRO_REV"
"$PYTHON" scripts/prepare_ood.py
echo 'Official Task/Position assets ready. Use scripts/evaluate_all.sh after starting a trained policy server.'

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
VARIANT="${1:?Usage: bash scripts/evaluate_all.sh baseline|spatial [port] [run_name]}"
PORT="${2:-8000}"
RUN_NAME="${3:-paper_seed42}"
[[ "$VARIANT" == baseline || "$VARIANT" == spatial ]] || { echo 'Invalid variant'; exit 1; }
for CONDITION in id task position; do
  for SUITE in libero_object libero_spatial; do
    "$PI05_ROOT/runs/env/simulator/bin/python" scripts/evaluate.py \
      --variant "$VARIANT" --suite "$SUITE" --condition "$CONDITION" \
      --port "$PORT" --run-name "$RUN_NAME"
  done
done
echo "All six evaluation commands finished; inspect errors and success rates under runs/evaluation/$RUN_NAME/."

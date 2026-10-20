#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
SCALE="${1:-69}"
FSDP_DEVICES="${2:-1}"
[[ "$SCALE" == 20 || "$SCALE" == 69 ]] || { echo 'Scale must be 20 or 69'; exit 1; }
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
exec > >(tee "$PI05_ROOT/runs/logs/pipeline_$STAMP.log") 2>&1
CURRENT_STAGE=initializing
trap 'status=$?; printf "stage=%s\nexit_code=%s\n" "$CURRENT_STAGE" "$status" > "$PI05_ROOT/runs/reports/pipeline_$STAMP.status"' EXIT
CURRENT_STAGE=setup; bash scripts/setup.sh
CURRENT_STAGE=simulator_setup; bash scripts/setup_simulator.sh
CURRENT_STAGE=ood_setup; bash scripts/setup_ood.sh
CURRENT_STAGE=smoke; bash scripts/smoke_test.sh
CURRENT_STAGE=download_weights; "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/download.py weights
CURRENT_STAGE=data; bash scripts/prepare_data.sh "$SCALE"
CURRENT_STAGE=baseline; bash scripts/train_baseline.sh --repo-id "local/pi05_3d_$SCALE" --fsdp-devices "$FSDP_DEVICES"
CURRENT_STAGE=spatial; bash scripts/train_spatial.sh --repo-id "local/pi05_3d_$SCALE" --fsdp-devices "$FSDP_DEVICES"
CURRENT_STAGE=training_finished
echo 'Both training commands finished. This does not mean benchmark reproduction has passed.'
echo 'Start a trained policy server, then run scripts/evaluate_all.sh baseline or spatial. See docs/evaluation_protocol.md.'

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
SCALE="${1:-69}"
[[ "$SCALE" == 20 || "$SCALE" == 69 ]] || { echo 'Scale must be 20 or 69'; exit 1; }
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/download.py data --scale "$SCALE"
"$PI05_ROOT/runs/env/simulator/bin/python" scripts/replay_dataset.py --scale "$SCALE"
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/convert_dataset.py --scale "$SCALE"
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/run.py norm baseline --repo-id "local/pi05_3d_$SCALE"
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/run.py norm spatial --repo-id "local/pi05_3d_$SCALE"

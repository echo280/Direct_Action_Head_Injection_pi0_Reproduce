#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
LIBERO_DIR="$PI05_ROOT/runs/cache/LIBERO"
LIBERO_REV=f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
if [[ ! -d "$LIBERO_DIR/.git" ]]; then
  git clone --no-checkout https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
fi
git -C "$LIBERO_DIR" checkout --detach "$LIBERO_REV"
uv venv --python 3.10 "$PI05_ROOT/runs/env/simulator"
uv pip install --python "$PI05_ROOT/runs/env/simulator/bin/python" -r "$PI05_ROOT/configs/requirements_sim.txt"
uv pip install --python "$PI05_ROOT/runs/env/simulator/bin/python" torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$PI05_ROOT/runs/env/simulator/bin/python" --no-deps -e "$LIBERO_DIR" -e "$PI05_ROOT/openpi/packages/openpi-client"
uv pip freeze --python "$PI05_ROOT/runs/env/simulator/bin/python" > "$PI05_ROOT/runs/reports/simulator_requirements_resolved.txt"
echo 'Simulator installed; EGL/headless rendering still needs a live check.'

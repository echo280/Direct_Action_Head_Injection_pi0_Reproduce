#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"$UV_PROJECT_ENVIRONMENT/bin/python" -m unittest discover -s tests -v
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/smoke_model.py

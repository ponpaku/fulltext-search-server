#!/usr/bin/env bash
set -euo pipefail

# System Version: 1.3.1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
VENV_DIR="${PROJECT_ROOT}/venv"
ACTIVATE_SH="${VENV_DIR}/bin/activate"

if [[ ! -f "${ACTIVATE_SH}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${ACTIVATE_SH}"
python -m pip install --upgrade pip
if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
  python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
fi

python "${PROJECT_ROOT}/app.py"

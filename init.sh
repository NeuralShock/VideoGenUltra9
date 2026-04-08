#!/usr/bin/env bash

set -euo pipefail

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQ_MARKER="${VENV_DIR}/.requirements_ready"
ASSET_MARKER="${VENV_DIR}/.assets_ready"

create_venv() {
  echo "Creating virtual environment at ${VENV_DIR}..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
}

activate_venv() {
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  echo "Virtual environment active: ${VIRTUAL_ENV}"
  echo "Python: $(command -v python)"
  echo "Environment and local assets are ready."
}

is_sourced() {
  [ "${BASH_SOURCE[0]}" != "$0" ]
}

requirements_need_refresh() {
  [ ! -f "${REQ_MARKER}" ] || [ "requirements.txt" -nt "${REQ_MARKER}" ]
}

assets_need_refresh() {
  [ ! -f "${ASSET_MARKER}" ] || [ "bootstrap_assets.py" -nt "${ASSET_MARKER}" ]
}

bootstrap_project() {
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  if requirements_need_refresh; then
    echo "Installing Python dependencies from requirements.txt..."
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
    touch "${REQ_MARKER}"
  else
    echo "Python dependencies already up to date."
  fi

  if assets_need_refresh; then
    echo "Downloading / validating local project assets..."
    python bootstrap_assets.py
    touch "${ASSET_MARKER}"
  else
    echo "Project assets already present."
  fi
}

if [ ! -d "${VENV_DIR}" ] || [ ! -f "${VENV_DIR}/bin/activate" ]; then
  create_venv
fi

bootstrap_project

if is_sourced; then
  activate_venv
else
  echo "init.sh was executed directly, so activation cannot modify your current shell."
  echo "Starting a new Bash shell with the project virtual environment active..."
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  exec "${SHELL:-/bin/bash}" -i
fi

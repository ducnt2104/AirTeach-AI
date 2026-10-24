#!/usr/bin/env bash
# ==============================================================================
# AirTeach AI - macOS Launcher
# ==============================================================================
set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
ROOT="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
cd "$ROOT"

mkdir -p logs data data/workspaces models

# Check virtual environment
if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
    PYTHON_CMD="python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
else
    echo "[ERROR] Python 3.10+ not found! Please run ./install_macos.sh first."
    exit 1
fi

exec "$PYTHON_CMD" main.py "$@"

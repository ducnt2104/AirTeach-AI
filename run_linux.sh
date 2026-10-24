#!/usr/bin/env bash
# ==============================================================================
# AirTeach AI - Linux Launcher
# ==============================================================================
set -e

# Resolve script root directory
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
ROOT="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
cd "$ROOT"

# Ensure runtime directories exist
mkdir -p logs data data/workspaces models

# Prefer local virtual environment (.venv)
if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
    PYTHON_CMD="python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
else
    echo "[ERROR] Python 3 not found! Please run ./install_linux.sh first."
    exit 1
fi

# Compatibility with Wayland / XWayland (e.g. Zorin OS 17, Ubuntu 22.04+)
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

# Run AirTeach AI
exec "$PYTHON_CMD" main.py "$@"

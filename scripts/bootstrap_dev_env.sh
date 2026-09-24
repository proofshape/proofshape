#!/usr/bin/env bash
set -eu

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
USE_MANAGED_ENV=0

# Lightning AI Studios expose one managed Conda environment named "cloudspace" and
# intentionally refuse creation of another venv/Conda environment. F-07 must use the
# same bootstrap command there, so reuse that managed environment instead of calling
# `python -m venv`. Normal macOS/Linux development still gets an isolated .venv.
if [ "${CONDA_DEFAULT_ENV:-}" = "cloudspace" ] && [ -n "${CONDA_PREFIX:-}" ]; then
  USE_MANAGED_ENV=1
  if [ -z "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python)"
  fi
  echo "Detected Lightning Studio managed Conda environment: ${CONDA_DEFAULT_ENV}"
fi

if [ -z "$PYTHON_BIN" ]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Python 3.11+ is required for ProofShape development. Install it first, then re-run this script." >&2
    exit 1
  fi
fi

if ! "$PYTHON_BIN" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
then
  echo "Python 3.11+ is required for ProofShape development. Install it first, then re-run this script." >&2
  exit 1
fi

if [ "$USE_MANAGED_ENV" -eq 1 ]; then
  ENV_PYTHON="$PYTHON_BIN"
else
  if [ ! -d .venv ]; then
    "$PYTHON_BIN" -m venv .venv
  fi
  ENV_PYTHON="$ROOT_DIR/.venv/bin/python"
fi

"$ENV_PYTHON" -m pip install --upgrade pip setuptools wheel
# requirements.txt already contains "-e .", so this installs the project itself
# plus every pinned development dependency exactly once.
"$ENV_PYTHON" -m pip install -r requirements.txt

cat <<EOF
Environment bootstrapped successfully.
Python: $("$ENV_PYTHON" --version 2>&1)
Environment Python: $ENV_PYTHON
Run tests with:
  "$ENV_PYTHON" -m pytest -q
EOF

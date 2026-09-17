#!/usr/bin/env bash
set -euo pipefail

# F-05 one-command workspace check. This intentionally does not install project dependencies;
# reproducible environment setup belongs to F-07. It only proves that this checkout is usable
# and that PyTorch can see the GPU in the shared cloud workspace.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "ProofShape GPU workspace check"
echo "Repository: $ROOT_DIR"
echo "Branch: $(git branch --show-current)"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "GPU CHECK: FAIL - nvidia-smi is not available."
    exit 1
fi

python3 scripts/check_gpu.py

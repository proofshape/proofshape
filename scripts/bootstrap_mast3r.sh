#!/usr/bin/env bash
set -euo pipefail

# R-11 MASt3R bootstrap for the shared Lightning Studio.
# Keep the Studio's working CUDA torch/torchvision build intact: upstream DUSt3R's
# requirements file lists both, so installing it wholesale could replace the managed build.

MAST3R_ROOT="${MAST3R_ROOT:-$HOME/third_party/mast3r}"
MAST3R_COMMIT="f5209afc300cec36239a7ac992263f36847bbba0"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "R-11 bootstrap expects a GPU-enabled Lightning Studio; CUDA is not available."
    )
print("Using existing PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY

python -m pip install --no-deps roma trimesh

mkdir -p "$(dirname "$MAST3R_ROOT")"
if [[ ! -d "$MAST3R_ROOT/.git" ]]; then
  git clone --recursive https://github.com/naver/mast3r.git "$MAST3R_ROOT"
fi

git -C "$MAST3R_ROOT" fetch origin
git -C "$MAST3R_ROOT" checkout "$MAST3R_COMMIT"
git -C "$MAST3R_ROOT" submodule update --init --recursive

current_commit="$(git -C "$MAST3R_ROOT" rev-parse HEAD)"
if [[ "$current_commit" != "$MAST3R_COMMIT" ]]; then
  echo "MASt3R pin mismatch: expected $MAST3R_COMMIT, got $current_commit" >&2
  exit 1
fi

echo "MASt3R ready at $MAST3R_ROOT ($current_commit)"
echo "Code license is CC BY-NC-SA 4.0; review checkpoint/training-data notices before non-course use."

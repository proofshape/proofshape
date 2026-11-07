#!/usr/bin/env bash
# A-12: build LaS-Comp's own environment on a CUDA host (Colab or Lightning), for
# scripts/lascomp_runner.py (D-057, D-058).
#
# LaS-Comp needs Python 3.10, torch 2.4.0+cu121 and CUDA-only extensions, none of which the
# service's Python 3.11 environment or its image (torch 2.8) can hold, so this builds a separate
# uv-managed venv and never touches the host's own torch. It automates A-10's working Colab
# install (stories/A/A-10.md), pinning every package A-10 left floating to the version in
# LaS-Comp's own requirements.txt freeze, so a re-run installs what was measured.
#
# Nothing of LaS-Comp is copied into this repository (its repository has no LICENSE file, D-057):
# it is cloned here, at the pinned commit, on the host that runs it.
#
#   bash scripts/setup_completion_env.sh
#
# Overrides: LASCOMP_VENV (default ~/lascomp-venv), LASCOMP_ROOT (default ~/LaS-Comp).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LASCOMP_VENV="${LASCOMP_VENV:-$HOME/lascomp-venv}"
LASCOMP_ROOT="${LASCOMP_ROOT:-$HOME/LaS-Comp}"
VENV_PYTHON="$LASCOMP_VENV/bin/python"

# D-057's pins. tests/test_setup_completion_env.py checks these equal recon/completion_contract.py's.
LASCOMP_REPO_URL="https://github.com/DavidYan2001/LaS-Comp.git"
LASCOMP_COMMIT="0da46f2569b484986d3737d97213428db962161f"
WEIGHTS_ID="microsoft/TRELLIS-image-large"
WEIGHTS_REVISION="25e0d31ffbebe4b5a97464dd851910efc3002d96"
CKPT_DIR="ckpt/ckpt_image_large"

TORCH_INDEX="https://download.pytorch.org/whl/cu121"
KAOLIN_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html"
UTILS3D_COMMIT="9a4eb15e4021b67b12c460c7057d642626897ec8"
# A prebuilt pytorch3d from a third-party builder, so its hash is checked before installing it.
PYTORCH3D_URL="https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.9/pytorch3d-0.7.9%2Bpt2.4.0cu121-cp310-cp310-linux_x86_64.whl"
PYTORCH3D_SHA256="786578d8cfe25f390a2d18a1a8075ed739d9247abe705ddeea7ffb4ffa65d1f2"

CACHE_DIR="$LASCOMP_ROOT/.proofshape-cache"
SETUP_STARTED=$SECONDS

step() {
  echo
  echo "== $1 (at $((SECONDS - SETUP_STARTED)) s)"
}

step "CUDA host"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "A-12 setup needs a CUDA host (Colab T4 or Lightning): nvidia-smi was not found." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if ! command -v uv >/dev/null 2>&1; then
  echo "A-12 setup needs uv (Colab ships it). Elsewhere: https://docs.astral.sh/uv/" >&2
  exit 1
fi
uv --version

step "Python 3.10 venv at $LASCOMP_VENV"
# only-managed: a conda Python 3.10 on Lightning must not be picked up by accident.
export UV_PYTHON_PREFERENCE=only-managed
uv python install 3.10
if [[ ! -x "$VENV_PYTHON" ]]; then
  uv venv --python 3.10 "$LASCOMP_VENV"
fi

step "torch 2.4.0+cu121 and LaS-Comp's CUDA extensions"
uv pip install --python "$VENV_PYTHON" torch==2.4.0 torchvision==0.19.0 --index-url "$TORCH_INDEX"
uv pip install --python "$VENV_PYTHON" numpy==1.26.4 xformers==0.0.27.post2 \
  --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple
uv pip install --python "$VENV_PYTHON" spconv-cu121==2.3.8
uv pip install --python "$VENV_PYTHON" kaolin==0.18.0 --find-links "$KAOLIN_LINKS"
wheel_dir="$(mktemp -d)"
pytorch3d_wheel="$wheel_dir/pytorch3d-0.7.9+pt2.4.0cu121-cp310-cp310-linux_x86_64.whl"
curl -fsSL -o "$pytorch3d_wheel" "$PYTORCH3D_URL"
echo "$PYTORCH3D_SHA256  $pytorch3d_wheel" | sha256sum -c -
uv pip install --python "$VENV_PYTHON" "$pytorch3d_wheel"
rm -rf "$wheel_dir"

step "LaS-Comp's Python dependencies"
# Versions from LaS-Comp's own requirements.txt freeze. Pillow is listed there only as a conda
# build (pillow @ file:///croot/pillow_1752524526621/work, July 2025), so it is pinned to 11.3.0,
# the Pillow release of that month.
# One exception: the freeze lists opencv-python-headless 4.12.0.88, which needs numpy >= 2 on
# Python 3.10, beside numpy 1.26.4. pip tolerates that with a warning; uv refuses it. 4.11.0.86
# accepts numpy 1.26.4, is what A-10's measured run resolved to (opencv was unpinned there), and
# is what this repo's own requirements.txt pins for the same reason.
uv pip install --python "$VENV_PYTHON" \
  numpy==1.26.4 pillow==11.3.0 imageio==2.37.3 easydict==1.13 \
  opencv-python-headless==4.11.0.86 scipy==1.15.3 scikit-image==0.25.2 \
  rembg==2.0.67 onnxruntime==1.22.1 open3d==0.19.0 trimesh==4.11.5 plyfile==1.1.2 \
  huggingface-hub==0.34.4 safetensors==0.6.2 tqdm==4.67.1 transformers==4.55.2 \
  igraph==0.11.9 pymeshfix==0.17.1 xatlas==0.0.11 \
  "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@$UTILS3D_COMMIT"

step "LaS-Comp at $LASCOMP_COMMIT"
if [[ ! -d "$LASCOMP_ROOT/.git" ]]; then
  git clone --quiet "$LASCOMP_REPO_URL" "$LASCOMP_ROOT"
fi
git -C "$LASCOMP_ROOT" fetch --quiet origin
git -C "$LASCOMP_ROOT" checkout --quiet "$LASCOMP_COMMIT"
current_commit="$(git -C "$LASCOMP_ROOT" rev-parse HEAD)"
if [[ "$current_commit" != "$LASCOMP_COMMIT" ]]; then
  echo "LaS-Comp pin mismatch: expected $LASCOMP_COMMIT, got $current_commit" >&2
  exit 1
fi

step "TRELLIS weights at $WEIGHTS_REVISION"
"$VENV_PYTHON" -I - "$WEIGHTS_ID" "$WEIGHTS_REVISION" "$LASCOMP_ROOT/$CKPT_DIR" <<'PY'
import sys

from huggingface_hub import snapshot_download

snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])
PY
# Always after the download: re-downloading can restore the published pipeline.json, which lacks
# the encoder entry LaS-Comp needs. The helper is idempotent.
"$VENV_PYTHON" -I "$REPO_ROOT/scripts/lascomp_runner.py" --ensure-encoder-entry \
  --lascomp-root "$LASCOMP_ROOT"

step "DINOv2 and u2net caches"
# Both download on first use otherwise: DINOv2 inside the model load, u2net inside the first
# sample's timed window. Prefetched here into caches the runner points at.
TORCH_HOME="$CACHE_DIR/torch" U2NET_HOME="$CACHE_DIR/u2net" \
  "$VENV_PYTHON" -I - "$LASCOMP_ROOT/$CKPT_DIR/pipeline.json" <<'PY'
import json
import sys

import torch

config = json.loads(open(sys.argv[1], encoding="utf-8").read())
torch.hub.load("facebookresearch/dinov2", config["args"]["image_cond_model"], pretrained=True)

import rembg

rembg.new_session("u2net")
PY

step "Pins file and freeze"
"$VENV_PYTHON" -I - "$LASCOMP_ROOT" "$LASCOMP_COMMIT" "$WEIGHTS_ID" "$WEIGHTS_REVISION" \
  "$CACHE_DIR" <<'PY'
import datetime
import json
import sys
from pathlib import Path

root, commit, weights, revision, cache = sys.argv[1:6]
pins = {
    "commit": commit,
    "weights": weights,
    "weights_revision": revision,
    "caches": {"torch_home": f"{cache}/torch", "u2net_home": f"{cache}/u2net"},
    "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
}
(Path(root) / "proofshape_pins.json").write_text(json.dumps(pins, indent=2) + "\n")
PY
uv pip freeze --python "$VENV_PYTHON" > "$LASCOMP_ROOT/proofshape_freeze.txt"

step "Checks"
if ! uv pip check --python "$VENV_PYTHON"; then
  echo "WARNING: uv pip check reports dependency conflicts (above); recorded, not fatal." >&2
fi
"$VENV_PYTHON" -I -c "import torch; assert torch.__version__ == '2.4.0+cu121', torch.__version__; \
assert torch.cuda.is_available(), 'CUDA is not available'"
"$VENV_PYTHON" -I "$REPO_ROOT/scripts/lascomp_runner.py" --check --lascomp-root "$LASCOMP_ROOT"

step "Done"
du -sh "$LASCOMP_VENV" "$LASCOMP_ROOT" 2>/dev/null || true
echo "Setup took $((SECONDS - SETUP_STARTED)) s. Point the service at the runner with:"
echo "PROOFSHAPE_COMPLETION_RUNNER=\"$VENV_PYTHON -I $REPO_ROOT/scripts/lascomp_runner.py --lascomp-root $LASCOMP_ROOT\""

"""A-12: static checks on scripts/setup_completion_env.sh.

The script needs a CUDA host and downloads several GB, so CI can't run it; its real run is A-12's
written Colab check. These pin what must not drift: D-057's pins, the order that keeps a non-CUDA
host from downloading anything, and the rule that it installs only into its own venv.
"""

from __future__ import annotations

import re
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

from recon import completion_contract as contract

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_completion_env.sh"
RUNNER_NS = runpy.run_path(str(SCRIPT.parent / "lascomp_runner.py"))
TEXT = SCRIPT.read_text(encoding="utf-8")
# Continuation lines joined, so a multi-line install reads as one command.
COMMANDS = TEXT.replace("\\\n", " ")


def _assigned(name: str) -> str:
    match = re.search(rf'^{name}="([^"]+)"$', TEXT, flags=re.MULTILINE)
    assert match, f"{name} is not assigned in {SCRIPT.name}"
    return match.group(1)


def test_the_pins_are_the_contracts():
    assert _assigned("LASCOMP_COMMIT") == contract.MODEL_COMMIT
    assert _assigned("WEIGHTS_ID") == contract.WEIGHTS_ID
    assert _assigned("WEIGHTS_REVISION") == contract.WEIGHTS_REVISION


def test_it_stops_on_the_first_error():
    commands = [
        line for line in TEXT.splitlines() if line.strip() and not line.startswith("#")
    ]
    assert commands[0] == "set -euo pipefail"


@pytest.mark.skipif(
    shutil.which("bash") is None, reason="needs bash (not on every Windows box)"
)
def test_it_parses():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_a_host_without_cuda_stops_before_downloading_anything():
    gpu_check = TEXT.index("command -v nvidia-smi")
    downloads = (
        "uv pip install",
        "git clone",
        "snapshot_download",
        "curl ",
        "fetch --quiet",
    )
    for download in downloads:
        assert gpu_check < TEXT.index(download), (
            f"{download!r} runs before the CUDA check"
        )


def test_every_install_goes_into_the_venv_and_torch_is_lascomps():
    installs = [line for line in COMMANDS.splitlines() if "pip install" in line]
    assert installs
    for line in installs:
        # Never the host's own pip, and never without the venv's interpreter: the host's
        # managed torch (Colab's, Lightning's) must survive this script.
        assert line.strip().startswith('uv pip install --python "$VENV_PYTHON"'), line
    torch_line = next(line for line in installs if "torch==" in line)
    assert "torch==2.4.0" in torch_line and "torchvision==0.19.0" in torch_line
    assert '--index-url "$TORCH_INDEX"' in torch_line
    assert _assigned("TORCH_INDEX").endswith("/cu121")


def test_every_package_is_pinned():
    for line in COMMANDS.splitlines():
        if "uv pip install" not in line or "$pytorch3d_wheel" in line:
            continue
        tokens = line.split('$VENV_PYTHON"', 1)[1].split()
        packages = []
        skip_value = False
        for token in tokens:
            if skip_value:  # the value of an option such as --index-url
                skip_value = False
                continue
            if token.startswith("--"):
                skip_value = True  # every option this script passes takes a value
                continue
            packages.append(token.strip('"'))
        for package in packages:
            if package in {"utils3d", "@"} or package.startswith("git+"):
                continue
            assert "==" in package, f"{package} is not pinned"


def test_the_third_party_pytorch3d_wheel_is_hash_checked_before_install():
    assert re.fullmatch(r"[0-9a-f]{64}", _assigned("PYTORCH3D_SHA256"))
    assert TEXT.index("sha256sum -c") < TEXT.index(
        'uv pip install --python "$VENV_PYTHON" "$pytorch3d_wheel"'
    )


def test_lascomp_is_verified_at_the_pin_and_patched_after_every_download():
    assert 'if [[ "$current_commit" != "$LASCOMP_COMMIT" ]]' in TEXT
    assert TEXT.index("snapshot_download") < TEXT.index("--ensure-encoder-entry")


def test_it_ends_with_the_runners_own_check():
    assert TEXT.index("--check --lascomp-root") > TEXT.index("proofshape_pins.json")
    assert TEXT.index("--check --lascomp-root") > TEXT.index("--ensure-encoder-entry")


def test_dinov2_is_pinned_to_the_commit_the_runner_checks():
    assert _assigned("DINOV2_COMMIT") == RUNNER_NS["DINOV2_COMMIT"]
    assert _assigned("DINOV2_CACHE_NAME") == RUNNER_NS["DINOV2_CACHE_NAME"]
    assert (
        _assigned("DINOV2_REPO_URL") == "https://github.com/facebookresearch/dinov2.git"
    )


def test_dinov2_is_verified_before_torch_hub_loads_it_and_recorded_in_the_pins():
    verified = TEXT.index('if [[ "$dinov2_commit" != "$DINOV2_COMMIT" ]]')
    assert verified < TEXT.index('torch.hub.load("facebookresearch/dinov2"')
    assert '"dinov2": {"commit": dinov2_commit' in TEXT

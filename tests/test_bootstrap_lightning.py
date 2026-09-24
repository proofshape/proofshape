"""Regression checks for the F-07 Lightning bootstrap path."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bootstrap_dev_env.sh"


def test_bootstrap_handles_lightning_managed_conda_before_venv_creation() -> None:
    content = SCRIPT.read_text(encoding="utf-8")

    lightning_check = 'CONDA_DEFAULT_ENV:-}" = "cloudspace"'
    managed_flag = "USE_MANAGED_ENV=1"
    venv_command = '"$PYTHON_BIN" -m venv .venv'

    assert lightning_check in content
    assert managed_flag in content
    assert venv_command in content
    assert content.index(lightning_check) < content.index(venv_command)


def test_bootstrap_uses_existing_python_for_managed_environment() -> None:
    content = SCRIPT.read_text(encoding="utf-8")

    assert 'ENV_PYTHON="$PYTHON_BIN"' in content
    assert '"$ENV_PYTHON" -m pip install -r requirements.txt' in content


def test_bootstrap_does_not_install_editable_project_twice() -> None:
    content = SCRIPT.read_text(encoding="utf-8")

    assert "python -m pip install -e ." not in content
    assert '"$ENV_PYTHON" -m pip install -e .' not in content

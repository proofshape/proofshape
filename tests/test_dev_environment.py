from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_script_exists_and_requires_supported_python():
    script = ROOT / "scripts" / "bootstrap_dev_env.sh"
    assert script.exists(), "F-07 requires a one-command bootstrap script"

    content = script.read_text(encoding="utf-8")
    assert "python3.11" in content
    assert "Python 3.11+" in content
    assert "python -m pip install --upgrade pip" in content
    assert "python -m pip install -r requirements.txt" in content
    assert "python -m pip install -e ." in content


def test_requirements_file_pins_versions_for_dev_setup():
    req = ROOT / "requirements.txt"
    assert req.exists(), "F-07 requires a committed, pinned dependency lock"

    lines = [
        line.strip()
        for line in req.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert any(line.startswith("-e .") for line in lines)
    assert any("pytest==" in line for line in lines)
    assert any("PyYAML==" in line for line in lines)
    assert any("jsonschema==" in line for line in lines)
    assert any("openapi-spec-validator==" in line for line in lines)
    assert any("ruff==" in line for line in lines), (
        "ruff must be pinned here too, or the pre-PR checklist in "
        "AGENTS_START_HERE.md (ruff check / ruff format --check) isn't reproducible "
        "from a clean checkout"
    )

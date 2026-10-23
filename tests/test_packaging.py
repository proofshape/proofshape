"""Regression test for D-030: recon installs as a package, inspect never shadows the stdlib.

If someone later adds inspect/__init__.py and registers it in pyproject.toml without
re-reading D-030, this fails immediately instead of silently reintroducing the shadow.

D-053 adds a second package, `inspection`, at inspect/inspection/ -- the two tests below are
that decision's own stated acceptance check: the standard library's `inspect` still resolves
under its real stdlib path, and `inspection` resolves under `inspect/inspection/`, not `inspect/`
itself (which would mean the directory got an `__init__.py` after all, quietly reopening D-030).
"""

import inspect
import sysconfig
from pathlib import Path


def test_recon_is_installed():
    import recon  # noqa: F401


def test_inspect_is_still_the_standard_library():
    assert "proofshape" not in inspect.__file__


def test_inspect_resolves_under_the_real_stdlib_path():
    stdlib_path = Path(sysconfig.get_path("stdlib")).resolve()
    assert Path(inspect.__file__).resolve().is_relative_to(stdlib_path)


def test_inspection_is_installed_under_inspect_inspection():
    import inspection

    repo_root = Path(__file__).resolve().parent.parent
    expected_root = repo_root / "inspect" / "inspection"
    assert Path(inspection.__file__).resolve().is_relative_to(expected_root)

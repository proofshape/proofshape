"""Regression test for D-030: recon installs as a package, inspect never shadows the stdlib.

If someone later adds inspect/__init__.py and registers it in pyproject.toml without
re-reading D-030, this fails immediately instead of silently reintroducing the shadow.
"""

import inspect


def test_recon_is_installed():
    import recon  # noqa: F401


def test_inspect_is_still_the_standard_library():
    assert "proofshape" not in inspect.__file__

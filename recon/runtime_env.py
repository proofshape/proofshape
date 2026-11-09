"""R-31: process-level runtime guards needed before reconstruction libraries import."""

from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping

MACOS_OPENMP_THREADS = "1"


def configure_macos_openmp(
    *,
    platform: str | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> bool:
    """Make Open3D and torch coexist on macOS before either OpenMP runtime initialises.

    The macOS wheels used by ProofShape each ship their own libomp.dylib. With more than one
    OpenMP worker, loading Open3D before torch deadlocks at torch's first threaded operation, while
    loading torch before Open3D can segfault (#261). OMP_NUM_THREADS=1 is the verified safe
    configuration for the real pipeline.

    This is intentionally forced, not setdefault: an inherited value such as 4 would recreate
    the bug. Non-macOS platforms are left untouched.
    """
    current_platform = sys.platform if platform is None else platform
    if current_platform != "darwin":
        return False

    target = os.environ if environ is None else environ
    target["OMP_NUM_THREADS"] = MACOS_OPENMP_THREADS
    return True

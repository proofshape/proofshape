"""Reconstruction engine: board detection, VGGT, fusion, and the FastAPI endpoint.

Scaffolding only — see stories R-01 onward for the actual implementation.
"""

from recon.runtime_env import configure_macos_openmp

# R-31: this must run at package import time, before any recon submodule can initialise either
# Open3D's or torch's bundled OpenMP runtime. On macOS those two runtimes deadlock/segfault when
# both are allowed to create worker pools in one process (#261). Linux is deliberately unchanged.
configure_macos_openmp()

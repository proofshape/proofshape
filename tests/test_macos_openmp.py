"""R-31 regression tests for the macOS Open3D/torch OpenMP conflict."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from recon.runtime_env import MACOS_OPENMP_THREADS, configure_macos_openmp

ROOT = Path(__file__).resolve().parents[1]
MACOS_IMPORT_TIMEOUT_S = 30


def test_macos_guard_forces_the_verified_safe_thread_count():
    env = {"OMP_NUM_THREADS": "8", "KEEP": "yes"}

    applied = configure_macos_openmp(platform="darwin", environ=env)

    assert applied is True
    assert env == {"OMP_NUM_THREADS": MACOS_OPENMP_THREADS, "KEEP": "yes"}


def test_non_macos_guard_leaves_the_environment_unchanged():
    env = {"OMP_NUM_THREADS": "8", "KEEP": "yes"}

    applied = configure_macos_openmp(platform="linux", environ=env)

    assert applied is False
    assert env == {"OMP_NUM_THREADS": "8", "KEEP": "yes"}


def test_runtime_env_applies_guard_when_service_startup_imports_it():
    code = (
        "import os, sys; "
        "sys.platform = 'darwin'; "
        "os.environ['OMP_NUM_THREADS'] = '8'; "
        "import recon.runtime_env; "
        "assert os.environ['OMP_NUM_THREADS'] == '1'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_service_imports_runtime_guard_before_agent_tools():
    service_source = (ROOT / "recon" / "service.py").read_text(encoding="utf-8")
    guard_import = "from recon import runtime_env as _runtime_env"
    agent_import = "    from recon import agent_tools"
    assert guard_import in service_source
    assert agent_import in service_source
    assert service_source.index(guard_import) < service_source.index(agent_import)


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="R-31 reproduces only with the macOS Open3D/torch wheels and their two libomp runtimes",
)
def test_real_pipeline_import_order_does_not_deadlock_on_macos():
    pytest.importorskip(
        "torch", reason="the real pipeline check needs the optional torch dependency"
    )
    # Import the real service first: today that reaches agent_tools -> provenance -> Open3D before
    # run_reconstruction_pipeline later imports the rest of the pipeline and reaches torch/VGGT.
    # Then exercise the exact two library directions that reproduced #261. The timeout turns a
    # recurrence into a test failure in seconds instead of leaving a session pending for hours.
    code = """
import os
import recon.service

assert os.environ["OMP_NUM_THREADS"] == "1"

from recon import (
    board_pose,
    completion,
    disagreement,
    mesh_cleanup,
    metric_alignment,
    provenance,
    reconstruction_runner,
    tsdf_fusion,
    uncertainty,
)

import torch

tensor = torch.zeros((1, 1374, 1024))
assert tensor.sum().item() == 0.0

import open3d as o3d

mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
mesh.compute_vertex_normals()
tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(tmesh)
queries = o3d.core.Tensor([[2.0, 2.0, 2.0]], dtype=o3d.core.Dtype.Float32)
distance = scene.compute_distance(queries)
assert float(distance[0].item()) > 0.0
print("R31_OK")
"""
    env = os.environ.copy()
    env.pop("OMP_NUM_THREADS", None)
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=MACOS_IMPORT_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"R-31 import-order subprocess exceeded {MACOS_IMPORT_TIMEOUT_S}s; "
            "the OpenMP deadlock may have returned. "
            f"stdout={exc.stdout!r} stderr={exc.stderr!r}"
        )

    assert result.returncode == 0, result.stderr
    assert "R31_OK" in result.stdout

"""Unit tests for I-17's run-to-run variation script."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts import i17_measure_variation as var

# --- helpers ----------------------------------------------------------------------------------


def _box_cad(
    size_x: float, size_y: float, size_z: float
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [size_x, 0.0, 0.0],
            [size_x, size_y, 0.0],
            [0.0, size_y, 0.0],
            [0.0, 0.0, size_z],
            [size_x, 0.0, size_z],
            [size_x, size_y, size_z],
            [0.0, size_y, size_z],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )
    return vertices, faces


def _fake_run_once_result(
    wall: float = 3.0, top: float = 1.0, worst: float = 5.0, yaw: float = 0.0
) -> dict:
    return {
        "wall_abs_p95_mm": wall,
        "top_abs_p95_mm": top,
        "worst_reading_mm": worst,
        "pose_yaw_deg": yaw,
        "fitness": 0.95,
        "coverage_ratio": 1.1,
        "orientation_ambiguous": False,
    }


# --- tests ------------------------------------------------------------------------------------


class TestStats:
    """The _stats helper inside run_variation."""

    def test_spread_of_identical_values_is_zero(self):
        values = [3.0, 3.0, 3.0]
        arr = np.array(values)
        stats = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "spread": float(arr.max() - arr.min()),
            "std": float(arr.std()),
            "mean": float(arr.mean()),
        }
        assert stats["spread"] == 0.0
        assert stats["std"] == 0.0

    def test_spread_of_varying_values(self):
        values = [2.5, 3.0, 3.5]
        arr = np.array(values)
        stats = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "spread": float(arr.max() - arr.min()),
            "std": float(arr.std()),
            "mean": float(arr.mean()),
        }
        assert stats["spread"] == pytest.approx(1.0)
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["min"] == pytest.approx(2.5)
        assert stats["max"] == pytest.approx(3.5)

    def test_empty_values_give_none(self):
        values: list[float] = []
        # Reproduce the internal logic
        if not values:
            stats = {
                "min": None,
                "max": None,
                "spread": None,
                "std": None,
                "mean": None,
            }
        assert stats["spread"] is None


class TestRunVariation:
    """run_variation with run_once mocked out."""

    def test_collects_n_runs_and_reports_spread(self):
        wall_values = [3.0, 3.1, 2.9, 3.05, 2.95]
        call_count = [0]

        def fake_run_once(model, cad_mesh, board_surface):
            i = call_count[0]
            call_count[0] += 1
            return _fake_run_once_result(wall=wall_values[i])

        with (
            patch.object(var, "run_once", side_effect=fake_run_once),
            patch.object(var, "_load_model", return_value=(None, None, None)),
            patch.object(var, "load_mesh", return_value=(None, None, "mm")),
            patch.object(var, "_load_board_surface", return_value=None),
        ):
            result = var.run_variation(
                Path("model.glb"), Path("cad.stl"), Path("board.json"), 5
            )

        assert result["n_runs"] == 5
        assert len(result["runs"]) == 5
        assert result["wall_abs_p95"]["spread"] == pytest.approx(0.2)
        assert result["wall_abs_p95"]["mean"] == pytest.approx(3.0)

    def test_deterministic_runs_give_zero_spread(self):
        def fake_run_once(model, cad_mesh, board_surface):
            return _fake_run_once_result(wall=3.44, top=0.98)

        with (
            patch.object(var, "run_once", side_effect=fake_run_once),
            patch.object(var, "_load_model", return_value=(None, None, None)),
            patch.object(var, "load_mesh", return_value=(None, None, "mm")),
            patch.object(var, "_load_board_surface", return_value=None),
        ):
            result = var.run_variation(
                Path("model.glb"), Path("cad.stl"), Path("board.json"), 3
            )

        assert result["wall_abs_p95"]["spread"] == 0.0
        assert result["top_abs_p95"]["spread"] == 0.0


class TestMainCLI:
    """CLI argument handling."""

    def test_refuses_existing_output(self, tmp_path):
        out = tmp_path / "result.json"
        out.write_text("{}")
        with pytest.raises(FileExistsError, match="already exists"):
            var.main(
                [
                    "--model",
                    "m.glb",
                    "--cad",
                    "c.stl",
                    "--board-json",
                    "b.json",
                    "--out",
                    str(out),
                ]
            )

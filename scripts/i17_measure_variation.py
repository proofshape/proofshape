"""I-17: measure run-to-run variation in registration + deviation on a saved model.

#440 found that Open3D's ICP is not fully deterministic under threads. This script
runs the registration-to-deviation pipeline N times on the same model and CAD, with
OMP_NUM_THREADS=1, and reports how much the wall and top-face 95th-percentile readings
vary across runs. The story then compares that spread with the 0.42 mm change between
commits to decide whether the change is systematic or within noise.

Usage (from the repository root):

    PYTHONPATH=.:inspect OMP_NUM_THREADS=1 .venv/bin/python scripts/i17_measure_variation.py \
        --model <path>/mesh_uncertainty.glb \
        --cad fixtures/reference_cad/<part>.stl \
        --board-json <path>/tsdf_fusion.json \
        --runs 10 \
        --out <new file>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from recon import runtime_env as _runtime_env  # noqa: F401  # isort: skip

from inspection import registration
from inspection.cad import load_mesh
from inspection.deviation import compute_deviation

from recon.mesh_cleanup import read_glb
from recon.part_segmentation import BoardSurface
from recon.provenance import LABEL_OBSERVED, colors_to_labels
from scripts.measure_part_accuracy import summarise_part


def _load_model(model_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices, faces, colors = read_glb(model_path)
    if colors is None:
        raise ValueError(f"{model_path} has no provenance colours.")
    labels = colors_to_labels(colors)
    return vertices, faces, labels


def _load_board_surface(board_json_path: Path) -> BoardSurface:
    report = json.loads(board_json_path.read_text(encoding="utf-8"))
    return BoardSurface(**report["board_surface"])


def run_once(
    model: tuple[np.ndarray, np.ndarray, np.ndarray],
    cad_mesh: tuple[np.ndarray, np.ndarray, str],
    board_surface: BoardSurface,
) -> dict[str, Any]:
    """Run registration + deviation once and return the summary."""
    vertices, faces, labels = model
    candidates = registration.global_register(model, cad_mesh)
    ambiguity = registration.check_ambiguity(candidates, model, cad_mesh)
    best = ambiguity.best
    deviation_mm, deviation_measured = compute_deviation(
        model, cad_mesh, best.transformation, board_surface
    )
    measure = {
        "vertices": vertices,
        "faces": faces,
        "observed": (labels == LABEL_OBSERVED),
        "deviation_mm": deviation_mm,
        "deviation_measured": deviation_measured,
        "cad_to_model": best.transformation,
        "cad_vertices": cad_mesh[0],
        "cad_faces": cad_mesh[1],
    }
    summary = summarise_part(measure)
    return {
        "wall_abs_p95_mm": summary["wall"]["abs_p95_mm"] if summary["wall"] else None,
        "top_abs_p95_mm": summary["top"]["abs_p95_mm"] if summary["top"] else None,
        "worst_reading_mm": summary["worst_reading_mm"],
        "pose_yaw_deg": summary["pose_yaw_deg"],
        "fitness": float(best.fitness),
        "coverage_ratio": ambiguity.coverage_ratio,
        "orientation_ambiguous": bool(ambiguity.orientation_ambiguous),
    }


def run_variation(
    model_path: Path,
    cad_path: Path,
    board_json_path: Path,
    n_runs: int,
) -> dict[str, Any]:
    """Run registration+deviation n_runs times and report the spread."""
    model = _load_model(model_path)
    cad_vertices, cad_faces, cad_units = load_mesh(cad_path)
    cad_mesh = (cad_vertices, cad_faces, cad_units)
    board_surface = _load_board_surface(board_json_path)

    runs = []
    for i in range(n_runs):
        result = run_once(model, cad_mesh, board_surface)
        result["run"] = i
        runs.append(result)
        wall = result["wall_abs_p95_mm"]
        top = result["top_abs_p95_mm"]
        print(
            f"  run {i:2d}: wall |95%| = {wall:.3f} mm, "
            f"top |95%| = {top:.3f} mm, "
            f"yaw = {result['pose_yaw_deg']:.2f}°",
            file=sys.stderr,
        )

    wall_values = [
        r["wall_abs_p95_mm"] for r in runs if r["wall_abs_p95_mm"] is not None
    ]
    top_values = [r["top_abs_p95_mm"] for r in runs if r["top_abs_p95_mm"] is not None]

    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"min": None, "max": None, "spread": None, "std": None, "mean": None}
        arr = np.array(values)
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "spread": float(arr.max() - arr.min()),
            "std": float(arr.std()),
            "mean": float(arr.mean()),
        }

    return {
        "model": str(model_path),
        "cad": str(cad_path),
        "board_json": str(board_json_path),
        "n_runs": n_runs,
        "wall_abs_p95": _stats(wall_values),
        "top_abs_p95": _stats(top_values),
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="I-17: measure run-to-run registration variation on a saved model."
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--cad", required=True, type=Path)
    parser.add_argument("--board-json", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.out.exists():
        raise FileExistsError(f"{args.out} already exists; pick a new --out.")

    result = run_variation(args.model, args.cad, args.board_json, args.runs)

    wall = result["wall_abs_p95"]
    top = result["top_abs_p95"]
    print(f"\n{args.runs} runs on {args.model.name}:")
    print(
        f"  wall |95%|: {wall['mean']:.3f} ± {wall['std']:.4f} mm "
        f"(spread {wall['spread']:.4f} mm)"
    )
    print(
        f"  top  |95%|: {top['mean']:.3f} ± {top['std']:.4f} mm "
        f"(spread {top['spread']:.4f} mm)"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()

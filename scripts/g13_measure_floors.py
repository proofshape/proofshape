"""G-13: measure the verdict's sigma floors and outline margin on the served pipeline.

Re-runs registration + deviation from saved models on current main (I-15 merged, R-44 off),
splits measured vertices into the verdict's own pools (height-percentile split, not CAD-face
split), and computes the floor-coverage statistic from D-077: the smallest sigma_floor at
which k * max(sigma, floor) covers 95% of a good part's readings in that pool.

Also compares outline margins (0, 1, 2 mm) to settle D-086 before the floors are measured.

Usage (from the repository root):

    PYTHONPATH=.:inspect OMP_NUM_THREADS=1 .venv/bin/python scripts/g13_measure_floors.py \
        --data-dir ~/proofshape-runs/i15-scratch-meshes-2026-11-23 \
        --out ~/proofshape-runs/g13-floors \
        --margins 0,1,2
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
from inspection.verdict import D077_K, TOP_REGION_PERCENTILE

from recon.cad_size_check import compute_height_above_board
from recon.mesh_cleanup import read_glb
from recon.part_segmentation import BoardSurface
from recon.provenance import LABEL_OBSERVED, colors_to_labels
from recon.uncertainty import read_sigma_npz
from scripts.measure_part_accuracy import (
    _spread,
    distance_outside_cad_mm,
    tolerance_sweep,
    verdict_pool_sizes,
)

PARTS = [
    "s-04",
    "s-05",
    "g05_00",
    "g05_02",
    "g02_01",
    "g02_02",
    "g02_03",
    "g02_04",
    "g02_05",
    "g02_06",
]

STRAY_DISTANCE_MM = 5.0

CANDIDATE_FLOORS_MM = tuple(round(v, 2) for v in np.arange(0.1, 10.05, 0.1).tolist())

DEFAULT_MARGINS_MM = (0.0, 1.0, 2.0)

CAD_FILES = {
    "s-04": "s04_triangular_prism_60-26-65x30mm.stl",
    "s-05": "s05_rect_prism_30x30x60mm.stl",
    "g05_00": "g05_00_control_50x32x20mm.stl",
    "g05_02": "g05_02_hole_shift_4mm.stl",
    "g02_01": "g02_01_square_20x20x15mm.stl",
    "g02_02": "g02_02_rect_50x20x25mm.stl",
    "g02_03": "g02_03_right_triangle_40x30x20mm.stl",
    "g02_04": "g02_04_pentagon_r22x25mm.stl",
    "g02_05": "g02_05_hexagon_r28x35mm.stl",
    "g02_06": "g02_06_rect_70x45x40mm.stl",
}


def _load_model(model_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices, faces, colors = read_glb(model_path)
    if colors is None:
        raise ValueError(f"{model_path} has no provenance colours.")
    labels = colors_to_labels(colors)
    return vertices, faces, labels


def _load_board_surface(board_json_path: Path) -> BoardSurface:
    report = json.loads(board_json_path.read_text(encoding="utf-8"))
    return BoardSurface(**report["board_surface"])


def _verdict_pool_split(
    vertices: np.ndarray,
    board_plane: tuple[float, float, float],
    judged_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split judged vertices into out-of-plane / in-plane using the verdict's own rule."""
    judged_verts = vertices[judged_mask]
    height = compute_height_above_board(judged_verts, board_plane)
    threshold = float(np.percentile(height, TOP_REGION_PERCENTILE))
    out_of_plane = np.zeros(len(vertices), dtype=bool)
    in_plane = np.zeros(len(vertices), dtype=bool)
    judged_indices = np.where(judged_mask)[0]
    out_of_plane[judged_indices[height >= threshold]] = True
    in_plane[judged_indices[height < threshold]] = True
    return out_of_plane, in_plane


def _floor_coverage(
    deviation_mm: np.ndarray,
    sigma_mm: np.ndarray,
    candidate_floor: float,
    k: float = D077_K,
) -> float:
    """Share of vertices within k * max(sigma, floor)."""
    if len(deviation_mm) == 0:
        return 0.0
    sigma_eff = np.fmax(sigma_mm, candidate_floor)
    threshold = k * sigma_eff
    within = np.abs(deviation_mm) <= threshold
    return float(np.count_nonzero(within) / len(deviation_mm))


def _smallest_covering_floor(
    deviation_mm: np.ndarray,
    sigma_mm: np.ndarray,
    target_coverage: float = 0.95,
    k: float = D077_K,
) -> float | None:
    """The smallest candidate floor at which coverage reaches target_coverage."""
    for floor in CANDIDATE_FLOORS_MM:
        if _floor_coverage(deviation_mm, sigma_mm, floor, k) >= target_coverage:
            return floor
    return None


def measure_part(
    data_dir: Path,
    part: str,
    margins_mm: tuple[float, ...],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Run registration + deviation for one part, then analyse at each margin.

    Returns (result_dict, pool_arrays) where pool_arrays is keyed by margin string
    and holds {oop_dev, oop_sig, ip_dev, ip_sig} numpy arrays for pooled analysis.
    """
    variant_dir = data_dir / part / "variants" / "base"
    model_path = variant_dir / "uncertainty" / "mesh_uncertainty.glb"
    sigma_path = variant_dir / "uncertainty" / "sigma_mm.npz"
    board_json_path = variant_dir / "fusion" / "tsdf_fusion.json"
    cad_path = Path("fixtures/reference_cad") / CAD_FILES[part]

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    vertices, faces, labels = _load_model(model_path)
    model = (vertices, faces, labels)
    cad_mesh = load_mesh(str(cad_path))
    board_surface = _load_board_surface(board_json_path)
    sigma_mm = read_sigma_npz(sigma_path)["sigma_mm"]

    candidates = registration.global_register(model, cad_mesh)
    ambiguity = registration.check_ambiguity(candidates, model, cad_mesh)
    best = ambiguity.best
    cad_to_model = best.transformation

    observed = labels == LABEL_OBSERVED
    board_plane = (
        board_surface.slope_x,
        board_surface.slope_y,
        board_surface.z_at_origin_mm,
    )

    margin_results = {}
    pool_arrays: dict[str, dict[str, np.ndarray]] = {}
    for margin in margins_mm:
        deviation_mm, deviation_measured = compute_deviation(
            model,
            cad_mesh,
            cad_to_model,
            board_surface,
            footprint_margin_mm=margin,
        )
        measured = deviation_measured > 0.5
        judged = observed & measured

        out_of_plane, in_plane = _verdict_pool_split(
            vertices,
            board_plane,
            judged,
        )

        dev_oop = deviation_mm[out_of_plane]
        dev_ip = deviation_mm[in_plane]
        sig_oop = sigma_mm[out_of_plane]
        sig_ip = sigma_mm[in_plane]

        pool_arrays[str(margin)] = {
            "oop_dev": dev_oop,
            "oop_sig": sig_oop,
            "ip_dev": dev_ip,
            "ip_sig": sig_ip,
        }

        # Surface outside CAD (stray)
        distance = distance_outside_cad_mm(
            vertices,
            cad_mesh[0],
            cad_mesh[1],
            cad_to_model,
        )
        stray_count = int(np.count_nonzero(measured & (distance > STRAY_DISTANCE_MM)))

        # Floor coverage sweep
        oop_coverage = {
            str(f): _floor_coverage(dev_oop, sig_oop, f) for f in CANDIDATE_FLOORS_MM
        }
        ip_coverage = {
            str(f): _floor_coverage(dev_ip, sig_ip, f) for f in CANDIDATE_FLOORS_MM
        }

        margin_results[str(margin)] = {
            "margin_mm": margin,
            "observed_count": int(np.count_nonzero(observed)),
            "measured_count": int(np.count_nonzero(measured)),
            "measured_share": (
                float(np.count_nonzero(measured) / np.count_nonzero(observed))
                if observed.any()
                else None
            ),
            "stray_count": stray_count,
            "stray_share": (
                float(stray_count / np.count_nonzero(observed))
                if observed.any()
                else None
            ),
            "out_of_plane": {
                "count": int(np.count_nonzero(out_of_plane)),
                "deviation": _spread(dev_oop),
                "sigma": _spread(sig_oop),
                "smallest_95_floor_mm": _smallest_covering_floor(dev_oop, sig_oop),
                "coverage_at_current_floor": _floor_coverage(dev_oop, sig_oop, 0.86),
            },
            "in_plane": {
                "count": int(np.count_nonzero(in_plane)),
                "deviation": _spread(dev_ip),
                "sigma": _spread(sig_ip),
                "smallest_95_floor_mm": _smallest_covering_floor(dev_ip, sig_ip),
                "coverage_at_current_floor": _floor_coverage(dev_ip, sig_ip, 5.278),
            },
            "floor_coverage": {
                "out_of_plane": oop_coverage,
                "in_plane": ip_coverage,
            },
        }

    # Also run the tolerance sweep and verdict pool sizes at margin=0
    deviation_mm_0, deviation_measured_0 = compute_deviation(
        model,
        cad_mesh,
        cad_to_model,
        board_surface,
        footprint_margin_mm=0.0,
    )
    measure_dict = {
        "vertices": vertices,
        "faces": faces,
        "observed": observed,
        "deviation_mm": deviation_mm_0,
        "deviation_measured": deviation_measured_0,
        "sigma_mm": sigma_mm,
        "cad_to_model": cad_to_model,
        "cad_vertices": cad_mesh[0],
        "cad_faces": cad_mesh[1],
        "board_plane": np.array(list(board_plane)),
    }
    sweep = tolerance_sweep(measure_dict)
    pools = verdict_pool_sizes(measure_dict)

    return (
        {
            "part": part,
            "fitness": float(best.fitness),
            "orientation_ambiguous": ambiguity.orientation_ambiguous,
            "coverage_ratio": ambiguity.coverage_ratio,
            "margins": margin_results,
            "tolerance_sweep": sweep,
            "verdict_pools": pools,
        },
        pool_arrays,
    )


def _print_summary_table(results: list[dict[str, Any]], margin: str) -> None:
    """Print a human-readable summary table for one margin."""
    print(f"\n{'=' * 80}")
    print(f"Margin = {margin} mm")
    print(f"{'=' * 80}")
    header = (
        f"{'Part':<10} {'Obs':>5} {'Meas':>5} {'M%':>5} "
        f"{'OOP':>4} {'IP':>5} "
        f"{'OOP dev p95':>11} {'IP dev p95':>10} "
        f"{'OOP σ p95':>9} {'IP σ p95':>8} "
        f"{'OOP fl95':>8} {'IP fl95':>7} "
        f"{'1stPass':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        m = r["margins"][margin]
        oop_dev = m["out_of_plane"]["deviation"]
        ip_dev = m["in_plane"]["deviation"]
        oop_sig = m["out_of_plane"]["sigma"]
        ip_sig = m["in_plane"]["sigma"]
        fp = r["tolerance_sweep"]["first_pass_mm"]
        print(
            f"{r['part']:<10} "
            f"{m['observed_count']:>5} {m['measured_count']:>5} "
            f"{m['measured_share'] * 100 if m['measured_share'] else 0:>5.1f} "
            f"{m['out_of_plane']['count']:>4} {m['in_plane']['count']:>5} "
            f"{oop_dev['abs_p95_mm'] if oop_dev else 0:>11.3f} "
            f"{ip_dev['abs_p95_mm'] if ip_dev else 0:>10.3f} "
            f"{oop_sig['p95_mm'] if oop_sig else 0:>9.3f} "
            f"{ip_sig['p95_mm'] if ip_sig else 0:>8.3f} "
            f"{m['out_of_plane']['smallest_95_floor_mm'] or 'N/A':>8} "
            f"{m['in_plane']['smallest_95_floor_mm'] or 'N/A':>7} "
            f"{fp if fp else 'N/A':>7}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="G-13: measure verdict floors and outline margin"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root of the i15-scratch-meshes directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--margins",
        type=str,
        default="0,1,2",
        help="Comma-separated outline margins in mm to compare (default: 0,1,2)",
    )
    parser.add_argument(
        "--parts",
        type=str,
        default="all",
        help="Comma-separated part names, or 'all' (default: all)",
    )
    parser.add_argument(
        "--floors",
        type=str,
        default=None,
        help="Override floor constants as OOP,IP in mm (e.g. '1.0,5.7'). "
        "Patches verdict.D073_SIGMA_FLOOR_BOARD_MM and D078_SIGMA_FLOOR_IN_PLANE_MM "
        "before measuring. Default: use the constants in verdict.py.",
    )
    args = parser.parse_args(argv)

    if args.out.exists() and any(args.out.iterdir()):
        print(
            f"ERROR: {args.out} already has files. Pick a new --out.", file=sys.stderr
        )
        return 1

    margins_mm = tuple(float(m) for m in args.margins.split(","))
    parts = PARTS if args.parts == "all" else args.parts.split(",")

    if args.floors is not None:
        import inspection.verdict as verdict_mod

        oop_floor, ip_floor = (float(f) for f in args.floors.split(","))
        verdict_mod.D073_SIGMA_FLOOR_BOARD_MM = oop_floor
        verdict_mod.D078_SIGMA_FLOOR_IN_PLANE_MM = ip_floor
        print(f"Patched floors: OOP={oop_floor}, IP={ip_floor}")

    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    all_pool_arrays: list[dict[str, dict[str, np.ndarray]]] = []
    for part in parts:
        print(f"\n--- {part} ---")
        result, pool_arrays = measure_part(args.data_dir, part, margins_mm)
        results.append(result)
        all_pool_arrays.append(pool_arrays)

        # Save per-part result
        part_file = args.out / f"{part}.json"
        part_file.write_text(
            json.dumps(result, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(
            f"  fitness={result['fitness']:.3f}, "
            f"ambiguous={result['orientation_ambiguous']}, "
            f"first_pass={result['tolerance_sweep']['first_pass_mm']}"
        )

    # Print summary tables
    for margin in [str(m) for m in margins_mm]:
        _print_summary_table(results, margin)

    # Pooled floor analysis
    print(f"\n{'=' * 80}")
    print(f"POOLED FLOOR ANALYSIS ({len(results)} parts combined)")
    print(f"{'=' * 80}")
    print("Statistic: smallest floor at which k * max(sigma, floor) covers 95%")
    print(f"k = {D077_K}, TOP_REGION_PERCENTILE = {TOP_REGION_PERCENTILE}")

    for margin in [str(m) for m in margins_mm]:
        all_dev_oop = [pa[margin]["oop_dev"] for pa in all_pool_arrays]
        all_sig_oop = [pa[margin]["oop_sig"] for pa in all_pool_arrays]
        all_dev_ip = [pa[margin]["ip_dev"] for pa in all_pool_arrays]
        all_sig_ip = [pa[margin]["ip_sig"] for pa in all_pool_arrays]

        pooled_dev_oop = np.concatenate(all_dev_oop) if all_dev_oop else np.array([])
        pooled_sig_oop = np.concatenate(all_sig_oop) if all_sig_oop else np.array([])
        pooled_dev_ip = np.concatenate(all_dev_ip) if all_dev_ip else np.array([])
        pooled_sig_ip = np.concatenate(all_sig_ip) if all_sig_ip else np.array([])

        pooled_oop_floor = _smallest_covering_floor(pooled_dev_oop, pooled_sig_oop)
        pooled_ip_floor = _smallest_covering_floor(pooled_dev_ip, pooled_sig_ip)
        pooled_oop_cov = _floor_coverage(pooled_dev_oop, pooled_sig_oop, 0.86)
        pooled_ip_cov = _floor_coverage(pooled_dev_ip, pooled_sig_ip, 5.278)

        print(f"\n  Margin = {margin} mm:")
        print(
            f"    OOP: {len(pooled_dev_oop)} verts, "
            f"pooled 95% floor = {pooled_oop_floor or 'N/A'} mm, "
            f"coverage at 0.86 = {pooled_oop_cov * 100:.1f}%"
        )
        print(
            f"    IP:  {len(pooled_dev_ip)} verts, "
            f"pooled 95% floor = {pooled_ip_floor or 'N/A'} mm, "
            f"coverage at 5.278 = {pooled_ip_cov * 100:.1f}%"
        )

    # Compute pooled stats for the report
    pooled: dict[str, Any] = {}
    for margin in [str(m) for m in margins_mm]:
        all_dev_oop = [pa[margin]["oop_dev"] for pa in all_pool_arrays]
        all_sig_oop = [pa[margin]["oop_sig"] for pa in all_pool_arrays]
        all_dev_ip = [pa[margin]["ip_dev"] for pa in all_pool_arrays]
        all_sig_ip = [pa[margin]["ip_sig"] for pa in all_pool_arrays]
        p_dev_oop = np.concatenate(all_dev_oop) if all_dev_oop else np.array([])
        p_sig_oop = np.concatenate(all_sig_oop) if all_sig_oop else np.array([])
        p_dev_ip = np.concatenate(all_dev_ip) if all_dev_ip else np.array([])
        p_sig_ip = np.concatenate(all_sig_ip) if all_sig_ip else np.array([])
        pooled[margin] = {
            "out_of_plane": {
                "count": len(p_dev_oop),
                "smallest_95_floor_mm": _smallest_covering_floor(p_dev_oop, p_sig_oop),
                "coverage_at_current_floor": _floor_coverage(
                    p_dev_oop, p_sig_oop, 0.86
                ),
            },
            "in_plane": {
                "count": len(p_dev_ip),
                "smallest_95_floor_mm": _smallest_covering_floor(p_dev_ip, p_sig_ip),
                "coverage_at_current_floor": _floor_coverage(p_dev_ip, p_sig_ip, 5.278),
            },
        }

    # Record the floors actually used (may have been patched by --floors)
    from inspection.verdict import (
        D073_SIGMA_FLOOR_BOARD_MM,
        D078_SIGMA_FLOOR_IN_PLANE_MM,
    )

    # Save full report
    report = {
        "commit": _current_commit(),
        "margins_mm": list(margins_mm),
        "parts": results,
        "pooled": pooled,
        "d077_k": D077_K,
        "top_region_percentile": TOP_REGION_PERCENTILE,
        "d073_floor_mm_used": D073_SIGMA_FLOOR_BOARD_MM,
        "d078_floor_mm_used": D078_SIGMA_FLOOR_IN_PLANE_MM,
    }
    report_file = args.out / "g13_report.json"
    report_file.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"\nFull report: {report_file}")
    return 0


def _current_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())

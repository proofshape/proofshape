"""I-16 real-capture record: run check_handedness on saved recon runs.

Loads each part's model mesh and its CAD, runs global_register to get the best pose,
then checks handedness with check_handedness.  Reports the result for each part.

Usage (from repo root):
    PYTHONPATH=inspect .venv/bin/python scripts/run_handedness_record.py

Optionally pass a sessions directory for the phone sessions:
    PYTHONPATH=inspect .venv/bin/python scripts/run_handedness_record.py \
        --sessions proofshape-runs/r44-sessions-2026-11-23
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from inspection import cad
from inspection.pose_judgement import check_handedness
from inspection.registration import global_register

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
RECON_RUNS = FIXTURES / "data" / "recon_runs"
CAD_DIR = FIXTURES / "reference_cad"

# Map run name to CAD file — the ten calipered parts.
RUN_TO_CAD = {
    "g02_01": "g02_01_square_20x20x15mm.stl",
    "g02_02": "g02_02_rect_50x20x25mm.stl",
    "g02_03": "g02_03_right_triangle_40x30x20mm.stl",
    "g02_04": "g02_04_pentagon_r22x25mm.stl",
    "g02_05": "g02_05_hexagon_r28x35mm.stl",
    "g02_06": "g02_06_rect_70x45x40mm.stl",
    "g05_00": "g05_00_control_50x32x20mm.stl",
    "g05_02": "g05_00_control_50x32x20mm.stl",  # registered against the control
    "s-04": "s04_triangular_prism_60-26-65x30mm.stl",
    "s-05": "s05_rect_prism_30x30x60mm.stl",
}

# Phone sessions: map session id to (CAD file, label).
PHONE_SESSIONS = {
    "sess_20201ffd675a": ("s04_triangular_prism_60-26-65x30mm.stl", "s-04 phone#1"),
    "sess_cdf9822969a4": ("s04_triangular_prism_60-26-65x30mm.stl", "s-04 phone#2"),
    "sess_bec6c7dad03e": ("s04_triangular_prism_60-26-65x30mm.stl", "s-04 phone#3"),
}


def load_model(run_dir: Path):
    mesh_path = run_dir / "mesh" / "mesh.ply"
    prov_dir = run_dir / "mesh" / "provenance"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)

    labels = None
    label_path = prov_dir / "vertex_labels.npy"
    if label_path.exists():
        labels = np.load(str(label_path))

    if labels is not None:
        return (vertices, faces, labels)
    return (vertices, faces)


def load_cad(cad_name: str):
    path = CAD_DIR / cad_name
    vertices, faces, units = cad.load_mesh(path)
    return (vertices, faces, units)


def run_one(label: str, run_dir: Path, cad_name: str):
    if not run_dir.exists():
        print(f"  {label:20s}  SKIP  (no run directory)")
        return

    model = load_model(run_dir)
    cad_mesh = load_cad(cad_name)

    model_v = np.asarray(model[0], dtype=np.float64)
    model_f = np.asarray(model[1], dtype=np.int64)
    cad_v = np.asarray(cad_mesh[0], dtype=np.float64)
    cad_f = np.asarray(cad_mesh[1], dtype=np.int64)

    try:
        candidates = global_register(model, cad_mesh)
    except (ValueError, RuntimeError) as e:
        print(f"  {label:20s}  ERROR  global_register: {e}")
        return

    best_pose = candidates[0].transformation
    result = check_handedness(model_v, model_f, cad_v, cad_f, best_pose)

    flag = "FLAG" if result.mismatched else "ok"
    handed = "chiral" if result.has_handedness else "achiral"
    print(
        f"  {label:20s}  {flag:5s}  {handed:7s}  "
        f"h7_model={result.h7_model:+.4e}  h7_cad={result.h7_cad:+.4e}  "
        f"reason={result.reason}"
    )


def main():
    parser = argparse.ArgumentParser(description="I-16 handedness record")
    parser.add_argument(
        "--sessions",
        type=Path,
        default=None,
        help="Directory containing phone session subdirectories",
    )
    args = parser.parse_args()

    print("I-16 handedness record")
    print("=" * 100)
    hdr = f"  {'part':20s}  {'hand':5s}  {'shape':7s}  {'h7_model':>14s}  {'h7_cad':>14s}  reason"
    print(hdr)
    print("-" * 100)

    for name, cad_name in sorted(RUN_TO_CAD.items()):
        label = f"{name} (golden)" if name == "s-04" else name
        run_one(label, RECON_RUNS / name, cad_name)

    if args.sessions:
        print()
        print("Phone sessions:")
        print("-" * 100)
        for sess_id, (cad_name, label) in sorted(PHONE_SESSIONS.items()):
            sess_dir = args.sessions / sess_id
            run_one(f"{label} ({sess_id})", sess_dir, cad_name)

    print()


if __name__ == "__main__":
    main()

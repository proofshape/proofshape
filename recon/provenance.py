"""R-09: label every mesh vertex Observed or Unobserved -- the mechanism behind the hard rule
("unobserved geometry can never pass or fail a part," `AGENTS.md`).

Per `docs/glossary.md`: **Observed** means two or more frames with adequate triangulation
baseline; everything else is **Unobserved**. A frame only counts as "seeing" a vertex if the
vertex also projects within a depth-consistency tolerance of that frame's *measured* depth at the
landed pixel -- the same occlusion check R-07's TSDF fusion uses (`tsdf_fusion.fuse_tsdf`), reused
here via `recon.metric_alignment.project_points_to_camera` rather than re-derived. Without this, a
vertex on the far (occluded) side of the part would be miscounted as "seen" by a frame that's
actually looking at the near side, straight through the object.

Labels travel as vertex colors inside the exported GLB (`contracts/openapi.yaml`: "labels and
vertex colors," not separate JSON fields) -- white for Observed (no color asserted yet; the
*deviation heatmap* that colors Observed vertices by CAD deviation is I-06, much later), grey for
Unobserved, per `submissions/2026-09-05-proposal.md`'s own "drawn gray" convention.

R-09 depends only on R-07, not R-08 -- both are R-07's siblings, not each other's prerequisite --
so `provenance_run` accepts either R-07's raw `mesh.ply` or R-08's cleaned `mesh_clean.glb`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from recon.mesh_cleanup import read_glb, write_glb
from recon.metric_alignment import (
    apply_similarity_transform_to_extrinsic,
    camera_centres,
    project_points_to_camera,
)
from recon.tsdf_fusion import read_mesh_ply

# Neither number has been measured; both are configurable, documented assumptions, same as every
# prior story's unmeasured defaults (R-06's height threshold, R-07's voxel size, R-08's triangle
# count). 2 degrees is a standard SfM rule-of-thumb minimum triangulation baseline.
DEFAULT_MIN_TRIANGULATION_ANGLE_DEG = 2.0
DEFAULT_DEPTH_CONSISTENCY_TOLERANCE_MM = 5.0

OBSERVED_COLOR = (255, 255, 255)
UNOBSERVED_COLOR = (128, 128, 128)

LABEL_OBSERVED = "Observed"
LABEL_UNOBSERVED = "Unobserved"


def compute_observations(
    vertices_mm: np.ndarray,
    extrinsic_board: np.ndarray,
    intrinsic: np.ndarray,
    depth_mm: np.ndarray,
    tolerance_mm: float = DEFAULT_DEPTH_CONSISTENCY_TOLERANCE_MM,
) -> dict[str, np.ndarray]:
    """For every (vertex, frame) pair where that frame actually observes that vertex, record it.

    "Observes" means: in front of the camera, projects inside the frame's image, and its
    camera-space depth is within `tolerance_mm` of that frame's *measured* depth at the landed
    pixel -- the occlusion check. Returns a flat table (one row per observation), not a per-vertex
    structure, since R-10 (per-vertex uncertainty) needs exactly this -- `measured_depth_mm` minus
    `z_cam_mm` is the depth residual its sigma computation needs -- so it's persisted as-is rather
    than summarized away.
    """
    vertices_mm = np.asarray(vertices_mm, dtype=np.float64)
    if vertices_mm.ndim != 2 or vertices_mm.shape[1] != 3:
        raise ValueError(f"vertices_mm must have shape (N,3), got {vertices_mm.shape}")
    extrinsic_board = np.asarray(extrinsic_board, dtype=np.float64)
    depth_mm = np.asarray(depth_mm, dtype=np.float64)
    if depth_mm.ndim != 3:
        raise ValueError(f"depth_mm must have shape (S,H,W), got {depth_mm.shape}")
    frame_count, height, width = depth_mm.shape
    if extrinsic_board.shape != (frame_count, 3, 4):
        raise ValueError(
            f"extrinsic_board must be ({frame_count},3,4), got {extrinsic_board.shape}"
        )
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape != (frame_count, 3, 3):
        raise ValueError(
            f"intrinsic must be ({frame_count},3,3), got {intrinsic.shape}"
        )
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("tolerance_mm must be a positive, finite number.")

    vertex_indices: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    z_cam_values: list[np.ndarray] = []
    measured_depth_values: list[np.ndarray] = []

    all_vertex_indices = np.arange(len(vertices_mm))

    for frame_index in range(frame_count):
        rotation = extrinsic_board[frame_index, :, :3]
        translation = extrinsic_board[frame_index, :, 3]
        u, v, z_cam = project_points_to_camera(
            vertices_mm, rotation, translation, intrinsic[frame_index]
        )

        in_front = z_cam > 1e-9
        u_index = np.round(u).astype(np.int64)
        v_index = np.round(v).astype(np.int64)
        in_bounds = (
            in_front
            & (u_index >= 0)
            & (u_index < width)
            & (v_index >= 0)
            & (v_index < height)
        )
        clamped_u = np.where(in_bounds, u_index, 0)
        clamped_v = np.where(in_bounds, v_index, 0)

        measured_depth = depth_mm[frame_index, clamped_v, clamped_u]
        has_depth = in_bounds & np.isfinite(measured_depth) & (measured_depth > 0.0)
        depth_consistent = has_depth & (np.abs(measured_depth - z_cam) <= tolerance_mm)

        observed_here = all_vertex_indices[depth_consistent]
        if len(observed_here) == 0:
            continue
        vertex_indices.append(observed_here)
        frame_indices.append(np.full(len(observed_here), frame_index, dtype=np.int64))
        z_cam_values.append(z_cam[depth_consistent])
        measured_depth_values.append(measured_depth[depth_consistent])

    if vertex_indices:
        return {
            "vertex_index": np.concatenate(vertex_indices),
            "frame_index": np.concatenate(frame_indices),
            "z_cam_mm": np.concatenate(z_cam_values),
            "measured_depth_mm": np.concatenate(measured_depth_values),
        }
    return {
        "vertex_index": np.empty(0, dtype=np.int64),
        "frame_index": np.empty(0, dtype=np.int64),
        "z_cam_mm": np.empty(0, dtype=np.float64),
        "measured_depth_mm": np.empty(0, dtype=np.float64),
    }


def triangulation_angle_deg(
    vertex_mm: np.ndarray,
    camera_center_a_mm: np.ndarray,
    camera_center_b_mm: np.ndarray,
) -> float:
    """Angle at `vertex_mm` between the rays to two camera centres, in degrees."""
    vertex_mm = np.asarray(vertex_mm, dtype=np.float64)
    ray_a = np.asarray(camera_center_a_mm, dtype=np.float64) - vertex_mm
    ray_b = np.asarray(camera_center_b_mm, dtype=np.float64) - vertex_mm
    norm_a = np.linalg.norm(ray_a)
    norm_b = np.linalg.norm(ray_b)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError(
            "A camera centre coincides with the vertex; triangulation angle is undefined."
        )
    cosine = np.dot(ray_a, ray_b) / (norm_a * norm_b)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def label_vertices(
    vertex_count: int,
    observations: dict[str, np.ndarray],
    camera_centers_mm: np.ndarray,
    vertices_mm: np.ndarray,
    min_triangulation_angle_deg: float = DEFAULT_MIN_TRIANGULATION_ANGLE_DEG,
) -> dict[str, np.ndarray]:
    """Observed iff >=2 observing frames AND their widest pair exceeds the triangulation-angle
    threshold. A vertex seen twice by two nearly-coincident cameras is still Unobserved -- the
    angle check is a real second gate, not a formality once view_count>=2.
    """
    if not np.isfinite(min_triangulation_angle_deg) or min_triangulation_angle_deg <= 0:
        raise ValueError(
            "min_triangulation_angle_deg must be a positive, finite number."
        )

    view_count = np.zeros(vertex_count, dtype=np.int64)
    max_triangulation_angle_deg = np.zeros(vertex_count, dtype=np.float64)
    labels = np.full(vertex_count, LABEL_UNOBSERVED, dtype=object)

    vertex_to_frames: dict[int, list[int]] = {}
    for vertex_index, frame_index in zip(
        observations["vertex_index"].tolist(),
        observations["frame_index"].tolist(),
        strict=True,
    ):
        vertex_to_frames.setdefault(vertex_index, []).append(frame_index)

    for vertex_index, frame_list in vertex_to_frames.items():
        distinct_frames = sorted(set(frame_list))
        view_count[vertex_index] = len(distinct_frames)
        if len(distinct_frames) < 2:
            continue

        best_angle = 0.0
        for i in range(len(distinct_frames)):
            for j in range(i + 1, len(distinct_frames)):
                angle = triangulation_angle_deg(
                    vertices_mm[vertex_index],
                    camera_centers_mm[distinct_frames[i]],
                    camera_centers_mm[distinct_frames[j]],
                )
                best_angle = max(best_angle, angle)
        max_triangulation_angle_deg[vertex_index] = best_angle

        if best_angle >= min_triangulation_angle_deg:
            labels[vertex_index] = LABEL_OBSERVED

    return {
        "label": labels,
        "view_count": view_count,
        "max_triangulation_angle_deg": max_triangulation_angle_deg,
    }


def label_faces(faces: np.ndarray, vertex_labels: np.ndarray) -> np.ndarray:
    """A face is Observed only if all three of its vertices are -- the conservative convention
    this story chose, not a measured or contract-mandated rule."""
    faces = np.asarray(faces, dtype=np.int64)
    is_observed_vertex = vertex_labels == LABEL_OBSERVED
    return np.where(
        is_observed_vertex[faces].all(axis=1), LABEL_OBSERVED, LABEL_UNOBSERVED
    )


def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    """Per-vertex label strings -> (N,3) uint8 RGB, per this module's documented color convention."""
    colors = np.empty((len(labels), 3), dtype=np.uint8)
    colors[labels == LABEL_OBSERVED] = OBSERVED_COLOR
    colors[labels == LABEL_UNOBSERVED] = UNOBSERVED_COLOR
    return colors


def colors_to_labels(colors: np.ndarray) -> np.ndarray:
    """(N,3) uint8 RGB -> per-vertex label strings, the inverse of labels_to_colors.

    Lets R-10 recover R-09's labels from `mesh_provenance.glb` directly -- R-09 only persists
    labels as GLB vertex colors, not separately, so this is how a downstream consumer reads them
    back without needing camera centres or any of R-09's own projection machinery.
    """
    colors = np.asarray(colors)
    is_observed = np.all(colors == np.asarray(OBSERVED_COLOR), axis=1)
    is_unobserved = np.all(colors == np.asarray(UNOBSERVED_COLOR), axis=1)
    unrecognised = ~(is_observed | is_unobserved)
    if np.any(unrecognised):
        raise ValueError(
            f"{int(np.count_nonzero(unrecognised))} color(s) match neither OBSERVED_COLOR "
            f"{OBSERVED_COLOR} nor UNOBSERVED_COLOR {UNOBSERVED_COLOR}"
        )
    labels = np.full(len(colors), LABEL_UNOBSERVED, dtype=object)
    labels[is_observed] = LABEL_OBSERVED
    return labels


def write_observations_npz(path: Path, observations: dict[str, np.ndarray]) -> None:
    """Persist the observations table for R-10 to reuse without recomputing projections."""
    np.savez_compressed(
        path,
        vertex_index=observations["vertex_index"],
        frame_index=observations["frame_index"],
        z_cam_mm=observations["z_cam_mm"],
        measured_depth_mm=observations["measured_depth_mm"],
    )


def read_observations_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {
            key: data[key]
            for key in ("vertex_index", "frame_index", "z_cam_mm", "measured_depth_mm")
        }


def write_provenance_report(
    path: Path,
    mesh_path: Path,
    vertex_labels: np.ndarray,
    face_labels: np.ndarray,
    min_triangulation_angle_deg: float,
    depth_consistency_tolerance_mm: float,
) -> dict[str, Any]:
    observed_vertex_count = int(np.count_nonzero(vertex_labels == LABEL_OBSERVED))
    observed_face_count = int(np.count_nonzero(face_labels == LABEL_OBSERVED))
    total_face_count = len(face_labels)
    report = {
        "story": "R-09",
        "source_mesh": str(mesh_path),
        "min_triangulation_angle_deg": min_triangulation_angle_deg,
        "depth_consistency_tolerance_mm": depth_consistency_tolerance_mm,
        "vertex_count": len(vertex_labels),
        "observed_vertex_count": observed_vertex_count,
        "unobserved_vertex_count": len(vertex_labels) - observed_vertex_count,
        "total_face_count": total_face_count,
        "observed_face_count": observed_face_count,
        "observed_fraction": (observed_face_count / total_face_count)
        if total_face_count
        else 0.0,
    }
    path = Path(path)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _read_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh_path = Path(mesh_path)
    if mesh_path.suffix == ".glb":
        vertices, faces, _colors = read_glb(mesh_path)
        return vertices, faces
    if mesh_path.suffix == ".ply":
        return read_mesh_ply(mesh_path)
    raise ValueError(
        f"Unrecognised mesh format {mesh_path.suffix!r}; expected .ply or .glb"
    )


def provenance_run(
    mesh_path: Path,
    reconstruction_run_dir: Path,
    alignment_json_path: Path,
    out_dir: Path,
    min_triangulation_angle_deg: float = DEFAULT_MIN_TRIANGULATION_ANGLE_DEG,
    depth_consistency_tolerance_mm: float = DEFAULT_DEPTH_CONSISTENCY_TOLERANCE_MM,
) -> dict[str, Any]:
    """Load a mesh (R-07's `.ply` or R-08's `.glb`) plus R-01/R-04 output, label every vertex,
    and write a colored GLB, a JSON report, and the raw observations table."""
    mesh_path = Path(mesh_path)
    reconstruction_run_dir = Path(reconstruction_run_dir)
    alignment_json_path = Path(alignment_json_path)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already has files in it. Pick a new --out-dir or delete it deliberately; "
            "overwriting silently would let outputs from two runs get mixed."
        )

    vertices_mm, faces = _read_mesh(mesh_path)

    with np.load(reconstruction_run_dir / "poses.npz") as poses:
        extrinsic = poses["extrinsic"].astype(np.float64)
        intrinsic = poses["intrinsic"].astype(np.float64)
    with np.load(reconstruction_run_dir / "depth.npz") as depth_data:
        depth = depth_data["depth"].astype(np.float64)

    alignment_json = json.loads(alignment_json_path.read_text(encoding="utf-8"))
    scale = float(alignment_json["scale"])
    rotation = np.asarray(alignment_json["rotation"], dtype=np.float64)
    translation = np.asarray(alignment_json["translation_mm"], dtype=np.float64)

    board_extrinsic = apply_similarity_transform_to_extrinsic(
        extrinsic, scale, rotation, translation
    )
    depth_mm = scale * depth
    camera_centers_mm = camera_centres(board_extrinsic)

    observations = compute_observations(
        vertices_mm,
        board_extrinsic,
        intrinsic,
        depth_mm,
        depth_consistency_tolerance_mm,
    )
    vertex_results = label_vertices(
        len(vertices_mm),
        observations,
        camera_centers_mm,
        vertices_mm,
        min_triangulation_angle_deg,
    )
    vertex_labels = vertex_results["label"]
    face_labels = label_faces(faces, vertex_labels)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_glb(
        out_dir / "mesh_provenance.glb",
        vertices_mm,
        faces,
        labels_to_colors(vertex_labels),
    )
    write_observations_npz(out_dir / "observations.npz", observations)
    return write_provenance_report(
        out_dir / "provenance.json",
        mesh_path,
        vertex_labels,
        face_labels,
        min_triangulation_angle_deg,
        depth_consistency_tolerance_mm,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label every mesh vertex Observed or Unobserved (R-09)."
    )
    parser.add_argument("mesh_path", type=Path)
    parser.add_argument("reconstruction_run_dir", type=Path)
    parser.add_argument(
        "alignment_json",
        type=Path,
        nargs="?",
        default=None,
        help="defaults to <reconstruction_run_dir>/metric_alignment.json (R-04's default output)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--min-triangulation-angle-deg",
        type=float,
        default=DEFAULT_MIN_TRIANGULATION_ANGLE_DEG,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    alignment_json = args.alignment_json or (
        args.reconstruction_run_dir / "metric_alignment.json"
    )
    out_dir = args.out_dir or args.mesh_path.with_name("provenance")
    try:
        report = provenance_run(
            args.mesh_path,
            args.reconstruction_run_dir,
            alignment_json,
            out_dir,
            min_triangulation_angle_deg=args.min_triangulation_angle_deg,
        )
    except FileExistsError as exc:
        print(f"R-09 PROVENANCE: FAIL - {exc}", file=sys.stderr)
        return 1

    print(
        f"R-09 PROVENANCE: PASS - {report['observed_vertex_count']}/{report['vertex_count']} "
        f"vertices Observed ({report['observed_fraction'] * 100:.1f}% of surface)"
    )
    print(f"Outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

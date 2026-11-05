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

R-24: a frame can only observe a vertex through a pixel VGGT's own confidence trusts, decided by
`recon.vggt_runner.confidence_keep_mask` -- the same predicate, at the same threshold (read from
the run's own `run.json` by default, via the same `resolve_min_confidence` helper as `fuse_run`),
that R-01 uses to build `points.ply` and R-07 uses to decide which pixels may be fused (R-20,
D-045). Before this, a pixel already excluded from the cloud and from the fused surface could
still count toward a vertex's two observing frames, so a vertex built from zero trusted pixels at
that spot could be labelled Observed -- the hard rule's own mechanism accepting evidence the rest
of the pipeline had decided not to trust.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from recon.mesh_cleanup import read_glb, write_glb
from recon.metric_alignment import (
    apply_similarity_transform_to_extrinsic,
    camera_centres,
    project_points_to_camera,
)
from recon.tsdf_fusion import read_mesh_ply
from recon.vggt_runner import confidence_keep_mask, resolve_min_confidence

# Neither number has been measured; both are configurable, documented assumptions, same as every
# prior story's unmeasured defaults (R-06's height threshold, R-07's voxel size, R-08's triangle
# count). 2 degrees is a standard SfM rule-of-thumb minimum triangulation baseline.
DEFAULT_MIN_TRIANGULATION_ANGLE_DEG = 2.0
DEFAULT_DEPTH_CONSISTENCY_TOLERANCE_MM = 5.0

OBSERVED_COLOR = (255, 255, 255)
UNOBSERVED_COLOR = (128, 128, 128)

LABEL_OBSERVED = "Observed"
LABEL_UNOBSERVED = "Unobserved"

# A-04: surface a generative model produced rather than a camera measured (proposal §6.6, D-006).
# Defined here, beside the two measured labels, so the next story to serve it (A-05 or whichever
# first shows completion) has one place to read it from. No served file carries it yet: A-04 only
# colours its own completion samples with it, and `colors_to_labels` deliberately still refuses
# it, so a GLB that ever carried it would fail loudly in R-10, next-view and the agent tools
# rather than be read as measured surface.
# This is a *source* colour, the encoding in the file. It is deliberately not capture's *display*
# orange (`capture/src/provenance-colors.ts`, [249, 115, 22]): the two are separate layers there,
# the same as white/grey are source colours that capture remaps for display. Saturated orange is
# far from both white and grey, so no blend of the two measured colours can pass for it.
LABEL_INFERRED = "Inferred"
INFERRED_COLOR = (255, 128, 0)


def vertex_normals(vertices_mm: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals. Degenerate/unreferenced vertices keep a zero normal.

    This is the same cross-product accumulation used by recon.next_view. It is kept local here
    so provenance does not import next_view's module-level Open3D dependency into the service
    path. The sign follows mesh winding; R-25 intentionally uses that sign for the camera-facing
    test rather than abs(dot).
    """
    vertices_mm = np.asarray(vertices_mm, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices_mm.ndim != 2 or vertices_mm.shape[1] != 3:
        raise ValueError(f"vertices_mm must have shape (N,3), got {vertices_mm.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F,3), got {faces.shape}")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices_mm)):
        raise ValueError("faces reference a vertex outside vertices_mm")
    normals = np.zeros_like(vertices_mm)
    if not len(faces):
        return normals
    a = vertices_mm[faces[:, 0]]
    b = vertices_mm[faces[:, 1]]
    c = vertices_mm[faces[:, 2]]
    face_normals = np.cross(b - a, c - a)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0.0
    normals[valid] /= lengths[valid, None]
    return normals


def _orient_faces_with_groups(
    faces: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Orient faces across shared manifold edges and return those winding groups.

    A winding group is connected by shared *edges*, not merely shared vertices. This distinction
    matters for bowtie vertices: two sheets that touch at one corner can have independent initial
    winding and must choose their normal sign independently.
    """
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F,3), got {faces.shape}")
    if len(faces) == 0:
        return faces.copy(), []

    edge_owners: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_index, triangle in enumerate(faces):
        for corner in range(3):
            start = int(triangle[corner])
            end = int(triangle[(corner + 1) % 3])
            edge = (start, end) if start < end else (end, start)
            direction = 1 if (start, end) == edge else -1
            edge_owners.setdefault(edge, []).append((face_index, direction))

    adjacency: list[list[tuple[int, bool]]] = [[] for _ in range(len(faces))]
    for edge, owners in edge_owners.items():
        if len(owners) > 2:
            raise ValueError(
                f"non-manifold edge {edge} belongs to {len(owners)} faces; "
                "cannot establish reliable signed facing"
            )
        if len(owners) != 2:
            continue
        (left_face, left_direction), (right_face, right_direction) = owners
        toggle = left_direction == right_direction
        adjacency[left_face].append((right_face, toggle))
        adjacency[right_face].append((left_face, toggle))

    flip = np.zeros(len(faces), dtype=bool)
    assigned = np.zeros(len(faces), dtype=bool)
    groups: list[np.ndarray] = []
    for root in range(len(faces)):
        if assigned[root]:
            continue
        assigned[root] = True
        stack = [root]
        group: list[int] = []
        while stack:
            current = stack.pop()
            group.append(current)
            for neighbour, toggle in adjacency[current]:
                required = bool(flip[current] ^ toggle)
                if assigned[neighbour]:
                    if bool(flip[neighbour]) != required:
                        raise ValueError(
                            "mesh winding is topologically contradictory; "
                            "cannot establish reliable signed facing"
                        )
                    continue
                flip[neighbour] = required
                assigned[neighbour] = True
                stack.append(neighbour)
        groups.append(np.asarray(group, dtype=np.int64))

    oriented = faces.copy()
    if np.any(flip):
        oriented[flip] = oriented[flip][:, [0, 2, 1]]
    return oriented, groups


def orient_faces_consistently(faces: np.ndarray) -> np.ndarray:
    """Return triangle winding made locally consistent across every shared manifold edge."""
    oriented, _groups = _orient_faces_with_groups(faces)
    return oriented


def _mesh_is_watertight(faces: np.ndarray) -> bool:
    if len(faces) == 0:
        return False
    edge_counts: dict[tuple[int, int], int] = {}
    for triangle in faces:
        for corner in range(3):
            first = int(triangle[corner])
            second = int(triangle[(corner + 1) % 3])
            edge = (first, second) if first < second else (second, first)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return bool(edge_counts) and all(count == 2 for count in edge_counts.values())


def _signed_mesh_volume(
    vertices_mm: np.ndarray, faces: np.ndarray
) -> tuple[float, float]:
    """Signed volume plus an absolute scale used to decide whether its sign is meaningful."""
    centre = np.mean(vertices_mm, axis=0)
    centred = vertices_mm - centre
    a = centred[faces[:, 0]]
    b = centred[faces[:, 1]]
    c = centred[faces[:, 2]]
    contributions = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    return float(np.sum(contributions)), float(np.sum(np.abs(contributions)))


def _point_inside_closed_mesh(
    point: np.ndarray,
    vertices_mm: np.ndarray,
    faces: np.ndarray,
) -> bool:
    """Odd/even ray test for a point against a closed triangle shell.

    The ray direction is deliberately non-axis-aligned, and coincident intersection distances are
    collapsed so hitting the diagonal/shared edge of a triangulated face does not count twice.
    This is used only to detect shell nesting; ambiguous/intersecting cases are handled
    conservatively by withholding trusted normals.
    """
    point = np.asarray(point, dtype=np.float64)
    vertices_mm = np.asarray(vertices_mm, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    direction = np.array([1.0, 0.3713906763541037, 0.17320508075688773])
    direction /= np.linalg.norm(direction)

    v0 = vertices_mm[faces[:, 0]]
    edge1 = vertices_mm[faces[:, 1]] - v0
    edge2 = vertices_mm[faces[:, 2]] - v0
    ray = np.broadcast_to(direction, edge2.shape)
    h = np.cross(ray, edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    epsilon = 1e-10

    usable = np.abs(determinant) > epsilon
    if not np.any(usable):
        return False

    inverse = np.zeros_like(determinant)
    inverse[usable] = 1.0 / determinant[usable]
    s = point - v0
    u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inverse * np.einsum("j,ij->i", direction, q)
    t = inverse * np.einsum("ij,ij->i", edge2, q)

    hit = (
        usable
        & (u >= -epsilon)
        & (v >= -epsilon)
        & (u + v <= 1.0 + epsilon)
        & (t > epsilon)
    )
    distances = np.sort(t[hit])
    if len(distances) == 0:
        return False

    unique_count = 1
    previous = float(distances[0])
    for distance in distances[1:]:
        distance = float(distance)
        tolerance = 1e-8 * max(1.0, abs(previous), abs(distance))
        if abs(distance - previous) > tolerance:
            unique_count += 1
            previous = distance
    return unique_count % 2 == 1


def _build_visibility_scene(
    vertices_mm: np.ndarray, faces: np.ndarray
) -> o3d.t.geometry.RaycastingScene:
    """Build one accelerated triangle scene for repeated provenance visibility queries."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(vertices_mm, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(faces, dtype=np.uint32)),
    )
    return scene


def _mesh_visible_pair_mask(
    vertices_mm: np.ndarray,
    camera_centers_mm: np.ndarray,
    vertex_index: np.ndarray,
    frame_index: np.ndarray,
    *,
    scene: o3d.t.geometry.RaycastingScene,
    self_hit_tolerance_mm: float = 1e-3,
) -> np.ndarray:
    """Return which camera->vertex pairs are not occluded by the mesh.

    One Open3D RaycastingScene accelerates all triangle intersections, and all requested rays are
    cast in one batch. A pair is visible when the first hit is at the target vertex (within a
    small numerical self-hit tolerance) or beyond it. Any earlier mesh hit is an occluder.

    This check is independent of measured depth, so noisy depth cannot make a hidden surface
    geometrically visible.
    """
    vertex_index = np.asarray(vertex_index, dtype=np.int64)
    frame_index = np.asarray(frame_index, dtype=np.int64)
    if vertex_index.shape != frame_index.shape or vertex_index.ndim != 1:
        raise ValueError("vertex/frame indices must be matching one-dimensional arrays")
    if len(vertex_index) == 0:
        return np.empty(0, dtype=bool)
    if (
        np.any(vertex_index < 0)
        or np.any(vertex_index >= len(vertices_mm))
        or np.any(frame_index < 0)
        or np.any(frame_index >= len(camera_centers_mm))
    ):
        raise ValueError("vertex/frame indices are out of range")
    if not np.isfinite(self_hit_tolerance_mm) or self_hit_tolerance_mm < 0.0:
        raise ValueError("self_hit_tolerance_mm must be finite and non-negative")

    targets = vertices_mm[vertex_index]
    origins = camera_centers_mm[frame_index]
    to_targets = targets - origins
    target_distance_mm = np.linalg.norm(to_targets, axis=1)
    if np.any(target_distance_mm <= 1e-12):
        raise ValueError("a camera centre coincides with a mesh vertex")
    directions = to_targets / target_distance_mm[:, None]
    rays = np.hstack([origins, directions]).astype(np.float32)

    first_hit_mm = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return first_hit_mm.astype(np.float64) >= (
        target_distance_mm - self_hit_tolerance_mm
    )


def _filter_observations_by_mesh_visibility(
    observations: dict[str, np.ndarray],
    vertices_mm: np.ndarray,
    camera_centers_mm: np.ndarray,
    *,
    scene: o3d.t.geometry.RaycastingScene,
) -> dict[str, np.ndarray]:
    """Drop final observation rows whose camera->vertex segment is mesh-occluded."""
    visible = _mesh_visible_pair_mask(
        vertices_mm,
        camera_centers_mm,
        observations["vertex_index"],
        observations["frame_index"],
        scene=scene,
    )
    result = dict(observations)
    for key in ("vertex_index", "frame_index", "z_cam_mm", "measured_depth_mm"):
        result[key] = observations[key][visible]
    return result


def _subset_mesh(
    vertices_mm: np.ndarray,
    faces: np.ndarray,
    face_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return compact component vertices/faces plus original vertex indices."""
    component_faces_global = faces[face_indices]
    vertex_indices = np.unique(component_faces_global)
    old_to_new = -np.ones(len(vertices_mm), dtype=np.int64)
    old_to_new[vertex_indices] = np.arange(len(vertex_indices))
    component_faces = old_to_new[component_faces_global]
    return vertices_mm[vertex_indices], component_faces, vertex_indices


def oriented_vertex_normals(
    vertices_mm: np.ndarray,
    faces: np.ndarray,
    camera_centers_mm: np.ndarray,
    *,
    visible_vertex_index: np.ndarray | None = None,
    visible_frame_index: np.ndarray | None = None,
    visibility_scene: o3d.t.geometry.RaycastingScene | None = None,
) -> np.ndarray:
    """Normals with a defensible sign for R-25's camera-facing test.

    Winding is first made locally consistent across shared manifold edges. Each resulting winding
    group then chooses its sign independently:
    - exterior closed group: signed volume chooses outward;
    - any group enclosed inside another closed shell: ambiguous/occluded material side, so no trusted normal;
    - exterior open group: only trusted, in-bounds, depth-consistent pairs that are also
      geometrically visible through the mesh may vote on that group's sign;
    - no usable or ambiguous evidence: that group's faces do not contribute a trusted normal.

    Faces from signed groups are flipped as needed, then vertex normals are computed once from all
    trusted faces. This is important at bowtie vertices shared by otherwise independent sheets:
    both sheets contribute only after their own winding/sign has been settled.
    """
    vertices_mm = np.asarray(vertices_mm, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    camera_centers_mm = np.asarray(camera_centers_mm, dtype=np.float64)
    oriented_faces, winding_groups = _orient_faces_with_groups(faces)

    if camera_centers_mm.ndim != 2 or camera_centers_mm.shape[1] != 3:
        raise ValueError("camera_centers_mm must have shape (S,3)")

    if visible_vertex_index is not None or visible_frame_index is not None:
        if visible_vertex_index is None or visible_frame_index is None:
            raise ValueError("visible vertex/frame indices must be supplied together")
        visible_vertex_index = np.asarray(visible_vertex_index, dtype=np.int64)
        visible_frame_index = np.asarray(visible_frame_index, dtype=np.int64)
        if visible_vertex_index.ndim != 1 or visible_frame_index.ndim != 1:
            raise ValueError("visible vertex/frame indices must be one-dimensional")
        if visible_vertex_index.shape != visible_frame_index.shape:
            raise ValueError("visible vertex/frame indices must have matching shapes")
        if (
            np.any(visible_vertex_index < 0)
            or np.any(visible_vertex_index >= len(vertices_mm))
            or np.any(visible_frame_index < 0)
            or np.any(visible_frame_index >= len(camera_centers_mm))
        ):
            raise ValueError("visible vertex/frame indices are out of range")

    if visible_vertex_index is not None and len(visible_vertex_index):
        if visibility_scene is None:
            visibility_scene = _build_visibility_scene(vertices_mm, oriented_faces)
        geometrically_visible = _mesh_visible_pair_mask(
            vertices_mm,
            camera_centers_mm,
            visible_vertex_index,
            visible_frame_index,
            scene=visibility_scene,
        )
        visible_vertex_index = visible_vertex_index[geometrically_visible]
        visible_frame_index = visible_frame_index[geometrically_visible]

    signed_faces = oriented_faces.copy()
    trusted_face_mask = np.zeros(len(oriented_faces), dtype=bool)

    group_records = []
    for face_indices in winding_groups:
        group_vertices, group_faces, vertex_indices = _subset_mesh(
            vertices_mm, oriented_faces, face_indices
        )
        group_records.append(
            (
                face_indices,
                group_vertices,
                group_faces,
                vertex_indices,
                _mesh_is_watertight(group_faces),
            )
        )

    enclosed_groups: set[int] = set()
    for group_index, record in enumerate(group_records):
        _face_indices, group_vertices, _group_faces, _vertex_indices, _is_closed = (
            record
        )
        if len(group_vertices) == 0:
            continue
        # A disconnected winding group whose geometry starts inside a closed shell is not
        # externally observable evidence, whether that group is itself closed (a cavity shell)
        # or open (a partial cavity patch).  Do not let camera-facing votes guess its material
        # side: the hard rule prefers zero trusted normals / Unobserved.
        probe = group_vertices[0]
        for other_index, other in enumerate(group_records):
            if other_index == group_index or not other[4]:
                continue
            other_vertices, other_faces = other[1], other[2]
            if _point_inside_closed_mesh(probe, other_vertices, other_faces):
                enclosed_groups.add(group_index)
                break

    for group_index, record in enumerate(group_records):
        face_indices, group_vertices, group_faces, vertex_indices, is_closed = record
        group_normals = vertex_normals(group_vertices, group_faces)
        if not np.any(np.linalg.norm(group_normals, axis=1) > 0.0):
            continue

        sign: float | None = None
        if group_index in enclosed_groups:
            # Closed cavity shells and open cavity patches both need topology/material semantics
            # to decide which side is trusted. Do not guess: zero normals keep them out of the
            # facing evidence.
            sign = None
        elif is_closed:
            signed_volume, volume_scale = _signed_mesh_volume(
                group_vertices, group_faces
            )
            if volume_scale > 0.0 and abs(signed_volume) > 1e-12 * volume_scale:
                sign = -1.0 if signed_volume < 0.0 else 1.0
        elif visible_vertex_index is not None and len(visible_vertex_index):
            in_group = np.isin(visible_vertex_index, vertex_indices)
            pair_vertices = visible_vertex_index[in_group]
            pair_frames = visible_frame_index[in_group]
            if len(pair_vertices):
                local_indices = np.searchsorted(vertex_indices, pair_vertices)
                pair_normals = group_normals[local_indices]
                pair_valid = np.linalg.norm(pair_normals, axis=1) > 0.0
                pair_vertices = pair_vertices[pair_valid]
                pair_frames = pair_frames[pair_valid]
                pair_normals = pair_normals[pair_valid]
                if len(pair_vertices):
                    to_camera = (
                        camera_centers_mm[pair_frames] - vertices_mm[pair_vertices]
                    )
                    signed_terms = np.einsum("ij,ij->i", pair_normals, to_camera)
                    score = float(np.sum(signed_terms))
                    score_scale = float(np.sum(np.abs(signed_terms)))
                    if score_scale > 0.0 and abs(score) > 1e-9 * score_scale:
                        sign = -1.0 if score < 0.0 else 1.0

        if sign is None:
            continue
        if sign < 0.0:
            signed_faces[face_indices] = signed_faces[face_indices][:, [0, 2, 1]]
        trusted_face_mask[face_indices] = True

    if not np.any(trusted_face_mask):
        return np.zeros_like(vertices_mm)

    return vertex_normals(vertices_mm, signed_faces[trusted_face_mask])


def observed_surface_area_fraction(
    vertices_mm: np.ndarray, faces: np.ndarray, vertex_labels: np.ndarray
) -> float:
    """Observed surface-area share using the final labels that are encoded in the served GLB."""
    vertices_mm = np.asarray(vertices_mm, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return 0.0
    face_labels = label_faces(faces, np.asarray(vertex_labels, dtype=object))
    a = vertices_mm[faces[:, 0]]
    b = vertices_mm[faces[:, 1]]
    c = vertices_mm[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total_area = float(np.sum(areas))
    if total_area <= 0.0:
        return 0.0
    return float(np.sum(areas[face_labels == LABEL_OBSERVED]) / total_area)


def compute_observations(
    vertices_mm: np.ndarray,
    extrinsic_board: np.ndarray,
    intrinsic: np.ndarray,
    depth_mm: np.ndarray,
    tolerance_mm: float = DEFAULT_DEPTH_CONSISTENCY_TOLERANCE_MM,
    *,
    trusted_pixel_mask: np.ndarray,
    vertex_normals_mm: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """For every (vertex, frame) pair where that frame actually observes that vertex, record it.

    "Observes" means: in front of the camera, projects inside the frame's image, lands on a pixel
    `trusted_pixel_mask` marks True, and its camera-space depth is within `tolerance_mm` of that
    frame's *measured* depth at that pixel -- the occlusion check. Returns a flat table (one row
    per observation), not a per-vertex structure, since R-10 (per-vertex uncertainty) needs exactly
    this -- `measured_depth_mm` minus `z_cam_mm` is the depth residual its sigma computation needs
    -- so it's persisted as-is rather than summarized away.

    `trusted_pixel_mask` is (S,H,W) boolean, `recon.vggt_runner.confidence_keep_mask`'s output
    (R-24). A pixel the rest of the pipeline refused to trust -- dropped from `points.ply`, never
    fused into the mesh -- must not be able to vouch for a vertex here either, or the hard rule's
    own mechanism would accept evidence everything upstream already rejected. It is required, not
    defaulted to "trust everything," so no caller can skip that check by forgetting to pass it; a
    caller that genuinely wants no confidence filtering passes an all-True mask and says so.
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
    trusted_pixel_mask = np.asarray(trusted_pixel_mask)
    # Rejected rather than coerced with astype(bool): passing the raw depth_conf array here by
    # mistake would coerce every nonzero confidence -- including VGGT's untrusted floor of 1.0 --
    # to True, silently trusting exactly the pixels this mask exists to exclude.
    if trusted_pixel_mask.dtype != np.bool_:
        raise ValueError(
            "trusted_pixel_mask must be a boolean array (confidence_keep_mask's output), got "
            f"dtype {trusted_pixel_mask.dtype}"
        )
    if trusted_pixel_mask.shape != depth_mm.shape:
        raise ValueError(
            f"trusted_pixel_mask must match depth_mm's shape {depth_mm.shape}, got "
            f"{trusted_pixel_mask.shape}"
        )

    if vertex_normals_mm is None:
        # Backward-compatible helper behaviour for direct callers. The production provenance_run
        # always supplies mesh-derived normals; R-25's hard-rule path therefore cannot skip facing.
        vertex_normals_mm = np.zeros_like(vertices_mm)
        enforce_facing = False
    else:
        vertex_normals_mm = np.asarray(vertex_normals_mm, dtype=np.float64)
        if vertex_normals_mm.shape != vertices_mm.shape:
            raise ValueError(
                f"vertex_normals_mm must match vertices_mm shape {vertices_mm.shape}, got "
                f"{vertex_normals_mm.shape}"
            )
        enforce_facing = True

    vertex_indices: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    z_cam_values: list[np.ndarray] = []
    measured_depth_values: list[np.ndarray] = []
    depth_gate_rejected_count = np.zeros(len(vertices_mm), dtype=np.int64)

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
        # R-24: looked up at the same clamped pixel as measured_depth, so "trusted" always
        # refers to the exact depth sample being compared, never a neighbour. Out-of-bounds
        # vertices read pixel (0,0) here, but in_bounds (inside has_depth) already excludes them.
        pixel_trusted = trusted_pixel_mask[frame_index, clamped_v, clamped_u]
        if enforce_facing:
            rotation_t = rotation.T
            camera_center_mm = -(rotation_t @ translation)
            to_camera = camera_center_mm[None, :] - vertices_mm
            facing = np.einsum("ij,ij->i", vertex_normals_mm, to_camera) > 0.0
        else:
            facing = np.ones(len(vertices_mm), dtype=bool)
        eligible_before_depth_gate = has_depth & pixel_trusted & facing
        residual_mm = np.abs(measured_depth - z_cam)
        depth_rejected = eligible_before_depth_gate & (residual_mm > tolerance_mm)
        depth_gate_rejected_count += depth_rejected.astype(np.int64)
        depth_consistent = eligible_before_depth_gate & (residual_mm <= tolerance_mm)

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
            "depth_gate_rejected_count": depth_gate_rejected_count,
        }
    return {
        "vertex_index": np.empty(0, dtype=np.int64),
        "frame_index": np.empty(0, dtype=np.int64),
        "z_cam_mm": np.empty(0, dtype=np.float64),
        "measured_depth_mm": np.empty(0, dtype=np.float64),
        "depth_gate_rejected_count": depth_gate_rejected_count,
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
    """Per-vertex label strings -> (N,3) uint8 RGB, per this module's documented color convention.

    A label outside the three raises. Before A-04 the array started from `np.empty` and only the
    two known labels were written, so any other label silently came back with whatever bytes the
    allocator left there -- a quiet wrong colour in a file whose colours *are* the provenance.
    """
    labels = np.asarray(labels)
    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    is_known = np.zeros(len(labels), dtype=bool)
    for label, color in (
        (LABEL_OBSERVED, OBSERVED_COLOR),
        (LABEL_UNOBSERVED, UNOBSERVED_COLOR),
        (LABEL_INFERRED, INFERRED_COLOR),
    ):
        matches = labels == label
        colors[matches] = color
        is_known |= matches
    if not np.all(is_known):
        unknown = sorted({str(label) for label in labels[~is_known]})
        raise ValueError(
            f"{int(np.count_nonzero(~is_known))} label(s) are not {LABEL_OBSERVED!r}, "
            f"{LABEL_UNOBSERVED!r} or {LABEL_INFERRED!r}: {unknown[:5]}"
        )
    return colors


def colors_to_labels(colors: np.ndarray) -> np.ndarray:
    """(N,3) uint8 RGB -> per-vertex label strings, the inverse of labels_to_colors.

    Lets R-10 recover R-09's labels from `mesh_provenance.glb` directly -- R-09 only persists
    labels as GLB vertex colors, not separately, so this is how a downstream consumer reads them
    back without needing camera centres or any of R-09's own projection machinery.

    It decodes the two measured labels only. INFERRED_COLOR is deliberately refused (A-04): no
    served file carries Inferred yet, and every reader of this function (R-10, next-view, the
    agent tools) assumes measured surface. The story that first serves Inferred vertices extends
    this together with capture's colour mapping and the manual.
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
        depth_gate_rejected_count=observations.get(
            "depth_gate_rejected_count", np.zeros(0, dtype=np.int64)
        ),
    )


def read_observations_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        result = {
            key: data[key]
            for key in ("vertex_index", "frame_index", "z_cam_mm", "measured_depth_mm")
        }
        if "depth_gate_rejected_count" in data.files:
            result["depth_gate_rejected_count"] = data["depth_gate_rejected_count"]
        return result


def write_provenance_report(
    path: Path,
    mesh_path: Path,
    vertex_labels: np.ndarray,
    face_labels: np.ndarray,
    min_triangulation_angle_deg: float,
    depth_consistency_tolerance_mm: float,
    *,
    min_confidence: float,
    min_confidence_source: str,
    confidence_dropped_pixel_count: int,
    observation_count: int,
) -> dict[str, Any]:
    observed_vertex_count = int(np.count_nonzero(vertex_labels == LABEL_OBSERVED))
    observed_face_count = int(np.count_nonzero(face_labels == LABEL_OBSERVED))
    total_face_count = len(face_labels)
    report = {
        "story": "R-09",
        "source_mesh": str(mesh_path),
        "min_triangulation_angle_deg": min_triangulation_angle_deg,
        "depth_consistency_tolerance_mm": depth_consistency_tolerance_mm,
        # R-24: which pixels were trusted to vouch for a vertex, and where that threshold came
        # from -- the same three fields, with the same meanings, as R-07's tsdf_fusion.json, so
        # the two reports can be compared side by side to confirm fusion and provenance agreed.
        "min_confidence": min_confidence,
        "min_confidence_source": min_confidence_source,
        "confidence_dropped_pixel_count": confidence_dropped_pixel_count,
        # Rows in observations.npz. Comparing it between a run at threshold 0 and one at the real
        # threshold, on the same mesh, is how many (vertex, frame) observations confidence removed.
        "observation_count": observation_count,
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
    min_confidence: float | None = None,
) -> dict[str, Any]:
    """Load a mesh (R-07's `.ply` or R-08's `.glb`) plus R-01/R-04 output, label every vertex,
    and write a colored GLB, a JSON report, and the raw observations table.

    R-24: `min_confidence=None` (the default) uses the threshold the source run itself was built
    with (`run.json`'s own `min_confidence`), resolved by the same helper R-07's `fuse_run` uses,
    so in the pipeline provenance trusts exactly the pixels fusion did. A `run.json` that is
    missing or predates that field is an explicit error, never a silent 0.0. Pass `min_confidence`
    explicitly to override it -- e.g. 0.0 to reproduce pre-R-24 labelling for comparison.
    """
    mesh_path = Path(mesh_path)
    reconstruction_run_dir = Path(reconstruction_run_dir)
    alignment_json_path = Path(alignment_json_path)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already has files in it. Pick a new --out-dir or delete it deliberately; "
            "overwriting silently would let outputs from two runs get mixed."
        )

    # Resolved before any mesh or depth loading, so an ambiguous run fails fast and cheaply.
    resolved_min_confidence, min_confidence_source = resolve_min_confidence(
        reconstruction_run_dir, min_confidence
    )

    vertices_mm, faces = _read_mesh(mesh_path)

    with np.load(reconstruction_run_dir / "poses.npz") as poses:
        extrinsic = poses["extrinsic"].astype(np.float64)
        intrinsic = poses["intrinsic"].astype(np.float64)
    with np.load(reconstruction_run_dir / "depth.npz") as depth_data:
        depth = depth_data["depth"].astype(np.float64)
        depth_conf = depth_data["depth_conf"].astype(np.float64)

    # Built unconditionally, even at threshold 0.0, rather than skipped as a "no-op": R-20's
    # review (PR #128) found that shortcut in fuse_run let NaN/inf-confidence pixels through,
    # which confidence_keep_mask's own isfinite check excludes at every threshold.
    trusted_pixel_mask = confidence_keep_mask(depth_conf, resolved_min_confidence)
    confidence_dropped_pixel_count = int(np.count_nonzero(~trusted_pixel_mask))

    alignment_json = json.loads(alignment_json_path.read_text(encoding="utf-8"))
    scale = float(alignment_json["scale"])
    rotation = np.asarray(alignment_json["rotation"], dtype=np.float64)
    translation = np.asarray(alignment_json["translation_mm"], dtype=np.float64)

    board_extrinsic = apply_similarity_transform_to_extrinsic(
        extrinsic, scale, rotation, translation
    )
    depth_mm = scale * depth
    camera_centers_mm = camera_centres(board_extrinsic)

    # Closed meshes have a volume-defined outward direction. Open meshes need evidence for the
    # global sign: first project without a facing gate and keep only trusted, depth-consistent
    # pairs. This is deliberately the same depth tolerance as the final observation pass; hidden
    # back faces must not get a vote merely because a camera exists on their side of the object.
    orientation_evidence = None
    oriented_faces = orient_faces_consistently(faces)
    visibility_scene = _build_visibility_scene(vertices_mm, faces)
    if not _mesh_is_watertight(oriented_faces):
        orientation_evidence = compute_observations(
            vertices_mm,
            board_extrinsic,
            intrinsic,
            depth_mm,
            depth_consistency_tolerance_mm,
            trusted_pixel_mask=trusted_pixel_mask,
        )
    normals_mm = oriented_vertex_normals(
        vertices_mm,
        faces,
        camera_centers_mm,
        visible_vertex_index=(
            orientation_evidence["vertex_index"]
            if orientation_evidence is not None
            else None
        ),
        visible_frame_index=(
            orientation_evidence["frame_index"]
            if orientation_evidence is not None
            else None
        ),
        visibility_scene=visibility_scene,
    )
    observations = compute_observations(
        vertices_mm,
        board_extrinsic,
        intrinsic,
        depth_mm,
        depth_consistency_tolerance_mm,
        trusted_pixel_mask=trusted_pixel_mask,
        vertex_normals_mm=normals_mm,
    )
    observations = _filter_observations_by_mesh_visibility(
        observations,
        vertices_mm,
        camera_centers_mm,
        scene=visibility_scene,
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
        min_confidence=resolved_min_confidence,
        min_confidence_source=min_confidence_source,
        confidence_dropped_pixel_count=confidence_dropped_pixel_count,
        observation_count=len(observations["vertex_index"]),
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
    parser.add_argument(
        "--depth-consistency-tolerance-mm",
        type=float,
        default=DEFAULT_DEPTH_CONSISTENCY_TOLERANCE_MM,
        help=(
            "maximum absolute mesh-vs-measured-depth residual that can count as an "
            "observation; default 5 mm is an explicit unmeasured assumption, not a "
            "calibrated accuracy claim"
        ),
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help=(
            "override the confidence threshold read from the run's own run.json "
            "(default: use whatever points.ply was already built with, D-045 / R-24)"
        ),
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
            depth_consistency_tolerance_mm=args.depth_consistency_tolerance_mm,
            min_confidence=args.min_confidence,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        # R-24: a run.json that is missing or predates min_confidence now stops provenance too
        # (FileNotFoundError/ValueError from resolve_min_confidence) -- reported as a named FAIL,
        # same as fuse_run's CLI, rather than an unexplained traceback.
        print(f"R-09 PROVENANCE: FAIL - {exc}", file=sys.stderr)
        return 1

    print(
        f"R-09 PROVENANCE: PASS - {report['observed_vertex_count']}/{report['vertex_count']} "
        f"vertices Observed ({report['observed_fraction'] * 100:.1f}% of surface), "
        f"min_confidence={report['min_confidence']} ({report['min_confidence_source']})"
    )
    print(f"Outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

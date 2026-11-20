"""I-06: signed point-to-mesh deviation from each Observed model vertex to the buyer's CAD,
written into the served GLB as two new vertex attributes (D-062).

D-062 already settled *how* the number travels: `_DEVIATION_MM` (one float32 per vertex, the
signed distance in millimetres, positive where the part has extra material and negative where
material is missing -- proposal 7.2) and `_DEVIATION_MEASURED` (1.0 exactly where a deviation was
computed, 0.0 elsewhere). This module is written against that decision, not a fresh design for
the data format -- only the computation itself (how the distance is found, and which vertices are
compared) is this story's own.

I-05's `check_ambiguity` already decided which CAD pose to measure against (`AmbiguityCheck.best`,
a CAD -> model rigid transform); this module takes that pose as given rather than re-deriving it.

Review on #309: "measures Observed vertices only" isn't just "every Observed vertex" -- D-062's
own wording excludes "surface outside the registered part" too. `compute_deviation` gates on
two things, re-reviewed against real golden data rather than assumed correct from the logic
alone: `board_surface`'s own, already-resolved `height_threshold_mm` (fusion's real cut for that
board), and the posed CAD's own xy footprint plus a declared margin. On real served meshes
(s-04, s-05), the height gate excludes essentially nothing -- fusion's own cut already removed
that surface before I-06 ever sees the mesh, so its real job is defending against degenerate
inputs (a vertex somehow surviving below the cut), not filtering real board skin. The footprint
margin is what actually does that: it excludes stray surface beyond the CAD's hull plus the
margin, but board skin sitting *within* that ring (#292's own measured case, s-02) is still
measured today -- a declared, recorded limitation (see `DEFAULT_FOOTPRINT_MARGIN_MM` and
`stories/I/I-06.md`'s notes for the real-data numbers), not something either gate actually
closes. Whether to close that ring, and how, is #292's or I-07's call, not re-decided here.

Scope, per the story's own notes: this is a library function (the computation) and a GLB writer.
`publish_deviation_glb` does the atomic "which file does glb_path point at" flip against a real
`SessionStore`, reusing its existing atomicity -- but nothing here is called from
`recon/service.py`'s real pipeline yet: no session has a per-order CAD to measure against until
R-34 (order store) exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from inspection.registration import _mesh_arrays, _transform_points
from recon.mesh_cleanup import read_glb, write_glb
from recon.part_segmentation import BoardSurface
from recon.provenance import LABEL_OBSERVED, _mesh_is_watertight, colors_to_labels

# The two D-062 vertex attribute names. Not this story's choice -- glTF's own convention for
# application-specific attributes (the leading underscore), already fixed by D-062 and by
# `recon.mesh_cleanup.write_glb`'s existing validation (every extra attribute name must start
# with "_").
DEVIATION_MM_ATTRIBUTE = "_DEVIATION_MM"
DEVIATION_MEASURED_ATTRIBUTE = "_DEVIATION_MEASURED"

# I-14 / D-086: the saved-session rerun found no positive xy margin that cleanly separates
# reconstruction flare from ordinary wall scatter. Use the CAD's actual outer top-view outline
# itself (0 mm extra allowance), rather than I-06's axis-aligned bounding box + 10 mm. Concave
# outer corners/notches are therefore excluded, while interior holes are deliberately ignored by
# the outline gate so reconstructed hole geometry stays measurable. Geometry outside this outline
# is explicitly unmeasured and cannot contribute to Pass or Fail (D-005/D-062).
DEFAULT_FOOTPRINT_MARGIN_MM = 0.0


def _oriented_edge_in_face(face: np.ndarray, u: int, v: int) -> tuple[int, int]:
    """Return edge (u, v) in the winding direction used by one triangle."""
    a, b, c = (int(value) for value in face)
    for start, end in ((a, b), (b, c), (c, a)):
        if start == u and end == v:
            return u, v
        if start == v and end == u:
            return v, u
    raise ValueError("edge is not present in face")


def _cad_outer_outline_loops(
    cad_vertices: np.ndarray, cad_faces: np.ndarray
) -> list[np.ndarray]:
    """Return the CAD's outer top-view silhouette loops, with interior holes omitted.

    The CAD is already required to be watertight. For an orthographic view along +z, silhouette
    edges separate front/back-facing triangles (or a facing triangle from an edge-on side).
    Choosing each edge's direction from the +z-facing side gives outer loops one winding sign and
    hole loops the opposite sign. The largest loop establishes the outer sign; all loops with that
    sign are kept, so disjoint solids remain supported while through-holes are deliberately filled
    for eligibility purposes.
    """
    triangles = cad_vertices[cad_faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normal_z = normals[:, 2]
    scale = max(float(np.linalg.norm(normals, axis=1).max(initial=0.0)), 1.0)
    eps = scale * 1e-12

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(cad_faces):
        a, b, c = (int(value) for value in face)
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            edge_faces.setdefault(key, []).append(face_index)

    directed_segments: dict[
        tuple[tuple[float, float], tuple[float, float]],
        tuple[tuple[float, float], tuple[float, float]],
    ] = {}
    for (u, v), adjacent in edge_faces.items():
        if len(adjacent) != 2:
            raise ValueError("CAD silhouette needs a two-manifold watertight mesh.")

        signs = [
            1 if normal_z[index] > eps else -1 if normal_z[index] < -eps else 0
            for index in adjacent
        ]
        if signs[0] == signs[1] or (signs[0] == 0 and signs[1] == 0):
            continue

        positive = [index for index in adjacent if normal_z[index] > eps]
        if positive:
            face_index = max(positive, key=lambda index: normal_z[index])
            start, end = _oriented_edge_in_face(cad_faces[face_index], u, v)
        else:
            negative = [index for index in adjacent if normal_z[index] < -eps]
            if not negative:
                continue
            face_index = min(negative, key=lambda index: normal_z[index])
            end, start = _oriented_edge_in_face(cad_faces[face_index], u, v)

        p = tuple(float(value) for value in np.round(cad_vertices[start, :2], 9))
        q = tuple(float(value) for value in np.round(cad_vertices[end, :2], 9))
        if p == q:
            continue
        undirected = (p, q) if p < q else (q, p)
        prior = directed_segments.get(undirected)
        if prior is not None and prior != (p, q):
            raise ValueError("CAD silhouette has inconsistent projected edge winding.")
        directed_segments[undirected] = (p, q)

    if not directed_segments:
        raise ValueError("CAD has no usable top-view outer outline.")

    next_point: dict[tuple[float, float], tuple[float, float]] = {}
    for p, q in directed_segments.values():
        prior = next_point.get(p)
        if prior is not None and prior != q:
            raise ValueError("CAD top-view silhouette is not a set of simple loops.")
        next_point[p] = q

    loops: list[np.ndarray] = []
    visited: set[tuple[float, float]] = set()
    for start in list(next_point):
        if start in visited:
            continue
        loop: list[tuple[float, float]] = []
        point = start
        while point not in visited:
            visited.add(point)
            loop.append(point)
            point = next_point.get(point)
            if point is None:
                raise ValueError("CAD top-view silhouette contains an open chain.")
        if point != start or len(loop) < 3:
            raise ValueError("CAD top-view silhouette does not close into simple loops.")
        loops.append(np.asarray(loop, dtype=np.float64))

    def signed_area(loop: np.ndarray) -> float:
        x = loop[:, 0]
        y = loop[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    areas = np.asarray([signed_area(loop) for loop in loops], dtype=np.float64)
    if not np.any(np.abs(areas) > 1e-12):
        raise ValueError("CAD top-view silhouette has zero projected area.")
    outer_area = areas[int(np.argmax(np.abs(areas)))]
    outer_loops = [
        loop for loop, area in zip(loops, areas, strict=True) if area * outer_area > 0.0
    ]
    if not outer_loops:
        raise ValueError("CAD top-view silhouette has no outer loop.")
    return outer_loops


def _points_in_polygon(points_xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized even-odd point-in-polygon with boundary points included."""
    points_xy = np.asarray(points_xy, dtype=np.float64)
    inside = np.zeros(len(points_xy), dtype=bool)
    boundary = np.zeros(len(points_xy), dtype=bool)
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    tol = 1e-9

    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        edge = b - a
        rel = points_xy - a
        cross = np.abs(rel[:, 0] * edge[1] - rel[:, 1] * edge[0])
        edge_len = max(float(np.linalg.norm(edge)), tol)
        on_line = cross <= tol * edge_len
        within_segment = (
            (points_xy[:, 0] >= min(a[0], b[0]) - tol)
            & (points_xy[:, 0] <= max(a[0], b[0]) + tol)
            & (points_xy[:, 1] >= min(a[1], b[1]) - tol)
            & (points_xy[:, 1] <= max(a[1], b[1]) + tol)
        )
        boundary |= on_line & within_segment

        crosses = (a[1] > y) != (b[1] > y)
        denominator = b[1] - a[1]
        safe_denominator = denominator if abs(denominator) > tol else tol
        x_intersection = a[0] + (y - a[1]) * edge[0] / safe_denominator
        inside ^= crosses & (x < x_intersection)

    return boundary | inside


def _distance_to_outline(points_xy: np.ndarray, loops: list[np.ndarray]) -> np.ndarray:
    distances = np.full(len(points_xy), np.inf, dtype=np.float64)
    for loop in loops:
        for i in range(len(loop)):
            a = loop[i]
            b = loop[(i + 1) % len(loop)]
            edge = b - a
            length_sq = float(np.dot(edge, edge))
            if length_sq == 0.0:
                continue
            t = np.clip(((points_xy - a) @ edge) / length_sq, 0.0, 1.0)
            nearest = a + t[:, None] * edge
            distances = np.minimum(
                distances, np.linalg.norm(points_xy - nearest, axis=1)
            )
    return distances


def _within_cad_outer_outline(
    points_xy: np.ndarray,
    outer_loops: list[np.ndarray],
    margin_mm: float,
) -> np.ndarray:
    inside = np.zeros(len(points_xy), dtype=bool)
    for loop in outer_loops:
        inside |= _points_in_polygon(points_xy, loop)
    if margin_mm > 0.0:
        inside |= _distance_to_outline(points_xy, outer_loops) <= margin_mm
    return inside


def compute_deviation(
    model: Any,
    cad: Any,
    cad_to_model_transform: np.ndarray,
    board_surface: BoardSurface,
    *,
    footprint_margin_mm: float = DEFAULT_FOOTPRINT_MARGIN_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """Signed point-to-mesh distance from each Observed model vertex to the CAD (D-062, 7.2).

    Returns (deviation_mm, deviation_measured), one value per model vertex, in the model's own
    vertex order; the model's vertices are never moved. `deviation_measured` is 1.0 exactly where
    a deviation was computed, and 0.0 everywhere else; `deviation_mm` is 0.0 wherever
    `deviation_measured` is 0.0. A vertex is measured only if all three hold (D-062: "I-06
    measures Observed vertices only" -- the latter two are "that excludes any it leaves out, such
    as surface outside the registered part"):

    - labelled Observed;
    - above `board_surface`'s own `height_threshold_mm` (fusion's real, already-resolved
      board-vs-part cut for this board, D-063's whole formula -- not a second, independently
      guessed threshold). Review on #309, second round: on real served meshes this excludes
      essentially nothing, since fusion's own cut already removed that surface before this
      function ever sees the mesh -- it defends against a degenerate input, not real skin;
    - within the registered CAD's actual outer xy outline, evaluated in CAD-local coordinates.
      Concave outer features such as G-05's notch are part of that outline; interior holes are
      intentionally filled for this eligibility test so a reconstructed hole/shift remains
      measurable. I-14/D-086 sets the default extra allowance to 0 mm because the real-session
      rerun found no positive margin that separates flare from normal wall scatter. Observed
      geometry outside that outer outline is served with `_DEVIATION_MEASURED = 0`: it is not
      Pass evidence and cannot Fail the part (D-005).

    Sign convention: positive where the part has extra material, negative where material is
    missing. This is Open3D's own signed-distance convention for a closed mesh (negative inside,
    outside positive) applied directly, once the CAD is posed into the model frame by
    `cad_to_model_transform` -- a model vertex sitting outside the CAD solid is exactly "the part
    has extra material there," and one sitting inside the CAD solid is exactly "material is
    missing," so no sign flip is needed. Open3D decides inside/outside by ray parity, which is
    only meaningful for a closed mesh, so `cad` must be watertight -- raised explicitly rather
    than silently misreading an open CAD's sign. `model` and `cad` accept anything
    `inspection.registration._mesh_arrays` does (I-01's `SyntheticSample`, `inspection.cad.
    load_mesh`'s `(vertices_mm, faces, units)` tuples, or raw `(vertices_mm, faces, labels)`).

    Raises if `model` carries no provenance labels: without them there is no way to find the
    Observed vertices this function's whole job is to measure.
    """
    model_vertices, _model_faces, model_labels = _mesh_arrays(model, name="model")
    cad_vertices, cad_faces, _cad_labels = _mesh_arrays(cad, name="cad")
    if model_labels is None:
        raise ValueError(
            "compute_deviation needs model provenance: without Observed/Unobserved labels there "
            "is no way to find the vertices this function measures."
        )
    if not _mesh_is_watertight(cad_faces):
        raise ValueError(
            "compute_deviation needs a watertight CAD: Open3D's signed-distance inside/outside "
            "decision (by ray parity) is only meaningful for a closed mesh, and a silently wrong "
            "sign on an open CAD is worse than refusing to guess."
        )

    cad_to_model_transform = np.asarray(cad_to_model_transform, dtype=np.float64)
    if cad_to_model_transform.shape != (4, 4):
        raise ValueError(
            f"cad_to_model_transform must be a 4x4 matrix, got {cad_to_model_transform.shape}"
        )
    if not np.isfinite(footprint_margin_mm) or footprint_margin_mm < 0.0:
        raise ValueError("footprint_margin_mm must be finite and non-negative.")

    posed_cad_vertices = _transform_points(cad_vertices, cad_to_model_transform)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(posed_cad_vertices.astype(np.float32)),
        o3d.core.Tensor(cad_faces.astype(np.uint32)),
    )

    # Evaluate the registered footprint in CAD-local coordinates against the actual outer
    # silhouette, not an axis-aligned box. Concave outer notches are excluded; interior holes are
    # deliberately ignored by _cad_outer_outline_loops so hole defects remain measurable.
    model_to_cad_transform = np.linalg.inv(cad_to_model_transform)
    model_vertices_cad = _transform_points(model_vertices, model_to_cad_transform)
    outer_loops = _cad_outer_outline_loops(cad_vertices, cad_faces)
    within_footprint = _within_cad_outer_outline(
        model_vertices_cad[:, :2], outer_loops, footprint_margin_mm
    )
    above_board = (
        board_surface.height_above_board_mm(model_vertices)
        > board_surface.height_threshold_mm
    )

    is_observed = model_labels == LABEL_OBSERVED
    eligible = is_observed & within_footprint & above_board
    deviation_mm = np.zeros(len(model_vertices), dtype=np.float32)
    deviation_measured = np.zeros(len(model_vertices), dtype=np.float32)
    if np.any(eligible):
        query_points = o3d.core.Tensor(model_vertices[eligible].astype(np.float32))
        signed_distance = scene.compute_signed_distance(query_points).numpy()
        deviation_mm[eligible] = signed_distance
        deviation_measured[eligible] = 1.0

    if not np.isfinite(deviation_mm).all():
        raise ValueError(
            "compute_deviation produced a non-finite distance; glTF forbids NaN in float "
            "accessors, so this must be caught here rather than written into the served GLB."
        )
    return deviation_mm, deviation_measured


def write_deviation_glb(
    existing_glb_path: Path,
    cad: Any,
    cad_to_model_transform: np.ndarray,
    board_surface: BoardSurface,
    output_path: Path,
) -> None:
    """Read a served GLB, compute its deviation, and write a new GLB beside it (D-062).

    Reads COLOR_0 from `existing_glb_path` (the served mesh's own provenance, R-09) to find which
    vertices are Observed -- `recon.provenance.colors_to_labels` is the established reader for
    exactly this. COLOR_0 is carried through unchanged into the new file; only the two new
    `_DEVIATION_MM`/`_DEVIATION_MEASURED` attributes are added. The new GLB is written to
    `output_path`, beside the original, rather than overwriting it in place -- flipping whichever
    record points at the served file is the caller's job (see `publish_deviation_glb` below for
    the one that does it against a real `SessionStore`); this function does not touch any session
    record itself.
    """
    vertices, faces, colors = read_glb(Path(existing_glb_path))
    if colors is None:
        raise ValueError(
            f"{existing_glb_path} has no COLOR_0 (provenance) attribute; deviation can only be "
            "computed against a mesh whose Observed vertices are known."
        )
    labels = colors_to_labels(colors)

    deviation_mm, deviation_measured = compute_deviation(
        (vertices, faces, labels), cad, cad_to_model_transform, board_surface
    )

    write_glb(
        Path(output_path),
        vertices,
        faces,
        colors=colors,
        extra_vertex_attributes={
            DEVIATION_MM_ATTRIBUTE: deviation_mm,
            DEVIATION_MEASURED_ATTRIBUTE: deviation_measured,
        },
    )


def publish_deviation_glb(
    store: Any,
    session_id: str,
    cad: Any,
    cad_to_model_transform: np.ndarray,
    board_surface: BoardSurface,
) -> Path:
    """Compute a session's deviation and atomically point its record at the result (D-062's
    third criterion: "flips the session record to it in one atomic write").

    `store` is a `recon.service.SessionStore` (imported lazily so this module doesn't pull in
    `recon.service`'s FastAPI/pydantic dependencies for callers who only want the computation).
    Writes the new GLB beside the session's existing one, then updates
    `record["reconstruction"]["glb_path"]` and calls `store.save(record)` -- reusing
    `SessionStore`'s own existing atomicity (a temp `session.json`, then `.replace()`) rather
    than inventing a second mechanism. Not called by the real pipeline yet: no session has a
    per-order CAD to measure against until R-34 (order store) exists; this exists so whichever
    story wires that up only has to call it.
    """
    record = store.load(session_id)
    if record is None:
        raise KeyError(f"no session {session_id!r}")
    reconstruction = record.get("reconstruction", {})
    if reconstruction.get("status") != "complete":
        raise ValueError(
            f"session {session_id!r} has no completed reconstruction to measure a deviation "
            "against."
        )

    existing_glb_path = Path(reconstruction["glb_path"])
    output_path = existing_glb_path.with_name(
        existing_glb_path.stem + "_deviation" + existing_glb_path.suffix
    )
    write_deviation_glb(
        existing_glb_path, cad, cad_to_model_transform, board_surface, output_path
    )

    record["reconstruction"]["glb_path"] = str(output_path)
    store.save(record)
    return output_path

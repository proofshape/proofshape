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

# Review on #309, second round: G-05's bumps grow upward from the top face (G-05.md: "8mm tall
# (top 28mm up)"), not outward in xy, so they can't justify this number -- the first version of
# this comment's reasoning didn't hold. This bounds the posed CAD's own xy footprint before a
# vertex counts as "within the registered part." Declared, not derived from a specific defect
# or measured source: on real golden data (s-04, s-05) the real part's own base flare reaches
# about 10 mm beyond the CAD's hull, so this isn't obviously too tight for legitimate geometry,
# but it also isn't validated to be the right number for anything in particular -- a plain
# declared parameter, same as R-06's height threshold or R-07's voxel size before they were
# measured.
DEFAULT_FOOTPRINT_MARGIN_MM = 10.0


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
    - within the posed CAD's own xy footprint, expanded by `footprint_margin_mm` on every side.
      This is what actually excludes real stray/board surface, but only beyond the margin --
      board skin within `footprint_margin_mm` of the CAD's hull is still measured today (#292's
      own measured case). Not closed here; #292/I-07 decide whether and how to close that ring.

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

    posed_cad_vertices = _transform_points(cad_vertices, cad_to_model_transform)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(posed_cad_vertices.astype(np.float32)),
        o3d.core.Tensor(cad_faces.astype(np.uint32)),
    )

    footprint_min_xy = posed_cad_vertices[:, :2].min(axis=0) - footprint_margin_mm
    footprint_max_xy = posed_cad_vertices[:, :2].max(axis=0) + footprint_margin_mm
    model_xy = model_vertices[:, :2]
    within_footprint = np.all(
        (model_xy >= footprint_min_xy) & (model_xy <= footprint_max_xy), axis=1
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

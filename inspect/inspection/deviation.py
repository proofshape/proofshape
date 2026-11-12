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

Scope, per the story's own notes: this is a library function (the computation) and a GLB writer,
not a wired-up pipeline stage. No session has a per-order CAD to measure against until R-34
(order store) exists, so nothing here is called from `recon/service.py` yet. Once it is, the
caller is responsible for the atomic "which file does glb_path point at" flip -- that's
`SessionStore.save`'s existing job (already atomic: write a temp session.json, then `.replace()`
it), the same pattern this module's own GLB write follows for the GLB file itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from inspection.registration import _mesh_arrays, _transform_points
from recon.mesh_cleanup import read_glb, write_glb
from recon.provenance import LABEL_OBSERVED, colors_to_labels

# The two D-062 vertex attribute names. Not this story's choice -- glTF's own convention for
# application-specific attributes (the leading underscore), already fixed by D-062 and by
# `recon.mesh_cleanup.write_glb`'s existing validation (every extra attribute name must start
# with "_").
DEVIATION_MM_ATTRIBUTE = "_DEVIATION_MM"
DEVIATION_MEASURED_ATTRIBUTE = "_DEVIATION_MEASURED"


def compute_deviation(
    model: Any,
    cad: Any,
    cad_to_model_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Signed point-to-mesh distance from each Observed model vertex to the CAD (D-062, 7.2).

    Returns (deviation_mm, deviation_measured), one value per model vertex, in the model's own
    vertex order; the model's vertices are never moved. `deviation_measured` is 1.0 exactly where
    a deviation was computed -- a model vertex labelled Observed -- and 0.0 everywhere else;
    `deviation_mm` is 0.0 wherever `deviation_measured` is 0.0. A vertex this function doesn't
    compare (not Observed) is left unmeasured, matching D-062's "I-06 measures Observed vertices
    only."

    Sign convention: positive where the part has extra material, negative where material is
    missing. This is Open3D's own signed-distance convention for a closed mesh (negative inside,
    outside positive) applied directly, once the CAD is posed into the model frame by
    `cad_to_model_transform` -- a model vertex sitting outside the CAD solid is exactly "the part
    has extra material there," and one sitting inside the CAD solid is exactly "material is
    missing," so no sign flip is needed. `model` and `cad` accept anything
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

    is_observed = model_labels == LABEL_OBSERVED
    deviation_mm = np.zeros(len(model_vertices), dtype=np.float32)
    deviation_measured = np.zeros(len(model_vertices), dtype=np.float32)
    if np.any(is_observed):
        query_points = o3d.core.Tensor(model_vertices[is_observed].astype(np.float32))
        signed_distance = scene.compute_signed_distance(query_points).numpy()
        deviation_mm[is_observed] = signed_distance
        deviation_measured[is_observed] = 1.0

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
    output_path: Path,
) -> None:
    """Read a served GLB, compute its deviation, and write a new GLB beside it (D-062).

    Reads COLOR_0 from `existing_glb_path` (the served mesh's own provenance, R-09) to find which
    vertices are Observed -- `recon.provenance.colors_to_labels` is the established reader for
    exactly this. COLOR_0 is carried through unchanged into the new file; only the two new
    `_DEVIATION_MM`/`_DEVIATION_MEASURED` attributes are added. The new GLB is written to
    `output_path`, beside the original, rather than overwriting it in place -- flipping whichever
    record points at the served file (`recon/service.py`'s `SessionStore`) to the new path is the
    caller's job, atomically, the same way `SessionStore.save` already writes `session.json`; this
    function does not touch any session record.
    """
    vertices, faces, colors = read_glb(Path(existing_glb_path))
    if colors is None:
        raise ValueError(
            f"{existing_glb_path} has no COLOR_0 (provenance) attribute; deviation can only be "
            "computed against a mesh whose Observed vertices are known."
        )
    labels = colors_to_labels(colors)

    deviation_mm, deviation_measured = compute_deviation(
        (vertices, faces, labels), cad, cad_to_model_transform
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

"""C-08: next-view scoring -- given the mesh so far, which camera position would see the most
surface that hasn't been properly observed yet?

Proposal §6.2 defines the method: candidate camera positions on a hemisphere over the part (radius
25-40 cm, at most 60 degrees from vertical), each looking at the part, each scored by how many
under-observed vertices it would see unoccluded, using Open3D raycasting on the CPU. D-022 puts this
in the capture lane; it lives in `recon/` anyway because Open3D can't run in the PWA, and nothing
here needs a GPU.

"Under-observed" is not redefined here. It is exactly R-09's `Unobserved` label after R-10's sigma
downgrade (`recon/provenance.py`, `recon/uncertainty.py`'s `downgrade_high_sigma_vertices`): the
labels this module is handed are already downgraded, and the only test applied to them is
`labels == LABEL_UNOBSERVED`, using R-09's own constant.

Board frame (`recon/board_pose.py`): millimetres, z = 0 on the paper and +z pointing *into* the
table, so "up" is -z and every camera above the board has a negative z relative to the part's base.

Four choices made while planning this story (2026-10-15, recorded in `stories/C/C-08.md`):

- **Raycaster: Open3D's `RaycastingScene`**, now a pinned `requirements.txt` dependency (D-047) so
  these tests run in CI rather than skipping. Imported at module level, like R-08's real
  dependencies, not lazily like R-07's optional Poisson fallback.
- **The hemisphere is centred on the part's footprint, not the literal board origin.** The story
  says "centred on the board origin", but `board_pose.py` puts that origin at the board's *outer
  corner*, about 100 mm from the board's centre. Centred there, a 25 cm candidate would be anywhere
  from roughly 15 to 35 cm from a part sitting mid-board. The default centre is the mesh's x/y
  bounding-box centre at the part's base (its *largest* z); both it and the look-at point can be
  overridden.
- **The score is the story's count, with a head-on tie-break.** A flat face is fully visible from
  anywhere in front of its plane, so on the story's own box test 8+ candidates tie at "every
  under-observed vertex visible". Ties are broken by the larger sum of |cos(incidence)| over the
  visible under-observed vertices (prefer the view that looks at the surface square-on, where depth
  is measured best), then by candidate index so the order is always deterministic. The absolute
  value means the mesh's winding order -- which nothing upstream guarantees -- can't flip it.
- **A thin CLI** over R-10's output directory, like the other `recon/` modules.

The board itself is never added to the raycasting scene, on purpose: every candidate and every part
vertex is on the camera side of the board plane, a half-space is convex, so the straight line
between them never crosses the board. It cannot occlude anything this module asks about.

Known, deliberate omissions (neither threshold has been measured, so neither is invented here):
the camera's field of view is not modelled -- a candidate is scored as if it sees every unoccluded
vertex -- and a vertex seen at a grazing angle still counts toward the score (the head-on
tie-break only orders views that already tie).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from recon.mesh_cleanup import read_glb
from recon.provenance import LABEL_OBSERVED, LABEL_UNOBSERVED, colors_to_labels

# Proposal §6.2's design parameters, not measurements: "candidate views on a hemisphere of 25-40 cm
# radius, up to 60 degrees from vertical". Board-frame millimetres, like every other recon module.
DEFAULT_MIN_RADIUS_MM = 250.0
DEFAULT_MAX_RADIUS_MM = 400.0
DEFAULT_MAX_ANGLE_FROM_VERTICAL_DEG = 60.0

# Sampling density -- unmeasured defaults, chosen so the candidate set is small enough to raycast
# in well under a second on a laptop CPU yet dense enough that the best view is never more than
# half a step from the true optimum. With these values: 3 radii x (1 pole + 4 rings x 12
# azimuths) = 147 candidates. Radius matters least (visibility barely changes between 25 and
# 40 cm), so it gets the fewest samples.
DEFAULT_RADIUS_STEPS = 3
DEFAULT_POLAR_STEP_DEG = 15.0
DEFAULT_AZIMUTH_STEP_DEG = 30.0

# A ray aimed at a vertex legitimately hits one of the triangles that *touch* that vertex, at a
# distance equal to the vertex's own distance give or take float32 rounding. Anything hit more than
# this much *before* the vertex is a genuine occluder. A quarter of R-07's 2 mm TSDF voxel: far
# above float32 rounding at 40 cm (~0.0001 mm), far below any real occluding feature on a mesh that
# coarse. Unmeasured, configurable, like every other recon default.
DEFAULT_SELF_HIT_TOLERANCE_MM = 0.5


def part_hemisphere_centre_and_target(
    vertices_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The default hemisphere centre and look-at point for a part mesh, both in board-frame mm.

    Centre: the x/y centre of the mesh's bounding box, at the part's base -- its *largest* z, since
    board +z points into the table. That keeps the hemisphere symmetric about the part wherever on
    the board it was placed, and its rim level with where the part meets the board (R-21: the real
    board surface doesn't land exactly at z = 0, so the mesh's own base is used, not z = 0).

    Look-at: the bounding-box centre in all three axes, so every candidate points at the middle of
    the part, not at its footprint.
    """
    vertices_mm = np.asarray(vertices_mm, dtype=np.float64)
    if vertices_mm.ndim != 2 or vertices_mm.shape[1] != 3 or len(vertices_mm) == 0:
        raise ValueError(
            f"vertices_mm must have shape (N,3) with N >= 1, got {vertices_mm.shape}"
        )
    if not np.all(np.isfinite(vertices_mm)):
        raise ValueError("vertices_mm contains non-finite coordinates.")

    bounding_min = vertices_mm.min(axis=0)
    bounding_max = vertices_mm.max(axis=0)
    look_at_mm = (bounding_min + bounding_max) / 2.0
    hemisphere_centre_mm = np.array([look_at_mm[0], look_at_mm[1], bounding_max[2]])
    return hemisphere_centre_mm, look_at_mm


def _polar_angles_deg(
    max_angle_from_vertical_deg: float, polar_step_deg: float
) -> list[float]:
    """0, step, 2*step, ... and always the limit itself, so the outermost ring sits exactly at the
    proposal's 60 degrees even when the step doesn't divide it evenly. A ring closer than a
    thousandth of a step to the limit is merged into it rather than emitted twice."""
    angles: list[float] = []
    angle = 0.0
    while angle < max_angle_from_vertical_deg - 1e-3 * polar_step_deg:
        angles.append(angle)
        angle += polar_step_deg
    angles.append(max_angle_from_vertical_deg)
    return angles


def _azimuths_deg(azimuth_step_deg: float) -> list[float]:
    """0, step, 2*step, ... strictly below 360, so 0 and 360 are never both emitted."""
    azimuths: list[float] = []
    azimuth = 0.0
    while azimuth < 360.0 - 1e-3 * azimuth_step_deg:
        azimuths.append(azimuth)
        azimuth += azimuth_step_deg
    return azimuths


def generate_candidate_views(
    hemisphere_centre_mm: np.ndarray,
    look_at_mm: np.ndarray,
    *,
    min_radius_mm: float = DEFAULT_MIN_RADIUS_MM,
    max_radius_mm: float = DEFAULT_MAX_RADIUS_MM,
    radius_steps: int = DEFAULT_RADIUS_STEPS,
    max_angle_from_vertical_deg: float = DEFAULT_MAX_ANGLE_FROM_VERTICAL_DEG,
    polar_step_deg: float = DEFAULT_POLAR_STEP_DEG,
    azimuth_step_deg: float = DEFAULT_AZIMUTH_STEP_DEG,
) -> dict[str, np.ndarray]:
    """Candidate camera positions on a hemisphere above `hemisphere_centre_mm`, each looking at
    `look_at_mm`. Returns per-candidate arrays (K rows each):

    - `position_mm` (K,3) and `look_at_mm` (K,3), board-frame mm.
    - `radius_mm`, `angle_from_vertical_deg` (0 = straight down onto the part), `azimuth_deg`
      (board frame, measured from +x toward +y).
    - `elevation_deg` = 90 - angle_from_vertical_deg: C-07's convention (0 = level with the part,
      90 = straight down), so whoever later wires this scorer into C-07's arrow doesn't have to
      convert. C-07's azimuth is relative to the first frame's compass heading, not the board's +x
      axis; mapping between the two is that wiring step's job, not this one's.

    Straight down is emitted once per radius, not once per azimuth: azimuth means nothing at the
    pole (C-07 exempts its straight-down slot from azimuth for the same reason), and 12 identical
    copies would just be 12 ties.
    """
    hemisphere_centre_mm = np.asarray(hemisphere_centre_mm, dtype=np.float64)
    look_at_mm = np.asarray(look_at_mm, dtype=np.float64)
    if hemisphere_centre_mm.shape != (3,) or not np.all(
        np.isfinite(hemisphere_centre_mm)
    ):
        raise ValueError(
            f"hemisphere_centre_mm must be 3 finite numbers, got {hemisphere_centre_mm}"
        )
    if look_at_mm.shape != (3,) or not np.all(np.isfinite(look_at_mm)):
        raise ValueError(f"look_at_mm must be 3 finite numbers, got {look_at_mm}")
    if not math.isfinite(min_radius_mm) or min_radius_mm <= 0:
        raise ValueError("min_radius_mm must be a positive, finite number.")
    if not math.isfinite(max_radius_mm) or max_radius_mm < min_radius_mm:
        raise ValueError("max_radius_mm must be finite and at least min_radius_mm.")
    if isinstance(radius_steps, bool) or not isinstance(radius_steps, int):
        raise TypeError(f"radius_steps must be an integer, got {radius_steps!r}")
    if radius_steps < 1:
        raise ValueError("radius_steps must be at least 1.")
    if radius_steps == 1 and max_radius_mm != min_radius_mm:
        # One sample can't cover a range; silently using only one end of it would quietly drop
        # half of what the caller asked for.
        raise ValueError(
            "radius_steps=1 only makes sense when min_radius_mm == max_radius_mm; "
            "use radius_steps >= 2 to sample a range."
        )
    if radius_steps > 1 and max_radius_mm == min_radius_mm:
        # The mirror of the check above (PR #144 review): several samples of a single radius are
        # exact duplicate candidates, which would only ever tie with each other and pad the ranking.
        raise ValueError(
            "radius_steps > 1 needs min_radius_mm < max_radius_mm; with equal radii every extra "
            "step would duplicate the same candidates. Use radius_steps=1 for a single radius."
        )
    # Strictly below 90: at 90 degrees the camera would sit level with the board, where the board
    # really could occlude the part -- and this module deliberately leaves the board out of the
    # scene on the grounds that it can't (see the module docstring).
    if (
        not math.isfinite(max_angle_from_vertical_deg)
        or max_angle_from_vertical_deg < 0
        or max_angle_from_vertical_deg >= 90
    ):
        raise ValueError("max_angle_from_vertical_deg must be in [0, 90).")
    if not math.isfinite(polar_step_deg) or polar_step_deg <= 0:
        raise ValueError("polar_step_deg must be a positive, finite number.")
    if (
        not math.isfinite(azimuth_step_deg)
        or azimuth_step_deg <= 0
        or azimuth_step_deg > 360
    ):
        raise ValueError("azimuth_step_deg must be in (0, 360].")

    radii_mm = np.linspace(min_radius_mm, max_radius_mm, radius_steps).tolist()
    polar_angles_deg = _polar_angles_deg(max_angle_from_vertical_deg, polar_step_deg)
    azimuths_deg = _azimuths_deg(azimuth_step_deg)

    positions: list[np.ndarray] = []
    radius_column: list[float] = []
    angle_column: list[float] = []
    azimuth_column: list[float] = []

    for radius_mm in radii_mm:
        for angle_deg in polar_angles_deg:
            if angle_deg == 0.0:
                ring_azimuths_deg = [0.0]
            else:
                ring_azimuths_deg = azimuths_deg
            theta = math.radians(angle_deg)
            for azimuth_deg in ring_azimuths_deg:
                phi = math.radians(azimuth_deg)
                # Up is -z in the board frame, hence the minus on the vertical component.
                offset = radius_mm * np.array(
                    [
                        math.sin(theta) * math.cos(phi),
                        math.sin(theta) * math.sin(phi),
                        -math.cos(theta),
                    ]
                )
                positions.append(hemisphere_centre_mm + offset)
                radius_column.append(radius_mm)
                angle_column.append(angle_deg)
                azimuth_column.append(azimuth_deg)

    candidate_count = len(positions)
    angle_from_vertical_deg = np.asarray(angle_column, dtype=np.float64)
    return {
        "position_mm": np.asarray(positions, dtype=np.float64).reshape(
            candidate_count, 3
        ),
        "look_at_mm": np.tile(look_at_mm, (candidate_count, 1)),
        "radius_mm": np.asarray(radius_column, dtype=np.float64),
        "angle_from_vertical_deg": angle_from_vertical_deg,
        "elevation_deg": 90.0 - angle_from_vertical_deg,
        "azimuth_deg": np.asarray(azimuth_column, dtype=np.float64),
    }


def _validate_mesh(
    vertices_mm: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    vertices_mm = np.asarray(vertices_mm, dtype=np.float64)
    if vertices_mm.ndim != 2 or vertices_mm.shape[1] != 3 or len(vertices_mm) == 0:
        raise ValueError(
            f"vertices_mm must have shape (N,3) with N >= 1, got {vertices_mm.shape}"
        )
    if not np.all(np.isfinite(vertices_mm)):
        raise ValueError("vertices_mm contains non-finite coordinates.")
    faces = np.asarray(faces)
    # A mesh with no faces has nothing to occlude with and no surface to orient a normal by; it
    # isn't a mesh this module can say anything honest about.
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"faces must have shape (F,3) with F >= 1, got {faces.shape}")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError(f"faces must hold integer vertex indices, got {faces.dtype}")
    faces = faces.astype(np.int64)
    if faces.min() < 0 or faces.max() >= len(vertices_mm):
        raise ValueError(
            f"faces reference vertex indices outside [0, {len(vertices_mm) - 1}]"
        )
    return vertices_mm, faces


def vertex_normals(vertices_mm: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals, (N,3). Unreferenced vertices get a zero normal.

    Each face contributes its un-normalised cross product, whose length is twice its area, so a
    big face pulls a shared vertex's normal harder than a sliver does. The *sign* of the result
    follows the mesh's winding order, which nothing upstream guarantees; callers here only ever
    use |normal . direction|, so it doesn't matter. Faces with inconsistent winding can partly
    cancel at a shared vertex -- that only weakens the head-on tie-break there, never the count.
    """
    vertices_mm, faces = _validate_mesh(vertices_mm, faces)
    corner_0 = vertices_mm[faces[:, 0]]
    corner_1 = vertices_mm[faces[:, 1]]
    corner_2 = vertices_mm[faces[:, 2]]
    area_weighted_face_normals = np.cross(corner_1 - corner_0, corner_2 - corner_0)

    normals = np.zeros_like(vertices_mm)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], area_weighted_face_normals)

    lengths = np.linalg.norm(normals, axis=1)
    has_normal = lengths > 0.0
    normals[has_normal] = normals[has_normal] / lengths[has_normal, None]
    return normals


def _validate_labels(labels: np.ndarray, vertex_count: int) -> np.ndarray:
    """Per-vertex provenance labels, exactly R-09's two strings. Anything else is refused rather
    than quietly treated as one or the other: in an inspection system a misread label is a wrong
    answer about what was observed."""
    labels = np.asarray(labels, dtype=object)
    if labels.shape != (vertex_count,):
        raise ValueError(
            f"labels must have one entry per vertex ({vertex_count}), got shape {labels.shape}"
        )
    is_known = (labels == LABEL_OBSERVED) | (labels == LABEL_UNOBSERVED)
    if not np.all(is_known):
        unknown = sorted({str(label) for label in labels[~is_known]})
        raise ValueError(
            f"{int(np.count_nonzero(~is_known))} label(s) are neither {LABEL_OBSERVED!r} nor "
            f"{LABEL_UNOBSERVED!r}: {unknown[:5]}"
        )
    return labels


def score_candidate_views(
    vertices_mm: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    candidates: dict[str, np.ndarray],
    self_hit_tolerance_mm: float = DEFAULT_SELF_HIT_TOLERANCE_MM,
) -> dict[str, np.ndarray]:
    """For each candidate, how many under-observed vertices it would see unoccluded, plus the
    head-on tie-break score. Returns per-candidate arrays (K rows each):

    - `under_observed_visible_count`: the story's score.
    - `head_on_sum`: sum of |normal . ray direction| over those same visible vertices; 1 per vertex
      seen square-on, toward 0 at grazing angles.

    `labels` must already carry R-10's sigma downgrade (`uncertainty.downgrade_high_sigma_vertices`
    or R-10's `mesh_uncertainty.glb`); "under-observed" is `labels == LABEL_UNOBSERVED` and nothing
    else.

    Rays are cast only *to under-observed vertices* -- an Observed vertex can't add to any score, so
    raycasting it would be wasted work -- but the scene holds the *whole* mesh, because Observed
    surface occludes just as well as Unobserved surface does.
    """
    vertices_mm, faces = _validate_mesh(vertices_mm, faces)
    labels = _validate_labels(labels, len(vertices_mm))
    if not math.isfinite(self_hit_tolerance_mm) or self_hit_tolerance_mm <= 0:
        raise ValueError("self_hit_tolerance_mm must be a positive, finite number.")
    positions_mm = np.asarray(candidates["position_mm"], dtype=np.float64)
    if positions_mm.ndim != 2 or positions_mm.shape[1] != 3:
        raise ValueError(
            f"candidates['position_mm'] must have shape (K,3), got {positions_mm.shape}"
        )

    candidate_count = len(positions_mm)
    visible_count = np.zeros(candidate_count, dtype=np.int64)
    head_on_sum = np.zeros(candidate_count, dtype=np.float64)

    under_observed_indices = np.flatnonzero(labels == LABEL_UNOBSERVED)
    if len(under_observed_indices) == 0:
        # Fully observed: every score is genuinely zero. No scene is built -- there is nothing to
        # aim a ray at.
        return {
            "under_observed_visible_count": visible_count,
            "head_on_sum": head_on_sum,
        }

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(vertices_mm.astype(np.float32)),
        o3d.core.Tensor(faces.astype(np.uint32)),
    )
    target_vertices_mm = vertices_mm[under_observed_indices]
    target_normals = vertex_normals(vertices_mm, faces)[under_observed_indices]

    for candidate_index in range(candidate_count):
        camera_mm = positions_mm[candidate_index]
        to_targets = target_vertices_mm - camera_mm
        target_distance_mm = np.linalg.norm(to_targets, axis=1)
        if np.any(target_distance_mm == 0.0):
            raise ValueError(
                f"Candidate {candidate_index} sits exactly on a mesh vertex; "
                "a ray direction to it is undefined."
            )
        ray_directions = to_targets / target_distance_mm[:, None]
        rays = np.hstack(
            [np.repeat(camera_mm[None, :], len(ray_directions), axis=0), ray_directions]
        )
        first_hit_mm = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))[
            "t_hit"
        ].numpy()

        # Visible iff the first surface the ray meets is the vertex itself (one of its own
        # triangles, within the self-hit tolerance) or nothing at all -- a miss reports t_hit = inf,
        # which can happen for a ray grazing exactly through a silhouette vertex. Anything hit
        # meaningfully *before* the vertex is an occluder.
        is_visible = first_hit_mm.astype(np.float64) >= (
            target_distance_mm - self_hit_tolerance_mm
        )
        visible_count[candidate_index] = int(np.count_nonzero(is_visible))
        incidence_cosines = np.abs(
            np.sum(target_normals[is_visible] * ray_directions[is_visible], axis=1)
        )
        head_on_sum[candidate_index] = float(np.sum(incidence_cosines))

    return {"under_observed_visible_count": visible_count, "head_on_sum": head_on_sum}


# Mirror-image candidates (say 30 and 330 degrees azimuth either side of an unobserved face) produce
# head-on sums that differ only in floating-point rounding. Compared raw, which of the two wins
# would be decided by that rounding noise rather than by the documented index tie-break.
_HEAD_ON_SUM_DECIMALS = 6


def rank_candidate_views(scores: dict[str, np.ndarray]) -> list[int]:
    """Candidate indices, best first: most under-observed vertices visible; then, among equal
    counts, the most head-on view; then the lower candidate index, so the order never depends on
    anything but the inputs."""
    visible_count = np.asarray(scores["under_observed_visible_count"])
    head_on_sum = np.round(np.asarray(scores["head_on_sum"]), _HEAD_ON_SUM_DECIMALS)

    def ranking_key(candidate_index: int) -> tuple[int, float, int]:
        return (
            -int(visible_count[candidate_index]),
            -float(head_on_sum[candidate_index]),
            candidate_index,
        )

    return sorted(range(len(visible_count)), key=ranking_key)


def score_next_views(
    vertices_mm: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    *,
    hemisphere_centre_mm: np.ndarray | None = None,
    look_at_mm: np.ndarray | None = None,
    min_radius_mm: float = DEFAULT_MIN_RADIUS_MM,
    max_radius_mm: float = DEFAULT_MAX_RADIUS_MM,
    radius_steps: int = DEFAULT_RADIUS_STEPS,
    max_angle_from_vertical_deg: float = DEFAULT_MAX_ANGLE_FROM_VERTICAL_DEG,
    polar_step_deg: float = DEFAULT_POLAR_STEP_DEG,
    azimuth_step_deg: float = DEFAULT_AZIMUTH_STEP_DEG,
    self_hit_tolerance_mm: float = DEFAULT_SELF_HIT_TOLERANCE_MM,
) -> list[dict[str, Any]]:
    """The story's pure core: a board-frame mesh plus per-vertex labels in, every candidate view
    out, ranked best first, as plain JSON-ready dicts.

    `hemisphere_centre_mm` and `look_at_mm` default to `part_hemisphere_centre_and_target`'s
    footprint-centred values; pass either to override it.
    """
    vertices_mm, faces = _validate_mesh(vertices_mm, faces)
    # Only work out the defaults when a caller actually needs one (PR #144 review): next_view_run
    # always passes both, having already computed them for its report.
    if hemisphere_centre_mm is None or look_at_mm is None:
        default_centre_mm, default_look_at_mm = part_hemisphere_centre_and_target(
            vertices_mm
        )
        if hemisphere_centre_mm is None:
            hemisphere_centre_mm = default_centre_mm
        if look_at_mm is None:
            look_at_mm = default_look_at_mm

    candidates = generate_candidate_views(
        hemisphere_centre_mm,
        look_at_mm,
        min_radius_mm=min_radius_mm,
        max_radius_mm=max_radius_mm,
        radius_steps=radius_steps,
        max_angle_from_vertical_deg=max_angle_from_vertical_deg,
        polar_step_deg=polar_step_deg,
        azimuth_step_deg=azimuth_step_deg,
    )
    scores = score_candidate_views(
        vertices_mm, faces, labels, candidates, self_hit_tolerance_mm
    )

    ranked: list[dict[str, Any]] = []
    for rank, candidate_index in enumerate(rank_candidate_views(scores), start=1):
        ranked.append(
            {
                "rank": rank,
                "candidate_index": candidate_index,
                "position_mm": candidates["position_mm"][candidate_index].tolist(),
                "look_at_mm": candidates["look_at_mm"][candidate_index].tolist(),
                "radius_mm": float(candidates["radius_mm"][candidate_index]),
                "angle_from_vertical_deg": float(
                    candidates["angle_from_vertical_deg"][candidate_index]
                ),
                "elevation_deg": float(candidates["elevation_deg"][candidate_index]),
                "azimuth_deg": float(candidates["azimuth_deg"][candidate_index]),
                "under_observed_visible_count": int(
                    scores["under_observed_visible_count"][candidate_index]
                ),
                "head_on_sum": float(scores["head_on_sum"][candidate_index]),
            }
        )
    return ranked


def next_view_run(
    uncertainty_dir: Path,
    out_dir: Path,
    *,
    min_radius_mm: float = DEFAULT_MIN_RADIUS_MM,
    max_radius_mm: float = DEFAULT_MAX_RADIUS_MM,
    radius_steps: int = DEFAULT_RADIUS_STEPS,
    max_angle_from_vertical_deg: float = DEFAULT_MAX_ANGLE_FROM_VERTICAL_DEG,
    polar_step_deg: float = DEFAULT_POLAR_STEP_DEG,
    azimuth_step_deg: float = DEFAULT_AZIMUTH_STEP_DEG,
    self_hit_tolerance_mm: float = DEFAULT_SELF_HIT_TOLERANCE_MM,
) -> dict[str, Any]:
    """Score next views for R-10's output and write `next_view.json`.

    Reads `mesh_uncertainty.glb` -- R-10's mesh, whose vertex colours already carry the sigma
    downgrade -- and decodes the labels with R-09's own `colors_to_labels`, so this CLI applies
    exactly the definition of "under-observed" the pure function does, with no re-derivation.
    """
    uncertainty_dir = Path(uncertainty_dir)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already has files in it. Pick a new --out-dir or delete it deliberately; "
            "overwriting silently would let outputs from two runs get mixed."
        )

    mesh_path = uncertainty_dir / "mesh_uncertainty.glb"
    vertices_mm, faces, colors = read_glb(mesh_path)
    if colors is None:
        raise ValueError(f"{mesh_path} has no vertex colors to decode labels from")
    labels = colors_to_labels(colors)

    hemisphere_centre_mm, look_at_mm = part_hemisphere_centre_and_target(vertices_mm)
    parameters = {
        "min_radius_mm": min_radius_mm,
        "max_radius_mm": max_radius_mm,
        "radius_steps": radius_steps,
        "max_angle_from_vertical_deg": max_angle_from_vertical_deg,
        "polar_step_deg": polar_step_deg,
        "azimuth_step_deg": azimuth_step_deg,
        "self_hit_tolerance_mm": self_hit_tolerance_mm,
    }
    ranked = score_next_views(
        vertices_mm,
        faces,
        labels,
        hemisphere_centre_mm=hemisphere_centre_mm,
        look_at_mm=look_at_mm,
        **parameters,
    )

    report = {
        "story": "C-08",
        "source_mesh": str(mesh_path),
        "parameters": parameters,
        "hemisphere_centre_mm": hemisphere_centre_mm.tolist(),
        "look_at_mm": look_at_mm.tolist(),
        "vertex_count": len(vertices_mm),
        "under_observed_vertex_count": int(
            np.count_nonzero(labels == LABEL_UNOBSERVED)
        ),
        "candidate_count": len(ranked),
        "ranked_candidates": ranked,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "next_view.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank candidate camera views by how much under-observed surface each would "
        "see (C-08)."
    )
    parser.add_argument(
        "uncertainty_dir",
        type=Path,
        help="R-10's output directory, holding mesh_uncertainty.glb",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-radius-mm", type=float, default=DEFAULT_MIN_RADIUS_MM)
    parser.add_argument("--max-radius-mm", type=float, default=DEFAULT_MAX_RADIUS_MM)
    parser.add_argument("--radius-steps", type=int, default=DEFAULT_RADIUS_STEPS)
    parser.add_argument(
        "--max-angle-from-vertical-deg",
        type=float,
        default=DEFAULT_MAX_ANGLE_FROM_VERTICAL_DEG,
    )
    parser.add_argument("--polar-step-deg", type=float, default=DEFAULT_POLAR_STEP_DEG)
    parser.add_argument(
        "--azimuth-step-deg", type=float, default=DEFAULT_AZIMUTH_STEP_DEG
    )
    parser.add_argument(
        "--self-hit-tolerance-mm", type=float, default=DEFAULT_SELF_HIT_TOLERANCE_MM
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    out_dir = args.out_dir or args.uncertainty_dir / "next_view"
    try:
        report = next_view_run(
            args.uncertainty_dir,
            out_dir,
            min_radius_mm=args.min_radius_mm,
            max_radius_mm=args.max_radius_mm,
            radius_steps=args.radius_steps,
            max_angle_from_vertical_deg=args.max_angle_from_vertical_deg,
            polar_step_deg=args.polar_step_deg,
            azimuth_step_deg=args.azimuth_step_deg,
            self_hit_tolerance_mm=args.self_hit_tolerance_mm,
        )
    except FileExistsError as exc:
        print(f"C-08 NEXT VIEW: FAIL - {exc}", file=sys.stderr)
        return 1

    best = report["ranked_candidates"][0]
    under_observed = report["under_observed_vertex_count"]
    if under_observed == 0:
        print(
            f"C-08 NEXT VIEW: PASS - no under-observed vertices; all "
            f"{report['candidate_count']} candidates score 0"
        )
    else:
        print(
            f"C-08 NEXT VIEW: PASS - best view sees "
            f"{best['under_observed_visible_count']}/{under_observed} under-observed vertices "
            f"(azimuth {best['azimuth_deg']:.0f} deg, {best['angle_from_vertical_deg']:.0f} deg "
            f"from vertical, {best['radius_mm']:.0f} mm)"
        )
    print(f"Outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

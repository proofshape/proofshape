"""I-03/I-04/I-05 registration tests on the real s-04/s-05 reference CAD and I-01 synthetic output."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from inspection import cad, registration, synthetic

from recon import provenance

ROOT = Path(__file__).resolve().parents[1]
S04 = ROOT / "fixtures" / "reference_cad" / "s04_triangular_prism_60-26-65x30mm.stl"
S05 = ROOT / "fixtures" / "reference_cad" / "s05_rect_prism_30x30x60mm.stl"

TEST_SAMPLE_COUNT = 700
TEST_YAW_STEPS = 36
TEST_CANDIDATE_COUNT = 6
TEST_RANSAC_ITERATIONS = 3_000


def _expected_transform(sample: synthetic.SyntheticSample) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(sample.sidecar["transform"]["rotation"], dtype=float)
    transform[:3, 3] = np.asarray(
        sample.sidecar["transform"]["translation_mm"], dtype=float
    )
    return transform


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _rotation_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    relative = actual @ expected.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _yaw_relative_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    """Yaw of `actual` about board z relative to `expected`, in [0, 360)."""
    relative = actual[:3, :3] @ expected[:3, :3].T
    return math.degrees(math.atan2(relative[1, 0], relative[0, 0])) % 360.0


def _symmetric_vertex_rmse(
    vertices: np.ndarray,
    actual: np.ndarray,
    expected: np.ndarray,
) -> float:
    actual_points = _transform_points(vertices, actual)
    expected_points = _transform_points(vertices, expected)
    distances = np.linalg.norm(
        actual_points[:, None, :] - expected_points[None, :, :],
        axis=2,
    )
    forward = distances.min(axis=1)
    backward = distances.min(axis=0)
    return float(np.sqrt(np.mean(np.concatenate([forward**2, backward**2]))))


def _copy_sample(
    sample: synthetic.SyntheticSample,
    *,
    vertices_mm: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> synthetic.SyntheticSample:
    copied_labels = np.asarray(
        sample.labels if labels is None else labels, dtype=object
    )
    return synthetic.SyntheticSample(
        vertices_mm=np.asarray(
            sample.vertices_mm if vertices_mm is None else vertices_mm,
            dtype=np.float64,
        ),
        faces=np.asarray(sample.faces, dtype=np.int64),
        labels=copied_labels,
        colors=provenance.labels_to_colors(copied_labels),
        sidecar=dict(sample.sidecar),
    )


def _register(model, cad_mesh, *, seed: int):
    return registration.global_register(
        model,
        cad_mesh,
        sample_count=TEST_SAMPLE_COUNT,
        yaw_steps=TEST_YAW_STEPS,
        candidate_count=TEST_CANDIDATE_COUNT,
        ransac_iterations=TEST_RANSAC_ITERATIONS,
        seed=seed,
    )


@pytest.mark.parametrize(
    ("cad_path", "seed"),
    [
        (S04, 41),
        (S05, 57),
    ],
)
def test_global_register_recovers_i01_pose(cad_path: Path, seed: int):
    vertices, faces, _units = cad.load_mesh(cad_path)
    sigma_mm = 0.3
    sample = synthetic.generate_synthetic_sample(
        vertices,
        faces,
        seed,
        sigma_mm=sigma_mm,
        patch_fraction=0.2,
    )

    candidates = _register(sample, (vertices, faces), seed=seed + 100)
    best = candidates[0]
    expected = _expected_transform(sample)

    assert len(candidates) >= TEST_CANDIDATE_COUNT
    assert 0.0 <= best.fitness <= 1.0
    assert math.isfinite(best.inlier_rmse)
    assert best.correspondence_count >= 3
    assert best.free_scale_residual is not None

    # s-05 is four-fold symmetric about board z, so a 90-degree-equivalent pose is every bit as
    # correct geometrically as the literal random-yaw sidecar matrix.  Compare transformed CAD
    # surfaces for both parts; for scalene s-04 we additionally pin the actual transform.
    surface_rmse = _symmetric_vertex_rmse(vertices, best.transformation, expected)
    assert surface_rmse <= 5.0 * sigma_mm

    if cad_path == S04:
        assert _rotation_error_deg(best.transformation[:3, :3], expected[:3, :3]) <= 3.0
        assert (
            np.linalg.norm(best.transformation[:2, 3] - expected[:2, 3])
            <= 5.0 * sigma_mm
        )
        assert abs(best.transformation[2, 3]) <= 1e-9


@pytest.mark.parametrize("cad_path", [S04, S05])
def test_global_register_true_size_free_scale_residual_is_near_zero_with_noise(
    cad_path: Path,
):
    vertices, faces, _units = cad.load_mesh(cad_path)
    sample = synthetic.generate_synthetic_sample(
        vertices,
        faces,
        seed=11,
        sigma_mm=0.3,
        patch_fraction=0.2,
    )

    best = _register(sample, (vertices, faces), seed=111)[0]

    assert best.free_scale_residual is not None
    assert abs(best.free_scale_residual) <= 0.015


@pytest.mark.parametrize("cad_path", [S04, S05])
@pytest.mark.parametrize(
    ("model_scale", "expected_residual"),
    [
        (1.03, 0.03),
        (1.05, 0.05),
    ],
)
def test_global_register_scaled_model_reports_positive_free_scale_residual_with_noise(
    cad_path: Path,
    model_scale: float,
    expected_residual: float,
):
    vertices, faces, _units = cad.load_mesh(cad_path)
    sample = synthetic.generate_synthetic_sample(
        vertices,
        faces,
        seed=12,
        sigma_mm=0.3,
        patch_fraction=0.2,
    )
    model_vertices = np.asarray(sample.vertices_mm, dtype=np.float64)
    centre = model_vertices.mean(axis=0)
    scaled_model = _copy_sample(
        sample,
        vertices_mm=centre + (model_vertices - centre) * model_scale,
    )

    best = _register(scaled_model, (vertices, faces), seed=112)[0]

    assert best.free_scale_residual is not None
    assert best.free_scale_residual == pytest.approx(expected_residual, abs=0.015)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(("model_scale", "expected"), [(1.0, 0.0), (1.03, 0.03)])
def test_s04_free_scale_residual_stays_accurate_across_hidden_patch_seeds(
    seed: int, model_scale: float, expected: float
):
    """Regression for the occlusion bias found in final review: a hidden CAD patch must not
    pull a true-size model small or reverse the sign of a +3% substitution."""
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices,
        faces,
        seed=seed,
        sigma_mm=0.3,
        patch_fraction=0.2,
    )
    model = sample
    if model_scale != 1.0:
        model_vertices = np.asarray(sample.vertices_mm, dtype=np.float64)
        centre = model_vertices.mean(axis=0)
        model = _copy_sample(
            sample,
            vertices_mm=centre + (model_vertices - centre) * model_scale,
        )

    best = _register(model, (vertices, faces), seed=seed + 200)[0]

    assert best.free_scale_residual is not None
    assert best.free_scale_residual == pytest.approx(expected, abs=0.015)


@pytest.mark.parametrize(
    ("seed", "patch_fraction"),
    [
        (22, 0.0),
        # With I-01's patch hidden too. #233's review found two mutations no other test caught:
        # dropping the rigid pre-pass, and shrinking the dense CAD target to 700 points. These
        # two seeds catch both (I-05 closed the note).
        (3, 0.2),
        (4, 0.2),
    ],
)
def test_free_scale_residual_stays_accurate_when_board_contact_face_is_unobserved(
    seed: int, patch_fraction: float
):
    """A real camera cannot see the board-contact face; hiding it must not bias the scale fit."""
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices,
        faces,
        seed=seed,
        sigma_mm=0.3,
        patch_fraction=patch_fraction,
    )
    expected_transform = _expected_transform(sample)
    model_vertices = np.asarray(sample.vertices_mm, dtype=np.float64)
    cad_approx = (model_vertices - expected_transform[:3, 3]) @ expected_transform[
        :3, :3
    ]
    contact = np.abs(cad_approx[:, 2]) <= 0.75
    assert np.count_nonzero(contact) > 0
    labels = np.asarray(sample.labels, dtype=object).copy()
    labels[contact] = provenance.LABEL_UNOBSERVED
    model = _copy_sample(sample, labels=labels)

    best = _register(model, (vertices, faces), seed=200 + seed)[0]

    assert best.free_scale_residual is not None
    assert best.free_scale_residual == pytest.approx(0.0, abs=0.015)


@pytest.mark.parametrize("cad_path", [S04, S05])
@pytest.mark.parametrize("sigma_mm", [0.3, 0.5])
def test_global_register_returns_none_when_only_top_face_is_observed(
    cad_path: Path, sigma_mm: float
):
    """Noisy single-plane evidence must not be mistaken for 3-D scale evidence."""
    vertices, faces, _units = cad.load_mesh(cad_path)
    sample = synthetic.generate_synthetic_sample(
        vertices,
        faces,
        seed=3,
        sigma_mm=sigma_mm,
        patch_fraction=0.0,
    )
    expected_transform = _expected_transform(sample)
    model_vertices = np.asarray(sample.vertices_mm, dtype=np.float64)
    cad_approx = (model_vertices - expected_transform[:3, 3]) @ expected_transform[
        :3, :3
    ]
    top_z = float(np.max(vertices[:, 2]))
    top = cad_approx[:, 2] >= top_z - 0.75
    assert np.count_nonzero(top) > 0
    labels = np.full(len(sample.labels), provenance.LABEL_UNOBSERVED, dtype=object)
    labels[top] = provenance.LABEL_OBSERVED
    model = _copy_sample(sample, labels=labels)

    best = _register(model, (vertices, faces), seed=303)[0]

    assert best.free_scale_residual is None


def test_free_scale_residual_is_unavailable_without_model_provenance():
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices,
        faces,
        seed=14,
        sigma_mm=0.3,
        patch_fraction=0.2,
    )

    best = _register(
        (np.asarray(sample.vertices_mm), np.asarray(sample.faces)),
        (vertices, faces),
        seed=114,
    )[0]

    assert best.free_scale_residual is None


@pytest.mark.parametrize(
    "label",
    [provenance.LABEL_UNOBSERVED, provenance.LABEL_INFERRED],
)
def test_free_scale_residual_is_unavailable_without_observed_faces(label: str):
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices,
        faces,
        seed=15,
        sigma_mm=0.3,
        patch_fraction=0.2,
    )
    hidden = _copy_sample(
        sample,
        labels=np.full(len(sample.labels), label, dtype=object),
    )

    best = _register(hidden, (vertices, faces), seed=115)[0]

    assert best.free_scale_residual is None


def test_unobserved_geometry_cannot_change_scale_diagnostic():
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices,
        faces,
        seed=16,
        sigma_mm=0.3,
        patch_fraction=0.2,
    )
    baseline = _register(sample, (vertices, faces), seed=116)[0]

    distorted_vertices = np.asarray(sample.vertices_mm, dtype=np.float64).copy()
    hidden = np.asarray(sample.labels) != provenance.LABEL_OBSERVED
    distorted_vertices[hidden] += np.array([500.0, -300.0, 200.0])
    distorted = _copy_sample(sample, vertices_mm=distorted_vertices)
    changed = _register(distorted, (vertices, faces), seed=116)[0]

    assert baseline.free_scale_residual is not None
    assert changed.free_scale_residual == pytest.approx(
        baseline.free_scale_residual,
        abs=1e-9,
    )


def _point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return cloud


def test_free_scale_residual_is_vertex_order_invariant_on_a_cube():
    cube = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [10.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [10.0, 0.0, 10.0],
            [0.0, 10.0, 10.0],
            [10.0, 10.0, 10.0],
        ]
    )
    candidate = registration.CandidatePose(
        transformation=np.eye(4),
        fitness=1.0,
        inlier_rmse=0.0,
        source="test",
        correspondence_count=len(cube),
    )

    residual = registration._free_scale_residual(
        candidate,
        _point_cloud(cube),
        _point_cloud(cube[::-1]),
        max_correspondence_distance_mm=2.0,
    )

    assert residual == pytest.approx(0.0, abs=1e-9)


def test_free_scale_residual_is_unavailable_without_icp_correspondences():
    source_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ]
    )
    target_points = source_points + np.array([100.0, 0.0, 0.0])
    candidate = registration.CandidatePose(
        transformation=np.eye(4),
        fitness=0.0,
        inlier_rmse=math.inf,
        source="test-no-correspondence",
        correspondence_count=0,
    )

    residual = registration._free_scale_residual(
        candidate,
        _point_cloud(source_points),
        _point_cloud(target_points),
        max_correspondence_distance_mm=0.001,
    )

    assert residual is None


def test_free_scale_residual_is_unavailable_for_planar_only_observed_geometry():
    axis = np.linspace(-20.0, 20.0, 9)
    xx, yy = np.meshgrid(axis, axis)
    plane = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    candidate = registration.CandidatePose(
        transformation=np.eye(4),
        fitness=1.0,
        inlier_rmse=0.0,
        source="test-planar-only",
        correspondence_count=len(plane),
    )

    residual = registration._free_scale_residual(
        candidate,
        _point_cloud(plane),
        _point_cloud(plane),
        max_correspondence_distance_mm=2.0,
    )

    assert residual is None


def _candidate_at_yaw(yaw_deg: float, fitness: float) -> registration.CandidatePose:
    transform = (
        registration._yaw_matrix(math.radians(yaw_deg)) @ registration.CAD_TO_BOARD_FLIP
    )
    return registration.CandidatePose(
        transformation=transform,
        fitness=fitness,
        inlier_rmse=1.0 - 0.1 * fitness,
        source=f"test:{yaw_deg}",
        correspondence_count=100,
    )


def test_orientation_diverse_selection_preserves_separate_yaw_basins():
    # The three strongest scores cluster around the same 0-degree basin. I-05 still needs the
    # genuinely different 90/180/270-degree alternatives, so score-only truncation is wrong.
    candidates = [
        _candidate_at_yaw(0.0, 1.00),
        _candidate_at_yaw(5.0, 0.99),
        _candidate_at_yaw(10.0, 0.98),
        _candidate_at_yaw(90.0, 0.70),
        _candidate_at_yaw(180.0, 0.80),
        _candidate_at_yaw(270.0, 0.75),
    ]

    selected = registration._select_orientation_diverse(candidates, 4)
    selected_degrees = sorted(
        round(math.degrees(registration._pose_yaw_rad(candidate))) % 360
        for candidate in selected
    )

    assert selected_degrees == [0, 90, 180, 270]


def test_s05_global_register_preserves_multiple_orientation_basins():
    vertices, faces, _units = cad.load_mesh(S05)
    sample = synthetic.generate_synthetic_sample(
        vertices,
        faces,
        seed=31,
        sigma_mm=0.3,
        patch_fraction=0.2,
    )

    candidates = _register(sample, (vertices, faces), seed=131)
    # Yaw relative to the sidecar's true pose, not absolute board yaw (#233's review): seed 31's
    # true yaw is 325.1 degrees, so absolute-yaw basins counted the wrong poses.
    expected = _expected_transform(sample)
    yaw_degrees = [
        _yaw_relative_deg(candidate.transformation, expected)
        for candidate in candidates
    ]
    basins = {
        round(yaw / 90.0) % 4
        for yaw in yaw_degrees
        if min(abs((yaw - k * 90.0 + 180.0) % 360.0 - 180.0) for k in range(4)) <= 15.0
    }

    assert len(basins) >= 3


def test_global_register_candidates_are_rigid_and_distinct():
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices,
        faces,
        seed=21,
        sigma_mm=0.0,
        patch_fraction=0.0,
    )

    candidates = _register(sample, (vertices, faces), seed=121)

    for candidate in candidates:
        rotation = candidate.transformation[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-8)
        assert candidate.transformation[2, 3] == pytest.approx(0.0, abs=1e-9)

    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            rotation_delta = registration._rotation_delta_deg(
                left.transformation, right.transformation
            )
            translation_delta = np.linalg.norm(
                left.transformation[:3, 3] - right.transformation[:3, 3]
            )
            assert rotation_delta > 2.0 or translation_delta > 0.25


# --- I-04: refine ----------------------------------------------------------------------------


def _assert_rigid(transform: np.ndarray) -> None:
    rotation = transform[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize(
    ("cad_path", "seed"),
    [
        (S04, 0),
        (S04, 1),
        (S05, 1),
        (S05, 2),
    ],
)
def test_refine_reduces_pose_error_from_the_i03_best_candidate(
    cad_path: Path, seed: int
):
    # The story's first test: start from I-03's best candidate on I-01 output (I-01's default
    # σ = 0.3 mm and 20% Unobserved patch) and check refinement gets closer to the sidecar truth.
    # Surface RMSE rather than the raw transform, so s-05's four-fold symmetric poses count as
    # equally right. These seeds' I-03 starts are 0.82/0.57 mm (s-04) and 2.13/1.55 mm (s-05) off,
    # so "below the error before it" is a real reduction, not a tie at zero.
    vertices, faces, _units = cad.load_mesh(cad_path)
    sigma_mm = 0.3
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, seed, sigma_mm=sigma_mm, patch_fraction=0.2
    )
    expected = _expected_transform(sample)
    best = _register(sample, (vertices, faces), seed=100 + seed)[0]

    refined = registration.refine(best, sample, (vertices, faces), seed=seed)

    before = _symmetric_vertex_rmse(vertices, best.transformation, expected)
    after = _symmetric_vertex_rmse(vertices, refined.transformation, expected)
    assert after < before
    # Within one σ of the injected noise. A bound for this test's synthetic setup, not an
    # inspection tolerance.
    assert after <= sigma_mm
    _assert_rigid(refined.transformation)
    assert refined.source == f"refined:{best.source}"
    assert refined.correspondence_count >= 3
    assert 0.0 < refined.fitness <= 1.0
    assert math.isfinite(refined.inlier_rmse)


def test_refine_requires_model_provenance():
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, 30, sigma_mm=0.3, patch_fraction=0.2
    )
    best = _register(sample, (vertices, faces), seed=130)[0]

    with pytest.raises(ValueError, match="provenance"):
        registration.refine(
            best,
            (np.asarray(sample.vertices_mm), np.asarray(sample.faces)),
            (vertices, faces),
        )


@pytest.mark.parametrize(
    "label",
    [provenance.LABEL_UNOBSERVED, provenance.LABEL_INFERRED],
)
def test_refine_refuses_a_model_with_no_observed_face(label: str):
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, 31, sigma_mm=0.3, patch_fraction=0.2
    )
    best = _register(sample, (vertices, faces), seed=131)[0]
    hidden = _copy_sample(
        sample, labels=np.full(len(sample.labels), label, dtype=object)
    )

    with pytest.raises(ValueError, match="fully Observed"):
        registration.refine(best, hidden, (vertices, faces))


def _displace_patch(
    sample: synthetic.SyntheticSample,
    *,
    patch_fraction: float,
    displacement_mm: float,
    patch_seed: int,
) -> synthetic.SyntheticSample:
    """Plant a defect: push one connected surface patch along its mean outward normal.

    Positive displacement is a bump, negative a dent. Every vertex stays labelled Observed, so
    the defect is measured geometry the fit has to cope with, not hidden geometry it can ignore.
    """
    vertices = np.asarray(sample.vertices_mm, dtype=np.float64).copy()
    faces = np.asarray(sample.faces, dtype=np.int64)
    patch = synthetic.select_patch_vertices(
        faces, len(vertices), patch_fraction, np.random.default_rng(patch_seed)
    )
    normals = np.asarray(registration._triangle_mesh(vertices, faces).vertex_normals)
    direction = normals[patch].mean(axis=0)
    direction /= np.linalg.norm(direction)
    vertices[patch] += displacement_mm * direction
    return _copy_sample(
        sample,
        vertices_mm=vertices,
        labels=np.full(len(vertices), provenance.LABEL_OBSERVED, dtype=object),
    )


@pytest.mark.parametrize(
    ("cad_path", "seed", "displacement_mm"),
    [
        (S04, 0, 4.0),
        (S04, 2, -4.0),
        (S05, 0, 4.0),
        (S05, 2, -4.0),
    ],
)
def test_refine_is_not_dragged_by_an_observed_defect(
    cad_path: Path, seed: int, displacement_mm: float, monkeypatch
):
    # The story's second test, and the reason I-04 exists (D-008): a 4 mm bump or dent over 10%
    # of the surface, all of it Observed, must not pull the pose toward itself. Tukey gives the
    # displaced points zero weight once they sit beyond k (1 voxel, 1.8 mm here).
    vertices, faces, _units = cad.load_mesh(cad_path)
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, seed, sigma_mm=0.3, patch_fraction=0.2
    )
    expected = _expected_transform(sample)
    best = _register(sample, (vertices, faces), seed=100 + seed)[0]
    defective = _displace_patch(
        sample,
        patch_fraction=0.10,
        displacement_mm=displacement_mm,
        patch_seed=seed,
    )

    robust = registration.refine(best, defective, (vertices, faces), seed=seed)
    robust_error = _symmetric_vertex_rmse(vertices, robust.transformation, expected)

    # The same refinement with Tukey swapped for plain least squares, through the identical code
    # path. It proves this defect is big enough to drag a non-robust fit; without that, the
    # bound on the Tukey fit below could pass vacuously.
    monkeypatch.setattr(
        registration.o3d.pipelines.registration,
        "TukeyLoss",
        lambda _k: o3d.pipelines.registration.L2Loss(),
    )
    least_squares = registration.refine(best, defective, (vertices, faces), seed=seed)
    least_squares_error = _symmetric_vertex_rmse(
        vertices, least_squares.transformation, expected
    )

    defect_size_mm = abs(displacement_mm)
    # "The pose error stays below the defect's size", with margin: a quarter of it.
    assert robust_error < 0.25 * defect_size_mm
    assert least_squares_error > 3.0 * robust_error
    _assert_rigid(robust.transformation)


def _plate_with_hole(
    hole_x_mm: float,
    *,
    width_mm: float = 60.0,
    depth_mm: float = 40.0,
    height_mm: float = 8.0,
    hole_mm: float = 10.0,
    hole_y_mm: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A plate with a square through-hole, base at z = 0, wound outward (the hole's walls face
    into the hole). Neither golden CAD has a hole, so the shifted-hole test builds its own."""
    half = hole_mm / 2.0
    outer = [(0.0, 0.0), (width_mm, 0.0), (width_mm, depth_mm), (0.0, depth_mm)]
    inner = [
        (hole_x_mm - half, hole_y_mm - half),
        (hole_x_mm + half, hole_y_mm - half),
        (hole_x_mm + half, hole_y_mm + half),
        (hole_x_mm - half, hole_y_mm + half),
    ]
    corners = outer + inner
    vertices = np.array(
        [(x, y, 0.0) for x, y in corners] + [(x, y, height_mm) for x, y in corners]
    )
    outer_bottom, inner_bottom = [0, 1, 2, 3], [4, 5, 6, 7]
    outer_top, inner_top = [8, 9, 10, 11], [12, 13, 14, 15]
    faces = []
    for k in range(4):
        n = (k + 1) % 4
        faces += [
            [outer_top[k], outer_top[n], inner_top[n]],
            [outer_top[k], inner_top[n], inner_top[k]],
            [outer_bottom[k], inner_bottom[n], outer_bottom[n]],
            [outer_bottom[k], inner_bottom[k], inner_bottom[n]],
            [outer_bottom[k], outer_bottom[n], outer_top[n]],
            [outer_bottom[k], outer_top[n], outer_top[k]],
            [inner_bottom[k], inner_top[n], inner_bottom[n]],
            [inner_bottom[k], inner_top[k], inner_top[n]],
        ]
    return vertices, np.array(faces, dtype=np.int64)


def test_plate_with_hole_fixture_is_a_closed_solid():
    vertices, faces = _plate_with_hole(20.0)
    mesh = registration._triangle_mesh(vertices, faces)
    assert mesh.is_watertight()
    assert mesh.get_volume() == pytest.approx(60.0 * 40.0 * 8.0 - 10.0 * 10.0 * 8.0)


@pytest.mark.parametrize("seed", [1, 2])
def test_refine_is_not_dragged_by_a_shifted_hole(seed: int, monkeypatch):
    # The story's shifted-hole case, the one D-008 names: the part's hole sits 3 mm from where
    # the CAD puts it. A hole's walls face sideways, so unlike a bump or dent they pull the pose
    # laterally, toward the misplaced hole. The model also keeps I-01's 20% Unobserved patch.
    # 3 mm sits inside the 2.5-voxel correspondence distance (4.5 mm here), where a non-robust
    # fit is dragged; a shift beyond it simply stops matching.
    #
    # Seeds 1 and 2 start I-03 within 2 mm of the true pose. Seed 0's best candidate is the
    # plate's 180-degree mirror (11 mm off): a hole that is the only asymmetry leaves two
    # basins, and choosing between candidates is I-05's job, not refine's.
    shift_mm = 3.0
    cad_vertices, cad_faces = _plate_with_hole(20.0)
    part_vertices, part_faces = _plate_with_hole(20.0 + shift_mm)
    sample = synthetic.generate_synthetic_sample(
        part_vertices, part_faces, seed, sigma_mm=0.3, patch_fraction=0.2
    )
    expected = _expected_transform(sample)
    best = _register(sample, (cad_vertices, cad_faces), seed=100 + seed)[0]

    robust = registration.refine(best, sample, (cad_vertices, cad_faces), seed=seed)
    robust_error = _symmetric_vertex_rmse(cad_vertices, robust.transformation, expected)

    monkeypatch.setattr(
        registration.o3d.pipelines.registration,
        "TukeyLoss",
        lambda _k: o3d.pipelines.registration.L2Loss(),
    )
    least_squares = registration.refine(
        best, sample, (cad_vertices, cad_faces), seed=seed
    )
    least_squares_error = _symmetric_vertex_rmse(
        cad_vertices, least_squares.transformation, expected
    )

    assert robust_error < 0.25 * shift_mm
    assert least_squares_error > 3.0 * robust_error
    _assert_rigid(robust.transformation)


@pytest.mark.parametrize("cad_path", [S04, S05])
def test_refine_ignores_unobserved_geometry(cad_path: Path):
    # I-01's 20% patch is labelled Unobserved but its geometry is still in the mesh. Moving that
    # hidden geometry 8 mm (far past the 4.5 mm correspondence distance, so a fit that used it
    # would notice) must leave the refined pose exactly unchanged: the hard rule (D-005) says
    # geometry nobody measured can't influence the pose a pass or fail is judged against. This
    # is the story's "I-01's missing patch" case: a missing patch is hidden geometry by
    # definition.
    vertices, faces, _units = cad.load_mesh(cad_path)
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, 33, sigma_mm=0.3, patch_fraction=0.2
    )
    best = _register(sample, (vertices, faces), seed=133)[0]
    baseline = registration.refine(best, sample, (vertices, faces), seed=33)

    moved_vertices = np.asarray(sample.vertices_mm, dtype=np.float64).copy()
    hidden = np.asarray(sample.labels) != provenance.LABEL_OBSERVED
    assert hidden.any()
    moved_vertices[hidden] += np.array([0.0, 0.0, -8.0])
    moved = _copy_sample(sample, vertices_mm=moved_vertices)
    changed = registration.refine(best, moved, (vertices, faces), seed=33)

    np.testing.assert_allclose(
        changed.transformation, baseline.transformation, atol=1e-9
    )
    assert changed.inlier_rmse == pytest.approx(baseline.inlier_rmse, abs=1e-12)


@pytest.mark.parametrize("cad_path", [S04, S05])
@pytest.mark.parametrize(
    ("model_scale", "expected_residual"),
    [
        (1.0, 0.0),
        (1.03, 0.03),
        (1.05, 0.05),
    ],
)
def test_refine_recomputes_the_free_scale_residual(
    cad_path: Path, model_scale: float, expected_residual: float
):
    # The AC's "plus the free-scale residual recomputed on the refined correspondences": the
    # refined candidate carries its own residual, computed from the refined pose, with D-039's
    # sign (+0.03 = the model came out 3% larger than the design). 1.03 sits just past D-007's
    # ~2% flag. The ±0.015 is this synthetic test's bound, not an inspection tolerance.
    vertices, faces, _units = cad.load_mesh(cad_path)
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, 34, sigma_mm=0.3, patch_fraction=0.2
    )
    scaled = _copy_sample(
        sample, vertices_mm=np.asarray(sample.vertices_mm) * model_scale
    )
    best = _register(scaled, (vertices, faces), seed=134)[0]
    # Only I-03's first candidate carries a residual; the runner-ups I-05 will refine carry None.
    # Refining one of those (simulated by clearing it) proves the residual is computed here from
    # the refined pose, not copied through from the candidate.
    runner_up_like = replace(best, free_scale_residual=None)

    refined = registration.refine(runner_up_like, scaled, (vertices, faces), seed=34)

    assert refined.free_scale_residual is not None
    assert refined.free_scale_residual == pytest.approx(expected_residual, abs=0.015)


def test_refine_fails_loudly_when_the_candidate_is_nowhere_near_the_model():
    # A candidate 500 mm away finds no correspondences. Open3D then reports fitness 0 and
    # inlier_rmse 0, and an inlier_rmse of 0 would read as a perfect fit to I-05's comparison.
    # No silent failures: refine raises instead.
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, 35, sigma_mm=0.3, patch_fraction=0.2
    )
    far = _expected_transform(sample)
    far[:3, 3] += np.array([500.0, 0.0, 0.0])
    candidate = registration.CandidatePose(
        transformation=far,
        fitness=0.0,
        inlier_rmse=math.inf,
        source="test:far",
        correspondence_count=0,
    )

    with pytest.raises(ValueError, match="too far from the model"):
        registration.refine(candidate, sample, (vertices, faces))


@pytest.mark.parametrize(
    "bad_keyword",
    [
        {"voxel_fraction": 0.0},
        {"tukey_k_factor": -1.0},
        {"tukey_k_factor": math.inf},
        {"correspondence_factor": math.nan},
        {"max_iterations": 0},
    ],
)
def test_refine_rejects_invalid_parameters(bad_keyword: dict):
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, 32, sigma_mm=0.3, patch_fraction=0.2
    )
    candidate = registration.CandidatePose(
        transformation=_expected_transform(sample),
        fitness=1.0,
        inlier_rmse=0.0,
        source="test",
        correspondence_count=100,
    )

    with pytest.raises(ValueError):
        registration.refine(candidate, sample, (vertices, faces), **bad_keyword)


# I-05 · orientation ambiguity (D-008, D-061)


def _solid_plate(
    *, width_mm: float = 60.0, depth_mm: float = 40.0, height_mm: float = 8.0
) -> tuple[np.ndarray, np.ndarray]:
    """`_plate_with_hole` without its hole: a part whose one symmetry-breaking feature is
    missing, which is D-008's case."""
    corners = [(0.0, 0.0), (width_mm, 0.0), (width_mm, depth_mm), (0.0, depth_mm)]
    vertices = np.array(
        [(x, y, 0.0) for x, y in corners] + [(x, y, height_mm) for x, y in corners]
    )
    faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]]
    for k in range(4):
        n = (k + 1) % 4
        faces += [[k, n, n + 4], [k, n + 4, k + 4]]
    return vertices, np.array(faces, dtype=np.int64)


def _check_ambiguity(sample, cad_mesh, *, seed: int, **kwargs):
    candidates = _register(sample, cad_mesh, seed=100 + seed)
    return registration.check_ambiguity(
        candidates, sample, cad_mesh, seed=seed, **kwargs
    )


def _far_candidate() -> registration.CandidatePose:
    transformation = np.eye(4)
    transformation[0, 3] = 1000.0
    return registration.CandidatePose(
        transformation=transformation,
        fitness=0.0,
        inlier_rmse=math.inf,
        source="far-away",
        correspondence_count=0,
    )


def test_solid_plate_fixture_is_a_closed_solid():
    vertices, faces = _solid_plate()
    mesh = registration._triangle_mesh(vertices, faces)
    assert mesh.is_watertight()
    assert mesh.get_volume() == pytest.approx(60.0 * 40.0 * 8.0)


@pytest.mark.parametrize("patch_fraction", [0.0, 0.2])
def test_check_ambiguity_flags_four_fold_s05(patch_fraction: float):
    # The story's first case: s-05 (30 x 30 x 60 mm, four-fold about z) is ambiguous, with and
    # without a hidden patch. The runner-up is a symmetric alternative, not a 5-degree neighbour.
    vertices, faces, _units = cad.load_mesh(S05)
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, 1, sigma_mm=0.3, patch_fraction=patch_fraction
    )

    result = _check_ambiguity(sample, (vertices, faces), seed=1)

    assert result.orientation_ambiguous
    assert result.reason_codes == (registration.REASON_ORIENTATION_AMBIGUOUS,)
    assert result.runner_up is not None
    assert (
        registration._rotation_delta_deg(
            result.best.transformation, result.runner_up.transformation
        )
        >= 80.0
    )
    assert result.rmse_ratio <= 1.0 + registration.DEFAULT_AMBIGUITY_THRESHOLD


@pytest.mark.parametrize("patch_fraction", [0.0, 0.2])
def test_check_ambiguity_clears_scalene_s04(patch_fraction: float):
    # The story's second case: s-04's scalene triangle has no symmetric alternative. Its
    # runner-up is about 4x worse, and the best is the true pose.
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, 0, sigma_mm=0.3, patch_fraction=patch_fraction
    )
    expected = _expected_transform(sample)

    result = _check_ambiguity(sample, (vertices, faces), seed=0)

    assert not result.orientation_ambiguous
    assert result.reason_codes == ()
    assert result.rmse_ratio > 1.0 + registration.DEFAULT_AMBIGUITY_THRESHOLD
    assert _symmetric_vertex_rmse(vertices, result.best.transformation, expected) <= 0.3


def test_check_ambiguity_keeps_symmetric_poses_separate():
    # Poses equivalent under the CAD's symmetry are kept separate, never merged: for the printed
    # part they are different poses, and the difference is where a defect would be. Merging them
    # would make this check a no-op on exactly the parts it exists for.
    vertices, faces, _units = cad.load_mesh(S05)
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, 2, sigma_mm=0.3, patch_fraction=0.2
    )
    expected = _expected_transform(sample)

    result = _check_ambiguity(sample, (vertices, faces), seed=2)

    basins = {
        round(_yaw_relative_deg(pose.transformation, expected) / 90.0) % 4
        for pose in result.refined
        if pose.fitness >= 0.99
    }
    assert len(basins) >= 3


def test_check_ambiguity_merges_candidates_whose_transforms_agree():
    # Only poses whose transforms agree are merged: the same candidate twice, and one nudged by
    # half a degree, refine to one pose, so there is no runner-up to be ambiguous with.
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, 0, sigma_mm=0.3, patch_fraction=0.2
    )
    best = _register(sample, (vertices, faces), seed=100)[0]
    half_degree = math.radians(0.5)
    nudge = np.eye(4)
    nudge[:2, :2] = [
        [math.cos(half_degree), -math.sin(half_degree)],
        [math.sin(half_degree), math.cos(half_degree)],
    ]
    nudged = replace(best, transformation=best.transformation @ nudge, source="nudged")

    result = registration.check_ambiguity(
        [best, best, nudged], sample, (vertices, faces), seed=0
    )

    assert len(result.refined) == 1
    assert result.runner_up is None
    assert result.rmse_ratio is None
    assert not result.orientation_ambiguous


@pytest.mark.parametrize("seed", [0, 3, 6, 7])
def test_check_ambiguity_flags_a_missing_symmetry_breaker_under_occlusion(seed: int):
    # D-008's own case: the one feature that breaks the CAD's 180-degree symmetry (the plate's
    # hole) is missing from the part, so the mirror pose fits as well as the true one. Under
    # I-01's 20% hidden patch, seeds 0, 3 and 6 read runner-up/best ratios of 1.10, 1.11 and 1.15,
    # which D-008's 10% missed (9/12 seeds caught at 10%, 12/12 at 20%). D-061 raised the default
    # to 20% for this case, so reverting it fails here. Seed 7 pins the ranking: the runner-up by
    # fitness has the lower RMSE (ratio 0.82), and ranking by RMSE alone would swap the two and
    # read 1.22, missing the ambiguity.
    cad_mesh = _plate_with_hole(20.0)
    part_vertices, part_faces = _solid_plate()
    sample = synthetic.generate_synthetic_sample(
        part_vertices, part_faces, seed, sigma_mm=0.3, patch_fraction=0.2
    )

    result = _check_ambiguity(sample, cad_mesh, seed=seed)

    assert result.orientation_ambiguous


@pytest.mark.parametrize("seed", [0, 1])
def test_check_ambiguity_flags_a_missing_symmetry_breaker_without_occlusion(seed: int):
    # The same case with nothing hidden: the ratios stay at 0.98-1.04 over 12 seeds.
    cad_mesh = _plate_with_hole(20.0)
    part_vertices, part_faces = _solid_plate()
    sample = synthetic.generate_synthetic_sample(
        part_vertices, part_faces, seed, sigma_mm=0.3, patch_fraction=0.0
    )

    assert _check_ambiguity(sample, cad_mesh, seed=seed).orientation_ambiguous


def test_check_ambiguity_recovers_from_an_i03_mirror_pick():
    # #251's shifted-hole seed 0: I-03's best candidate is the plate's 180-degree mirror, which
    # refine alone can't fix. Refining every candidate and ranking them again does, and the
    # observed hole then makes the mirror clearly worse (ratio 1.29), so it isn't ambiguous.
    cad_vertices, cad_faces = _plate_with_hole(20.0)
    part_vertices, part_faces = _plate_with_hole(23.0)
    sample = synthetic.generate_synthetic_sample(
        part_vertices, part_faces, 0, sigma_mm=0.3, patch_fraction=0.2
    )
    expected = _expected_transform(sample)
    candidates = _register(sample, (cad_vertices, cad_faces), seed=100)
    assert (
        _symmetric_vertex_rmse(cad_vertices, candidates[0].transformation, expected)
        > 5.0
    )

    result = registration.check_ambiguity(
        candidates, sample, (cad_vertices, cad_faces), seed=0
    )

    assert (
        _symmetric_vertex_rmse(cad_vertices, result.best.transformation, expected)
        <= 0.25
    )
    assert not result.orientation_ambiguous


def test_check_ambiguity_clears_a_plate_whose_hole_is_in_place():
    # The hole breaks the symmetry and is where the CAD puts it: the mirror is 1.7-2.3x worse.
    cad_mesh = _plate_with_hole(20.0)
    sample = synthetic.generate_synthetic_sample(
        *cad_mesh, 1, sigma_mm=0.3, patch_fraction=0.2
    )

    result = _check_ambiguity(sample, cad_mesh, seed=1)

    assert not result.orientation_ambiguous
    assert result.reason_codes == ()


def test_check_ambiguity_records_a_candidate_it_cannot_refine():
    # I-03's ransac:0 is often an identity pose at fitness 0. A candidate that can't be refined
    # is reported with its reason, never dropped silently, and the check still answers.
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, 0, sigma_mm=0.3, patch_fraction=0.2
    )
    candidates = _register(sample, (vertices, faces), seed=100)

    result = registration.check_ambiguity(
        [*candidates, _far_candidate()], sample, (vertices, faces), seed=0
    )

    reasons = {source: reason for source, reason in result.failed}
    assert "too far from the model" in reasons["far-away"]
    assert not result.orientation_ambiguous


def test_check_ambiguity_raises_when_no_candidate_can_be_refined():
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, 0, sigma_mm=0.3, patch_fraction=0.2
    )

    with pytest.raises(ValueError, match="No candidate could be refined"):
        registration.check_ambiguity([_far_candidate()], sample, (vertices, faces))


def test_check_ambiguity_requires_model_provenance():
    # A structural input problem raises once, up front, instead of reading as "every candidate
    # failed to refine".
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, 0, sigma_mm=0.3, patch_fraction=0.2
    )
    unlabelled = (np.asarray(sample.vertices_mm), np.asarray(sample.faces))

    with pytest.raises(ValueError, match="^check_ambiguity needs model provenance"):
        registration.check_ambiguity([_far_candidate()], unlabelled, (vertices, faces))


@pytest.mark.parametrize(
    "label", [provenance.LABEL_UNOBSERVED, provenance.LABEL_INFERRED]
)
def test_check_ambiguity_refuses_a_model_with_no_observed_face(label: str):
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, 0, sigma_mm=0.3, patch_fraction=0.2
    )
    hidden = _copy_sample(
        sample, labels=np.full(len(sample.vertices_mm), label, dtype=object)
    )

    with pytest.raises(
        ValueError, match="^check_ambiguity needs at least one fully Observed"
    ):
        registration.check_ambiguity([_far_candidate()], hidden, (vertices, faces))


def test_reason_code_is_the_contracts():
    example = json.loads(
        (
            ROOT
            / "contracts"
            / "examples"
            / "getVerdict.response-200-rescan-unobserved.json"
        ).read_text(encoding="utf-8")
    )
    assert example["orientation_ambiguous"] is True
    assert registration.REASON_ORIENTATION_AMBIGUOUS in example["reason_codes"]


@pytest.mark.parametrize("threshold", [-0.1, math.nan, math.inf])
def test_check_ambiguity_rejects_an_invalid_threshold(threshold: float):
    with pytest.raises(ValueError, match="ambiguity_threshold"):
        registration.check_ambiguity(
            [_far_candidate()], None, None, ambiguity_threshold=threshold
        )


def test_check_ambiguity_rejects_no_candidates():
    with pytest.raises(ValueError, match="at least one candidate"):
        registration.check_ambiguity([], None, None)


def _hide_bottom_and_half(sample: synthetic.SyntheticSample, vertices_mm: np.ndarray):
    """Hide the board-contact face and half the part along its long horizontal axis."""
    height = -vertices_mm[:, 2]  # board frame: z points into the board
    centred_xy = vertices_mm[:, :2] - vertices_mm[:, :2].mean(axis=0)
    long_axis = np.linalg.svd(centred_xy, full_matrices=False)[2][0]
    labels = np.asarray(sample.labels, dtype=object).copy()
    labels[(height < 0.5) | (centred_xy @ long_axis > 0.0)] = (
        provenance.LABEL_UNOBSERVED
    )
    return _copy_sample(sample, vertices_mm=vertices_mm, labels=labels)


@pytest.mark.parametrize("seed", [2, 17, 20, 21, 23])
def test_free_scale_residual_is_never_absurd_under_heavy_occlusion(seed: int):
    # #233's open note, closed in I-05: with the bottom face and half the part hidden, s-04 x1.03
    # (true residual +0.03) read +0.29 to +6.36 on these seeds. A reading beyond the sanity bound
    # is a failed similarity fit, so it now comes back as None. Inside the bound, occlusion this
    # heavy can still misread; this pins only that nothing absurd reaches D-039.
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, seed, sigma_mm=0.3, patch_fraction=0.2
    )
    model_vertices = np.asarray(sample.vertices_mm, dtype=np.float64)
    centre = model_vertices.mean(axis=0)
    model = _hide_bottom_and_half(sample, centre + (model_vertices - centre) * 1.03)

    best = _register(model, (vertices, faces), seed=100 + seed)[0]
    refined = registration.refine(best, model, (vertices, faces), seed=seed)

    bound = registration.DEFAULT_SCALE_RESIDUAL_SANITY_BOUND
    for residual in (best.free_scale_residual, refined.free_scale_residual):
        assert residual is None or abs(residual) <= bound

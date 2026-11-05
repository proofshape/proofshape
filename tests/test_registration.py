"""I-03 registration tests on the real s-04/s-05 reference CAD and I-01 synthetic output."""

from __future__ import annotations

import math
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


def test_free_scale_residual_stays_accurate_when_board_contact_face_is_unobserved():
    """A real camera cannot see the board-contact face; hiding it must not bias the scale fit."""
    vertices, faces, _units = cad.load_mesh(S04)
    sample = synthetic.generate_synthetic_sample(
        vertices,
        faces,
        seed=22,
        sigma_mm=0.3,
        patch_fraction=0.0,
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

    best = _register(model, (vertices, faces), seed=222)[0]

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
    yaw_degrees = [
        math.degrees(registration._pose_yaw_rad(candidate)) % 360.0
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

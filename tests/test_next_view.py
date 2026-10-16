"""Unit tests for C-08's next-view scoring. All synthetic meshes -- no GPU, no golden capture."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from recon import mesh_cleanup, next_view, provenance, uncertainty

# --- synthetic meshes ----------------------------------------------------------------------------
#
# Board frame (recon/board_pose.py): +z points INTO the table, so a box sitting on the board spans
# z in [-height, 0] and its top face is the one at z = -height.


def _subdivided_box(x_range, y_range, z_range, cells_per_edge=4):
    """A closed, watertight box with outward winding and `cells_per_edge`^2 quads per face, so each
    face has interior vertices of its own (a plain 8-vertex box would make every vertex a corner
    shared by three faces, and "label one face unobserved" would be meaningless)."""
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range
    length_x = np.array([x1 - x0, 0.0, 0.0])
    length_y = np.array([0.0, y1 - y0, 0.0])
    length_z = np.array([0.0, 0.0, z1 - z0])
    # (origin, u, v) per face, with u x v pointing outward.
    face_frames = [
        (np.array([x1, y0, z0]), length_y, length_z),  # +x
        (np.array([x0, y0, z0]), length_z, length_y),  # -x
        (np.array([x0, y1, z0]), length_z, length_x),  # +y
        (np.array([x0, y0, z0]), length_x, length_z),  # -y
        (np.array([x0, y0, z1]), length_x, length_y),  # +z: the bottom, on the board
        (np.array([x0, y0, z0]), length_y, length_x),  # -z: the top
    ]

    vertices = []
    faces = []
    for origin, u, v in face_frames:
        base_index = len(vertices)
        for i in range(cells_per_edge + 1):
            for j in range(cells_per_edge + 1):
                vertices.append(
                    origin + (i / cells_per_edge) * u + (j / cells_per_edge) * v
                )
        for i in range(cells_per_edge):
            for j in range(cells_per_edge):
                a = base_index + i * (cells_per_edge + 1) + j
                b = base_index + (i + 1) * (cells_per_edge + 1) + j
                c = base_index + (i + 1) * (cells_per_edge + 1) + (j + 1)
                d = base_index + i * (cells_per_edge + 1) + (j + 1)
                faces.append([a, b, c])
                faces.append([a, c, d])

    # Merge the vertices each face grid duplicates along shared edges, so the box is watertight.
    vertices = np.asarray(vertices)
    unique_vertices, inverse = np.unique(
        np.round(vertices, 6), axis=0, return_inverse=True
    )
    faces = inverse.reshape(-1)[np.asarray(faces)]
    return unique_vertices, faces.astype(np.int64)


# A hand-sized part centred over the board's x/y origin: 60 x 40 mm footprint, 30 mm tall.
BOX_X = (-30.0, 30.0)
BOX_Y = (-20.0, 20.0)
BOX_Z = (-30.0, 0.0)


# --- part_hemisphere_centre_and_target ---------------------------------------------------------


def test_hemisphere_centre_is_the_footprint_centre_at_the_parts_base():
    # A part sitting mid-board rather than over the board's corner origin.
    vertices, _faces = _subdivided_box((50.0, 110.0), (30.0, 70.0), (-30.0, 0.0))

    centre, look_at = next_view.part_hemisphere_centre_and_target(vertices)

    np.testing.assert_allclose(centre, [80.0, 50.0, 0.0])
    np.testing.assert_allclose(look_at, [80.0, 50.0, -15.0])


def test_hemisphere_centre_uses_the_meshs_own_base_not_z_zero():
    # R-21: the real board surface doesn't land exactly at z = 0, so neither does the part's base.
    vertices, _faces = _subdivided_box(BOX_X, BOX_Y, (-32.0, -2.0))

    centre, _look_at = next_view.part_hemisphere_centre_and_target(vertices)

    assert centre[2] == pytest.approx(-2.0)


@pytest.mark.parametrize(
    "vertices",
    [np.zeros((0, 3)), np.zeros((4, 2)), np.array([[0.0, 0.0, np.nan]])],
)
def test_hemisphere_centre_rejects_empty_or_malformed_vertices(vertices):
    with pytest.raises(ValueError, match="vertices_mm"):
        next_view.part_hemisphere_centre_and_target(vertices)


# --- generate_candidate_views ---------------------------------------------------------------------


def _default_candidates(centre=(0.0, 0.0, 0.0), look_at=(0.0, 0.0, -15.0)):
    return next_view.generate_candidate_views(np.array(centre), np.array(look_at))


def test_default_candidate_count():
    candidates = _default_candidates()

    # 3 radii x (1 pole + rings at 15, 30, 45, 60 degrees x 12 azimuths each).
    assert len(candidates["position_mm"]) == 3 * (1 + 4 * 12)
    for key in (
        "look_at_mm",
        "radius_mm",
        "angle_from_vertical_deg",
        "elevation_deg",
        "azimuth_deg",
    ):
        assert len(candidates[key]) == 147, key


def test_every_candidate_is_on_the_hemisphere_within_the_story_limits():
    centre = np.array([80.0, 60.0, -1.0])
    candidates = _default_candidates(centre=centre)

    offsets = candidates["position_mm"] - centre
    distances = np.linalg.norm(offsets, axis=1)
    np.testing.assert_allclose(distances, candidates["radius_mm"])
    assert distances.min() == pytest.approx(next_view.DEFAULT_MIN_RADIUS_MM)
    assert distances.max() == pytest.approx(next_view.DEFAULT_MAX_RADIUS_MM)

    # Angle from vertical recomputed from the positions themselves, with up = -z.
    recomputed_angle_deg = np.degrees(np.arccos(-offsets[:, 2] / distances))
    np.testing.assert_allclose(
        recomputed_angle_deg, candidates["angle_from_vertical_deg"], atol=1e-9
    )
    assert recomputed_angle_deg.max() <= 60.0 + 1e-9
    # Every candidate is above the part's base (more negative z), never level with or below it.
    assert np.all(candidates["position_mm"][:, 2] < centre[2])


def test_straight_down_appears_once_per_radius():
    candidates = _default_candidates()

    is_pole = candidates["angle_from_vertical_deg"] == 0.0
    assert np.count_nonzero(is_pole) == next_view.DEFAULT_RADIUS_STEPS
    np.testing.assert_allclose(
        sorted(candidates["radius_mm"][is_pole]), [250.0, 325.0, 400.0]
    )


def test_outer_ring_sits_exactly_at_the_limit_when_the_step_does_not_divide_it():
    candidates = next_view.generate_candidate_views(
        np.zeros(3),
        np.zeros(3),
        max_angle_from_vertical_deg=50.0,
        polar_step_deg=15.0,
    )

    rings = sorted(set(candidates["angle_from_vertical_deg"].tolist()))
    assert rings == [0.0, 15.0, 30.0, 45.0, 50.0]


def test_elevation_follows_c07s_convention():
    candidates = _default_candidates()

    # C-07: 0 = level with the part, 90 = straight down.
    np.testing.assert_allclose(
        candidates["elevation_deg"], 90.0 - candidates["angle_from_vertical_deg"]
    )
    is_pole = candidates["angle_from_vertical_deg"] == 0.0
    assert np.all(candidates["elevation_deg"][is_pole] == 90.0)


def test_azimuth_is_measured_from_board_x_toward_board_y():
    candidates = _default_candidates()

    at_90 = (candidates["azimuth_deg"] == 90.0) & (
        candidates["angle_from_vertical_deg"] == 60.0
    )
    offsets = candidates["position_mm"][at_90]
    assert len(offsets) == 3
    assert np.all(offsets[:, 1] > 0.0)
    np.testing.assert_allclose(offsets[:, 0], 0.0, atol=1e-9)


def test_every_candidate_carries_the_look_at_point():
    look_at = np.array([1.0, 2.0, -3.0])
    candidates = _default_candidates(look_at=look_at)

    np.testing.assert_allclose(
        candidates["look_at_mm"], np.tile(look_at, (147, 1)), atol=0
    )


def test_overrides_are_honoured():
    candidates = next_view.generate_candidate_views(
        np.zeros(3),
        np.zeros(3),
        min_radius_mm=300.0,
        max_radius_mm=300.0,
        radius_steps=1,
        max_angle_from_vertical_deg=30.0,
        polar_step_deg=30.0,
        azimuth_step_deg=90.0,
    )

    # 1 radius x (1 pole + one ring at 30 degrees x 4 azimuths).
    assert len(candidates["position_mm"]) == 5
    np.testing.assert_allclose(candidates["radius_mm"], 300.0)
    assert sorted(set(candidates["azimuth_deg"].tolist())) == [0.0, 90.0, 180.0, 270.0]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"min_radius_mm": 0.0}, "min_radius_mm"),
        ({"min_radius_mm": math.nan}, "min_radius_mm"),
        ({"max_radius_mm": 200.0}, "max_radius_mm"),
        ({"radius_steps": 0}, "radius_steps"),
        # One sample can't cover a 250-400 mm range; dropping half of it silently is refused.
        ({"radius_steps": 1}, "radius_steps=1"),
        # The mirror case (PR #144 review): several steps over one radius would only duplicate.
        ({"min_radius_mm": 300.0, "max_radius_mm": 300.0}, "radius_steps > 1"),
        ({"max_angle_from_vertical_deg": 90.0}, "max_angle_from_vertical_deg"),
        ({"max_angle_from_vertical_deg": -1.0}, "max_angle_from_vertical_deg"),
        ({"polar_step_deg": 0.0}, "polar_step_deg"),
        ({"azimuth_step_deg": 0.0}, "azimuth_step_deg"),
        ({"azimuth_step_deg": 361.0}, "azimuth_step_deg"),
    ],
)
def test_generate_candidate_views_rejects_bad_parameters(overrides, message):
    with pytest.raises(ValueError, match=message):
        next_view.generate_candidate_views(np.zeros(3), np.zeros(3), **overrides)


@pytest.mark.parametrize("radius_steps", [2.0, True])
def test_generate_candidate_views_rejects_a_non_integer_radius_step_count(radius_steps):
    # bool is an int subclass in Python, so True would otherwise slip through as "1 step".
    with pytest.raises(TypeError, match="radius_steps"):
        next_view.generate_candidate_views(
            np.zeros(3), np.zeros(3), radius_steps=radius_steps
        )


@pytest.mark.parametrize(
    ("centre", "look_at", "message"),
    [
        (np.zeros(2), np.zeros(3), "hemisphere_centre_mm"),
        (np.zeros(3), np.array([0.0, np.inf, 0.0]), "look_at_mm"),
    ],
)
def test_generate_candidate_views_rejects_bad_points(centre, look_at, message):
    with pytest.raises(ValueError, match=message):
        next_view.generate_candidate_views(centre, look_at)


# --- vertex_normals ------------------------------------------------------------------------------


def test_vertex_normals_point_out_of_each_box_face():
    vertices, faces = _subdivided_box(BOX_X, BOX_Y, BOX_Z)

    normals = next_view.vertex_normals(vertices, faces)

    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    # A vertex in the interior of the +x face (not on any edge) has the face's own normal.
    plus_x_interior = (
        np.isclose(vertices[:, 0], 30.0)
        & (np.abs(vertices[:, 1]) < 19.0)
        & (vertices[:, 2] > -29.0)
        & (vertices[:, 2] < -1.0)
    )
    assert np.count_nonzero(plus_x_interior) > 0
    np.testing.assert_allclose(normals[plus_x_interior], [[1.0, 0.0, 0.0]] * 9)
    # Top-face interior: up is -z in the board frame.
    top_interior = (
        np.isclose(vertices[:, 2], -30.0)
        & (np.abs(vertices[:, 0]) < 29.0)
        & (np.abs(vertices[:, 1]) < 19.0)
    )
    np.testing.assert_allclose(normals[top_interior], [[0.0, 0.0, -1.0]] * 9)


def test_vertex_normals_flip_with_winding_and_are_zero_when_unreferenced():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [99.0, 99.0, 99.0]]
    )
    faces = np.array([[0, 1, 2]])

    normals = next_view.vertex_normals(vertices, faces)
    flipped = next_view.vertex_normals(vertices, faces[:, ::-1])

    np.testing.assert_allclose(normals[:3], [[0.0, 0.0, 1.0]] * 3)
    np.testing.assert_allclose(flipped[:3], [[0.0, 0.0, -1.0]] * 3)
    np.testing.assert_array_equal(normals[3], [0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("faces", "message"),
    [
        (np.zeros((0, 3), dtype=np.int64), "F >= 1"),
        (np.array([[0, 1, 5]]), "outside"),
        (np.array([[0, -1, 2]]), "outside"),
        (np.array([[0.0, 1.0, 2.0]]), "integer"),
        (np.array([[0, 1, 2, 3]]), r"\(F,3\)"),
    ],
)
def test_vertex_normals_rejects_malformed_faces(faces, message):
    vertices = np.zeros((4, 3))
    with pytest.raises(ValueError, match=message):
        next_view.vertex_normals(vertices, faces)


# --- score_candidate_views / score_next_views: the story's three required cases ------------------


def _labels_with_unobserved(vertex_count, unobserved_mask):
    labels = np.full(vertex_count, provenance.LABEL_OBSERVED, dtype=object)
    labels[unobserved_mask] = provenance.LABEL_UNOBSERVED
    return labels


def _box_with_unobserved_plus_x_face():
    vertices, faces = _subdivided_box(BOX_X, BOX_Y, BOX_Z)
    plus_x_face = np.isclose(vertices[:, 0], BOX_X[1])
    return vertices, faces, _labels_with_unobserved(len(vertices), plus_x_face)


def test_top_candidate_faces_the_one_unobserved_side_face():
    vertices, faces, labels = _box_with_unobserved_plus_x_face()
    unobserved_count = np.count_nonzero(labels == provenance.LABEL_UNOBSERVED)

    ranked = next_view.score_next_views(vertices, faces, labels)

    best = ranked[0]
    assert best["rank"] == 1
    assert best["under_observed_visible_count"] == unobserved_count
    # Looking straight at the +x face: azimuth 0 (board +x), from as far round the side as the
    # story's 60-degree limit allows -- the head-on tie-break is what picks this out of the many
    # candidates that also see the whole face.
    assert best["azimuth_deg"] == 0.0
    assert best["angle_from_vertical_deg"] == 60.0
    # And on the face's outward side, beyond its plane.
    assert best["position_mm"][0] > BOX_X[1]


def test_top_candidate_is_straight_down_when_only_the_top_face_is_unobserved():
    vertices, faces = _subdivided_box(BOX_X, BOX_Y, BOX_Z)
    top_face = np.isclose(vertices[:, 2], BOX_Z[0])  # -z is up
    labels = _labels_with_unobserved(len(vertices), top_face)

    ranked = next_view.score_next_views(vertices, faces, labels)

    assert ranked[0]["angle_from_vertical_deg"] == 0.0
    assert ranked[0]["under_observed_visible_count"] == np.count_nonzero(top_face)


def _merge_meshes(first, second):
    first_vertices, first_faces = first
    second_vertices, second_faces = second
    vertices = np.vstack([first_vertices, second_vertices])
    faces = np.vstack([first_faces, second_faces + len(first_vertices)])
    return vertices, faces


def test_an_occluder_in_front_of_the_unobserved_face_drops_that_views_score():
    box_vertices, box_faces, box_labels = _box_with_unobserved_plus_x_face()
    # Pin the hemisphere so adding the wall (which widens the bounding box) can't move the
    # candidates: the comparison has to be the same view, with and without the occluder.
    fixed_geometry = {
        "hemisphere_centre_mm": np.array([0.0, 0.0, 0.0]),
        "look_at_mm": np.array([0.0, 0.0, -15.0]),
    }
    without_wall = next_view.score_next_views(
        box_vertices, box_faces, box_labels, **fixed_geometry
    )
    best_view = without_wall[0]

    # An Observed wall 15 mm in front of the +x face, taller and wider than the box.
    wall = _subdivided_box((45.0, 50.0), (-40.0, 40.0), (-80.0, 0.0))
    vertices, faces = _merge_meshes((box_vertices, box_faces), wall)
    labels = np.concatenate(
        [
            box_labels,
            np.full(len(wall[0]), provenance.LABEL_OBSERVED, dtype=object),
        ]
    )
    with_wall = next_view.score_next_views(vertices, faces, labels, **fixed_geometry)

    same_view_with_wall = next(
        view
        for view in with_wall
        if view["candidate_index"] == best_view["candidate_index"]
    )
    assert (
        same_view_with_wall["under_observed_visible_count"]
        < best_view["under_observed_visible_count"]
    )


def test_a_fully_observed_mesh_scores_zero_everywhere():
    vertices, faces = _subdivided_box(BOX_X, BOX_Y, BOX_Z)
    labels = _labels_with_unobserved(len(vertices), np.zeros(len(vertices), bool))

    ranked = next_view.score_next_views(vertices, faces, labels)

    assert len(ranked) == 147
    assert all(view["under_observed_visible_count"] == 0 for view in ranked)
    assert all(view["head_on_sum"] == 0.0 for view in ranked)
    # Every candidate ties, so the order falls through to candidate index.
    assert [view["candidate_index"] for view in ranked] == list(range(147))


def test_explicit_centre_and_look_at_skip_the_default_computation(monkeypatch):
    # PR #144 review: next_view_run always passes both, so working out the defaults anyway was
    # wasted work. Any call to the default helper now fails the test.
    vertices, faces, labels = _box_with_unobserved_plus_x_face()

    def must_not_be_called(_vertices_mm):
        raise AssertionError("defaults computed although both were given")

    monkeypatch.setattr(
        next_view, "part_hemisphere_centre_and_target", must_not_be_called
    )

    ranked = next_view.score_next_views(
        vertices,
        faces,
        labels,
        hemisphere_centre_mm=np.zeros(3),
        look_at_mm=np.array([0.0, 0.0, -15.0]),
    )

    assert ranked[0]["azimuth_deg"] == 0.0


def test_overriding_only_the_look_at_point_keeps_the_default_centre():
    # A part sitting mid-board: the default centre is its footprint centre at its base.
    vertices, faces = _subdivided_box((50.0, 110.0), (30.0, 70.0), (-30.0, 0.0))
    labels = _labels_with_unobserved(len(vertices), np.zeros(len(vertices), bool))
    look_at = np.array([1.0, 2.0, -3.0])

    ranked = next_view.score_next_views(vertices, faces, labels, look_at_mm=look_at)

    straight_down = [view for view in ranked if view["angle_from_vertical_deg"] == 0.0]
    for view in straight_down:
        np.testing.assert_allclose(
            view["position_mm"], [80.0, 50.0, -view["radius_mm"]]
        )
    assert all(view["look_at_mm"] == look_at.tolist() for view in ranked)


# --- score_candidate_views: reuse of R-10, winding, occlusion, validation ------------------------


def test_a_vertex_downgraded_by_r10_counts_as_under_observed():
    # Every vertex starts Observed by R-09; R-10's own downgrade function (not a re-derived check)
    # relabels the +x face Unobserved because its sigma is over the threshold.
    vertices, faces = _subdivided_box(BOX_X, BOX_Y, BOX_Z)
    r09_labels = _labels_with_unobserved(len(vertices), np.zeros(len(vertices), bool))
    sigma_mm = np.full(len(vertices), 0.5)
    plus_x_face = np.isclose(vertices[:, 0], BOX_X[1])
    sigma_mm[plus_x_face] = 10.0
    r10_labels, downgraded = uncertainty.downgrade_high_sigma_vertices(
        r09_labels, sigma_mm, threshold_mm=3.0
    )
    assert np.array_equal(downgraded, plus_x_face)

    ranked = next_view.score_next_views(vertices, faces, r10_labels)

    assert ranked[0]["under_observed_visible_count"] == np.count_nonzero(plus_x_face)
    assert ranked[0]["azimuth_deg"] == 0.0


def test_reversing_the_winding_changes_nothing():
    vertices, faces, labels = _box_with_unobserved_plus_x_face()

    ranked = next_view.score_next_views(vertices, faces, labels)
    ranked_reversed = next_view.score_next_views(vertices, faces[:, ::-1], labels)

    assert [view["candidate_index"] for view in ranked] == [
        view["candidate_index"] for view in ranked_reversed
    ]
    np.testing.assert_allclose(
        [view["head_on_sum"] for view in ranked],
        [view["head_on_sum"] for view in ranked_reversed],
    )


def _plus_x_top_edge_count(vertices):
    """The +x face's top row of vertices is also the top face's edge, so it's visible from anywhere
    above the box -- the only part of the +x face that is, from the far side or from straight
    down."""
    on_top_edge = np.isclose(vertices[:, 0], BOX_X[1]) & np.isclose(
        vertices[:, 2], BOX_Z[0]
    )
    return int(np.count_nonzero(on_top_edge))


def test_from_the_far_side_the_box_hides_all_but_the_shared_top_edge():
    vertices, faces, labels = _box_with_unobserved_plus_x_face()
    candidates = next_view.generate_candidate_views(
        np.zeros(3), np.array([0.0, 0.0, -15.0])
    )

    scores = next_view.score_candidate_views(vertices, faces, labels, candidates)

    # Azimuth 180 looks at the -x face. The box hides the whole +x face from there except the top
    # edge, seen across the top.
    from_the_far_side = candidates["azimuth_deg"] == 180.0
    assert _plus_x_top_edge_count(vertices) == 5
    np.testing.assert_array_equal(
        scores["under_observed_visible_count"][from_the_far_side], 5
    )


def test_observed_surface_still_occludes_unobserved_vertices():
    vertices, faces, labels = _box_with_unobserved_plus_x_face()
    candidates = next_view.generate_candidate_views(
        np.zeros(3), np.array([0.0, 0.0, -15.0])
    )

    scores = next_view.score_candidate_views(vertices, faces, labels, candidates)

    unobserved_count = np.count_nonzero(labels == provenance.LABEL_UNOBSERVED)
    assert scores["under_observed_visible_count"].max() == unobserved_count
    # Straight down looks at the top face, all Observed: it adds nothing to the count, but its
    # triangles still hide the +x face below them -- everything except the shared top edge.
    from_above = candidates["angle_from_vertical_deg"] == 0.0
    np.testing.assert_array_equal(
        scores["under_observed_visible_count"][from_above],
        _plus_x_top_edge_count(vertices),
    )


@pytest.mark.parametrize(
    ("labels_override", "message"),
    [
        (lambda labels: labels[:-1], "one entry per vertex"),
        (lambda labels: np.where(labels == "Observed", "observed", labels), "neither"),
        (lambda labels: np.full(len(labels), None, dtype=object), "neither"),
    ],
)
def test_score_candidate_views_rejects_bad_labels(labels_override, message):
    vertices, faces, labels = _box_with_unobserved_plus_x_face()
    candidates = _default_candidates()

    with pytest.raises(ValueError, match=message):
        next_view.score_candidate_views(
            vertices, faces, labels_override(labels), candidates
        )


def test_score_candidate_views_rejects_bad_tolerance_and_positions():
    vertices, faces, labels = _box_with_unobserved_plus_x_face()

    with pytest.raises(ValueError, match="self_hit_tolerance_mm"):
        next_view.score_candidate_views(
            vertices, faces, labels, _default_candidates(), self_hit_tolerance_mm=0.0
        )
    with pytest.raises(ValueError, match="position_mm"):
        next_view.score_candidate_views(
            vertices, faces, labels, {"position_mm": np.zeros((4, 2))}
        )
    on_a_vertex = {"position_mm": vertices[labels == provenance.LABEL_UNOBSERVED][:1]}
    with pytest.raises(ValueError, match="sits exactly on a mesh vertex"):
        next_view.score_candidate_views(vertices, faces, labels, on_a_vertex)


# --- rank_candidate_views -----------------------------------------------------------------------


def test_ranking_is_count_then_head_on_then_index():
    scores = {
        "under_observed_visible_count": np.array([5, 7, 7, 0]),
        "head_on_sum": np.array([9.0, 2.0, 3.0, 0.0]),
    }

    assert next_view.rank_candidate_views(scores) == [2, 1, 0, 3]


def test_ranking_ignores_floating_point_noise_in_the_head_on_sum():
    # Mirror-image views differ only in rounding; the lower index must win, not the noise.
    scores = {
        "under_observed_visible_count": np.array([4, 4]),
        "head_on_sum": np.array([2.0, 2.0 + 1e-12]),
    }

    assert next_view.rank_candidate_views(scores) == [0, 1]


# --- next_view_run / main: the thin CLI over R-10's output ---------------------------------------


def _write_synthetic_uncertainty_output(uncertainty_dir, vertices, faces, labels):
    """What R-10 leaves behind: a GLB whose vertex colours carry the downgraded labels."""
    uncertainty_dir.mkdir()
    mesh_cleanup.write_glb(
        uncertainty_dir / "mesh_uncertainty.glb",
        vertices.astype(np.float32),
        faces.astype(np.int32),
        provenance.labels_to_colors(labels),
    )


def test_next_view_run_ranks_views_from_r10s_glb(tmp_path):
    vertices, faces, labels = _box_with_unobserved_plus_x_face()
    uncertainty_dir = tmp_path / "uncertainty"
    _write_synthetic_uncertainty_output(uncertainty_dir, vertices, faces, labels)
    out_dir = tmp_path / "next_view_out"

    report = next_view.next_view_run(uncertainty_dir, out_dir)

    assert report["story"] == "C-08"
    assert report["vertex_count"] == len(vertices)
    assert report["under_observed_vertex_count"] == np.count_nonzero(
        labels == provenance.LABEL_UNOBSERVED
    )
    assert report["candidate_count"] == 147
    np.testing.assert_allclose(report["hemisphere_centre_mm"], [0.0, 0.0, 0.0])
    best = report["ranked_candidates"][0]
    assert best["azimuth_deg"] == 0.0
    assert best["angle_from_vertical_deg"] == 60.0
    assert [view["rank"] for view in report["ranked_candidates"]] == list(range(1, 148))

    on_disk = json.loads((out_dir / "next_view.json").read_text(encoding="utf-8"))
    assert on_disk == report


def test_next_view_run_passes_overrides_through(tmp_path):
    vertices, faces, labels = _box_with_unobserved_plus_x_face()
    uncertainty_dir = tmp_path / "uncertainty"
    _write_synthetic_uncertainty_output(uncertainty_dir, vertices, faces, labels)

    report = next_view.next_view_run(
        uncertainty_dir,
        tmp_path / "out",
        min_radius_mm=300.0,
        max_radius_mm=300.0,
        radius_steps=1,
        max_angle_from_vertical_deg=30.0,
        polar_step_deg=30.0,
        azimuth_step_deg=90.0,
    )

    assert report["candidate_count"] == 5
    assert report["parameters"]["max_angle_from_vertical_deg"] == 30.0


def test_next_view_run_refuses_to_overwrite_existing_output(tmp_path):
    vertices, faces, labels = _box_with_unobserved_plus_x_face()
    uncertainty_dir = tmp_path / "uncertainty"
    _write_synthetic_uncertainty_output(uncertainty_dir, vertices, faces, labels)
    out_dir = tmp_path / "out"
    next_view.next_view_run(uncertainty_dir, out_dir)

    with pytest.raises(FileExistsError, match="already has files"):
        next_view.next_view_run(uncertainty_dir, out_dir)


def test_next_view_run_rejects_a_mesh_without_label_colours(tmp_path):
    vertices, faces, _labels = _box_with_unobserved_plus_x_face()
    uncertainty_dir = tmp_path / "uncertainty"
    uncertainty_dir.mkdir()
    mesh_cleanup.write_glb(
        uncertainty_dir / "mesh_uncertainty.glb",
        vertices.astype(np.float32),
        faces.astype(np.int32),
    )

    with pytest.raises(ValueError, match="no vertex colors"):
        next_view.next_view_run(uncertainty_dir, tmp_path / "out")


def test_main_prints_the_best_view_and_writes_to_the_default_out_dir(tmp_path, capsys):
    vertices, faces, labels = _box_with_unobserved_plus_x_face()
    uncertainty_dir = tmp_path / "uncertainty"
    _write_synthetic_uncertainty_output(uncertainty_dir, vertices, faces, labels)

    exit_code = next_view.main([str(uncertainty_dir)])

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert (
        "C-08 NEXT VIEW: PASS - best view sees 25/25 under-observed vertices" in printed
    )
    assert "azimuth 0 deg, 60 deg from vertical" in printed
    assert (uncertainty_dir / "next_view" / "next_view.json").exists()


def test_main_reports_a_fully_observed_mesh_plainly(tmp_path, capsys):
    vertices, faces = _subdivided_box(BOX_X, BOX_Y, BOX_Z)
    labels = _labels_with_unobserved(len(vertices), np.zeros(len(vertices), bool))
    uncertainty_dir = tmp_path / "uncertainty"
    _write_synthetic_uncertainty_output(uncertainty_dir, vertices, faces, labels)

    assert next_view.main([str(uncertainty_dir)]) == 0
    assert "no under-observed vertices; all 147 candidates score 0" in (
        capsys.readouterr().out
    )


def test_main_fails_loudly_instead_of_overwriting(tmp_path, capsys):
    vertices, faces, labels = _box_with_unobserved_plus_x_face()
    uncertainty_dir = tmp_path / "uncertainty"
    _write_synthetic_uncertainty_output(uncertainty_dir, vertices, faces, labels)
    out_dir = tmp_path / "out"
    assert next_view.main([str(uncertainty_dir), "--out-dir", str(out_dir)]) == 0

    exit_code = next_view.main([str(uncertainty_dir), "--out-dir", str(out_dir)])

    assert exit_code == 1
    assert "C-08 NEXT VIEW: FAIL" in capsys.readouterr().err

"""Unit tests for I-01's synthetic mesh generator. All synthetic meshes -- no GPU, no golden
capture, no CAD files (that's I-02's job; I-01 takes (vertices_mm, faces) arrays directly, per
its own notes)."""

from __future__ import annotations

import numpy as np
import pytest
from inspection import synthetic

from recon.mesh_cleanup import read_glb
from recon.provenance import LABEL_OBSERVED, LABEL_UNOBSERVED, colors_to_labels

# --- a tiny closed CAD-like solid, z-up from z = 0 (ordinary CAD convention) ----------------------
#
# Deliberately scalene (10, 7, 4 mm legs), not the equal-legs tetrahedron this file used to use:
# equal legs are symmetric under swapping any two axes, so that shape's own mirror image looks
# identical to itself -- exactly why the original mirror-vs-rotation bug (PR #170 review) survived
# every test here. A genuinely chiral fixture is the only way a test can tell a mirror from a
# rotation at all.


def _tetrahedron():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 7.0, 0.0],
            [0.0, 0.0, 4.0],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ]
    )
    return vertices, faces


def _undirected_edge_counts(faces):
    counts: dict[tuple[int, int], int] = {}
    for v0, v1, v2 in faces:
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _signed_volume(vertices, faces):
    """Six times the enclosed volume of a closed, consistently-wound mesh (the divergence-theorem
    formula) -- translation-invariant, and its *sign* flips under a reflection but is preserved by
    any proper rotation, regardless of whether the shape itself has any symmetry."""
    total = 0.0
    for v0, v1, v2 in faces:
        a, b, c = vertices[v0], vertices[v1], vertices[v2]
        total += np.dot(a, np.cross(b, c))
    return total / 6.0


# --- a standard closest-point-on-triangle, for verifying the recovered transform ------------------


def _closest_point_on_triangle(p, a, b, c):
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + (d1 / (d1 - d3)) * ab
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    denom = 1.0 / (va + vb + vc)
    return a + ab * (vb * denom) + ac * (vc * denom)


def _distance_to_mesh(point, vertices, faces):
    return min(
        np.linalg.norm(
            point
            - _closest_point_on_triangle(
                point, vertices[f[0]], vertices[f[1]], vertices[f[2]]
            )
        )
        for f in faces
    )


# --- densify_mesh ----------------------------------------------------------------------------


def test_densify_reaches_target_edge_length():
    vertices, faces = _tetrahedron()
    dense_vertices, dense_faces = synthetic.densify_mesh(
        vertices, faces, target_edge_length_mm=2.0
    )

    assert synthetic._edge_lengths(dense_vertices, dense_faces).max() <= 2.0


def test_densify_stays_watertight():
    vertices, faces = _tetrahedron()
    dense_vertices, dense_faces = synthetic.densify_mesh(
        vertices, faces, target_edge_length_mm=3.0
    )

    counts = _undirected_edge_counts(dense_faces)
    assert all(count == 2 for count in counts.values())
    assert len(dense_vertices) == len(
        {tuple(v) for v in dense_vertices}
    )  # no duplicate positions


def test_densify_is_deterministic():
    vertices, faces = _tetrahedron()
    first = synthetic.densify_mesh(vertices, faces, 2.0)
    second = synthetic.densify_mesh(vertices, faces, 2.0)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_densify_rejects_non_positive_target():
    vertices, faces = _tetrahedron()
    with pytest.raises(ValueError, match="target_edge_length_mm"):
        synthetic.densify_mesh(vertices, faces, 0.0)


# --- board_frame_transform ---------------------------------------------------------------------


def test_board_frame_transform_flips_z_for_any_seed():
    # A pure CAD z-up point, with no xy offset, must always land at -z in board frame -- the
    # fixed flip holds regardless of the random yaw.
    for seed in range(20):
        rng = np.random.default_rng(seed)
        rotation, _translation = synthetic.board_frame_transform(rng)
        rotated = rotation @ np.array([0.0, 0.0, 7.0])
        assert rotated[2] == pytest.approx(-7.0)


def test_board_frame_transform_rotation_is_orthogonal():
    rng = np.random.default_rng(0)
    rotation, _translation = synthetic.board_frame_transform(rng)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)


def test_board_frame_transform_is_a_proper_rotation_for_any_seed():
    # det == +1, not -1: a mirror is orthogonal too, so orthogonality alone (above) can't tell
    # them apart. This is the PR #170 review's bug, caught here directly rather than only through
    # its downstream symptoms.
    for seed in range(20):
        rng = np.random.default_rng(seed)
        rotation, _translation = synthetic.board_frame_transform(rng)
        assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_board_frame_transform_preserves_signed_volume():
    vertices, faces = _tetrahedron()
    original_volume = _signed_volume(vertices, faces)

    rng = np.random.default_rng(11)
    rotation, translation = synthetic.board_frame_transform(rng)
    posed_vertices = vertices @ rotation.T + translation
    posed_volume = _signed_volume(posed_vertices, faces)

    assert posed_volume == pytest.approx(original_volume)


# --- apply_gaussian_noise ----------------------------------------------------------------------


def test_zero_sigma_leaves_vertices_unchanged():
    vertices, faces = _tetrahedron()
    rng = np.random.default_rng(0)
    noisy = synthetic.apply_gaussian_noise(vertices, faces, 0.0, rng)
    np.testing.assert_array_equal(noisy, vertices)


def test_noise_rejects_negative_sigma():
    vertices, faces = _tetrahedron()
    with pytest.raises(ValueError, match="sigma_mm"):
        synthetic.apply_gaussian_noise(vertices, faces, -0.1, np.random.default_rng(0))


# --- select_patch_vertices ---------------------------------------------------------------------


def test_patch_fraction_zero_selects_nothing():
    vertices, faces = synthetic.densify_mesh(*_tetrahedron(), target_edge_length_mm=2.0)
    mask = synthetic.select_patch_vertices(
        faces, len(vertices), 0.0, np.random.default_rng(0)
    )
    assert not mask.any()


def test_patch_fraction_one_selects_everything():
    vertices, faces = synthetic.densify_mesh(*_tetrahedron(), target_edge_length_mm=2.0)
    mask = synthetic.select_patch_vertices(
        faces, len(vertices), 1.0, np.random.default_rng(0)
    )
    assert mask.all()


def test_patch_is_connected_and_the_requested_size():
    vertices, faces = synthetic.densify_mesh(*_tetrahedron(), target_edge_length_mm=2.0)
    rng = np.random.default_rng(0)
    mask = synthetic.select_patch_vertices(faces, len(vertices), 0.2, rng)

    expected_size = round(0.2 * len(vertices))
    assert int(mask.sum()) == expected_size

    adjacency: dict[int, set[int]] = {i: set() for i in range(len(vertices))}
    for v0, v1, v2 in faces:
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            adjacency[a].add(b)
            adjacency[b].add(a)
    patch = set(np.flatnonzero(mask).tolist())
    start = next(iter(patch))
    reached = {start}
    frontier = [start]
    while frontier:
        vertex = frontier.pop()
        for neighbor in adjacency[vertex]:
            if neighbor in patch and neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    assert reached == patch  # one connected component, not several disjoint blobs


def test_patch_rejects_out_of_range_fraction():
    vertices, faces = _tetrahedron()
    with pytest.raises(ValueError, match="patch_fraction"):
        synthetic.select_patch_vertices(
            faces, len(vertices), 1.5, np.random.default_rng(0)
        )


# --- generate_synthetic_sample / write_synthetic_sample -----------------------------------------


def test_same_seed_is_byte_identical(tmp_path):
    vertices, faces = _tetrahedron()
    sample_a = synthetic.generate_synthetic_sample(vertices, faces, seed=42)
    sample_b = synthetic.generate_synthetic_sample(vertices, faces, seed=42)

    glb_a, glb_b = tmp_path / "a.glb", tmp_path / "b.glb"
    synthetic.write_synthetic_sample(sample_a, glb_a, tmp_path / "a.json")
    synthetic.write_synthetic_sample(sample_b, glb_b, tmp_path / "b.json")

    assert glb_a.read_bytes() == glb_b.read_bytes()


def test_glb_round_trips_through_read_glb_and_colors_to_labels(tmp_path):
    vertices, faces = _tetrahedron()
    sample = synthetic.generate_synthetic_sample(vertices, faces, seed=1)
    glb_path = tmp_path / "sample.glb"
    synthetic.write_synthetic_sample(sample, glb_path, tmp_path / "sample.json")

    read_vertices, read_faces, read_colors = read_glb(glb_path)
    read_labels = colors_to_labels(read_colors)

    np.testing.assert_allclose(read_vertices, sample.vertices_mm, atol=1e-4)
    np.testing.assert_array_equal(read_faces, sample.faces)
    np.testing.assert_array_equal(read_labels, sample.labels)


def test_patch_vertices_are_exactly_the_unobserved_ones():
    vertices, faces = _tetrahedron()
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, seed=7, patch_fraction=0.3
    )

    unobserved_indices = set(np.flatnonzero(sample.labels == LABEL_UNOBSERVED).tolist())
    observed_count = int(np.count_nonzero(sample.labels == LABEL_OBSERVED))

    # Set equality against the sidecar's own recorded indices, not just a count match -- a
    # same-sized but different patch (e.g. labels[np.roll(patch_mask, 1)]) would pass a count-only
    # check (PR #170 review).
    assert unobserved_indices == set(sample.sidecar["patch_vertex_indices"])
    assert len(unobserved_indices) == sample.sidecar["patch_vertex_count"]
    assert len(unobserved_indices) + observed_count == len(sample.labels)


def test_sidecar_transform_recovers_the_original_cad_within_noise():
    vertices, faces = _tetrahedron()
    sigma_mm = 0.3
    sample = synthetic.generate_synthetic_sample(
        vertices, faces, seed=3, sigma_mm=sigma_mm
    )

    rotation = np.array(sample.sidecar["transform"]["rotation"])
    translation = np.array(sample.sidecar["transform"]["translation_mm"])
    recovered = (
        sample.vertices_mm - translation
    ) @ rotation  # rotation is orthogonal: R^-1 = R^T

    max_distance = max(_distance_to_mesh(point, vertices, faces) for point in recovered)
    assert (
        max_distance < 6 * sigma_mm
    )  # generous: a 6-sigma miss essentially never happens

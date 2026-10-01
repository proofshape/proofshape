"""Unit tests for R-08's mesh cleanup and GLB export. All synthetic data -- no GPU, no fixtures."""

from __future__ import annotations

import json

import numpy as np
import pytest

from recon import mesh_cleanup, tsdf_fusion

# A closed tetrahedron: vertices A=0,B=1,C=2,D=3; faces chosen so every edge is shared by exactly
# two faces in opposite directions (a valid, consistently-wound closed 2-manifold).
_TETRAHEDRON_VERTICES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
_TETRAHEDRON_FACES_CLOSED = np.array(
    [[0, 1, 2], [0, 3, 1], [1, 3, 2], [0, 2, 3]], dtype=np.int32
)
_TETRAHEDRON_FACES_OPEN = _TETRAHEDRON_FACES_CLOSED[
    :3
]  # drop face (0,2,3) -> one open triangle


def _pentagon_cone(radius: float = 1.0):
    """A 5-triangle cone with no base -- one closed 5-vertex boundary loop (the rim)."""
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    base = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros(5)], axis=1
    )
    vertices = np.vstack([[[0.0, 0.0, 1.0]], base]).astype(np.float32)
    faces = np.array([[0, i, (i % 5) + 1] for i in range(1, 6)], dtype=np.int32)
    return vertices, faces


def _flat_grid(n: int = 20):
    """An n x n flat grid: one huge boundary loop (the whole rectangle's rim), no interior holes."""
    xs, ys = np.meshgrid(np.linspace(0.0, 10.0, n), np.linspace(0.0, 10.0, n))
    vertices = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1).astype(
        np.float32
    )
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    return vertices, np.array(faces, dtype=np.int32)


# --- keep_largest_connected_component -----------------------------------------------------------


def test_keep_largest_connected_component_drops_disconnected_junk():
    main_vertices, main_faces = _flat_grid(n=4)  # 4x4 grid: 9 quads = 18 faces
    junk_vertices = np.array(
        [[100.0, 100.0, 100.0], [101.0, 100.0, 100.0], [100.0, 101.0, 100.0]]
    )
    vertices = np.vstack([main_vertices, junk_vertices]).astype(np.float32)
    junk_face = np.array(
        [[len(main_vertices), len(main_vertices) + 1, len(main_vertices) + 2]]
    )
    faces = np.vstack([main_faces, junk_face]).astype(np.int32)

    kept_vertices, kept_faces = mesh_cleanup.keep_largest_connected_component(
        vertices, faces
    )

    assert len(kept_faces) == len(main_faces)
    assert kept_vertices.shape[0] == len(main_vertices)
    assert kept_faces.max() < len(kept_vertices)  # re-indexed, no dangling references
    assert not np.any(
        np.all(kept_vertices[:, :2] > 50.0, axis=1)
    )  # junk corner is gone


def test_keep_largest_connected_component_rejects_bad_shape():
    with pytest.raises(ValueError, match="faces"):
        mesh_cleanup.keep_largest_connected_component(
            np.zeros((3, 3)), np.zeros((1, 4))
        )


# --- find_boundary_loops --------------------------------------------------------------------


def test_find_boundary_loops_detects_one_open_triangle():
    loops = mesh_cleanup.find_boundary_loops(_TETRAHEDRON_FACES_OPEN)

    assert len(loops) == 1
    assert sorted(loops[0]) == [0, 2, 3]


def test_find_boundary_loops_empty_for_closed_mesh():
    loops = mesh_cleanup.find_boundary_loops(_TETRAHEDRON_FACES_CLOSED)

    assert loops == []


def test_find_boundary_loops_detects_pentagon_rim():
    _vertices, faces = _pentagon_cone()

    loops = mesh_cleanup.find_boundary_loops(faces)

    assert len(loops) == 1
    assert sorted(loops[0]) == [1, 2, 3, 4, 5]


# --- fill_small_holes ------------------------------------------------------------------------


def test_fill_small_holes_closes_a_small_triangle_gap_with_no_new_vertex():
    vertices, faces = mesh_cleanup.fill_small_holes(
        _TETRAHEDRON_VERTICES, _TETRAHEDRON_FACES_OPEN, max_loop_length=8
    )

    assert len(faces) == len(_TETRAHEDRON_FACES_OPEN) + 1
    assert len(vertices) == len(
        _TETRAHEDRON_VERTICES
    )  # no centroid needed for a 3-loop
    assert mesh_cleanup.find_boundary_loops(faces) == []


def test_fill_small_holes_closes_a_pentagon_with_a_centroid_fan():
    pentagon_vertices, pentagon_faces = _pentagon_cone()

    vertices, faces = mesh_cleanup.fill_small_holes(
        pentagon_vertices, pentagon_faces, max_loop_length=8
    )

    assert len(faces) == len(pentagon_faces) + 5  # one centroid fan of 5 new triangles
    assert len(vertices) == len(pentagon_vertices) + 1  # the added centroid
    assert mesh_cleanup.find_boundary_loops(faces) == []


def test_fill_small_holes_leaves_long_loops_open():
    pentagon_vertices, pentagon_faces = _pentagon_cone()

    vertices, faces = mesh_cleanup.fill_small_holes(
        pentagon_vertices, pentagon_faces, max_loop_length=4
    )

    assert len(faces) == len(
        pentagon_faces
    )  # untouched: the rim is length 5 > max_loop_length 4
    assert len(vertices) == len(pentagon_vertices)
    assert len(mesh_cleanup.find_boundary_loops(faces)) == 1


def test_fill_small_holes_rejects_too_small_max_loop_length():
    with pytest.raises(ValueError, match="max_loop_length"):
        mesh_cleanup.fill_small_holes(_TETRAHEDRON_VERTICES, _TETRAHEDRON_FACES_OPEN, 2)


# --- decimate_mesh ---------------------------------------------------------------------------


def test_decimate_mesh_reduces_toward_target():
    vertices, faces = _flat_grid(n=20)  # 722 faces

    new_vertices, new_faces = mesh_cleanup.decimate_mesh(
        vertices, faces, target_triangle_count=50
    )

    assert len(new_faces) <= 50
    assert len(new_faces) < len(faces)
    assert new_faces.max() < len(new_vertices)


def test_decimate_mesh_is_a_no_op_when_already_under_target():
    vertices = _TETRAHEDRON_VERTICES
    faces = _TETRAHEDRON_FACES_CLOSED

    new_vertices, new_faces = mesh_cleanup.decimate_mesh(
        vertices, faces, target_triangle_count=100
    )

    np.testing.assert_array_equal(new_faces, faces)
    np.testing.assert_allclose(new_vertices, vertices)


def test_decimate_mesh_rejects_non_positive_target():
    with pytest.raises(ValueError, match="target_triangle_count"):
        mesh_cleanup.decimate_mesh(_TETRAHEDRON_VERTICES, _TETRAHEDRON_FACES_CLOSED, 0)


# --- write_glb / read_glb ---------------------------------------------------------------------


def test_glb_round_trips(tmp_path):
    path = tmp_path / "mesh.glb"

    mesh_cleanup.write_glb(path, _TETRAHEDRON_VERTICES, _TETRAHEDRON_FACES_CLOSED)
    read_vertices, read_faces, read_colors = mesh_cleanup.read_glb(path)

    np.testing.assert_allclose(read_vertices, _TETRAHEDRON_VERTICES)
    np.testing.assert_array_equal(read_faces, _TETRAHEDRON_FACES_CLOSED)
    assert read_colors is None


def test_glb_round_trips_with_colors(tmp_path):
    path = tmp_path / "mesh.glb"
    colors = np.array(
        [[255, 255, 255], [128, 128, 128], [255, 255, 255], [128, 128, 128]],
        dtype=np.uint8,
    )

    mesh_cleanup.write_glb(
        path, _TETRAHEDRON_VERTICES, _TETRAHEDRON_FACES_CLOSED, colors=colors
    )
    read_vertices, read_faces, read_colors = mesh_cleanup.read_glb(path)

    np.testing.assert_allclose(read_vertices, _TETRAHEDRON_VERTICES)
    np.testing.assert_array_equal(read_faces, _TETRAHEDRON_FACES_CLOSED)
    np.testing.assert_array_equal(read_colors, colors)


def test_write_glb_rejects_mismatched_colors_shape():
    with pytest.raises(ValueError, match="colors"):
        mesh_cleanup.write_glb(
            "unused.glb",
            _TETRAHEDRON_VERTICES,
            _TETRAHEDRON_FACES_CLOSED,
            colors=np.zeros((2, 3)),
        )


def test_write_glb_rejects_bad_shape():
    with pytest.raises(ValueError, match="faces"):
        mesh_cleanup.write_glb("unused.glb", _TETRAHEDRON_VERTICES, np.zeros((1, 4)))


# --- check_file_size ---------------------------------------------------------------------------


def test_check_file_size(tmp_path):
    path = tmp_path / "mesh.glb"
    mesh_cleanup.write_glb(path, _TETRAHEDRON_VERTICES, _TETRAHEDRON_FACES_CLOSED)
    size = path.stat().st_size

    assert mesh_cleanup.check_file_size(path, max_bytes=size)
    assert not mesh_cleanup.check_file_size(path, max_bytes=size - 1)


# --- cleanup_run: end to end on a synthetic mesh -------------------------------------------------


def test_cleanup_run_writes_glb_and_report(tmp_path):
    main_vertices, main_faces = _flat_grid(n=20)  # 722 faces, one huge boundary loop
    junk_vertices = np.array(
        [[100.0, 100.0, 100.0], [101.0, 100.0, 100.0], [100.0, 101.0, 100.0]]
    )
    vertices = np.vstack([main_vertices, junk_vertices]).astype(np.float32)
    junk_face = np.array(
        [[len(main_vertices), len(main_vertices) + 1, len(main_vertices) + 2]]
    )
    faces = np.vstack([main_faces, junk_face]).astype(np.int32)
    mesh_ply_path = tmp_path / "mesh.ply"
    tsdf_fusion.write_mesh_ply(mesh_ply_path, vertices, faces)
    out_path = tmp_path / "mesh_clean.glb"

    report = mesh_cleanup.cleanup_run(
        mesh_ply_path, out_path, target_triangle_count=100, max_hole_loop_length=8
    )

    assert report["faces_before_component_filter"] == len(faces)
    assert report["faces_after_component_filter"] == len(
        main_faces
    )  # junk triangle dropped
    assert (
        report["holes_filled"] == 0
    )  # the grid's rim (length ~76) is far above max_loop_length
    assert report["holes_left_open"] == 1
    assert report["triangle_count_after_decimation"] <= 100
    assert out_path.exists()
    assert report["file_size_bytes"] == out_path.stat().st_size
    read_vertices, read_faces, _read_colors = mesh_cleanup.read_glb(out_path)
    assert len(read_faces) == report["face_count"]
    assert len(read_vertices) == report["vertex_count"]
    on_disk_report = json.loads(out_path.with_suffix(".json").read_text())
    assert on_disk_report == report


def test_cleanup_run_refuses_to_overwrite_existing_output(tmp_path):
    mesh_ply_path = tmp_path / "mesh.ply"
    tsdf_fusion.write_mesh_ply(
        mesh_ply_path, _TETRAHEDRON_VERTICES, _TETRAHEDRON_FACES_CLOSED
    )
    out_path = tmp_path / "mesh_clean.glb"

    mesh_cleanup.cleanup_run(mesh_ply_path, out_path)

    with pytest.raises(FileExistsError, match="already exists"):
        mesh_cleanup.cleanup_run(mesh_ply_path, out_path)

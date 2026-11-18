"""Unit tests for R-08's mesh cleanup and GLB export. All synthetic data -- no GPU, no fixtures."""

from __future__ import annotations

import json
import multiprocessing

import numpy as np
import pygltflib
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

# A minimal non-orientable manifold: 5 vertices on a ring, each face (i, i+1, i+2 mod 5). Every
# edge has exactly 2 owners (manifold -- drop_non_manifold_faces leaves it untouched), but the
# winding can't be made consistent by any combination of per-face flips (confirmed by hand
# against _find_orientation_conflict while writing this fix). Vertex positions are arbitrary --
# this is a purely combinatorial defect, not a geometric one.
_NON_ORIENTABLE_RING_VERTICES = np.array(
    [[np.cos(2 * np.pi * i / 5), np.sin(2 * np.pi * i / 5), 0.0] for i in range(5)],
    dtype=np.float32,
)
_NON_ORIENTABLE_RING_FACES = np.array(
    [[i, (i + 1) % 5, (i + 2) % 5] for i in range(5)], dtype=np.int32
)


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


# --- drop_non_manifold_faces (R-30, review on #270) ----------------------------------------------


def test_drop_non_manifold_faces_removes_every_face_on_the_bad_edge_only():
    # Add a fifth vertex and a fourth face reusing edge (0,1), already owned by two of the
    # tetrahedron's own faces -- now 3 faces share it, which is what R-30's re-anchoring left on
    # the real s-03 mesh at one part/board seam.
    vertices = np.vstack([_TETRAHEDRON_VERTICES, [[2.0, 2.0, 2.0]]]).astype(np.float32)
    extra_face = np.array([[0, 1, 4]], dtype=np.int32)
    faces = np.vstack([_TETRAHEDRON_FACES_CLOSED, extra_face]).astype(np.int32)

    kept_vertices, kept_faces, dropped_count = mesh_cleanup.drop_non_manifold_faces(
        vertices, faces
    )

    assert (
        dropped_count == 3
    )  # the two original faces on edge (0,1), plus the extra one
    np.testing.assert_array_equal(kept_vertices, vertices)  # vertices are left as-is
    remaining = {tuple(sorted(int(i) for i in face)) for face in kept_faces}
    assert remaining == {(1, 2, 3), (0, 2, 3)}


def test_drop_non_manifold_faces_leaves_a_manifold_mesh_untouched():
    kept_vertices, kept_faces, dropped_count = mesh_cleanup.drop_non_manifold_faces(
        _TETRAHEDRON_VERTICES, _TETRAHEDRON_FACES_CLOSED
    )

    assert dropped_count == 0
    np.testing.assert_array_equal(kept_faces, _TETRAHEDRON_FACES_CLOSED)
    np.testing.assert_array_equal(kept_vertices, _TETRAHEDRON_VERTICES)


def test_drop_non_manifold_faces_rejects_bad_shape():
    with pytest.raises(ValueError, match="faces"):
        mesh_cleanup.drop_non_manifold_faces(np.zeros((3, 3)), np.zeros((1, 4)))


# --- orientation (#300's other half, found on review @TabeenRaoof) ------------------------------


def test_find_orientation_conflict_detects_the_non_orientable_ring():
    conflict = mesh_cleanup._find_orientation_conflict(_NON_ORIENTABLE_RING_FACES)

    assert conflict is not None


def test_find_orientation_conflict_is_none_for_a_closed_manifold():
    assert mesh_cleanup._find_orientation_conflict(_TETRAHEDRON_FACES_CLOSED) is None
    assert mesh_cleanup._find_orientation_conflict(_TETRAHEDRON_FACES_OPEN) is None


def test_drop_faces_until_orientable_resolves_the_non_orientable_ring():
    _kept_vertices, kept_faces, dropped_count = (
        mesh_cleanup.drop_faces_until_orientable(
            _NON_ORIENTABLE_RING_VERTICES, _NON_ORIENTABLE_RING_FACES
        )
    )

    assert (
        dropped_count == 2
    )  # the ring resolves by dropping exactly one conflicting pair
    assert len(kept_faces) == len(_NON_ORIENTABLE_RING_FACES) - 2
    assert mesh_cleanup._find_orientation_conflict(kept_faces) is None
    assert _max_edge_owners(kept_faces) <= 2


def test_drop_faces_until_orientable_leaves_an_already_orientable_mesh_untouched():
    kept_vertices, kept_faces, dropped_count = mesh_cleanup.drop_faces_until_orientable(
        _TETRAHEDRON_VERTICES, _TETRAHEDRON_FACES_CLOSED
    )

    assert dropped_count == 0
    np.testing.assert_array_equal(kept_faces, _TETRAHEDRON_FACES_CLOSED)
    np.testing.assert_array_equal(kept_vertices, _TETRAHEDRON_VERTICES)


def test_drop_faces_until_orientable_raises_once_past_the_dropped_fraction_cap():
    # Found on review (@TabeenRaoof, PR #381): two independent non-orientable rings need two
    # separate drops (4 faces of 10 total); a 10% cap only allows one.
    second_ring_offset = len(_NON_ORIENTABLE_RING_VERTICES)
    vertices = np.vstack(
        [
            _NON_ORIENTABLE_RING_VERTICES,
            _NON_ORIENTABLE_RING_VERTICES + np.array([5.0, 0.0, 0.0]),
        ]
    )
    faces = np.vstack(
        [
            _NON_ORIENTABLE_RING_FACES,
            _NON_ORIENTABLE_RING_FACES + second_ring_offset,
        ]
    )

    with pytest.raises(ValueError, match="faces dropped"):
        mesh_cleanup.drop_faces_until_orientable(
            vertices, faces, max_dropped_fraction=0.1
        )


def test_drop_faces_until_orientable_rejects_non_positive_fraction():
    with pytest.raises(ValueError, match="max_dropped_fraction"):
        mesh_cleanup.drop_faces_until_orientable(
            _TETRAHEDRON_VERTICES, _TETRAHEDRON_FACES_CLOSED, max_dropped_fraction=0.0
        )


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


def test_filter_connected_components_keeps_legitimate_second_piece_and_reports_junk():
    main_vertices, main_faces = _flat_grid(n=4)  # 18 faces
    second_vertices, second_faces = _flat_grid(n=3)  # 8 faces
    second_vertices = second_vertices + np.array([30.0, 0.0, 0.0], dtype=np.float32)
    second_faces = second_faces + len(main_vertices)

    junk_vertices = np.array(
        [[100.0, 100.0, 100.0], [101.0, 100.0, 100.0], [100.0, 101.0, 100.0]],
        dtype=np.float32,
    )
    junk_offset = len(main_vertices) + len(second_vertices)
    junk_faces = np.array(
        [[junk_offset, junk_offset + 1, junk_offset + 2]], dtype=np.int32
    )

    vertices = np.vstack([main_vertices, second_vertices, junk_vertices])
    faces = np.vstack([main_faces, second_faces, junk_faces])

    kept_vertices, kept_faces, report = mesh_cleanup.filter_connected_components(
        vertices,
        faces,
        min_face_fraction=0.4,
    )

    assert report["component_count_before"] == 3
    assert report["component_count_after"] == 2
    assert report["dropped_component_count"] == 1
    assert report["dropped_face_count"] == 1
    assert len(kept_faces) == len(main_faces) + len(second_faces)
    assert np.max(kept_vertices[:, 0]) < 100.0


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


def _assert_no_directed_edge_is_reused(faces):
    directed_edges = []
    for triangle in faces:
        for i in range(3):
            directed_edges.append((int(triangle[i]), int(triangle[(i + 1) % 3])))
    assert len(directed_edges) == len(set(directed_edges))


def test_fill_small_holes_preserves_surrounding_winding():
    _vertices, tetra_faces = mesh_cleanup.fill_small_holes(
        _TETRAHEDRON_VERTICES,
        _TETRAHEDRON_FACES_OPEN,
        max_loop_length=8,
    )
    _assert_no_directed_edge_is_reused(tetra_faces)

    pentagon_vertices, pentagon_faces = _pentagon_cone()
    _vertices, filled_pentagon_faces = mesh_cleanup.fill_small_holes(
        pentagon_vertices,
        pentagon_faces,
        max_loop_length=8,
    )
    _assert_no_directed_edge_is_reused(filled_pentagon_faces)


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


def test_write_glb_custom_float_attributes_are_aligned_for_odd_vertex_count(tmp_path):
    path = tmp_path / "odd.glb"
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 1]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]], dtype=np.int32)
    colors = np.array(
        [
            [255, 255, 255],
            [128, 128, 128],
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
        ],
        dtype=np.uint8,
    )
    deviation = np.array([-2.0, 0.0, 1.0, 0.0, 3.0], dtype=np.float32)
    measured = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    mesh_cleanup.write_glb(
        path,
        vertices,
        faces,
        colors=colors,
        extra_vertex_attributes={
            "_DEVIATION_MM": deviation,
            "_DEVIATION_MEASURED": measured,
        },
    )

    document = pygltflib.GLTF2().load_binary(str(path))
    attrs = document.meshes[0].primitives[0].attributes
    for name, expected in (
        ("_DEVIATION_MM", deviation),
        ("_DEVIATION_MEASURED", measured),
    ):
        accessor_index = getattr(attrs, name)
        accessor = document.accessors[accessor_index]
        view = document.bufferViews[accessor.bufferView]
        assert view.byteOffset % 4 == 0
        actual = np.frombuffer(
            document.binary_blob(),
            dtype=np.float32,
            count=accessor.count,
            offset=view.byteOffset,
        )
        np.testing.assert_allclose(actual, expected)


def test_write_glb_rejects_bad_custom_attributes(tmp_path):
    with pytest.raises(ValueError, match="must start"):
        mesh_cleanup.write_glb(
            tmp_path / "bad-name.glb",
            _TETRAHEDRON_VERTICES,
            _TETRAHEDRON_FACES_CLOSED,
            extra_vertex_attributes={"DEVIATION_MM": np.zeros(4)},
        )

    with pytest.raises(ValueError, match="one scalar per vertex"):
        mesh_cleanup.write_glb(
            tmp_path / "bad-shape.glb",
            _TETRAHEDRON_VERTICES,
            _TETRAHEDRON_FACES_CLOSED,
            extra_vertex_attributes={"_DEVIATION_MM": np.zeros(3)},
        )

    bad = np.zeros(4, dtype=np.float32)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        mesh_cleanup.write_glb(
            tmp_path / "nan.glb",
            _TETRAHEDRON_VERTICES,
            _TETRAHEDRON_FACES_CLOSED,
            extra_vertex_attributes={"_DEVIATION_MM": bad},
        )


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


def test_write_glb_with_size_limit_redecimates_until_under_limit(tmp_path):
    vertices, faces = _flat_grid(n=20)
    baseline = tmp_path / "baseline.glb"
    mesh_cleanup.write_glb(baseline, vertices, faces)
    baseline_size = baseline.stat().st_size

    out = tmp_path / "limited.glb"
    limited_vertices, limited_faces, size_bytes, rounds, dropped, oriented_dropped = (
        mesh_cleanup.write_glb_with_size_limit(
            out,
            vertices,
            faces,
            max_file_size_bytes=int(baseline_size * 0.75),
        )
    )

    assert rounds >= 1
    assert size_bytes <= int(baseline_size * 0.75)
    assert len(limited_faces) < len(faces)
    assert len(limited_vertices) <= len(vertices)
    assert dropped == 0  # a plain flat grid never goes non-manifold under decimation
    assert oriented_dropped == 0  # nor non-orientable


def test_write_glb_with_size_limit_raises_when_limit_is_impossible(tmp_path):
    with pytest.raises(ValueError, match="over the 1-byte limit"):
        mesh_cleanup.write_glb_with_size_limit(
            tmp_path / "too-small.glb",
            _TETRAHEDRON_VERTICES,
            _TETRAHEDRON_FACES_CLOSED,
            max_file_size_bytes=1,
            min_triangle_count=len(_TETRAHEDRON_FACES_CLOSED),
        )


def _edge_owner_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for triangle in np.asarray(faces):
        for i in range(3):
            a, b = int(triangle[i]), int(triangle[(i + 1) % 3])
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _max_edge_owners(faces: np.ndarray) -> int:
    """The largest number of faces sharing any one undirected edge -- >2 means non-manifold."""
    return max(_edge_owner_counts(faces).values(), default=0)


def _run_stalled_decimation_in_subprocess(
    path_str: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    max_file_size_bytes: int,
    result_queue,
) -> None:
    """Module-level (picklable) subprocess target for the #263 regression below.

    Patches decimate_mesh *inside this process* -- pytest's monkeypatch fixture only touches
    the parent process's copy of the module, which a spawned child never sees. Reports the
    outcome through result_queue rather than letting a real infinite loop hang this process
    with no way for the parent to tell what happened.
    """
    from pathlib import Path

    from recon import mesh_cleanup

    def _stalled_decimate(vertices, faces, target_triangle_count):
        return vertices, faces

    mesh_cleanup.decimate_mesh = _stalled_decimate
    try:
        mesh_cleanup.write_glb_with_size_limit(
            Path(path_str), vertices, faces, max_file_size_bytes=max_file_size_bytes
        )
        result_queue.put(("ok", None))
    except ValueError as exc:
        result_queue.put(("value_error", str(exc)))
    except Exception as exc:  # noqa: BLE001 -- defensive: report any crash, don't re-raise
        result_queue.put(("other_error", str(exc)))


def test_write_glb_with_size_limit_raises_instead_of_looping_forever_on_a_stalled_decimation(
    tmp_path,
):
    # #263 reproduction: decimate_mesh that never reduces the face count would otherwise make
    # the while-True loop in write_glb_with_size_limit spin forever. Run it in a real OS
    # process, not a thread: a thread stuck in an infinite loop cannot be killed, and
    # concurrent.futures joins its worker threads at interpreter exit regardless of any
    # future.result() timeout, so a regression there would still hang the whole test run. A
    # process can be terminated outright, so a regression here fails this one test instead.
    vertices, faces = _flat_grid(n=20)
    out = tmp_path / "stalled.glb"
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_stalled_decimation_in_subprocess,
        args=(str(out), vertices, faces, 1, result_queue),
    )
    process.start()
    process.join(timeout=5.0)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail(
            "write_glb_with_size_limit looped past the 5s timeout -- #263 regression"
        )

    assert not result_queue.empty(), "subprocess exited without reporting an outcome"
    outcome, message = result_queue.get()
    assert outcome == "value_error", (outcome, message)
    assert "cannot be decimated further" in message


def test_write_glb_with_size_limit_restores_manifoldness_after_each_round(
    tmp_path, monkeypatch
):
    # #300 reproduction, built synthetically (the story's own Tests section allows this):
    # decimate_mesh here stands in for fast_simplification actually doing this on real meshes.
    # It still decimates for real, then injects one extra face onto an already-shared edge --
    # pushing that edge from 2 owners to 3, the same shape of damage a real edge collapse can
    # cause -- while leaving the rest of the mesh untouched, so the drop afterward only has to
    # remove a handful of faces, not the whole mesh.
    vertices, faces = _flat_grid(n=20)
    baseline = tmp_path / "baseline.glb"
    mesh_cleanup.write_glb(baseline, vertices, faces)
    baseline_size = baseline.stat().st_size
    real_decimate_mesh = mesh_cleanup.decimate_mesh

    def _decimate_then_add_a_third_owner_on_a_shared_edge(
        vertices, faces, target_triangle_count
    ):
        new_vertices, new_faces = real_decimate_mesh(
            vertices, faces, target_triangle_count
        )
        shared_edge = next(
            edge for edge, count in _edge_owner_counts(new_faces).items() if count == 2
        )
        a, b = shared_edge
        third_vertex = int(new_faces[0][0])
        if third_vertex in (a, b):
            third_vertex = int(new_faces[0][1])
        extra_face = np.array([[a, b, third_vertex]], dtype=new_faces.dtype)
        corrupted_faces = np.vstack([new_faces, extra_face])
        assert _max_edge_owners(corrupted_faces) >= 3  # confirms the repro is real
        return new_vertices, corrupted_faces

    monkeypatch.setattr(
        mesh_cleanup, "decimate_mesh", _decimate_then_add_a_third_owner_on_a_shared_edge
    )
    out = tmp_path / "cleaned.glb"

    _, result_faces, size_bytes, rounds, dropped, _oriented_dropped = (
        mesh_cleanup.write_glb_with_size_limit(
            out, vertices, faces, max_file_size_bytes=int(baseline_size * 0.75)
        )
    )

    assert rounds >= 1
    assert size_bytes <= int(baseline_size * 0.75)
    assert _max_edge_owners(result_faces) <= 2
    assert dropped > 0  # confirms the per-round non-manifold drop actually ran


def test_write_glb_with_size_limit_restores_orientability_after_each_round(
    tmp_path, monkeypatch
):
    # Found on review (@TabeenRaoof, PR #381), on R-32's real MASt3R s-05 "all" run: decimation
    # can leave an already-manifold mesh non-orientable. Reproduced synthetically by decimating
    # for real, then grafting the non-orientable ring onto 5 brand-new vertices -- isolated from
    # the rest of the mesh, the same way the existing non-manifold-injection test above adds
    # disconnected junk rather than reusing real grid vertices.
    vertices, faces = _flat_grid(n=20)
    baseline = tmp_path / "baseline.glb"
    mesh_cleanup.write_glb(baseline, vertices, faces)
    baseline_size = baseline.stat().st_size
    real_decimate_mesh = mesh_cleanup.decimate_mesh

    def _decimate_then_graft_a_non_orientable_ring(
        vertices, faces, target_triangle_count
    ):
        new_vertices, new_faces = real_decimate_mesh(
            vertices, faces, target_triangle_count
        )
        ring_vertex_offset = len(new_vertices)
        grafted_vertices = np.vstack(
            [new_vertices, _NON_ORIENTABLE_RING_VERTICES + np.array([0, 0, 100.0])]
        )
        grafted_ring_faces = _NON_ORIENTABLE_RING_FACES + ring_vertex_offset
        grafted_faces = np.vstack([new_faces, grafted_ring_faces]).astype(
            new_faces.dtype
        )
        assert mesh_cleanup._find_orientation_conflict(grafted_faces) is not None
        return grafted_vertices.astype(np.float32), grafted_faces

    monkeypatch.setattr(
        mesh_cleanup, "decimate_mesh", _decimate_then_graft_a_non_orientable_ring
    )
    out = tmp_path / "cleaned.glb"

    _, result_faces, size_bytes, rounds, _dropped, oriented_dropped = (
        mesh_cleanup.write_glb_with_size_limit(
            out, vertices, faces, max_file_size_bytes=int(baseline_size * 0.75)
        )
    )

    assert rounds >= 1
    assert size_bytes <= int(baseline_size * 0.75)
    assert oriented_dropped > 0
    assert mesh_cleanup._find_orientation_conflict(result_faces) is None


def test_mesh_cleanup_cli_returns_nonzero_for_impossible_size_limit(tmp_path, capsys):
    mesh_ply_path = tmp_path / "mesh.ply"
    out_path = tmp_path / "mesh_clean.glb"
    tsdf_fusion.write_mesh_ply(
        mesh_ply_path,
        _TETRAHEDRON_VERTICES,
        _TETRAHEDRON_FACES_CLOSED,
    )

    exit_code = mesh_cleanup.main(
        [
            str(mesh_ply_path),
            "--out",
            str(out_path),
            "--max-file-size-bytes",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL" in captured.err
    assert "PASS" not in captured.out


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


def test_cleanup_run_restores_manifoldness_after_its_own_decimation(
    tmp_path, monkeypatch
):
    # #300 reproduction at cleanup_run's own decimate_mesh call site (line 679's original, not
    # write_glb_with_size_limit's re-decimation loop), built synthetically the same way as the
    # write_glb_with_size_limit case above.
    vertices, faces = _flat_grid(n=20)
    mesh_ply_path = tmp_path / "mesh.ply"
    tsdf_fusion.write_mesh_ply(mesh_ply_path, vertices, faces)
    out_path = tmp_path / "mesh_clean.glb"
    real_decimate_mesh = mesh_cleanup.decimate_mesh

    def _decimate_then_add_a_third_owner_on_a_shared_edge(
        vertices, faces, target_triangle_count
    ):
        new_vertices, new_faces = real_decimate_mesh(
            vertices, faces, target_triangle_count
        )
        shared_edge = next(
            edge for edge, count in _edge_owner_counts(new_faces).items() if count == 2
        )
        a, b = shared_edge
        third_vertex = int(new_faces[0][0])
        if third_vertex in (a, b):
            third_vertex = int(new_faces[0][1])
        extra_face = np.array([[a, b, third_vertex]], dtype=new_faces.dtype)
        corrupted_faces = np.vstack([new_faces, extra_face])
        assert _max_edge_owners(corrupted_faces) >= 3  # confirms the repro is real
        return new_vertices, corrupted_faces

    monkeypatch.setattr(
        mesh_cleanup, "decimate_mesh", _decimate_then_add_a_third_owner_on_a_shared_edge
    )

    report = mesh_cleanup.cleanup_run(
        mesh_ply_path, out_path, target_triangle_count=100, max_hole_loop_length=8
    )

    assert report["non_manifold_faces_dropped_after_decimation"] > 0
    # the corruption only happens at cleanup_run's own decimate_mesh call in this test, not
    # inside write_glb_with_size_limit's re-decimation loop -- kept apart on purpose (see
    # write_cleanup_report), not summed into the field above.
    assert report["non_manifold_faces_dropped_during_size_redecimation"] == 0
    _read_vertices, read_faces, _read_colors = mesh_cleanup.read_glb(out_path)
    assert _max_edge_owners(read_faces) <= 2
    assert len(read_faces) == report["face_count"]


def test_cleanup_run_restores_orientability_after_its_own_decimation(
    tmp_path, monkeypatch
):
    # Same case as the write_glb_with_size_limit orientation test above, at cleanup_run's own
    # decimate_mesh call site.
    vertices, faces = _flat_grid(n=20)
    mesh_ply_path = tmp_path / "mesh.ply"
    tsdf_fusion.write_mesh_ply(mesh_ply_path, vertices, faces)
    out_path = tmp_path / "mesh_clean.glb"
    real_decimate_mesh = mesh_cleanup.decimate_mesh

    def _decimate_then_graft_a_non_orientable_ring(
        vertices, faces, target_triangle_count
    ):
        new_vertices, new_faces = real_decimate_mesh(
            vertices, faces, target_triangle_count
        )
        ring_vertex_offset = len(new_vertices)
        grafted_vertices = np.vstack(
            [new_vertices, _NON_ORIENTABLE_RING_VERTICES + np.array([0, 0, 100.0])]
        )
        grafted_ring_faces = _NON_ORIENTABLE_RING_FACES + ring_vertex_offset
        grafted_faces = np.vstack([new_faces, grafted_ring_faces]).astype(
            new_faces.dtype
        )
        assert mesh_cleanup._find_orientation_conflict(grafted_faces) is not None
        return grafted_vertices.astype(np.float32), grafted_faces

    monkeypatch.setattr(
        mesh_cleanup, "decimate_mesh", _decimate_then_graft_a_non_orientable_ring
    )

    report = mesh_cleanup.cleanup_run(
        mesh_ply_path, out_path, target_triangle_count=100, max_hole_loop_length=8
    )

    assert report["non_orientable_faces_dropped_after_decimation"] > 0
    assert report["non_orientable_faces_dropped_during_size_redecimation"] == 0
    _read_vertices, read_faces, _read_colors = mesh_cleanup.read_glb(out_path)
    assert mesh_cleanup._find_orientation_conflict(read_faces) is None
    assert len(read_faces) == report["face_count"]


def test_cleanup_run_refuses_to_overwrite_existing_output(tmp_path):
    mesh_ply_path = tmp_path / "mesh.ply"
    tsdf_fusion.write_mesh_ply(
        mesh_ply_path, _TETRAHEDRON_VERTICES, _TETRAHEDRON_FACES_CLOSED
    )
    out_path = tmp_path / "mesh_clean.glb"

    mesh_cleanup.cleanup_run(mesh_ply_path, out_path)

    with pytest.raises(FileExistsError, match="already exists"):
        mesh_cleanup.cleanup_run(mesh_ply_path, out_path)

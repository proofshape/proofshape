from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pygltflib
import pytest

from inspection.cad import load_mesh
from recon.mesh_cleanup import write_glb

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "reference_cad"


@pytest.mark.parametrize(
    ("filename", "expected_vertices", "expected_faces"),
    [
        ("s04_triangular_prism_60-26-65x30mm.stl", 6, 8),
        ("s05_rect_prism_30x30x60mm.stl", 8, 12),
    ],
)
def test_load_ascii_stl_fixtures(filename, expected_vertices, expected_faces):
    vertices, faces, units = load_mesh(FIXTURES / filename)

    assert vertices.shape == (expected_vertices, 3)
    assert faces.shape == (expected_faces, 3)
    assert units == "mm"
    assert np.isfinite(vertices).all()
    assert faces.min() >= 0
    assert faces.max() < len(vertices)


def _write_binary_stl(path: Path) -> np.ndarray:
    triangles = np.array(
        [
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        ],
        dtype=np.float32,
    )
    raw = bytearray(b"binary proofshape test".ljust(80, b" "))
    raw.extend(struct.pack("<I", len(triangles)))
    for triangle in triangles:
        raw.extend(np.zeros(3, dtype="<f4").tobytes())
        raw.extend(triangle.astype("<f4").tobytes())
        raw.extend(struct.pack("<H", 0))
    path.write_bytes(raw)
    return triangles


def test_load_binary_stl_deduplicates_triangle_soup(tmp_path):
    path = tmp_path / "mesh.stl"
    triangles = _write_binary_stl(path)

    vertices, faces, units = load_mesh(path)

    assert units == "mm"
    assert vertices.shape == (4, 3)
    assert faces.shape == (2, 3)
    reconstructed = vertices[faces]
    np.testing.assert_allclose(reconstructed, triangles)


def test_binary_stl_header_may_begin_with_solid(tmp_path):
    path = tmp_path / "mesh.stl"
    triangles = _write_binary_stl(path)
    raw = bytearray(path.read_bytes())
    raw[:5] = b"solid"
    path.write_bytes(raw)

    vertices, faces, _units = load_mesh(path)

    np.testing.assert_allclose(vertices[faces], triangles)


def test_write_glb_output_round_trips_coordinates_unchanged(tmp_path):
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 3, 1]], dtype=np.int32)
    path = tmp_path / "mesh.glb"
    write_glb(path, vertices, faces)

    loaded_vertices, loaded_faces, units = load_mesh(path)

    assert units == "assumed_mm"
    np.testing.assert_allclose(loaded_vertices[loaded_faces], vertices[faces])


def test_explicit_scale_is_the_only_coordinate_conversion(tmp_path):
    path = tmp_path / "mesh.glb"
    vertices = np.array(
        [[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 0.02, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    write_glb(path, vertices, faces)

    stored, _, units = load_mesh(path)
    converted, _, converted_units = load_mesh(path, scale=1000.0)

    assert units == converted_units == "assumed_mm"
    np.testing.assert_allclose(stored, vertices)
    np.testing.assert_allclose(converted, vertices * 1000.0)


def test_nonidentity_glb_node_scale_is_rejected(tmp_path):
    path = tmp_path / "mesh.glb"
    vertices = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
        dtype=np.float32,
    )
    write_glb(path, vertices, np.array([[0, 1, 2]], dtype=np.int32))

    document = pygltflib.GLTF2().load_binary(str(path))
    document.nodes[0].scale = [2.0, 1.0, 1.0]
    document.save_binary(str(path))

    with pytest.raises(ValueError, match="non-identity node scale"):
        load_mesh(path)


@pytest.mark.parametrize("suffix", [".obj", ".ply", ".gltf", ""])
def test_unsupported_formats_name_the_three_supported_inputs(tmp_path, suffix):
    path = tmp_path / ("mesh" + suffix)
    path.write_bytes(b"not a supported mesh")

    with pytest.raises(ValueError) as exc_info:
        load_mesh(path)

    message = str(exc_info.value)
    assert "ASCII STL" in message
    assert "binary STL" in message
    assert "GLB" in message


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf])
def test_scale_must_be_finite_and_positive(tmp_path, scale):
    path = tmp_path / "mesh.stl"
    path.write_text(
        """solid x
facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet
endsolid x
""",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="finite positive"):
        load_mesh(path, scale=scale)

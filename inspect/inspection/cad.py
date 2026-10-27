"""I-02/I-11: load buyer CAD meshes without silently changing their coordinate scale.

Neither STL nor OBJ has unit metadata, so ProofShape treats both as millimetres by project
convention. glTF/GLB nominally uses metres, but ProofShape's own reconstruction GLBs store raw
millimetres; I-02 therefore reads GLB coordinates exactly as stored and reports ``assumed_mm``.
A caller that knows an input needs conversion must pass an explicit ``scale``.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pygltflib

from recon.cad_size_check import read_ascii_stl

_STL_UNITS = "mm"
_OBJ_UNITS = "mm"
_GLB_UNITS = "assumed_mm"
_SUPPORTED_MESSAGE = "Supported formats are ASCII STL, binary STL, OBJ, and GLB."

_GLTF_COMPONENT_DTYPES: dict[int, np.dtype] = {
    pygltflib.BYTE: np.dtype(np.int8),
    pygltflib.UNSIGNED_BYTE: np.dtype(np.uint8),
    pygltflib.SHORT: np.dtype("<i2"),
    pygltflib.UNSIGNED_SHORT: np.dtype("<u2"),
    pygltflib.UNSIGNED_INT: np.dtype("<u4"),
    pygltflib.FLOAT: np.dtype("<f4"),
}
_GLTF_TYPE_WIDTH = {
    pygltflib.SCALAR: 1,
    pygltflib.VEC2: 2,
    pygltflib.VEC3: 3,
    pygltflib.VEC4: 4,
}


def _validate_scale(scale: float) -> float:
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite positive number.")
    return scale


def _deduplicate_triangle_soup(
    triangle_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn consecutive triangle vertices into an indexed mesh, preserving first-seen order."""
    triangle_vertices = np.asarray(triangle_vertices, dtype=np.float64)
    if (
        triangle_vertices.ndim != 2
        or triangle_vertices.shape[1] != 3
        or len(triangle_vertices) == 0
        or len(triangle_vertices) % 3 != 0
    ):
        raise ValueError(
            "Triangle soup must contain a non-empty multiple of three (x, y, z) vertices."
        )
    if not np.isfinite(triangle_vertices).all():
        raise ValueError("CAD vertices must all be finite.")

    index_by_vertex: dict[tuple[float, float, float], int] = {}
    vertices: list[np.ndarray] = []
    indices: list[int] = []
    for vertex in triangle_vertices:
        key = (float(vertex[0]), float(vertex[1]), float(vertex[2]))
        index = index_by_vertex.get(key)
        if index is None:
            index = len(vertices)
            index_by_vertex[key] = index
            vertices.append(vertex.copy())
        indices.append(index)

    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(indices, dtype=np.int32).reshape(-1, 3),
    )


def _read_binary_stl(path: Path, raw: bytes) -> np.ndarray:
    if len(raw) < 84:
        raise ValueError(f"{path} is too short to be a binary STL.")
    triangle_count = struct.unpack_from("<I", raw, 80)[0]
    expected_size = 84 + triangle_count * 50
    if expected_size != len(raw):
        raise ValueError(
            f"{path} has a binary-STL triangle count of {triangle_count}, which requires "
            f"{expected_size} bytes; file has {len(raw)} bytes."
        )

    vertices = np.empty((triangle_count * 3, 3), dtype=np.float64)
    offset = 84
    for triangle_index in range(triangle_count):
        triangle = np.frombuffer(raw, dtype="<f4", count=12, offset=offset)
        vertices[triangle_index * 3 : triangle_index * 3 + 3] = triangle[3:12].reshape(
            3, 3
        )
        offset += 50
    if triangle_count == 0:
        raise ValueError(f"{path} contains no triangles.")
    return vertices


def _load_stl(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    raw = path.read_bytes()

    is_binary = False
    if len(raw) >= 84:
        triangle_count = struct.unpack_from("<I", raw, 80)[0]
        is_binary = 84 + triangle_count * 50 == len(raw)

    if is_binary:
        soup = _read_binary_stl(path, raw)
    else:
        try:
            soup = read_ascii_stl(path)
        except ValueError as exc:
            raise ValueError(
                f"Could not read {path} as STL. {_SUPPORTED_MESSAGE}"
            ) from exc

    vertices, faces = _deduplicate_triangle_soup(soup)
    return vertices, faces, _STL_UNITS


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"Could not read {path} as OBJ. {_SUPPORTED_MESSAGE}") from exc

    positions: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tag, _, rest = line.partition(" ")
        rest = rest.strip()

        if tag == "v":
            parts = rest.split()
            if len(parts) < 3:
                raise ValueError(
                    f"{path}:{line_number}: OBJ 'v' line needs at least 3 coordinates."
                )
            try:
                positions.append(tuple(float(value) for value in parts[:3]))
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: OBJ 'v' line has a non-numeric coordinate."
                ) from exc
        elif tag == "f":
            corners = rest.split()
            if len(corners) < 3:
                raise ValueError(
                    f"{path}:{line_number}: OBJ 'f' line needs at least 3 vertices."
                )
            face_indices: list[int] = []
            for corner in corners:
                # `v`, `v/vt`, `v/vt/vn`, and `v//vn` all name the vertex in the first slot;
                # only that index is used (I-11: vt/vn are parsed past, not acted on).
                vertex_token = corner.split("/", 1)[0]
                try:
                    vertex_index = int(vertex_token)
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: OBJ 'f' line has a non-integer vertex index."
                    ) from exc
                if vertex_index <= 0:
                    raise ValueError(
                        f"{path}:{line_number}: negative or relative OBJ face indices are "
                        "not supported."
                    )
                face_indices.append(vertex_index - 1)
            # Fan triangulation from the face's first vertex (I-11: adequate for this
            # project's convex prism-shaped specimens, not a general polygon triangulator).
            for i in range(1, len(face_indices) - 1):
                triangles.append(
                    (face_indices[0], face_indices[i], face_indices[i + 1])
                )
        # vn, vt, g, o, mtllib, usemtl, and any other OBJ directive: parsed past, not acted on.

    if not positions or not triangles:
        raise ValueError(
            f"{path} contains no usable OBJ geometry. {_SUPPORTED_MESSAGE}"
        )

    vertex_array = np.asarray(positions, dtype=np.float64)
    if not np.isfinite(vertex_array).all():
        raise ValueError(f"{path} has a non-finite OBJ vertex coordinate.")

    face_array = np.asarray(triangles, dtype=np.int64)
    if face_array.min() < 0 or face_array.max() >= len(vertex_array):
        raise ValueError(f"{path} has an OBJ face index outside the vertex list.")

    soup = vertex_array[face_array.reshape(-1)]
    vertices, faces = _deduplicate_triangle_soup(soup)
    return vertices, faces, _OBJ_UNITS


def _accessor_array(
    document: pygltflib.GLTF2, accessor_index: int, blob: bytes
) -> np.ndarray:
    accessor = document.accessors[accessor_index]
    if accessor.sparse is not None:
        raise ValueError("Sparse GLB accessors are not supported for CAD loading.")
    if accessor.bufferView is None:
        raise ValueError("GLB accessor has no bufferView.")

    view = document.bufferViews[accessor.bufferView]
    if view.buffer != 0:
        raise ValueError("GLB CAD must keep mesh data in the binary buffer.")

    dtype = _GLTF_COMPONENT_DTYPES.get(accessor.componentType)
    width = _GLTF_TYPE_WIDTH.get(accessor.type)
    if dtype is None or width is None:
        raise ValueError(
            f"Unsupported GLB accessor component/type: {accessor.componentType}/{accessor.type}."
        )

    element_bytes = dtype.itemsize * width
    stride = view.byteStride or element_bytes
    if stride != element_bytes:
        raise ValueError(
            "Interleaved GLB vertex buffers are not supported for CAD loading."
        )

    offset = int(view.byteOffset or 0) + int(accessor.byteOffset or 0)
    count = int(accessor.count)
    array = np.frombuffer(blob, dtype=dtype, count=count * width, offset=offset)
    if array.size != count * width:
        raise ValueError("GLB accessor points outside the binary buffer.")
    if width == 1:
        return array.copy()
    return array.reshape(count, width).copy()


def _node_has_nonidentity_scale(node: pygltflib.Node) -> bool:
    if node.scale is not None:
        scale = np.asarray(node.scale, dtype=np.float64)
        if scale.shape != (3,) or not np.allclose(
            scale, np.ones(3), atol=0.0, rtol=0.0
        ):
            return True

    if node.matrix is not None:
        matrix_values = np.asarray(node.matrix, dtype=np.float64)
        if matrix_values.shape != (16,):
            return True
        # glTF stores matrices column-major. The upper-left 3x3 must be a proper rotation:
        # orthonormal with determinant +1. That rejects scale, shear and reflection instead of
        # silently dropping an authored handedness/shape change.
        matrix = matrix_values.reshape(4, 4, order="F")
        basis = matrix[:3, :3]
        if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-9, rtol=1e-9):
            return True
        if not np.isclose(np.linalg.det(basis), 1.0, atol=1e-9, rtol=1e-9):
            return True

    return False


def _node_has_nonidentity_transform(node: pygltflib.Node) -> bool:
    if node.matrix is not None:
        matrix_values = np.asarray(node.matrix, dtype=np.float64)
        if matrix_values.shape != (16,):
            return True
        matrix = matrix_values.reshape(4, 4, order="F")
        return not np.allclose(matrix, np.eye(4), atol=1e-9, rtol=1e-9)

    translation = (
        np.zeros(3, dtype=np.float64)
        if node.translation is None
        else np.asarray(node.translation, dtype=np.float64)
    )
    rotation = (
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        if node.rotation is None
        else np.asarray(node.rotation, dtype=np.float64)
    )
    scale = (
        np.ones(3, dtype=np.float64)
        if node.scale is None
        else np.asarray(node.scale, dtype=np.float64)
    )
    if translation.shape != (3,) or rotation.shape != (4,) or scale.shape != (3,):
        return True
    return not (
        np.allclose(translation, np.zeros(3), atol=1e-9, rtol=1e-9)
        and np.allclose(
            rotation,
            np.array([0.0, 0.0, 0.0, 1.0]),
            atol=1e-9,
            rtol=1e-9,
        )
        and np.allclose(scale, np.ones(3), atol=1e-9, rtol=1e-9)
    )


def _validate_glb_scene_nodes(document: pygltflib.GLTF2) -> None:
    mesh_nodes = [node for node in document.nodes if node.mesh is not None]
    mesh_refs = [int(node.mesh) for node in mesh_nodes]

    if len(mesh_refs) != len(set(mesh_refs)):
        raise ValueError(
            "GLB reuses one mesh from multiple nodes. Instance transforms are not supported; "
            "export a single baked mesh before loading."
        )

    referenced_meshes = set(mesh_refs)
    expected_meshes = set(range(len(document.meshes)))
    if referenced_meshes != expected_meshes:
        raise ValueError(
            "GLB contains mesh data that is not referenced exactly once by a node. "
            "ProofShape will not silently merge orphan geometry; export a single baked mesh "
            "before loading."
        )

    if len(mesh_nodes) > 1 and any(
        _node_has_nonidentity_transform(node) for node in document.nodes
    ):
        raise ValueError(
            "GLB contains multiple mesh-bearing nodes with transforms. ProofShape does not "
            "silently discard assembly placement; export a single baked mesh before loading."
        )


def _load_glb(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        document = pygltflib.GLTF2().load_binary(str(path))
    except Exception as exc:
        raise ValueError(f"Could not read {path} as GLB.") from exc

    if any(_node_has_nonidentity_scale(node) for node in document.nodes):
        raise ValueError(
            "GLB contains a non-identity node scale. ProofShape will not silently apply it; "
            "export coordinates at the intended scale or pass load_mesh(..., scale=...) "
            "explicitly after removing the node scale."
        )
    _validate_glb_scene_nodes(document)

    blob = document.binary_blob()
    if blob is None:
        raise ValueError("GLB has no binary mesh buffer.")

    vertex_blocks: list[np.ndarray] = []
    face_blocks: list[np.ndarray] = []
    vertex_offset = 0

    for mesh in document.meshes:
        for primitive in mesh.primitives:
            mode = primitive.mode if primitive.mode is not None else pygltflib.TRIANGLES
            if mode != pygltflib.TRIANGLES:
                raise ValueError("GLB CAD must use triangle primitives.")

            position_index = primitive.attributes.POSITION
            if position_index is None:
                raise ValueError("GLB CAD primitive has no POSITION attribute.")
            position_accessor = document.accessors[position_index]
            if (
                position_accessor.componentType != pygltflib.FLOAT
                or position_accessor.type != pygltflib.VEC3
            ):
                raise ValueError("GLB POSITION must be FLOAT VEC3.")

            positions = _accessor_array(document, position_index, blob).astype(
                np.float64, copy=False
            )
            if not np.isfinite(positions).all():
                raise ValueError("GLB CAD vertices must all be finite.")

            if primitive.indices is None:
                indices = np.arange(len(positions), dtype=np.int64)
            else:
                index_accessor = document.accessors[primitive.indices]
                if (
                    index_accessor.type != pygltflib.SCALAR
                    or index_accessor.componentType
                    not in {
                        pygltflib.UNSIGNED_BYTE,
                        pygltflib.UNSIGNED_SHORT,
                        pygltflib.UNSIGNED_INT,
                    }
                ):
                    raise ValueError(
                        "GLB triangle indices must be unsigned SCALAR values."
                    )
                indices = _accessor_array(document, primitive.indices, blob).astype(
                    np.int64, copy=False
                )

            if len(indices) == 0 or len(indices) % 3 != 0:
                raise ValueError(
                    "GLB triangle primitive must contain a non-empty multiple of 3 indices."
                )
            if indices.min() < 0 or indices.max() >= len(positions):
                raise ValueError("GLB triangle index is outside the POSITION array.")

            vertex_blocks.append(positions)
            face_blocks.append(indices.reshape(-1, 3) + vertex_offset)
            vertex_offset += len(positions)

    if not vertex_blocks:
        raise ValueError("GLB contains no triangle mesh primitives.")

    vertices = np.vstack(vertex_blocks)
    faces = np.vstack(face_blocks).astype(np.int32)

    soup = vertices[faces.reshape(-1)]
    vertices, faces = _deduplicate_triangle_soup(soup)
    return vertices, faces, _GLB_UNITS


def load_mesh(
    path: Path | str, *, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load CAD as ``(vertices_mm, faces, units)`` without implicit unit conversion.

    ``.stl`` may be ASCII or binary. STL has no units in the format, so ProofShape's project
    convention is millimetres and the returned units string is ``"mm"``.

    ``.obj`` is read for its ``v``/``f`` lines only; ``vn``, ``vt``, and other directives
    (``g``, ``o``, ``mtllib``, ``usemtl``, comments) are parsed past, not acted on. A face with
    more than 3 vertices is triangulated by a simple fan from its first vertex. OBJ has no unit
    metadata either, so it is treated as millimetres by the same convention as STL.

    ``.glb`` coordinates are read exactly as stored and reported as ``"assumed_mm"``. This is
    deliberate: ProofShape's own GLBs contain raw millimetres even though glTF normally defines
    metres. If a caller knows a buyer file requires conversion, it must opt in with ``scale``
    (for example, ``scale=1000`` for metres to millimetres). A non-identity GLB node ``scale`` is
    rejected rather than silently folded into the coordinates.
    """
    path = Path(path)
    scale = _validate_scale(scale)
    suffix = path.suffix.lower()

    if suffix == ".stl":
        vertices, faces, units = _load_stl(path)
    elif suffix == ".obj":
        vertices, faces, units = _load_obj(path)
    elif suffix == ".glb":
        vertices, faces, units = _load_glb(path)
    else:
        raise ValueError(
            f"Unsupported CAD format {suffix or '<none>'}. {_SUPPORTED_MESSAGE}"
        )

    if scale != 1.0:
        vertices = vertices * scale
    return vertices, faces, units

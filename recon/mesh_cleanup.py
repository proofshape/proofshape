"""R-08: clean up R-07's raw fused mesh and export a GLB a browser can load.

Order matches the acceptance criteria: keep connected components above a declared relative size
(drop only small disconnected junk), fill small holes with winding consistent with the surrounding
mesh, decimate to a target triangle count, and enforce the final GLB byte limit before export.

Decimation uses `fast_simplification` (a real quadric-decimation implementation) and GLB I/O uses
`pygltflib` -- both are core to this story's primary acceptance criteria, not a fallback path like
R-07's Open3D/Poisson, so both are real `requirements.txt`/CI dependencies rather than lazily
imported and untested.

No measured numbers exist yet for "a target triangle count," "small holes," or "a stated [file
size] limit" -- the DEFAULT_* constants below are documented assumptions (common web-asset
budgets), configurable, not measured figures, matching R-06/R-07's height-threshold/voxel-size
precedent.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import fast_simplification
import numpy as np
import pygltflib
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from recon.tsdf_fusion import read_mesh_ply

DEFAULT_TARGET_TRIANGLE_COUNT = 50_000
DEFAULT_MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024

# R-27 assumption, not a measured threshold: retain any connected component with at least 10% as
# many faces as the largest one. This is intentionally conservative compared with R-08's old
# largest-only rule so a legitimate observed second piece is not discarded automatically.
DEFAULT_MIN_COMPONENT_FACE_FRACTION = 0.10

# A boundary loop this long or shorter is treated as a "small hole" and closed. Longer openings
# are left alone on purpose: a large opening more likely means a genuinely unreconstructed
# region than a small gap, and silently closing it would assert geometry nothing ever observed --
# the same spirit as the hard rule, even though this story runs before per-vertex provenance
# (R-09) exists to label anything Observed/Unobserved.
DEFAULT_MAX_HOLE_LOOP_LENGTH = 8


def _component_labels(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, int]:
    """Connected-component label per vertex, using mesh edges as undirected adjacency."""
    edge_from = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    edge_to = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    adjacency = coo_matrix(
        (np.ones(len(edge_from)), (edge_from, edge_to)),
        shape=(len(vertices), len(vertices)),
    )
    component_count, labels = connected_components(adjacency, directed=False)
    return labels, component_count


def drop_non_manifold_faces(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop every face incident to an edge shared by more than two faces.

    R-30 review on #270 (@TabeenRaoof): per-frame re-anchoring can leave a handful of faces,
    right where a part meets the re-cut board, sharing an edge with a third face -- not just
    two -- which `find_boundary_loops` and provenance's winding/facing checks both assume never
    happens. Measured on the real s-03 mesh: 6 such edges touch only 17 of 43,043 faces,
    clustered at one part/board seam; dropping those 17 faces (not Open3D's
    `remove_non_manifold_edges()`, which deleted 14,751 faces on the same mesh) leaves a gap
    small enough for `fill_small_holes` to close normally afterward. Vertices are left as-is --
    `filter_connected_components`, which runs right after this, already drops any vertex no
    face references any more.

    Returns `(vertices, faces, dropped_face_count)`.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F,3), got {faces.shape}")
    if len(faces) == 0:
        return vertices, faces.astype(np.int32), 0

    edge_owner_count: dict[tuple[int, int], int] = {}
    for triangle in faces:
        for i in range(3):
            a, b = int(triangle[i]), int(triangle[(i + 1) % 3])
            key = (a, b) if a < b else (b, a)
            edge_owner_count[key] = edge_owner_count.get(key, 0) + 1

    bad_edges = {edge for edge, count in edge_owner_count.items() if count > 2}
    if not bad_edges:
        return vertices, faces.astype(np.int32), 0

    keep_face = np.ones(len(faces), dtype=bool)
    for index, triangle in enumerate(faces):
        for i in range(3):
            a, b = int(triangle[i]), int(triangle[(i + 1) % 3])
            key = (a, b) if a < b else (b, a)
            if key in bad_edges:
                keep_face[index] = False
                break

    kept_faces = faces[keep_face].astype(np.int32)
    return vertices, kept_faces, int(len(faces) - len(kept_faces))


def filter_connected_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    min_face_fraction: float = DEFAULT_MIN_COMPONENT_FACE_FRACTION,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Keep each face-component large enough relative to the largest and report drops."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N,3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F,3), got {faces.shape}")
    if not np.isfinite(min_face_fraction) or not (0.0 <= min_face_fraction <= 1.0):
        raise ValueError("min_face_fraction must be finite and between 0 and 1.")

    if len(faces) == 0:
        report = {
            "component_count_before": 0,
            "component_count_after": 0,
            "largest_component_face_count": 0,
            "dropped_component_count": 0,
            "dropped_face_count": 0,
            "min_component_face_fraction": float(min_face_fraction),
        }
        return vertices, faces.astype(np.int32), report

    labels, component_count = _component_labels(vertices, faces)
    face_labels = labels[faces[:, 0]]
    face_counts = np.bincount(face_labels, minlength=component_count)
    active_labels = np.flatnonzero(face_counts > 0)
    largest_count = int(face_counts[active_labels].max())
    keep_labels = active_labels[
        face_counts[active_labels] >= min_face_fraction * largest_count
    ]
    keep_face = np.isin(face_labels, keep_labels)
    kept_faces_global = faces[keep_face]

    kept_vertex_indices = np.unique(kept_faces_global)
    old_to_new = -np.ones(len(vertices), dtype=np.int64)
    old_to_new[kept_vertex_indices] = np.arange(len(kept_vertex_indices))
    new_vertices = vertices[kept_vertex_indices]
    new_faces = old_to_new[kept_faces_global].astype(np.int32)

    report = {
        "component_count_before": len(active_labels),
        "component_count_after": len(keep_labels),
        "largest_component_face_count": largest_count,
        "dropped_component_count": int(len(active_labels) - len(keep_labels)),
        "dropped_face_count": int(len(faces) - len(new_faces)),
        "min_component_face_fraction": float(min_face_fraction),
    }
    return new_vertices, new_faces, report


def keep_largest_connected_component(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility helper that keeps exactly the single largest face-component."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N,3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F,3), got {faces.shape}")
    if len(faces) == 0:
        return vertices, faces.astype(np.int32)

    labels, component_count = _component_labels(vertices, faces)
    face_labels = labels[faces[:, 0]]
    face_counts = np.bincount(face_labels, minlength=component_count)
    largest_label = int(np.argmax(face_counts))
    keep_face = face_labels == largest_label
    kept_faces_global = faces[keep_face]

    kept_vertex_indices = np.unique(kept_faces_global)
    old_to_new = -np.ones(len(vertices), dtype=np.int64)
    old_to_new[kept_vertex_indices] = np.arange(len(kept_vertex_indices))
    return (
        vertices[kept_vertex_indices],
        old_to_new[kept_faces_global].astype(np.int32),
    )


def find_boundary_loops(faces: np.ndarray) -> list[list[int]]:
    """Ordered vertex loops along the mesh's open boundary (edges used by exactly one face).

    Assumes a manifold, consistently-wound mesh -- what this pipeline's own marching-cubes output
    is -- and that each hole's boundary is a single simple loop (no vertex touching two different
    boundary loops). A loop that can't be walked back to its start under that assumption is
    skipped rather than guessed at.
    """
    faces = np.asarray(faces, dtype=np.int64)
    undirected_count: dict[tuple[int, int], int] = {}
    for triangle in faces:
        for i in range(3):
            a, b = int(triangle[i]), int(triangle[(i + 1) % 3])
            key = (a, b) if a < b else (b, a)
            undirected_count[key] = undirected_count.get(key, 0) + 1

    # directed_next[a] = b follows the direction the boundary edge has in its one owning face.
    # A fill face sharing that edge must traverse it in the opposite direction; R-08's old filler
    # reused this direction and therefore wound the patch inward.
    directed_next: dict[int, int] = {}
    for triangle in faces:
        for i in range(3):
            a, b = int(triangle[i]), int(triangle[(i + 1) % 3])
            key = (a, b) if a < b else (b, a)
            if undirected_count[key] == 1:
                directed_next[a] = b

    loops: list[list[int]] = []
    visited: set[int] = set()
    for start in directed_next:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        current = directed_next.get(start)
        walk_broke = False
        while current is not None and current != start:
            if current in visited:
                walk_broke = True
                break
            loop.append(current)
            visited.add(current)
            current = directed_next.get(current)
        if not walk_broke and current == start:
            loops.append(loop)
    return loops


def fill_small_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_loop_length: int = DEFAULT_MAX_HOLE_LOOP_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Close boundary loops no longer than `max_loop_length`; leave longer ones open."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if max_loop_length < 3:
        raise ValueError(
            "max_loop_length must be at least 3 -- a triangle is the smallest loop."
        )

    loops = find_boundary_loops(faces)
    new_vertices = vertices
    new_face_blocks = [faces]
    for loop in loops:
        if len(loop) < 3 or len(loop) > max_loop_length:
            continue
        if len(loop) == 3:
            new_face_blocks.append(np.array([list(reversed(loop))], dtype=np.int32))
        else:
            centroid = new_vertices[loop].mean(axis=0)
            centroid_index = len(new_vertices)
            new_vertices = np.vstack([new_vertices, centroid[None, :]])
            fan = [
                [loop[(i + 1) % len(loop)], loop[i], centroid_index]
                for i in range(len(loop))
            ]
            new_face_blocks.append(np.array(fan, dtype=np.int32))

    new_faces = np.vstack(new_face_blocks)
    return new_vertices, new_faces


def decimate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_triangle_count: int = DEFAULT_TARGET_TRIANGLE_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce to `target_triangle_count` triangles. A no-op if already at or under the target --
    `fast_simplification` itself refuses a target at or above the current count, so that case is
    handled here rather than surfacing as a confusing library error."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if target_triangle_count <= 0:
        raise ValueError("target_triangle_count must be a positive integer.")
    if len(faces) <= target_triangle_count:
        return vertices, faces

    new_vertices, new_faces = fast_simplification.simplify(
        vertices.astype(np.float64),
        faces.astype(np.int64),
        target_count=target_triangle_count,
    )
    return new_vertices.astype(np.float32), new_faces.astype(np.int32)


def write_glb_with_size_limit(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    min_triangle_count: int = 4,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Write a GLB that actually satisfies the configured byte limit.

    If the first write is oversized, progressively re-decimate to 75% of the current triangle
    count and rewrite. If the mesh cannot be reduced further while keeping at least
    min_triangle_count triangles, fail loudly rather than returning an oversized PASS.
    """
    if max_file_size_bytes <= 0:
        raise ValueError("max_file_size_bytes must be positive.")
    if min_triangle_count < 1:
        raise ValueError("min_triangle_count must be positive.")

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    redecimation_rounds = 0

    while True:
        write_glb(path, vertices, faces)
        file_size_bytes = Path(path).stat().st_size
        if file_size_bytes <= max_file_size_bytes:
            return vertices, faces, file_size_bytes, redecimation_rounds

        if len(faces) <= min_triangle_count:
            raise ValueError(
                f"GLB is {file_size_bytes} bytes, over the {max_file_size_bytes}-byte limit, "
                "and cannot be decimated further safely."
            )

        next_target = max(min_triangle_count, int(len(faces) * 0.75))
        if next_target >= len(faces):
            next_target = len(faces) - 1
        vertices, faces = decimate_mesh(vertices, faces, next_target)
        redecimation_rounds += 1


def _append_aligned_blob(buffer: bytearray, blob: bytes) -> tuple[int, int]:
    """Append one GLB bufferView payload at a 4-byte-aligned offset."""
    while len(buffer) % 4:
        buffer.append(0)
    offset = len(buffer)
    buffer.extend(blob)
    return offset, len(blob)


def write_glb(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
    extra_vertex_attributes: dict[str, np.ndarray] | None = None,
) -> None:
    """Write a minimal GLB with optional COLOR_0 and scalar float32 custom attributes.

    C-14/D-062 uses this for _DEVIATION_MM and _DEVIATION_MEASURED. Every
    bufferView begins at a four-byte boundary so a tightly packed RGB COLOR_0 with an odd
    vertex count cannot misalign the following float accessors.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N,3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F,3), got {faces.shape}")
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        if colors.shape != vertices.shape:
            raise ValueError(
                f"colors must have the same shape as vertices {vertices.shape}, got {colors.shape}"
            )

    extras: dict[str, np.ndarray] = {}
    for name, values in (extra_vertex_attributes or {}).items():
        if not name.startswith("_"):
            raise ValueError(
                f"custom vertex attribute {name!r} must start with '_' per glTF semantics"
            )
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.shape != (len(vertices),):
            raise ValueError(
                f"{name} must have one scalar per vertex ({len(vertices)},), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite float32 values")
        extras[name] = array

    binary = bytearray()
    buffer_views: list[pygltflib.BufferView] = []
    accessors: list[pygltflib.Accessor] = []

    index_offset, index_length = _append_aligned_blob(binary, faces.tobytes())
    buffer_views.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=index_offset,
            byteLength=index_length,
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        )
    )
    accessors.append(
        pygltflib.Accessor(
            bufferView=0,
            componentType=pygltflib.UNSIGNED_INT,
            count=int(faces.size),
            type=pygltflib.SCALAR,
            min=[int(faces.min())] if faces.size else [0],
            max=[int(faces.max())] if faces.size else [0],
        )
    )

    position_offset, position_length = _append_aligned_blob(binary, vertices.tobytes())
    buffer_views.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=position_offset,
            byteLength=position_length,
            target=pygltflib.ARRAY_BUFFER,
        )
    )
    accessors.append(
        pygltflib.Accessor(
            bufferView=1,
            componentType=pygltflib.FLOAT,
            count=len(vertices),
            type=pygltflib.VEC3,
            min=vertices.min(axis=0).tolist() if len(vertices) else [0.0, 0.0, 0.0],
            max=vertices.max(axis=0).tolist() if len(vertices) else [0.0, 0.0, 0.0],
        )
    )
    attributes = pygltflib.Attributes(POSITION=1)

    if colors is not None:
        color_offset, color_length = _append_aligned_blob(binary, colors.tobytes())
        color_view_index = len(buffer_views)
        buffer_views.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=color_offset,
                byteLength=color_length,
                target=pygltflib.ARRAY_BUFFER,
            )
        )
        color_accessor_index = len(accessors)
        accessors.append(
            pygltflib.Accessor(
                bufferView=color_view_index,
                componentType=pygltflib.UNSIGNED_BYTE,
                count=len(colors),
                type=pygltflib.VEC3,
                normalized=True,
                min=colors.min(axis=0).tolist() if len(colors) else [0, 0, 0],
                max=colors.max(axis=0).tolist() if len(colors) else [0, 0, 0],
            )
        )
        attributes.COLOR_0 = color_accessor_index

    for name, values in extras.items():
        offset, length = _append_aligned_blob(binary, values.tobytes())
        view_index = len(buffer_views)
        buffer_views.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=offset,
                byteLength=length,
                target=pygltflib.ARRAY_BUFFER,
            )
        )
        accessor_index = len(accessors)
        accessors.append(
            pygltflib.Accessor(
                bufferView=view_index,
                componentType=pygltflib.FLOAT,
                count=len(values),
                type=pygltflib.SCALAR,
                min=[float(values.min())] if len(values) else [0.0],
                max=[float(values.max())] if len(values) else [0.0],
            )
        )
        setattr(attributes, name, accessor_index)

    document = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[
            pygltflib.Mesh(
                primitives=[pygltflib.Primitive(attributes=attributes, indices=0)]
            )
        ],
        buffers=[pygltflib.Buffer(byteLength=len(binary))],
        bufferViews=buffer_views,
        accessors=accessors,
    )
    document.set_binary_blob(bytes(binary))
    document.save_binary(str(path))


def read_glb(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read back a GLB written by write_glb. The automated proxy for "loads in a browser": a
    genuinely malformed GLB won't parse here either. Only understands this module's own minimal
    single-mesh, POSITION-plus-indices(-plus-optional-COLOR_0) layout, not general glTF content.

    Returns (vertices, faces, colors) -- `colors` is None when the file has no COLOR_0 attribute.
    """
    document = pygltflib.GLTF2().load_binary(str(path))
    blob = document.binary_blob()

    index_accessor = document.accessors[0]
    position_accessor = document.accessors[1]
    index_view = document.bufferViews[index_accessor.bufferView]
    position_view = document.bufferViews[position_accessor.bufferView]

    faces = np.frombuffer(
        blob, dtype=np.uint32, count=index_accessor.count, offset=index_view.byteOffset
    ).reshape(-1, 3)
    vertices = np.frombuffer(
        blob,
        dtype=np.float32,
        count=position_accessor.count * 3,
        offset=position_view.byteOffset,
    ).reshape(-1, 3)

    colors = None
    color_accessor_index = document.meshes[0].primitives[0].attributes.COLOR_0
    if color_accessor_index is not None:
        color_accessor = document.accessors[color_accessor_index]
        color_view = document.bufferViews[color_accessor.bufferView]
        colors = np.frombuffer(
            blob,
            dtype=np.uint8,
            count=color_accessor.count * 3,
            offset=color_view.byteOffset,
        ).reshape(-1, 3)

    return vertices, faces.astype(np.int32), colors


def read_glb_extra_attributes(
    path: Path, names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Read back write_glb's optional scalar float32 extra_vertex_attributes by name (I-06).

    Additive sibling to read_glb, not a change to it: read_glb's 3-tuple return is unpacked
    positionally by six existing callers, so a fourth element there would be a silent breaking
    change for every one of them. A name read_glb's document never wrote (the mesh carries no
    such semantic) is simply absent from the returned dict, the same way read_glb reports no
    COLOR_0 as None rather than raising -- D-062's own "a GLB without these attributes stays
    valid" applies here too.
    """
    document = pygltflib.GLTF2().load_binary(str(path))
    blob = document.binary_blob()
    attributes = document.meshes[0].primitives[0].attributes

    found: dict[str, np.ndarray] = {}
    for name in names:
        accessor_index = getattr(attributes, name, None)
        if accessor_index is None:
            continue
        accessor = document.accessors[accessor_index]
        view = document.bufferViews[accessor.bufferView]
        found[name] = np.frombuffer(
            blob, dtype=np.float32, count=accessor.count, offset=view.byteOffset
        ).copy()
    return found


def check_file_size(path: Path, max_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES) -> bool:
    """True if `path`'s size is at or under `max_bytes`."""
    return Path(path).stat().st_size <= max_bytes


def write_cleanup_report(
    path: Path,
    mesh_ply_path: Path,
    faces_before: int,
    faces_after_component_filter: int,
    holes_filled: int,
    holes_left_open: int,
    triangle_count_before_decimation: int,
    triangle_count_after_decimation: int,
    target_triangle_count: int,
    vertex_count: int,
    face_count: int,
    file_size_bytes: int,
    max_file_size_bytes: int,
    component_report: dict[str, int | float],
    size_redecimation_rounds: int,
    non_manifold_faces_dropped: int = 0,
) -> dict[str, Any]:
    report = {
        "story": "R-08",
        "source_mesh_ply": str(mesh_ply_path),
        # R-30 review on #270: faces dropped by drop_non_manifold_faces, before the component
        # filter below -- almost always 0; non-zero means re-anchoring left a seam provenance
        # couldn't have labelled anyway.
        "non_manifold_faces_dropped": non_manifold_faces_dropped,
        "faces_before_component_filter": faces_before,
        "faces_after_component_filter": faces_after_component_filter,
        "holes_filled": holes_filled,
        "holes_left_open": holes_left_open,
        "triangle_count_before_decimation": triangle_count_before_decimation,
        "triangle_count_after_decimation": triangle_count_after_decimation,
        "target_triangle_count": target_triangle_count,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "file_size_bytes": file_size_bytes,
        "max_file_size_bytes": max_file_size_bytes,
        "within_file_size_limit": file_size_bytes <= max_file_size_bytes,
        "size_redecimation_rounds": size_redecimation_rounds,
        **component_report,
    }
    path = Path(path)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def cleanup_run(
    mesh_ply_path: Path,
    out_path: Path,
    target_triangle_count: int = DEFAULT_TARGET_TRIANGLE_COUNT,
    max_hole_loop_length: int = DEFAULT_MAX_HOLE_LOOP_LENGTH,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    min_component_face_fraction: float = DEFAULT_MIN_COMPONENT_FACE_FRACTION,
) -> dict[str, Any]:
    """Load an R-07 mesh.ply, clean it up, export a GLB, and write the report next to it."""
    mesh_ply_path = Path(mesh_ply_path)
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists. Pick a new --out or delete it deliberately; "
            "overwriting silently would let outputs from two runs get mixed."
        )

    vertices, faces = read_mesh_ply(mesh_ply_path)
    faces_before = len(faces)

    vertices, faces, non_manifold_faces_dropped = drop_non_manifold_faces(
        vertices, faces
    )

    vertices, faces, component_report = filter_connected_components(
        vertices, faces, min_face_fraction=min_component_face_fraction
    )
    faces_after_component_filter = len(faces)

    loops = find_boundary_loops(faces)
    holes_filled = sum(1 for loop in loops if 3 <= len(loop) <= max_hole_loop_length)
    holes_left_open = sum(1 for loop in loops if len(loop) > max_hole_loop_length)
    vertices, faces = fill_small_holes(vertices, faces, max_hole_loop_length)

    triangle_count_before_decimation = len(faces)
    vertices, faces = decimate_mesh(vertices, faces, target_triangle_count)
    triangle_count_after_decimation = len(faces)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces, file_size_bytes, size_redecimation_rounds = (
        write_glb_with_size_limit(
            out_path,
            vertices,
            faces,
            max_file_size_bytes=max_file_size_bytes,
        )
    )
    triangle_count_after_decimation = len(faces)

    report_path = out_path.with_suffix(".json")
    return write_cleanup_report(
        report_path,
        mesh_ply_path,
        faces_before,
        faces_after_component_filter,
        holes_filled,
        holes_left_open,
        triangle_count_before_decimation,
        triangle_count_after_decimation,
        target_triangle_count,
        vertex_count=len(vertices),
        face_count=len(faces),
        file_size_bytes=file_size_bytes,
        max_file_size_bytes=max_file_size_bytes,
        component_report=component_report,
        size_redecimation_rounds=size_redecimation_rounds,
        non_manifold_faces_dropped=non_manifold_faces_dropped,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean up an R-07 mesh and export a GLB (R-08)."
    )
    parser.add_argument("mesh_ply_path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--target-triangle-count", type=int, default=DEFAULT_TARGET_TRIANGLE_COUNT
    )
    parser.add_argument(
        "--min-component-face-fraction",
        type=float,
        default=DEFAULT_MIN_COMPONENT_FACE_FRACTION,
        help="keep components with at least this fraction of the largest component's faces",
    )
    parser.add_argument(
        "--max-file-size-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_SIZE_BYTES,
        help="enforced final GLB byte limit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    out_path = args.out or args.mesh_ply_path.with_name("mesh_clean.glb")
    try:
        report = cleanup_run(
            args.mesh_ply_path,
            out_path,
            target_triangle_count=args.target_triangle_count,
            min_component_face_fraction=args.min_component_face_fraction,
            max_file_size_bytes=args.max_file_size_bytes,
        )
    except (FileExistsError, ValueError) as exc:
        print(f"R-08 MESH CLEANUP: FAIL - {exc}", file=sys.stderr)
        return 1

    limit_note = "within" if report["within_file_size_limit"] else "OVER"
    print(
        f"R-08 MESH CLEANUP: PASS - {report['face_count']} faces, "
        f"{report['file_size_bytes']} bytes ({limit_note} the {report['max_file_size_bytes']}-byte limit)"
    )
    print(f"Output written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

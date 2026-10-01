"""R-08: clean up R-07's raw fused mesh and export a GLB a browser can load.

Order matches the acceptance criteria: keep the largest connected component (drop disconnected
junk -- stray voxels, TSDF noise far from the part), fill small holes (close small gaps without
asserting geometry that was never really observed), decimate to a target triangle count, export.

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

# A boundary loop this long or shorter is treated as a "small hole" and closed. Longer openings
# are left alone on purpose: a large opening more likely means a genuinely unreconstructed
# region than a small gap, and silently closing it would assert geometry nothing ever observed --
# the same spirit as the hard rule, even though this story runs before per-vertex provenance
# (R-09) exists to label anything Observed/Unobserved.
DEFAULT_MAX_HOLE_LOOP_LENGTH = 8


def keep_largest_connected_component(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the faces (and their vertices, re-indexed) in the mesh's largest component."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N,3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F,3), got {faces.shape}")
    if len(faces) == 0:
        return vertices, faces.astype(np.int32)

    edge_from = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    edge_to = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    vertex_count = len(vertices)
    adjacency = coo_matrix(
        (np.ones(len(edge_from)), (edge_from, edge_to)),
        shape=(vertex_count, vertex_count),
    )
    _, labels = connected_components(adjacency, directed=False)

    largest_label = int(np.argmax(np.bincount(labels)))
    keep_vertex = labels == largest_label
    keep_face = keep_vertex[faces].all(axis=1)

    old_to_new = -np.ones(vertex_count, dtype=np.int64)
    kept_vertex_indices = np.nonzero(keep_vertex)[0]
    old_to_new[kept_vertex_indices] = np.arange(len(kept_vertex_indices))

    new_vertices = vertices[kept_vertex_indices]
    new_faces = old_to_new[faces[keep_face]].astype(np.int32)
    return new_vertices, new_faces


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

    # `directed_next[a] = b` for each boundary edge, in the direction it appeared in its one
    # owning face -- this is the boundary's orientation induced by the interior triangles' own
    # winding, so a hole filled by walking this direction keeps the surrounding mesh's winding.
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
            new_face_blocks.append(np.array([loop], dtype=np.int32))
        else:
            centroid = new_vertices[loop].mean(axis=0)
            centroid_index = len(new_vertices)
            new_vertices = np.vstack([new_vertices, centroid[None, :]])
            fan = [
                [loop[i], loop[(i + 1) % len(loop)], centroid_index]
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


def write_glb(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a minimal, valid GLB: one mesh, POSITION + triangle indices, no materials."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N,3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F,3), got {faces.shape}")

    indices_blob = faces.tobytes()
    positions_blob = vertices.tobytes()

    document = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(POSITION=1), indices=0
                    )
                ]
            )
        ],
        buffers=[pygltflib.Buffer(byteLength=len(indices_blob) + len(positions_blob))],
        bufferViews=[
            pygltflib.BufferView(
                buffer=0,
                byteOffset=0,
                byteLength=len(indices_blob),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(indices_blob),
                byteLength=len(positions_blob),
                target=pygltflib.ARRAY_BUFFER,
            ),
        ],
        accessors=[
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.UNSIGNED_INT,
                count=int(faces.size),
                type=pygltflib.SCALAR,
                min=[int(faces.min())] if faces.size else [0],
                max=[int(faces.max())] if faces.size else [0],
            ),
            pygltflib.Accessor(
                bufferView=1,
                componentType=pygltflib.FLOAT,
                count=len(vertices),
                type=pygltflib.VEC3,
                min=vertices.min(axis=0).tolist() if len(vertices) else [0.0, 0.0, 0.0],
                max=vertices.max(axis=0).tolist() if len(vertices) else [0.0, 0.0, 0.0],
            ),
        ],
    )
    document.set_binary_blob(indices_blob + positions_blob)
    document.save_binary(str(path))


def read_glb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read back a GLB written by write_glb. The automated proxy for "loads in a browser": a
    genuinely malformed GLB won't parse here either. Only understands this module's own minimal
    single-mesh, POSITION-plus-indices layout, not general glTF content.
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
    return vertices, faces.astype(np.int32)


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
) -> dict[str, Any]:
    report = {
        "story": "R-08",
        "source_mesh_ply": str(mesh_ply_path),
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

    vertices, faces = keep_largest_connected_component(vertices, faces)
    faces_after_component_filter = len(faces)

    loops = find_boundary_loops(faces)
    holes_filled = sum(1 for loop in loops if 3 <= len(loop) <= max_hole_loop_length)
    holes_left_open = sum(1 for loop in loops if len(loop) > max_hole_loop_length)
    vertices, faces = fill_small_holes(vertices, faces, max_hole_loop_length)

    triangle_count_before_decimation = len(faces)
    vertices, faces = decimate_mesh(vertices, faces, target_triangle_count)
    triangle_count_after_decimation = len(faces)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_glb(out_path, vertices, faces)
    file_size_bytes = out_path.stat().st_size

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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    out_path = args.out or args.mesh_ply_path.with_name("mesh_clean.glb")
    try:
        report = cleanup_run(
            args.mesh_ply_path,
            out_path,
            target_triangle_count=args.target_triangle_count,
        )
    except FileExistsError as exc:
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

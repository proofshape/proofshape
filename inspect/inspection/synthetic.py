"""I-01: a synthetic mesh generator that produces exactly what reconstruction will eventually
produce, months earlier (sprint plan Section 4, item 3): load a CAD part, apply a random rigid
pose, add Gaussian noise, mark a surface patch Unobserved, and export it in reconstruction's own
GLB format. This is what lets I-03..I-05 (registration, ICP, ambiguity) be built and unit-tested
with no GPU, no capture hardware, and no fixtures.

Frame convention. CAD files are authored z-up from z = 0 (base at z = 0, body at z > 0) --
ordinary CAD practice. Reconstruction's own board frame uses the opposite sign for height:
`recon/cad_size_check.py`'s own comment puts it plainly ("height above the board is -z relative
to the fitted plane"). So turning a loaded CAD into something that looks like real reconstruction
output means negating z once, as a *fixed* step -- not a random one. That fixed flip is composed
with the random pose below into one rotation + translation, which is what the sidecar records as
"the transform": a single affine map from the CAD's own coordinates to this module's output.

Random pose. Not a full random SO(3) rotation -- the part rests on the board, so (as I-03's own
acceptance criteria puts it) "the free pose is yaw and an xy shift." A random rotation about z
plus a random xy translation is the only motion that keeps a part sitting on a board sitting on a
board.

"Deletes a surface patch" (the sprint plan's own phrasing) is implemented as *labeling* a patch
of vertices Unobserved, not literally removing mesh geometry. Real reconstruction output never
has holes either -- small ones are filled by R-08, and unobserved regions stay in the mesh,
colored grey (`docs/glossary.md`: "stay grey rather than neutral") -- and every downstream reader
(`recon.mesh_cleanup.read_glb` + `recon.provenance.colors_to_labels`) expects exactly that shape:
a fully triangulated, watertight mesh with per-vertex color only.

None of the default parameters below (target edge length, noise sigma, patch fraction) are
measured claims about the real method's noise -- they are declared defaults, same as every prior
story's unmeasured constants (R-06's height threshold, R-07's voxel size, R-08's triangle count).
G-04 measures the real thing, later.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from recon.mesh_cleanup import write_glb
from recon.next_view import vertex_normals
from recon.provenance import LABEL_OBSERVED, LABEL_UNOBSERVED, labels_to_colors

DEFAULT_TARGET_EDGE_LENGTH_MM = 5.0
DEFAULT_NOISE_SIGMA_MM = 0.3
DEFAULT_PATCH_FRACTION = 0.2
DEFAULT_TRANSLATION_RANGE_MM = 50.0

# A generous cap: the fixture CADs (6-8 vertices) need only a few passes to pass a few-mm edge
# target, and each pass roughly quadruples the triangle count, so this is a safety bound against
# an unreachable target, not a number ever expected to bind in real use.
_MAX_DENSIFY_PASSES = 10


def _validate_mesh(
    vertices_mm: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    vertices_mm = np.asarray(vertices_mm, dtype=np.float64)
    if vertices_mm.ndim != 2 or vertices_mm.shape[1] != 3 or len(vertices_mm) == 0:
        raise ValueError(
            f"vertices_mm must have shape (N,3) with N >= 1, got {vertices_mm.shape}"
        )
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"faces must have shape (F,3) with F >= 1, got {faces.shape}")
    faces = faces.astype(np.int64)
    if faces.min() < 0 or faces.max() >= len(vertices_mm):
        raise ValueError(
            f"faces reference vertex indices outside [0, {len(vertices_mm) - 1}]"
        )
    return vertices_mm, faces


def _edge_lengths(vertices_mm: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = (
        vertices_mm[faces[:, 0]],
        vertices_mm[faces[:, 1]],
        vertices_mm[faces[:, 2]],
    )
    return np.concatenate(
        [
            np.linalg.norm(a - b, axis=1),
            np.linalg.norm(b - c, axis=1),
            np.linalg.norm(c - a, axis=1),
        ]
    )


def densify_mesh(
    vertices_mm: np.ndarray, faces: np.ndarray, target_edge_length_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly subdivide every triangle (4-way, via edge midpoints) until the mesh's longest
    edge is at or below `target_edge_length_mm`. Shared edges get one shared midpoint vertex
    each pass -- not a separate one per adjacent face -- so the result stays watertight.
    """
    if target_edge_length_mm <= 0:
        raise ValueError(
            f"target_edge_length_mm must be positive, got {target_edge_length_mm}"
        )
    vertices_mm, faces = _validate_mesh(vertices_mm, faces)

    for _ in range(_MAX_DENSIFY_PASSES):
        if _edge_lengths(vertices_mm, faces).max() <= target_edge_length_mm:
            return vertices_mm, faces

        vertices_list = list(vertices_mm)
        midpoint_cache: dict[tuple[int, int], int] = {}
        new_faces = []

        for v0, v1, v2 in faces:
            midpoints = []
            for i, j in ((v0, v1), (v1, v2), (v2, v0)):
                key = (i, j) if i < j else (j, i)
                index = midpoint_cache.get(key)
                if index is None:
                    index = len(vertices_list)
                    vertices_list.append((vertices_mm[i] + vertices_mm[j]) / 2.0)
                    midpoint_cache[key] = index
                midpoints.append(index)
            m01, m12, m20 = midpoints
            new_faces.append((v0, m01, m20))
            new_faces.append((v1, m12, m01))
            new_faces.append((v2, m20, m12))
            new_faces.append((m01, m12, m20))

        vertices_mm = np.array(vertices_list, dtype=np.float64)
        faces = np.array(new_faces, dtype=np.int64)

    raise ValueError(
        f"Could not reach target edge length {target_edge_length_mm} mm within "
        f"{_MAX_DENSIFY_PASSES} subdivision passes."
    )


def board_frame_transform(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """The fixed CAD->board-frame z-flip, composed with a random yaw-about-z and a random xy
    translation -- the only motion a part resting on the board actually has. Returns (rotation
    (3,3), translation_mm (3,)) such that `board_vertex = rotation @ cad_vertex + translation`.
    """
    yaw_rad = rng.uniform(0.0, 2.0 * np.pi)
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
    yaw_rotation = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]]
    )
    z_flip = np.diag([1.0, 1.0, -1.0])
    rotation = yaw_rotation @ z_flip

    translation_xy = rng.uniform(
        -DEFAULT_TRANSLATION_RANGE_MM, DEFAULT_TRANSLATION_RANGE_MM, 2
    )
    translation = np.array([translation_xy[0], translation_xy[1], 0.0])
    return rotation, translation


def apply_gaussian_noise(
    vertices_mm: np.ndarray,
    faces: np.ndarray,
    sigma_mm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add per-vertex Gaussian noise along each vertex's own unit normal -- reconstruction's real
    depth noise is predominantly along the viewing/depth direction, not isotropic in 3D."""
    if sigma_mm < 0:
        raise ValueError(f"sigma_mm must be non-negative, got {sigma_mm}")
    vertices_mm, faces = _validate_mesh(vertices_mm, faces)
    normals = vertex_normals(vertices_mm, faces)
    offsets = rng.normal(0.0, sigma_mm, size=len(vertices_mm))
    return vertices_mm + offsets[:, None] * normals


def select_patch_vertices(
    faces: np.ndarray,
    vertex_count: int,
    patch_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """A connected patch of vertices, grown by face-adjacency flood fill from a random seed
    vertex until it reaches `patch_fraction` of all vertices. Returns a boolean mask, True where
    a vertex falls inside the patch (to be labelled Unobserved)."""
    if not 0.0 <= patch_fraction <= 1.0:
        raise ValueError(f"patch_fraction must be in [0, 1], got {patch_fraction}")
    faces = np.asarray(faces, dtype=np.int64)
    target_size = round(patch_fraction * vertex_count)
    if target_size <= 0:
        return np.zeros(vertex_count, dtype=bool)

    adjacency: list[set[int]] = [set() for _ in range(vertex_count)]
    for v0, v1, v2 in faces:
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            adjacency[a].add(b)
            adjacency[b].add(a)

    seed_vertex = int(rng.integers(vertex_count))
    visited = {seed_vertex}
    frontier = [seed_vertex]
    while frontier and len(visited) < target_size:
        next_frontier = []
        for vertex in frontier:
            for neighbor in adjacency[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
                    if len(visited) >= target_size:
                        break
            if len(visited) >= target_size:
                break
        frontier = next_frontier

    mask = np.zeros(vertex_count, dtype=bool)
    mask[list(visited)] = True
    return mask


@dataclass
class SyntheticSample:
    vertices_mm: np.ndarray
    faces: np.ndarray
    labels: np.ndarray
    colors: np.ndarray
    sidecar: dict[str, Any]


def generate_synthetic_sample(
    vertices_mm: np.ndarray,
    faces: np.ndarray,
    seed: int,
    *,
    target_edge_length_mm: float = DEFAULT_TARGET_EDGE_LENGTH_MM,
    sigma_mm: float = DEFAULT_NOISE_SIGMA_MM,
    patch_fraction: float = DEFAULT_PATCH_FRACTION,
) -> SyntheticSample:
    """Load CAD -> densify -> pose (fixed z-flip + random yaw/xy) -> add noise -> label a patch
    Unobserved -> ready to export. Deterministic for a given seed and the same inputs."""
    vertices_mm, faces = _validate_mesh(vertices_mm, faces)
    rng = np.random.default_rng(seed)

    dense_vertices, dense_faces = densify_mesh(
        vertices_mm, faces, target_edge_length_mm
    )

    rotation, translation = board_frame_transform(rng)
    posed_vertices = dense_vertices @ rotation.T + translation

    noisy_vertices = apply_gaussian_noise(posed_vertices, dense_faces, sigma_mm, rng)

    patch_mask = select_patch_vertices(
        dense_faces, len(dense_vertices), patch_fraction, rng
    )
    labels = np.full(len(dense_vertices), LABEL_OBSERVED, dtype=object)
    labels[patch_mask] = LABEL_UNOBSERVED
    colors = labels_to_colors(labels)

    sidecar = {
        "story": "I-01",
        "seed": seed,
        "target_edge_length_mm": target_edge_length_mm,
        "sigma_mm": sigma_mm,
        "patch_fraction": patch_fraction,
        "patch_vertex_count": int(np.count_nonzero(patch_mask)),
        "vertex_count": len(dense_vertices),
        "face_count": len(dense_faces),
        "transform": {
            "rotation": rotation.tolist(),
            "translation_mm": translation.tolist(),
        },
    }
    return SyntheticSample(
        vertices_mm=noisy_vertices,
        faces=dense_faces,
        labels=labels,
        colors=colors,
        sidecar=sidecar,
    )


def write_synthetic_sample(
    sample: SyntheticSample, glb_path: Path, sidecar_path: Path
) -> dict[str, Any]:
    """Write the sample's GLB (positions + colors) and its sidecar JSON. Returns the sidecar
    dict, same convention as `recon.provenance.write_provenance_report`."""
    write_glb(Path(glb_path), sample.vertices_mm, sample.faces, sample.colors)
    sidecar_path = Path(sidecar_path)
    sidecar_path.write_text(
        json.dumps(sample.sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sample.sidecar


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="I-01: generate a synthetic reconstruction-like mesh from a CAD part."
    )
    parser.add_argument(
        "cad_vertices_npy", type=Path, help="(N,3) float array, millimetres"
    )
    parser.add_argument(
        "cad_faces_npy", type=Path, help="(F,3) int array of vertex indices"
    )
    parser.add_argument("seed", type=int)
    parser.add_argument("out_glb", type=Path)
    parser.add_argument("out_sidecar_json", type=Path)
    parser.add_argument(
        "--target-edge-length-mm", type=float, default=DEFAULT_TARGET_EDGE_LENGTH_MM
    )
    parser.add_argument("--sigma-mm", type=float, default=DEFAULT_NOISE_SIGMA_MM)
    parser.add_argument("--patch-fraction", type=float, default=DEFAULT_PATCH_FRACTION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        vertices_mm = np.load(args.cad_vertices_npy)
        faces = np.load(args.cad_faces_npy)
        sample = generate_synthetic_sample(
            vertices_mm,
            faces,
            args.seed,
            target_edge_length_mm=args.target_edge_length_mm,
            sigma_mm=args.sigma_mm,
            patch_fraction=args.patch_fraction,
        )
        write_synthetic_sample(sample, args.out_glb, args.out_sidecar_json)
    except (FileNotFoundError, ValueError) as exc:
        print(f"I-01 SYNTHETIC: FAIL - {exc}", file=sys.stderr)
        return 1
    print(f"I-01 SYNTHETIC: PASS - wrote {args.out_glb} and {args.out_sidecar_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

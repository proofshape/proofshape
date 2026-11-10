"""G-05: write the defect ladder's ten CAD files into `fixtures/reference_cad/`.

Every part shares one base: an L-shaped plate 50 x 32 x 20 mm (a 16 x 11 mm notch cut from one
corner) with a 12 mm square through-hole. The control is that base. The nine variants each add one
defect of 2, 4 or 8 mm:

- the hole shifted along +x (toward the plate's centre),
- a 10 x 10 mm pocket in the top face (missing material),
- a 10 x 10 mm bump on the same spot of the top face (excess material).

Why each dimension is what it is -- the notch, the 20 mm height, the hole and the defect site --
is in `stories/G/G-05.md`, with the measurements behind it.

Every part is 2.5D: a height over each cell of a grid whose lines are the features' x and y
bounds. Each mesh is closed and consistently wound by construction, with no T-junctions, because
every wall is split at every height the part has (`build_mesh`).

    python scripts/make_g05_defect_ladder.py          # (re)write the ten files
    python scripts/make_g05_defect_ladder.py --check  # exit 1 if a committed file differs

The files are committed, and a test checks they are exactly what this script writes. Once a part
has been printed, its file is frozen: a change becomes a new file, never an edit to that one.
"""

from __future__ import annotations

import argparse
import math
import sys
from itertools import pairwise
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "fixtures" / "reference_cad"

# Millimetres, z up, base at z = 0. Rectangles are (x_min, x_max, y_min, y_max).
PLATE_X_MM = 50.0
PLATE_Y_MM = 32.0
PLATE_HEIGHT_MM = 20.0
NOTCH_XY_MM = (34.0, 50.0, 21.0, 32.0)
HOLE_XY_MM = (6.0, 18.0, 10.0, 22.0)
DEFECT_SITE_XY_MM = (35.0, 45.0, 5.0, 15.0)

# (file name, hole shift along +x in mm, height of the defect site in mm or None for plain top)
PARTS = (
    ("g05_00_control_50x32x20mm.stl", 0.0, None),
    ("g05_01_hole_shift_2mm.stl", 2.0, None),
    ("g05_02_hole_shift_4mm.stl", 4.0, None),
    ("g05_03_hole_shift_8mm.stl", 8.0, None),
    ("g05_04_pocket_2mm.stl", 0.0, PLATE_HEIGHT_MM - 2.0),
    ("g05_05_pocket_4mm.stl", 0.0, PLATE_HEIGHT_MM - 4.0),
    ("g05_06_pocket_8mm.stl", 0.0, PLATE_HEIGHT_MM - 8.0),
    ("g05_07_bump_2mm.stl", 0.0, PLATE_HEIGHT_MM + 2.0),
    ("g05_08_bump_4mm.stl", 0.0, PLATE_HEIGHT_MM + 4.0),
    ("g05_09_bump_8mm.stl", 0.0, PLATE_HEIGHT_MM + 8.0),
)


def _inside(rectangle, x: float, y: float) -> bool:
    return (
        rectangle is not None
        and rectangle[0] < x < rectangle[1]
        and rectangle[2] < y < rectangle[3]
    )


def build_mesh(hole_xy_mm, site_height_mm, notch_xy_mm=NOTCH_XY_MM):
    """Return (vertices, faces) for one part: lists of (x, y, z) floats and (a, b, c) indices.

    `hole_xy_mm` or `notch_xy_mm` may be None (no hole, no notch); the tests use that to show what
    the notch is for. Faces are wound counter-clockwise seen from outside.
    """
    xs = {0.0, PLATE_X_MM}
    ys = {0.0, PLATE_Y_MM}
    features = [notch_xy_mm, hole_xy_mm]
    if site_height_mm is not None:
        features.append(DEFECT_SITE_XY_MM)
    for rectangle in features:
        if rectangle is not None:
            xs.update(rectangle[:2])
            ys.update(rectangle[2:])
    xs = sorted(xs)
    ys = sorted(ys)

    def height(i: int, j: int) -> float:
        # Cells are indexed from 0; anything outside the grid is empty space.
        if not (0 <= i < len(xs) - 1 and 0 <= j < len(ys) - 1):
            return 0.0
        x = (xs[i] + xs[i + 1]) / 2.0
        y = (ys[j] + ys[j + 1]) / 2.0
        if _inside(notch_xy_mm, x, y) or _inside(hole_xy_mm, x, y):
            return 0.0
        if site_height_mm is not None and _inside(DEFECT_SITE_XY_MM, x, y):
            return site_height_mm
        return PLATE_HEIGHT_MM

    levels = sorted(
        {0.0} | {height(i, j) for i in range(len(xs) - 1) for j in range(len(ys) - 1)}
    )
    index: dict[tuple[float, float, float], int] = {}
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    def vertex(x: float, y: float, z: float) -> int:
        key = (x, y, z)
        if key not in index:
            index[key] = len(vertices)
            vertices.append(key)
        return index[key]

    def quad(a: int, b: int, c: int, d: int) -> None:
        faces.append((a, b, c))
        faces.append((a, c, d))

    def wall_levels(h1: float, h2: float) -> list[float]:
        # Every height the part has between the two cells, so that walls meeting on one vertical
        # line share every vertex on it -- no T-junctions, which `load_mesh` would leave open.
        low, high = min(h1, h2), max(h1, h2)
        return [z for z in levels if low <= z <= high]

    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            h = height(i, j)
            if h == 0.0:
                continue
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
            quad(
                vertex(x0, y0, h),
                vertex(x1, y0, h),
                vertex(x1, y1, h),
                vertex(x0, y1, h),
            )
            quad(
                vertex(x0, y0, 0.0),
                vertex(x0, y1, 0.0),
                vertex(x1, y1, 0.0),
                vertex(x1, y0, 0.0),
            )

    # Walls on each vertical grid line x = xs[k], between cell k-1 (left) and cell k (right).
    for k, x in enumerate(xs):
        for j in range(len(ys) - 1):
            left, right = height(k - 1, j), height(k, j)
            if left == right:
                continue
            y0, y1 = ys[j], ys[j + 1]
            zs = wall_levels(left, right)
            for z0, z1 in pairwise(zs):
                if left > right:  # solid on the left: the wall faces +x
                    quad(
                        vertex(x, y0, z0),
                        vertex(x, y1, z0),
                        vertex(x, y1, z1),
                        vertex(x, y0, z1),
                    )
                else:  # solid on the right: the wall faces -x
                    quad(
                        vertex(x, y0, z0),
                        vertex(x, y0, z1),
                        vertex(x, y1, z1),
                        vertex(x, y1, z0),
                    )

    # Walls on each horizontal grid line y = ys[k], between cell k-1 (below) and cell k (above).
    for k, y in enumerate(ys):
        for i in range(len(xs) - 1):
            below, above = height(i, k - 1), height(i, k)
            if below == above:
                continue
            x0, x1 = xs[i], xs[i + 1]
            zs = wall_levels(below, above)
            for z0, z1 in pairwise(zs):
                if below > above:  # solid below: the wall faces +y
                    quad(
                        vertex(x0, y, z0),
                        vertex(x0, y, z1),
                        vertex(x1, y, z1),
                        vertex(x1, y, z0),
                    )
                else:  # solid above: the wall faces -y
                    quad(
                        vertex(x0, y, z0),
                        vertex(x1, y, z0),
                        vertex(x1, y, z1),
                        vertex(x0, y, z1),
                    )

    return vertices, faces


def part_mesh(hole_shift_mm: float, site_height_mm):
    """The mesh for one row of `PARTS`."""
    x0, x1, y0, y1 = HOLE_XY_MM
    return build_mesh((x0 + hole_shift_mm, x1 + hole_shift_mm, y0, y1), site_height_mm)


def _unit_normal(a, b, c) -> tuple[float, float, float]:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    # `+ 0.0` turns a -0.0 into 0.0, so a normal prints the same however it was reached.
    return nx / length + 0.0, ny / length + 0.0, nz / length + 0.0


def stl_text(name: str, vertices, faces) -> str:
    """ASCII STL in the same layout as the other files in `fixtures/reference_cad/`."""
    lines = [f"solid {name}"]
    for face in faces:
        corners = [vertices[i] for i in face]
        nx, ny, nz = _unit_normal(*corners)
        lines.append(f"  facet normal {nx:e} {ny:e} {nz:e}")
        lines.append("    outer loop")
        lines.extend(f"      vertex {x:f} {y:f} {z:f}" for x, y, z in corners)
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    return "\n".join(lines) + "\n"


def ladder_stl_texts() -> dict[str, str]:
    """Every part's file name and the exact text this script writes for it."""
    texts = {}
    for filename, hole_shift_mm, site_height_mm in PARTS:
        vertices, faces = part_mesh(hole_shift_mm, site_height_mm)
        texts[filename] = stl_text(Path(filename).stem, vertices, faces)
    return texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 if any committed file differs from what this script writes",
    )
    args = parser.parse_args(argv)

    texts = ladder_stl_texts()
    if args.check:
        stale = [
            filename
            for filename, text in texts.items()
            if not (OUTPUT_DIR / filename).is_file()
            or (OUTPUT_DIR / filename).read_bytes() != text.encode("utf-8")
        ]
        for filename in stale:
            print(f"differs from the generator: {OUTPUT_DIR / filename}")
        if not stale:
            print(f"all {len(texts)} G-05 files match the generator")
        return 1 if stale else 0

    for filename, text in texts.items():
        (OUTPUT_DIR / filename).write_bytes(text.encode("utf-8"))
        print(f"wrote {OUTPUT_DIR / filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

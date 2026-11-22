"""G-11: write the robustness set's thin-walled part into `fixtures/reference_cad/`.

The part is G-02's pentagon specimen (`g02_04_pentagon_r22x25mm.stl`: a regular pentagon of
circumradius 22 mm, 25 mm tall) hollowed into an open-top tray. It keeps that specimen's outer
footprint, outer height and position exactly, so the solid grey `g02_04` that G-04 already
captured and measured is its control. The only change is the walls: 1.2 mm thick, measured
perpendicular to each face, standing on a 2.0 mm floor.

- **Why the pentagon:** G-04 kept `g02_04` among the five specimens whose height it trusts, and
  its pose search didn't fail on it (the square and the triangle did; R-37).
- **Why 1.2 mm:** three 0.4 mm nozzle widths, so the wall prints solid with G-01's slicer
  settings, while still being about 1/20 of the wall's height.
- **Why a 2.0 mm floor:** the floor sits within the board-clearance cut
  (`DEFAULT_MIN_BOARD_CLEARANCE_MM`, 2.0 mm in `recon/part_segmentation.py`), so only the walls
  are left for reconstruction to find. That is the case under test.

The glossy and dark parts of the set need no new file. They are `g02_04` itself, printed in a
different filament (`stories/G/G-11.md`).

    python scripts/make_g11_thin_wall.py          # (re)write the file
    python scripts/make_g11_thin_wall.py --check  # exit 1 if the committed file differs

The file is committed, and a test checks it is exactly what this script writes. Once printed,
it is frozen: a change becomes a new file, never an edit to this one.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "fixtures" / "reference_cad"
FILENAME = "g11_01_thin_wall_pentagon_r22x25mm.stl"

# Millimetres, z up, base at z = 0, the pentagon centred on the origin with a vertex on +y:
# the same placement as g02_04_pentagon_r22x25mm.stl.
SIDES = 5
OUTER_CIRCUMRADIUS_MM = 22.0
HEIGHT_MM = 25.0
WALL_MM = 1.2
FLOOR_MM = 2.0


def inner_circumradius_mm(
    outer_mm: float = OUTER_CIRCUMRADIUS_MM, wall_mm: float = WALL_MM
) -> float:
    """Circumradius of the cavity, so each wall is `wall_mm` thick perpendicular to its faces."""
    return outer_mm - wall_mm / math.cos(math.pi / SIDES)


def _ring(radius_mm: float, z_mm: float) -> list[tuple[float, float, float]]:
    # Counter-clockwise seen from above, starting at +y. Rounding first, then `+ 0.0`, so a
    # coordinate that is mathematically zero prints as 0.000000, never -0.000000.
    points = []
    for k in range(SIDES):
        angle = math.pi / 2.0 + 2.0 * math.pi * k / SIDES
        x = round(radius_mm * math.cos(angle), 9) + 0.0
        y = round(radius_mm * math.sin(angle), 9) + 0.0
        points.append((x, y, z_mm))
    return points


def build_mesh(wall_mm: float = WALL_MM, floor_mm: float = FLOOR_MM):
    """Return (vertices, faces): the tray as a closed solid, faces counter-clockwise from outside."""
    inner_mm = inner_circumradius_mm(wall_mm=wall_mm)
    outer_base = _ring(OUTER_CIRCUMRADIUS_MM, 0.0)
    outer_top = _ring(OUTER_CIRCUMRADIUS_MM, HEIGHT_MM)
    inner_top = _ring(inner_mm, HEIGHT_MM)
    inner_floor = _ring(inner_mm, floor_mm)
    vertices = outer_base + outer_top + inner_top + inner_floor
    ob, ot, it, fl = (list(range(s, s + SIDES)) for s in range(0, 4 * SIDES, SIDES))

    faces: list[tuple[int, int, int]] = []

    def quad(a: int, b: int, c: int, d: int) -> None:
        faces.append((a, b, c))
        faces.append((a, c, d))

    for k in range(1, SIDES - 1):
        faces.append((ob[0], ob[k + 1], ob[k]))  # base, facing -z
        faces.append((fl[0], fl[k], fl[k + 1]))  # cavity floor, facing +z
    for k in range(SIDES):
        n = (k + 1) % SIDES
        quad(ob[k], ob[n], ot[n], ot[k])  # outer wall, facing away from the axis
        quad(ot[k], ot[n], it[n], it[k])  # rim, facing +z
        quad(it[k], it[n], fl[n], fl[k])  # inner wall, facing the axis
    return vertices, faces


def _unit_normal(a, b, c) -> tuple[float, float, float]:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    # Rounding first, then `+ 0.0`, so a component that is mathematically zero prints as zero.
    return (
        round(nx / length, 9) + 0.0,
        round(ny / length, 9) + 0.0,
        round(nz / length, 9) + 0.0,
    )


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


def thin_wall_stl_text() -> str:
    """The exact text this script writes for `FILENAME`."""
    vertices, faces = build_mesh()
    return stl_text(Path(FILENAME).stem, vertices, faces)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 if the committed file differs from what this script writes",
    )
    args = parser.parse_args(argv)

    path = OUTPUT_DIR / FILENAME
    text = thin_wall_stl_text()
    if args.check:
        if path.is_file() and path.read_bytes() == text.encode("utf-8"):
            print(f"{path} matches the generator")
            return 0
        print(f"differs from the generator: {path}")
        return 1

    path.write_bytes(text.encode("utf-8"))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

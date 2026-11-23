"""Render the expo poster's 3D figures from one real reconstruction session.

    PYTHONPATH=. .venv/bin/python scripts/render_expo_figures.py <session_dir> <out_dir>

Every pixel comes from the session's own artifacts: the served mesh with its per-vertex signed
deviation (`_DEVIATION_MM`, `_DEVIATION_MEASURED`, I-06), its provenance colours (R-09), and the
ChArUco board poses of the frames (R-02). Nothing is retouched. Colours follow the capture app's
heatmap (`capture/src/deviation-heatmap.ts`): blue is missing material, red is extra, white is
on-CAD, and grey is "Unobserved or not measured", the viewer's own words. A triangle is grey
unless all three of its vertices are Observed *and* have a measured deviation.

It writes `figure-facts.json` beside the images: the session's counts and coverage, each image's
pixel size, and where the board, a camera and the part land in the hero image, so the poster
can draw its labels as text instead of baking them into the picture.

matplotlib draws the figures but isn't in requirements.txt (D-036), so it is imported only by the
functions that draw. What the tests check (which vertices are Observed and measured, the face
colours and the coverage figure) imports without it, so CI runs them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from recon.board_pose import (
    BOARD_SQUARE_MM,
    BOARD_SQUARES_X,
    BOARD_SQUARES_Y,
)
from recon.mesh_cleanup import read_glb, read_glb_extra_attributes
from recon.provenance import LABEL_OBSERVED, colors_to_labels

NEG = np.array([37, 99, 235]) / 255
ZERO = np.array([250, 250, 250]) / 255
POS = np.array([185, 28, 28]) / 255
NOT_EVALUATED = np.array([100, 116, 139]) / 255
NAVY = "#0b3a66"
GOLD = "#c9a14a"

DEVIATION_GLB = Path("artifacts") / "uncertainty" / "mesh_uncertainty_deviation.glb"
UNCERTAINTY_JSON = Path("artifacts") / "uncertainty" / "uncertainty.json"
BOARD_POSES = Path("artifacts") / "board_pose" / "board_poses.npz"


def dev_rgb(d: np.ndarray, rng: float) -> np.ndarray:
    t = np.clip(d / rng, -1, 1)[:, None]
    neg = ZERO + (NEG - ZERO) * (-t)
    pos = ZERO + (POS - ZERO) * t
    return np.where(t < 0, neg, pos)


LIGHT = np.array([-0.35, -0.5, 0.8])


def shade(
    rgb: np.ndarray, tris: np.ndarray, light: np.ndarray = LIGHT, amb: float = 0.45
) -> np.ndarray:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = light / np.linalg.norm(light)
    lam = np.abs(n @ light)  # two-sided: open meshes have no reliable outside
    return np.clip(rgb * (amb + (1 - amb) * lam[:, None]), 0, 1)


def observed_and_measured(glb: Path) -> tuple[np.ndarray, ...]:
    """Vertices, faces, the Observed mask, the measured mask and the signed deviation.

    Provenance is read with R-09's own decoder. A file without provenance colours is refused
    rather than treated as all-Observed: on this poster, Observed is a claim about what the
    camera saw.
    """
    v, f, c = read_glb(glb)
    if c is None:
        raise ValueError(
            f"{glb} has no provenance colours, so its Observed surface is unknown"
        )
    observed = colors_to_labels(c) == LABEL_OBSERVED
    names = ["_DEVIATION_MM", "_DEVIATION_MEASURED"]
    ex = read_glb_extra_attributes(glb, names)
    missing = [name for name in names if name not in ex]
    if missing:
        raise ValueError(f"{glb} has no {', '.join(missing)} (I-06's deviation mesh)")
    measured = (ex["_DEVIATION_MEASURED"] >= 0.5) & observed
    return v.astype(float), f, observed, measured, ex["_DEVIATION_MM"]


def observed_area_fraction(session: Path) -> float:
    """The service's own coverage figure: the area of fully Observed faces over the mesh's area
    (`uncertainty.json`'s `observed_fraction`, R-10). The poster quotes this one, not a vertex
    count."""
    data = json.loads((session / UNCERTAINTY_JSON).read_text())
    return float(data["observed_fraction"])


def load_session(session: Path):
    v, f, observed, measured, dev = observed_and_measured(session / DEVIATION_GLB)
    bp = np.load(session / BOARD_POSES)
    ok = bp["pose_found"].astype(bool)
    E = bp["extrinsic"][ok]
    R, t = E[:, :, :3], E[:, :, 3]
    centres = -np.einsum("nji,nj->ni", R, t)
    return v, f, observed, measured, dev, R, centres


def up(p: np.ndarray) -> np.ndarray:
    """Board frame has +z into the board; flip so the part stands up."""
    q = np.array(p, dtype=float).copy()
    q[..., 2] *= -1
    return q


def face_colours(f, measured, dev, rng):
    tri_measured = measured[f].all(axis=1)
    d = dev[f].mean(axis=1)
    return np.where(tri_measured[:, None], dev_rgb(d, rng), NOT_EVALUATED)


def board_polys():
    polys, cols = [], []
    for i in range(BOARD_SQUARES_X):
        for j in range(BOARD_SQUARES_Y):
            x0, y0 = i * BOARD_SQUARE_MM, j * BOARD_SQUARE_MM
            s = BOARD_SQUARE_MM
            polys.append(
                [[x0, y0, 0], [x0 + s, y0, 0], [x0 + s, y0 + s, 0], [x0, y0 + s, 0]]
            )
            cols.append((0.13, 0.15, 0.18) if (i + j) % 2 == 0 else (0.96, 0.95, 0.92))
    # paper margin around the board
    m = 14
    W, H = BOARD_SQUARES_X * BOARD_SQUARE_MM, BOARD_SQUARES_Y * BOARD_SQUARE_MM
    paper = [[-m, -m, 0.05], [W + m, -m, 0.05], [W + m, H + m, 0.05], [-m, H + m, 0.05]]
    return paper, polys, cols


def frustum(Rw, c, scale=24.0, aspect=4 / 3):
    """Camera pyramid in board coordinates (camera looks along +z of its own frame)."""
    w, h, d = scale * aspect / 2, scale / 2, scale * 1.15
    corners_cam = np.array([[-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d]])
    return corners_cam @ Rw + c  # x_w = R^T x_c + c


def _pyplot():
    """matplotlib's pyplot on the file-only Agg backend, imported on first use."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _image_xy(fig, ax, point) -> np.ndarray:
    """Where a 3D point lands in the saved image, in pixels from the top-left corner."""
    from mpl_toolkits.mplot3d import proj3d

    x, y, _ = proj3d.proj_transform(point[0], point[1], point[2], ax.get_proj())
    dx, dy = ax.transData.transform((x, y))
    return np.array([dx, fig.bbox.height - dy])


def _save_trimmed(fig, out: Path, anchors: dict[str, np.ndarray]) -> dict:
    """Save the whole canvas, trim the transparent margin, and express the anchors as fractions
    of the trimmed image."""
    fig.savefig(out, transparent=True, dpi=fig.dpi)
    _pyplot().close(fig)
    im = Image.open(out)
    left, top, right, bottom = im.getbbox()
    im.crop((left, top, right, bottom)).save(out)
    w, h = right - left, bottom - top
    return {
        "px": [w, h],
        "anchors": {
            name: [
                round(float((p[0] - left) / w), 4),
                round(float((p[1] - top) / h), 4),
            ]
            for name, p in anchors.items()
        },
    }


def render_scene(
    session: Path,
    out: Path,
    rng: float,
    *,
    figsize: tuple[float, float] = (14, 10),
    zoom: float = 1.75,
    mid_z: float = 75.0,
    elev: float = 32.0,
    azim: float = -112.0,
) -> dict:
    """The board, the part and the cameras that photographed it. The defaults were picked by
    eye on sess_f82401bb6852: the board fills the frame, the part shows its top and a side, and
    the cameras above it are cut off at the top edge."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    v, f, _observed, measured, dev, R, C = load_session(session)
    tris = v[f]
    rgb = shade(face_colours(f, measured, dev, rng), up(tris))

    fig = _pyplot().figure(figsize=figsize, dpi=200)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", computed_zorder=False)
    ax.set_facecolor((0, 0, 0, 0))
    ax.set_proj_type("persp", focal_length=0.6)

    # frame the board and the part; cameras past the edge are cut off
    W, H = BOARD_SQUARES_X * BOARD_SQUARE_MM, BOARD_SQUARES_Y * BOARD_SQUARE_MM
    mid = np.array([W / 2, H / 2, mid_z])
    half = 150.0
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.canvas.draw()
    cc = up(C)
    camera_xy = np.array([_image_xy(fig, ax, p) for p in cc])
    on_canvas = (
        (camera_xy[:, 0] > 0.04 * fig.bbox.width)
        & (camera_xy[:, 0] < 0.96 * fig.bbox.width)
        & (camera_xy[:, 1] > 0.04 * fig.bbox.height)
    )
    # Rays only from the cameras above the part, in the middle of the frame. A low side shot's
    # ray runs almost flat across the whole picture and reads as a stray line.
    with_ray = (
        (camera_xy[:, 0] > 0.15 * fig.bbox.width)
        & (camera_xy[:, 0] < 0.85 * fig.bbox.width)
        & (camera_xy[:, 1] > 0.04 * fig.bbox.height)
        & (camera_xy[:, 1] < 0.5 * fig.bbox.height)
    )

    paper, bpolys, bcols = board_polys()
    ax.add_collection3d(
        Poly3DCollection(
            [up(np.array(paper))],
            facecolors=[(1, 1, 1)],
            edgecolors=[(0.75, 0.72, 0.65)],
            linewidths=0.8,
            zorder=0,
        )
    )
    ax.add_collection3d(
        Poly3DCollection(
            [up(np.array(p)) for p in bpolys],
            facecolors=bcols,
            edgecolors="none",
            zorder=1,
        )
    )

    # part: painter's sort within one collection
    ax.add_collection3d(
        Poly3DCollection(
            up(tris), facecolors=rgb, edgecolors=rgb * 0.92, linewidths=0.15, zorder=3
        )
    )

    # cameras: small, light pyramids, so 29 of them don't crowd out the board and the part, and
    # faint rays to the part from the ones chosen above
    centre = v[measured].mean(axis=0) if measured.any() else v.mean(axis=0)
    rays, pyr_lines, pyr_faces = [], [], []
    for Ri, ci, ray in zip(R, C, with_ray):
        corners = frustum(Ri, ci, scale=18.0)
        if ray:
            rays.append([up(ci), up(centre)])
        for k in range(4):
            pyr_lines.append([up(ci), up(corners[k])])
            pyr_lines.append([up(corners[k]), up(corners[(k + 1) % 4])])
        pyr_faces.append(up(corners))
    ax.add_collection3d(
        Line3DCollection(
            rays, colors=[(0.79, 0.63, 0.29, 0.16)], linewidths=0.6, zorder=2
        )
    )
    ax.add_collection3d(
        Poly3DCollection(
            pyr_faces, facecolors=[(0.85, 0.9, 1.0, 0.3)], edgecolors="none", zorder=4
        )
    )
    ax.add_collection3d(
        Line3DCollection(
            pyr_lines, colors=[(0.04, 0.23, 0.4, 0.55)], linewidths=0.9, zorder=5
        )
    )
    ax.scatter(cc[:, 0], cc[:, 1], cc[:, 2], s=10, c=GOLD, depthshade=False, zorder=6)

    part = up(v[measured] if measured.any() else v)
    part_top = np.array([part[:, 0].mean(), part[:, 1].mean(), part[:, 2].max()])
    board_corners = up(np.array([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]], float))
    corner_xy = np.array([_image_xy(fig, ax, p) for p in board_corners])
    near = board_corners[int(np.argmax(corner_xy[:, 1]))]  # lowest on screen: nearest
    board_label = near + 0.18 * (board_corners.mean(axis=0) - near)
    pick = np.flatnonzero(on_canvas)[int(np.argmin(camera_xy[on_canvas, 1]))]
    anchors = {
        "part": _image_xy(fig, ax, part_top),
        "board": _image_xy(fig, ax, board_label),
        "camera": camera_xy[pick],
    }
    saved = _save_trimmed(fig, out, anchors)
    return {"cameras": len(C), "vertices": len(v), "faces": len(f), **saved}


def render_closeup(session: Path, out: Path, rng: float, azim=-60, elev=30) -> dict:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    v, f, observed, measured, dev, _, _ = load_session(session)
    tris = v[f]
    rgb = shade(face_colours(f, measured, dev, rng), up(tris))
    fig = _pyplot().figure(figsize=(10, 8), dpi=260)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", computed_zorder=False)
    ax.set_facecolor((0, 0, 0, 0))
    ax.set_proj_type("persp", focal_length=0.7)
    ax.add_collection3d(
        Poly3DCollection(
            up(tris), facecolors=rgb, edgecolors=rgb * 0.9, linewidths=0.2, zorder=2
        )
    )
    p = up(v)
    lo, hi = p.min(0), p.max(0)
    mid, half = (lo + hi) / 2, (hi - lo).max() / 2 * 0.62
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.canvas.draw()
    saved = _save_trimmed(fig, out, {})
    d = dev[measured]
    return {
        "px": saved["px"],
        "observed_vertex_fraction": float(observed.mean()),
        "measured_vertices": int(measured.sum()),
        "dev_p5_p50_p95_mm": [float(x) for x in np.percentile(d, [5, 50, 95])],
        "abs_dev_median_mm": float(np.median(np.abs(d))),
        "range_mm": rng,
    }


def colourbar(out: Path, rng: float) -> None:
    """A plain colour strip; the poster draws its tick labels as text."""
    x = np.linspace(-rng, rng, 1200)
    rgb = (dev_rgb(x, rng) * 255).round().astype(np.uint8)
    Image.fromarray(np.repeat(rgb[None], 60, axis=0)).save(out)


def main() -> None:
    session, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rng = 4.0
    facts = {
        "session": session.name,
        "observed_area_fraction": observed_area_fraction(session),
    }
    facts["scene"] = render_scene(session, out / "hero-scene.png", rng)
    facts["closeup"] = render_closeup(session, out / "heatmap-closeup.png", rng)
    colourbar(out / "heatmap-bar.png", rng)
    (out / "figure-facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()

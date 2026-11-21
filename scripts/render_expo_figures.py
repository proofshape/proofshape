"""Render the expo poster's 3D figures from one real reconstruction session.

    PYTHONPATH=. .venv/bin/python scripts/render_expo_figures.py <session_dir> <out_dir>

Every pixel comes from the session's own artifacts: the served mesh with its per-vertex signed
deviation (`_DEVIATION_MM`, `_DEVIATION_MEASURED`, I-06), its provenance colours (R-09), and the
ChArUco board poses of the frames (R-02). Nothing is retouched. Colours follow the capture app's
heatmap (`capture/src/deviation-heatmap.ts`): blue is missing material, red is extra, white is
on-CAD, slate is never observed (measured nowhere, judged nowhere).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from recon.board_pose import (
    BOARD_SQUARE_MM,
    BOARD_SQUARES_X,
    BOARD_SQUARES_Y,
)
from recon.mesh_cleanup import read_glb, read_glb_extra_attributes

NEG = np.array([37, 99, 235]) / 255
ZERO = np.array([250, 250, 250]) / 255
POS = np.array([185, 28, 28]) / 255
UNOBSERVED = np.array([100, 116, 139]) / 255
NAVY = "#0b3a66"
GOLD = "#c9a14a"


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


def load_session(session: Path):
    art = session / "artifacts"
    glb = art / "uncertainty" / "mesh_uncertainty_deviation.glb"
    v, f, c = read_glb(glb)
    ex = read_glb_extra_attributes(glb, ["_DEVIATION_MM", "_DEVIATION_MEASURED"])
    observed = np.all(c == 255, axis=1) if c is not None else np.ones(len(v), bool)
    measured = (ex["_DEVIATION_MEASURED"] >= 0.5) & observed
    dev = ex["_DEVIATION_MM"]
    bp = np.load(art / "board_pose" / "board_poses.npz")
    ok = bp["pose_found"].astype(bool)
    E = bp["extrinsic"][ok]
    R, t = E[:, :, :3], E[:, :, 3]
    centres = -np.einsum("nji,nj->ni", R, t)
    return v.astype(float), f, observed, measured, dev, R, centres


def up(p: np.ndarray) -> np.ndarray:
    """Board frame has +z into the board; flip so the part stands up."""
    q = np.array(p, dtype=float).copy()
    q[..., 2] *= -1
    return q


def face_colours(v, f, measured, dev, rng):
    tri_measured = measured[f].all(axis=1)
    d = dev[f].mean(axis=1)
    rgb = np.where(tri_measured[:, None], dev_rgb(d, rng), UNOBSERVED)
    return rgb


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
    corners = (
        corners_cam @ Rw + c
    )  # Rw is world->cam R; R^T maps cam->world: x_w = R^T x_c + c
    return corners


def render_scene(session: Path, out: Path, rng: float) -> dict:
    v, f, _observed, measured, dev, R, C = load_session(session)
    tris = v[f]
    rgb = shade(face_colours(v, f, measured, dev, rng), up(tris))

    fig = plt.figure(figsize=(16, 10), dpi=220)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", computed_zorder=False)
    ax.set_facecolor((0, 0, 0, 0))
    ax.set_proj_type("persp", focal_length=0.55)

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
    mesh = Poly3DCollection(
        up(tris), facecolors=rgb, edgecolors=rgb * 0.92, linewidths=0.15, zorder=3
    )
    ax.add_collection3d(mesh)

    # cameras: rays to the part, then pyramids
    centre = v[measured].mean(axis=0) if measured.any() else v.mean(axis=0)
    rays, pyr_lines, pyr_faces = [], [], []
    for Ri, ci in zip(R, C):
        corners = frustum(Ri, ci)
        rays.append([up(ci), up(centre)])
        for k in range(4):
            pyr_lines.append([up(ci), up(corners[k])])
            pyr_lines.append([up(corners[k]), up(corners[(k + 1) % 4])])
        pyr_faces.append(up(corners))
    ax.add_collection3d(
        Line3DCollection(
            rays, colors=[(0.79, 0.63, 0.29, 0.28)], linewidths=0.7, zorder=2
        )
    )
    ax.add_collection3d(
        Poly3DCollection(
            pyr_faces, facecolors=[(0.85, 0.9, 1.0, 0.55)], edgecolors="none", zorder=4
        )
    )
    ax.add_collection3d(
        Line3DCollection(pyr_lines, colors=[NAVY], linewidths=1.3, zorder=5)
    )
    cc = up(C)
    ax.scatter(cc[:, 0], cc[:, 1], cc[:, 2], s=14, c=GOLD, depthshade=False, zorder=6)

    allp = np.vstack([up(v), cc, up(np.array(paper))])
    lo, hi = allp.min(0), allp.max(0)
    mid, half = (lo + hi) / 2, (hi - lo).max() / 2 * 0.6
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(lo[2] - 5, lo[2] - 5 + 2 * half)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-118)
    ax.set_axis_off()
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return {"cameras": len(C), "vertices": len(v), "faces": len(f)}


def render_closeup(session: Path, out: Path, rng: float, azim=-60, elev=30) -> dict:
    v, f, observed, measured, dev, _, _ = load_session(session)
    tris = v[f]
    rgb = shade(face_colours(v, f, measured, dev, rng), up(tris))
    fig = plt.figure(figsize=(10, 8), dpi=260)
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
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    d = dev[measured]
    return {
        "observed_vertex_fraction": float(observed.mean()),
        "measured_vertices": int(measured.sum()),
        "dev_p5_p50_p95_mm": [float(x) for x in np.percentile(d, [5, 50, 95])],
        "abs_dev_median_mm": float(np.median(np.abs(d))),
        "range_mm": rng,
    }


def colourbar(out: Path, rng: float) -> None:
    """A plain colour strip; the poster draws its tick labels as text."""
    from PIL import Image

    x = np.linspace(-rng, rng, 1200)
    rgb = (dev_rgb(x, rng) * 255).round().astype(np.uint8)
    Image.fromarray(np.repeat(rgb[None], 60, axis=0)).save(out)


def main() -> None:
    session, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rng = 4.0
    facts = {"session": session.name}
    facts["scene"] = render_scene(session, out / "hero-scene.png", rng)
    facts["closeup"] = render_closeup(session, out / "heatmap-closeup.png", rng)
    colourbar(out / "heatmap-bar.png", rng)
    for name in (
        "hero-scene.png",
        "heatmap-closeup.png",
    ):  # trim the transparent margin
        from PIL import Image

        im = Image.open(out / name)
        im.crop(im.getbbox()).save(out / name)
    (out / "figure-facts.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()

"""G-05: the defect ladder's CAD files are exactly what the generator writes, are closed solids of
the designed shape, differ from the plate only by their own defect, and register to the control.
"""

from __future__ import annotations

import math
import re
import runpy
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from inspection import registration, synthetic
from inspection.cad import load_mesh

from recon import provenance
from recon.board_pose import MIN_CORNERS_FOR_POSE, make_board
from recon.cad_size_check import prism_from_cad
from recon.part_segmentation import DEFAULT_FOOTPRINT_MM

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "reference_cad"
LADDER = runpy.run_path(str(ROOT / "scripts" / "make_g05_defect_ladder.py"))
main = LADDER["main"]
build_mesh = LADDER["build_mesh"]
TEXTS = LADDER["ladder_stl_texts"]()

CONTROL = "g05_00_control_50x32x20mm.stl"

# Written out here rather than read from the generator, so a change to its constants fails these
# tests instead of quietly moving the expectations with it.
# (file, volume mm^3, top of the part mm, hole x range mm, height over the defect site mm)
EXPECTED = [
    (CONTROL, 25600.0, 20.0, (6.0, 18.0), 20.0),
    ("g05_01_hole_shift_2mm.stl", 25600.0, 20.0, (8.0, 20.0), 20.0),
    ("g05_02_hole_shift_4mm.stl", 25600.0, 20.0, (10.0, 22.0), 20.0),
    ("g05_03_hole_shift_8mm.stl", 25600.0, 20.0, (14.0, 26.0), 20.0),
    ("g05_04_pocket_2mm.stl", 25400.0, 20.0, (6.0, 18.0), 18.0),
    ("g05_05_pocket_4mm.stl", 25200.0, 20.0, (6.0, 18.0), 16.0),
    ("g05_06_pocket_8mm.stl", 24800.0, 20.0, (6.0, 18.0), 12.0),
    ("g05_07_bump_2mm.stl", 25800.0, 22.0, (6.0, 18.0), 22.0),
    ("g05_08_bump_4mm.stl", 26000.0, 24.0, (6.0, 18.0), 24.0),
    ("g05_09_bump_8mm.stl", 26400.0, 28.0, (6.0, 18.0), 28.0),
]
FILES = [row[0] for row in EXPECTED]
PLATE_HEIGHT_MM = 20.0
HOLE_Y_MM = (10.0, 22.0)
NOTCH_XY_MM = (34.0, 50.0, 21.0, 32.0)
SITE_XY_MM = (35.0, 45.0, 5.0, 15.0)

# I-05's test settings (tests/test_registration.py), not the production defaults.
TEST_SAMPLE_COUNT = 700
TEST_YAW_STEPS = 36
TEST_CANDIDATE_COUNT = 6
TEST_RANSAC_ITERATIONS = 3_000

_FACET = re.compile(
    r"facet normal (\S+) (\S+) (\S+)\s+outer loop\s+"
    r"vertex (\S+) (\S+) (\S+)\s+vertex (\S+) (\S+) (\S+)\s+vertex (\S+) (\S+) (\S+)"
)


def _arrays(mesh) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = mesh
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _is_closed_and_consistently_wound(faces: np.ndarray) -> bool:
    # G-02's topological check (#269): every directed edge appears exactly once and its reverse
    # exists, so one flipped or missing facet anywhere fails it, whatever plane it lies in.
    directed_edges: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            directed_edges[(u, v)] = directed_edges.get((u, v), 0) + 1
    if any(count != 1 for count in directed_edges.values()):
        return False
    return all((v, u) in directed_edges for (u, v) in directed_edges)


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    # About a point on none of the parts' facet planes, which all lie at whole millimetres, so no
    # facet's contribution vanishes whatever its winding: the origin is on the base and on two
    # walls, the blind spot #269's review found in an origin-referenced sum.
    triangles = vertices[faces] - np.array([math.pi, math.e, math.sqrt(2.0)])
    return float(
        np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum()
        / 6.0
    )


def _top_heights(vertices: np.ndarray, faces: np.ndarray):
    """The part's top straight down at every 1 mm cell centre, 0 where the ray meets nothing.

    Probes the loaded geometry itself, so it doesn't care how a file happens to be triangulated.
    """
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(vertices.astype(np.float32)),
        o3d.core.Tensor(faces.astype(np.uint32)),
    )
    xs, ys = np.meshgrid(np.arange(50) + 0.5, np.arange(32) + 0.5, indexing="ij")
    start_mm = 100.0
    rays = np.stack(
        [
            xs,
            ys,
            np.full_like(xs, start_mm),
            np.zeros_like(xs),
            np.zeros_like(xs),
            -np.ones_like(xs),
        ],
        axis=-1,
    )
    t_hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()
    return xs, ys, np.where(np.isfinite(t_hit), start_mm - t_hit, 0.0)


def _inside(rectangle, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return (
        (xs > rectangle[0])
        & (xs < rectangle[1])
        & (ys > rectangle[2])
        & (ys < rectangle[3])
    )


def _designed_heights(xs, ys, hole_x_mm, site_mm) -> np.ndarray:
    heights = np.full(xs.shape, PLATE_HEIGHT_MM)
    heights[_inside(SITE_XY_MM, xs, ys)] = site_mm
    heights[_inside(NOTCH_XY_MM, xs, ys)] = 0.0
    heights[_inside((*hole_x_mm, *HOLE_Y_MM), xs, ys)] = 0.0
    return heights


@pytest.mark.parametrize("filename", FILES)
def test_committed_file_is_exactly_what_the_generator_writes(filename):
    assert sorted(TEXTS) == sorted(FILES)
    assert (FIXTURES / filename).read_bytes() == TEXTS[filename].encode("utf-8")


def test_check_mode_passes_on_the_committed_files_and_catches_a_changed_one(
    tmp_path, monkeypatch, capsys
):
    assert main(["--check"]) == 0

    monkeypatch.setitem(main.__globals__, "OUTPUT_DIR", tmp_path)
    assert main([]) == 0
    assert main(["--check"]) == 0
    changed = tmp_path / CONTROL
    changed.write_bytes(
        changed.read_bytes().replace(b"vertex 6.000000", b"vertex 7.000000", 1)
    )
    capsys.readouterr()

    assert main(["--check"]) == 1
    assert CONTROL in capsys.readouterr().out


@pytest.mark.parametrize(("filename", "_volume", "top_mm", "_hole", "_site"), EXPECTED)
def test_part_loads_in_mm_with_its_base_on_z0_and_its_designed_extent(
    filename, _volume, top_mm, _hole, _site
):
    vertices, _faces, units = load_mesh(FIXTURES / filename)

    assert units == "mm"
    # Exactly 0: I-01's synthetic generator refuses anything else, and I-03 assumes it.
    np.testing.assert_array_equal(vertices.min(axis=0), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(vertices.max(axis=0), [50.0, 32.0, top_mm])


@pytest.mark.parametrize("filename", FILES)
def test_part_is_deliberately_not_an_r22_prism(filename):
    # The hole fails prism_from_cad's convex-hull check, and a pocket or bump adds a z level, so
    # R-22's size check can't measure these parts. Pinned, so nobody expects it to.
    vertices, _faces, _units = load_mesh(FIXTURES / filename)

    with pytest.raises(ValueError):
        prism_from_cad(vertices)


@pytest.mark.parametrize(("filename", "volume_mm3", "_top", "_hole", "_site"), EXPECTED)
def test_part_is_a_closed_outward_wound_solid_of_its_designed_volume(
    filename, volume_mm3, _top, _hole, _site
):
    vertices, faces, _units = load_mesh(FIXTURES / filename)

    assert _is_closed_and_consistently_wound(faces)
    assert _signed_volume(vertices, faces) == pytest.approx(volume_mm3, rel=1e-9)


@pytest.mark.parametrize("filename", FILES)
def test_stored_facet_normals_are_unit_and_agree_with_the_winding(filename):
    # load_mesh ignores the stored normals, but a slicer may not; nothing else checks them.
    facets = _FACET.findall((FIXTURES / filename).read_text(encoding="utf-8"))
    assert len(facets) == len(load_mesh(FIXTURES / filename)[1])

    for values in facets:
        normal = np.array(values[:3], dtype=np.float64)
        a, b, c = np.array(values[3:], dtype=np.float64).reshape(3, 3)
        winding = np.cross(b - a, c - a)
        assert np.linalg.norm(normal) == pytest.approx(1.0)
        assert normal @ (winding / np.linalg.norm(winding)) == pytest.approx(1.0)


def test_the_mesh_checks_catch_a_flipped_base_facet_and_a_missing_face():
    vertices, faces, _units = load_mesh(FIXTURES / CONTROL)
    base = int(np.flatnonzero(np.all(vertices[faces][:, :, 2] == 0.0, axis=1))[0])
    flipped = faces.copy()
    flipped[base] = flipped[base][::-1]

    assert not _is_closed_and_consistently_wound(flipped)
    assert _signed_volume(vertices, flipped) != pytest.approx(25600.0, rel=1e-9)
    assert not _is_closed_and_consistently_wound(np.delete(faces, 0, axis=0))


@pytest.mark.parametrize(
    ("filename", "_volume", "_top", "hole_x_mm", "site_mm"), EXPECTED
)
def test_part_differs_from_the_plain_plate_only_by_its_own_defect(
    filename, _volume, _top, hole_x_mm, site_mm
):
    xs, ys, heights = _top_heights(*load_mesh(FIXTURES / filename)[:2])

    np.testing.assert_allclose(
        heights, _designed_heights(xs, ys, hole_x_mm, site_mm), atol=1e-4
    )


def test_the_height_probe_catches_a_moved_defect_site(monkeypatch):
    # The same 4 mm pocket, built with its site 2 mm along +x, must not pass for the real one.
    moved = (37.0, 47.0, 5.0, 15.0)
    monkeypatch.setitem(build_mesh.__globals__, "DEFECT_SITE_XY_MM", moved)
    xs, ys, heights = _top_heights(*_arrays(LADDER["part_mesh"](0.0, 16.0)))

    assert not np.allclose(heights, _designed_heights(xs, ys, (6.0, 18.0), 16.0))


@pytest.mark.parametrize("filename", FILES)
def test_part_hides_few_enough_board_corners_for_r23(filename):
    # G-02's argument (#269), with the bounding box taken from the vertices: prism_from_cad
    # rejects these parts. Straight overhead with the part centred; an oblique frame hides more
    # behind the part, so this is a floor under the real count, not a margin above it.
    all_corners_mm = np.asarray(make_board().getChessboardCorners(), dtype=np.float64)[
        :, :2
    ]
    board_center_mm = np.asarray(DEFAULT_FOOTPRINT_MM, dtype=np.float64) / 2.0
    vertices, _faces, _units = load_mesh(FIXTURES / filename)
    half_bbox_mm = (vertices[:, :2].max(axis=0) - vertices[:, :2].min(axis=0)) / 2.0

    hidden = np.all(
        (all_corners_mm >= board_center_mm - half_bbox_mm)
        & (all_corners_mm <= board_center_mm + half_bbox_mm),
        axis=1,
    )
    visible_count = len(all_corners_mm) - int(np.count_nonzero(hidden))

    assert visible_count >= MIN_CORNERS_FOR_POSE * 2


def _expected_transform(sample: synthetic.SyntheticSample) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(sample.sidecar["transform"]["rotation"], dtype=float)
    transform[:3, 3] = np.asarray(
        sample.sidecar["transform"]["translation_mm"], dtype=float
    )
    return transform


def _yaw_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    relative = actual[:3, :3] @ expected[:3, :3].T
    yaw = math.degrees(math.atan2(relative[1, 0], relative[0, 0])) % 360.0
    return min(yaw, 360.0 - yaw)


def _centre_error_mm(cad_vertices, actual: np.ndarray, expected: np.ndarray) -> float:
    centre = cad_vertices.mean(axis=0)
    moved = actual[:3, :3] @ centre + actual[:3, 3]
    true = expected[:3, :3] @ centre + expected[:3, 3]
    return float(np.linalg.norm(moved - true))


def _check_ambiguity(part, cad, *, seed: int):
    """I-05 on an I-01 synthetic scan of `part`, with the board-contact face hidden as well as
    I-01's random patch, since a real scan never sees the face the part stands on (D-061)."""
    sample = synthetic.generate_synthetic_sample(
        *part, seed, sigma_mm=0.3, patch_fraction=0.2
    )
    expected = _expected_transform(sample)
    cad_frame = (sample.vertices_mm - expected[:3, 3]) @ expected[:3, :3]
    labels = np.asarray(sample.labels, dtype=object).copy()
    labels[np.abs(cad_frame[:, 2]) <= 0.75] = provenance.LABEL_UNOBSERVED
    model = synthetic.SyntheticSample(
        vertices_mm=sample.vertices_mm,
        faces=sample.faces,
        labels=labels,
        colors=provenance.labels_to_colors(labels),
        sidecar=dict(sample.sidecar),
    )
    candidates = registration.global_register(
        model,
        cad,
        sample_count=TEST_SAMPLE_COUNT,
        yaw_steps=TEST_YAW_STEPS,
        candidate_count=TEST_CANDIDATE_COUNT,
        ransac_iterations=TEST_RANSAC_ITERATIONS,
        seed=100 + seed,
    )
    return registration.check_ambiguity(candidates, model, cad, seed=seed), expected


# Each part as the scanned part, the control as the buyer's CAD. The comment is the runner-up's
# RMSE over the best's, measured 2026-11-09 with these settings; D-061 flags at 1.20 or less. Over
# seeds 0-7 none of the ten is flagged; the closest is the 8 mm shift's seed 6, at 1.22 (notes).
REGISTRATION_CASES = [
    (CONTROL, 0),  # 4.07
    ("g05_01_hole_shift_2mm.stl", 0),  # 1.85
    ("g05_02_hole_shift_4mm.stl", 0),  # 2.12
    ("g05_02_hole_shift_4mm.stl", 2),  # 2.69 -- M4's part, so a second seed
    ("g05_03_hole_shift_8mm.stl", 0),  # 2.19
    ("g05_04_pocket_2mm.stl", 0),  # 2.41
    ("g05_05_pocket_4mm.stl", 0),  # 2.14
    ("g05_06_pocket_8mm.stl", 0),  # 3.45
    ("g05_07_bump_2mm.stl", 0),  # 3.16
    ("g05_08_bump_4mm.stl", 0),  # 2.94
    ("g05_09_bump_8mm.stl", 0),  # 1.85
]


@pytest.mark.parametrize(("filename", "seed"), REGISTRATION_CASES)
def test_part_registers_to_its_true_pose_on_the_control_without_ambiguity(
    filename, seed
):
    cad = load_mesh(FIXTURES / CONTROL)[:2]

    check, expected = _check_ambiguity(
        load_mesh(FIXTURES / filename)[:2], cad, seed=seed
    )

    assert _yaw_error_deg(check.best.transformation, expected) < 1.0
    assert _centre_error_mm(cad[0], check.best.transformation, expected) < 0.5
    assert not check.orientation_ambiguous
    assert check.reason_codes == ()


def test_the_notch_fixes_orientation_where_a_plain_plate_cannot():
    # D-008's case: the part has lost the hole that would break its symmetry. On the L-plate the
    # notch still fixes the pose (ratio 1.85); the same plate without its notch, against its own
    # holed CAD, is ambiguous (0.96, seed 0) -- which is what the notch is for.
    cad = load_mesh(FIXTURES / CONTROL)[:2]
    check, expected = _check_ambiguity(_arrays(build_mesh(None, None)), cad, seed=0)

    assert not check.orientation_ambiguous
    assert _yaw_error_deg(check.best.transformation, expected) < 1.0

    plain_cad = _arrays(build_mesh(LADDER["HOLE_XY_MM"], None, None))
    plain_check, _expected = _check_ambiguity(
        _arrays(build_mesh(None, None, None)), plain_cad, seed=0
    )

    assert plain_check.orientation_ambiguous

"""A-04: the file contract between the completion stage and its runner (recon.completion_contract)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from recon import completion_contract as contract
from recon import tsdf_fusion

CONTRACT_SOURCE = Path(contract.__file__)


def _request(seeds=(0, 1, 2)) -> contract.CompletionRequest:
    return contract.CompletionRequest(seeds=tuple(seeds), budget_s=60.0)


def _result_payload(**overrides):
    payload = {
        "schema_version": 1,
        "status": "complete",
        "model": contract.ExpectedModel().as_dict(),
        "load_s": 2.0,
        "samples": [
            {"k": 0, "seed": 0, "time_s": 1.5},
            {"k": 1, "seed": 1, "time_s": 1.25},
        ],
        "error": None,
    }
    payload.update(overrides)
    return payload


def _write_result(tmp_path: Path, payload) -> Path:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- the frame change -----------------------------------------------------------------------------


def test_board_to_z_up_is_a_rotation_that_is_its_own_inverse():
    # det +1 is the I-01 lesson: diag(1, 1, -1) also puts z up, but mirrors the part.
    assert np.linalg.det(contract.BOARD_TO_Z_UP) == pytest.approx(1.0)
    points = np.array([[10.0, 20.0, -5.0], [1.0, -2.0, 3.0]])

    np.testing.assert_allclose(
        contract.z_up_to_board(contract.board_to_z_up(points)), points
    )


def test_board_to_z_up_puts_a_point_above_the_board_at_positive_z():
    # The board frame's +z points into the table, so a point 5 mm above the paper has z = -5.
    above_board = np.array([[30.0, 40.0, -5.0]])

    z_up = contract.board_to_z_up(above_board)

    np.testing.assert_allclose(z_up, [[30.0, -40.0, 5.0]])


# --- request.json ---------------------------------------------------------------------------------


def test_request_round_trips_with_relative_fixed_paths(tmp_path):
    request = contract.CompletionRequest(
        seeds=(3, 7), budget_s=120.0, sparse_structure_steps=50, slat_steps=None
    )

    path = contract.write_request(tmp_path, request)
    written = json.loads(path.read_text(encoding="utf-8"))

    assert contract.read_request(path) == request
    # Relative paths are what let the whole folder be copied to a CUDA host and back (A-12).
    assert written["partial_points"] == "input/partial_zup.ply"
    assert written["image"] == "input/view.png"
    assert written["output_dir"] == "runner"
    assert written["expected_model"]["commit"] == contract.MODEL_COMMIT


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data.update(extra=1), "extra"),
        (lambda data: data.update(image="/etc/passwd"), "'image'"),
        (lambda data: data.update(seeds=[1, 1]), "distinct"),
        (lambda data: data.update(seeds=[True]), "seed"),
        (lambda data: data.update(budget_s=float("nan")), "budget_s"),
        (lambda data: data["model_options"].update(slat_steps=0), "slat_steps"),
    ],
)
def test_read_request_rejects_anything_off_contract(tmp_path, change, message):
    path = contract.write_request(tmp_path, _request())
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(contract.ContractError, match=message):
        contract.read_request(path)


# --- runner/result.json ---------------------------------------------------------------------------


def test_parse_result_accepts_a_well_formed_result(tmp_path):
    result = contract.parse_result(
        _write_result(tmp_path, _result_payload()), _request()
    )

    assert result.status == "complete"
    assert [(sample.k, sample.seed) for sample in result.samples] == [(0, 0), (1, 1)]


def test_result_payload_round_trips_through_parse_result(tmp_path):
    result = contract.RunnerResult(
        status="budget_reached",
        model=contract.ExpectedModel().as_dict(),
        load_s=1.0,
        samples=(contract.SampleRecord(k=0, seed=0, time_s=2.0),),
        error=None,
    )

    parsed = contract.parse_result(
        _write_result(tmp_path, contract.result_payload(result)), _request()
    )

    assert parsed == result


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"extra": True}, "extra"),
        ({"status": "done"}, "status"),
        ({"model": {**contract.ExpectedModel().as_dict(), "commit": "abc"}}, "pins"),
        ({"samples": [{"k": 0, "seed": 5, "time_s": 1.0}]}, "seed"),
        ({"samples": [{"k": 1, "seed": 1, "time_s": 1.0}]}, "k=1"),
        ({"samples": [{"k": 0, "seed": 0, "time_s": -1.0}]}, "time_s"),
        ({"samples": [{"k": 0, "seed": 0, "time_s": 1.0, "path": "x"}]}, "extra"),
        ({"status": "failed"}, "error"),
        ({"error": "boom"}, "error"),
        ({"load_s": None}, "load_s"),
        ({"schema_version": 2}, "schema_version"),
    ],
)
def test_parse_result_rejects_anything_off_contract(tmp_path, overrides, message):
    path = _write_result(tmp_path, _result_payload(**overrides))

    with pytest.raises(contract.ContractError, match=message):
        contract.parse_result(path, _request())


def test_parse_result_rejects_more_samples_than_requested(tmp_path):
    path = _write_result(tmp_path, _result_payload())

    with pytest.raises(contract.ContractError, match="more than"):
        contract.parse_result(path, _request(seeds=(0,)))


def test_parse_result_reports_a_missing_or_unreadable_file(tmp_path):
    with pytest.raises(contract.ContractError, match="does not exist"):
        contract.parse_result(tmp_path / "result.json", _request())

    (tmp_path / "result.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(contract.ContractError, match="not readable JSON"):
        contract.parse_result(tmp_path / "result.json", _request())


def test_write_json_atomic_leaves_no_temporary_file(tmp_path):
    contract.write_json_atomic(tmp_path / "result.json", {"a": 1})

    assert sorted(path.name for path in tmp_path.iterdir()) == ["result.json"]


# --- the mesh file --------------------------------------------------------------------------------


def _tetrahedron():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return vertices, faces


def test_mesh_writer_matches_tsdf_fusions_layout_byte_for_byte(tmp_path):
    # The contract duplicates the writer because the runner can't import recon.tsdf_fusion;
    # this pins the copy to the original.
    vertices, faces = _tetrahedron()
    contract.write_mesh_ply(tmp_path / "contract.ply", vertices, faces)
    tsdf_fusion.write_mesh_ply(tmp_path / "tsdf.ply", vertices, faces)

    assert (tmp_path / "contract.ply").read_bytes() == (
        tmp_path / "tsdf.ply"
    ).read_bytes()


def test_strict_mesh_reader_round_trips(tmp_path):
    vertices, faces = _tetrahedron()
    contract.write_mesh_ply(tmp_path / "mesh.ply", vertices, faces)

    read_vertices, read_faces = contract.read_mesh_ply_strict(tmp_path / "mesh.ply")

    np.testing.assert_allclose(read_vertices, vertices)
    np.testing.assert_array_equal(read_faces, faces)


def _corrupt_truncated(path: Path) -> None:
    path.write_bytes(path.read_bytes()[:-5])


def _corrupt_nan(path: Path) -> None:
    vertices, faces = _tetrahedron()
    vertices[1, 0] = np.nan
    contract.write_mesh_ply(path, vertices, faces)


def _corrupt_no_faces(path: Path) -> None:
    vertices, _ = _tetrahedron()
    contract.write_mesh_ply(path, vertices, np.zeros((0, 3), dtype=np.int32))


def _corrupt_bad_index(path: Path) -> None:
    vertices, faces = _tetrahedron()
    faces[0, 0] = 99
    contract.write_mesh_ply(path, vertices, faces)


def _corrupt_extra_property(path: Path) -> None:
    raw = path.read_bytes()
    path.write_bytes(
        raw.replace(b"property float z\n", b"property float z\nproperty float nx\n")
    )


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_corrupt_truncated, "payload"),
        (_corrupt_nan, "non-finite"),
        (_corrupt_no_faces, "0 faces"),
        (_corrupt_bad_index, "face index"),
        (_corrupt_extra_property, "layout"),
    ],
)
def test_strict_mesh_reader_rejects_foreign_or_broken_files(tmp_path, corrupt, message):
    path = tmp_path / "mesh.ply"
    contract.write_mesh_ply(path, *_tetrahedron())
    corrupt(path)

    with pytest.raises(contract.ContractError, match=message):
        contract.read_mesh_ply_strict(path)


def test_points_reader_reads_what_the_service_writes(tmp_path):
    from recon import vggt_runner

    points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.5, 6.25]], dtype=np.float32)
    colors = np.full(points.shape, 255, dtype=np.uint8)
    vggt_runner.write_ply(tmp_path / "partial.ply", points, colors)

    np.testing.assert_allclose(
        contract.read_points_ply(tmp_path / "partial.ply"), points
    )


def test_points_reader_rejects_a_mesh_file(tmp_path):
    contract.write_mesh_ply(tmp_path / "mesh.ply", *_tetrahedron())

    with pytest.raises(contract.ContractError, match="points layout"):
        contract.read_points_ply(tmp_path / "mesh.ply")


# --- what the runner can import -------------------------------------------------------------------


def test_contract_module_is_python_310_syntax():
    # A-12's runner runs it under LaS-Comp's Python 3.10 (D-057).
    ast.parse(CONTRACT_SOURCE.read_text(encoding="utf-8"), feature_version=(3, 10))


def test_contract_module_imports_only_the_standard_library_and_numpy():
    tree = ast.parse(CONTRACT_SOURCE.read_text(encoding="utf-8"))
    allowed = {
        "__future__",
        "json",
        "math",
        "os",
        "tempfile",
        "dataclasses",
        "pathlib",
        "typing",
    }
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])

    assert imported - allowed == {"numpy"}


def test_pins_match_the_decision_that_set_them():
    decisions = (CONTRACT_SOURCE.parents[1] / "docs" / "decisions.md").read_text(
        encoding="utf-8"
    )

    assert contract.MODEL_COMMIT in decisions
    assert contract.WEIGHTS_REVISION in decisions

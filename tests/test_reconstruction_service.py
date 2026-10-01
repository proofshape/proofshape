"""R-14 tests for the file-backed reconstruction HTTP service."""

import json
from pathlib import Path

import yaml
from fastapi.testclient import TestClient
from jsonschema import Draft7Validator, RefResolver

from recon import service

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO_ROOT / "contracts" / "openapi.yaml"


def _fake_pipeline(capture_dir: Path, artifact_dir: Path) -> dict:
    assert len(list(capture_dir.glob("*.jpg"))) == 3
    artifact_dir.mkdir(parents=True, exist_ok=True)
    glb_path = artifact_dir / "model.glb"
    glb_path.write_bytes(b"glTF-test")
    return {
        "glb_path": str(glb_path),
        "reference_tier_used": "charuco_board",
        "metric": True,
        "stage_timings_s": {
            "board_detection": 0.1,
            "reconstruction": 0.2,
            "fusion": 0.3,
            "export": 0.1,
        },
        "warnings": [],
    }


def _upload_three_frames(client: TestClient, session_id: str) -> None:
    gyro = json.dumps({"alpha": 1.0, "beta": 2.0, "gamma": 3.0})
    for frame_index in range(3):
        response = client.post(
            f"/v1/sessions/{session_id}/frames",
            files={
                "frame": (f"frame{frame_index}.jpg", b"jpeg-data", "image/jpeg"),
                # D-041: gyro is its own JSON Blob and arrives as an upload part.
                "gyro": ("blob", gyro, "application/json"),
            },
        )
        assert response.status_code == 202
        assert response.json() == {"accepted": True, "frame_index": frame_index}


def _validate_contract_response(operation_id: str, status: str, body: dict) -> None:
    spec = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    resolver = RefResolver.from_schema(spec)
    for path_item in spec["paths"].values():
        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue
            if operation.get("operationId") != operation_id:
                continue
            response = operation["responses"][status]
            schema = next(iter(response["content"].values()))["schema"]
            Draft7Validator(schema, resolver=resolver).validate(body)
            return
    raise AssertionError(f"No contract response found for {operation_id} {status}")


def test_session_frames_finish_and_reconstruction_match_contract(tmp_path):
    app = service.create_app(
        tmp_path,
        pipeline_runner=_fake_pipeline,
        known_order_codes={"PO-1"},
    )
    with TestClient(app) as client:
        created = client.post("/v1/sessions", json={"order_code": "PO-1"})
        assert created.status_code == 201
        _validate_contract_response("createSession", "201", created.json())
        session_id = created.json()["session_id"]

        _upload_three_frames(client, session_id)

        finished = client.post(f"/v1/sessions/{session_id}/finish")
        assert finished.status_code == 202
        _validate_contract_response("finishSession", "202", finished.json())

        result = client.get(f"/v1/sessions/{session_id}/reconstruction")
        assert result.status_code == 200
        body = result.json()
        _validate_contract_response("getReconstruction", "200", body)
        assert body["status"] == "complete"
        assert body["metric"] is True
        assert body["reference_tier_used"] == "charuco_board"
        assert body["stage_timings_s"]["reconstruction"] == 0.2
        assert body["warnings"] == []
        assert "observed_fraction" not in body
        assert body["glb_url"].endswith(f"/v1/sessions/{session_id}/model.glb")

        model = client.get(f"/v1/sessions/{session_id}/model.glb")
        assert model.status_code == 200
        assert model.headers["content-type"] == "model/gltf-binary"
        assert model.content == b"glTF-test"


def test_contract_error_paths_are_structured(tmp_path):
    app = service.create_app(
        tmp_path,
        pipeline_runner=_fake_pipeline,
        known_order_codes={"PO-1"},
    )
    with TestClient(app) as client:
        unknown_order = client.post("/v1/sessions", json={"order_code": "BAD"})
        assert unknown_order.status_code == 400
        _validate_contract_response("createSession", "400", unknown_order.json())
        assert unknown_order.json()["code"] == "order_code_unknown"

        created = client.post("/v1/sessions", json={"order_code": "PO-1"})
        session_id = created.json()["session_id"]

        before_finish = client.get(f"/v1/sessions/{session_id}/reconstruction")
        assert before_finish.status_code == 404
        _validate_contract_response("getReconstruction", "404", before_finish.json())
        assert before_finish.json()["code"] == "session_not_found"

        too_few = client.post(f"/v1/sessions/{session_id}/finish")
        assert too_few.status_code == 409
        _validate_contract_response("finishSession", "409", too_few.json())
        assert too_few.json()["code"] == "too_few_frames"


def test_r02_failure_is_preserved_without_breaking_frozen_schema(tmp_path):
    def failing_pipeline(capture_dir: Path, artifact_dir: Path) -> dict:
        raise service.ReconstructionFailure(
            "board_not_detected",
            "The board was not visible in enough frames.",
        )

    app = service.create_app(tmp_path, pipeline_runner=failing_pipeline)
    with TestClient(app) as client:
        created = client.post("/v1/sessions", json={"order_code": "PO-1"})
        session_id = created.json()["session_id"]
        _upload_three_frames(client, session_id)
        assert client.post(f"/v1/sessions/{session_id}/finish").status_code == 202

        result = client.get(f"/v1/sessions/{session_id}/reconstruction")
        assert result.status_code == 200
        _validate_contract_response("getReconstruction", "200", result.json())
        assert result.json() == {
            "status": "failed",
            "warnings": [
                "board_not_detected: The board was not visible in enough frames."
            ],
        }

        record = app.state.store.load(session_id)
        assert record["reconstruction"]["failure"] == {
            "code": "board_not_detected",
            "message": "The board was not visible in enough frames.",
        }


def test_r02_and_r04_failure_mapping_is_explicit():
    failure = service._classify_board_failure(
        {"posed_count": 0, "failure_counts": {"board_not_detected": 4}}
    )
    assert failure is not None
    assert failure.code == "board_not_detected"

    failure = service._classify_board_failure(
        {"posed_count": 1, "failure_counts": {"board_partially_occluded": 2}}
    )
    assert failure is not None
    assert failure.code == "board_partially_occluded"

    failure = service._classify_board_failure({"posed_count": 2, "failure_counts": {}})
    assert failure is not None
    assert failure.code == "insufficient_baseline"

    assert (
        service._classify_board_failure({"posed_count": 3, "failure_counts": {}})
        is None
    )
    assert (
        service._map_alignment_error(
            ValueError("Need at least 3 paired camera poses")
        ).code
        == "insufficient_baseline"
    )
    assert (
        service._map_alignment_error(ValueError("singular transform")).code
        == "metric_alignment_failed"
    )


def test_no_cad_verdict_path_uses_frozen_error_envelope(tmp_path):
    app = service.create_app(tmp_path, pipeline_runner=_fake_pipeline)
    with TestClient(app) as client:
        created = client.post("/v1/sessions", json={"order_code": "PO-1"})
        session_id = created.json()["session_id"]
        response = client.get(f"/v1/sessions/{session_id}/verdict")
        assert response.status_code == 404
        _validate_contract_response("getVerdict", "404", response.json())
        assert response.json()["code"] == "no_cad_uploaded"


def test_public_operation_ids_match_the_frozen_contract(tmp_path):
    app = service.create_app(tmp_path, pipeline_runner=_fake_pipeline)
    generated = app.openapi()
    operation_ids = set()
    for path_item in generated["paths"].values():
        for operation in path_item.values():
            operation_ids.add(operation["operationId"])

    assert operation_ids == {
        "createSession",
        "uploadFrame",
        "finishSession",
        "getReconstruction",
        "getVerdict",
    }

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


def test_missing_multipart_part_uses_error_detail_contract(tmp_path):
    app = service.create_app(tmp_path, pipeline_runner=_fake_pipeline)
    with TestClient(app) as client:
        created = client.post("/v1/sessions", json={"order_code": "PO-1"})
        session_id = created.json()["session_id"]

        response = client.post(
            f"/v1/sessions/{session_id}/frames",
            files={"frame": ("frame.jpg", b"jpeg-data", "image/jpeg")},
        )

        assert response.status_code == 422
        _validate_contract_response("uploadFrame", "422", response.json())
        assert response.json()["code"] == "vlm_veto"
        assert "frame and gyro" in response.json()["message"]


def test_pending_reconstruction_is_failed_after_service_restart(tmp_path):
    first_app = service.create_app(tmp_path, pipeline_runner=_fake_pipeline)
    record = first_app.state.store.create("PO-1")
    record["status"] = "finished"
    record["reconstruction"] = {"status": "pending"}
    first_app.state.store.save(record)

    restarted_app = service.create_app(tmp_path, pipeline_runner=_fake_pipeline)
    recovered = restarted_app.state.store.load(record["session_id"])

    assert restarted_app.state.orphaned_pending_failed == 1
    assert recovered["reconstruction"]["status"] == "failed"
    assert recovered["reconstruction"]["failure"]["code"] == "metric_alignment_failed"
    assert "service restart" in recovered["reconstruction"]["failure"]["message"]


def test_pipeline_runs_provenance_and_uncertainty_before_serving_glb(
    tmp_path,
    monkeypatch,
):
    from recon import (
        board_pose,
        mesh_cleanup,
        metric_alignment,
        provenance,
        reconstruction_runner,
        tsdf_fusion,
        uncertainty,
    )

    capture_dir = tmp_path / "capture"
    artifact_dir = tmp_path / "artifacts"
    capture_dir.mkdir()

    monkeypatch.setattr(
        board_pose,
        "estimate_board_poses",
        lambda capture, out: {
            "posed_count": 3,
            "failure_counts": {},
        },
    )
    monkeypatch.setattr(
        reconstruction_runner,
        "run_on_capture",
        lambda *args, **kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(
        metric_alignment,
        "load_and_align",
        lambda *args, **kwargs: {"scale": 1.0},
    )
    monkeypatch.setattr(
        metric_alignment, "write_alignment", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(tsdf_fusion, "fuse_run", lambda *args, **kwargs: {})
    monkeypatch.setattr(mesh_cleanup, "cleanup_run", lambda *args, **kwargs: {})

    seen = {}

    def fake_provenance(mesh_path, reconstruction_dir, alignment_path, out_dir):
        seen["provenance_mesh"] = mesh_path
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "mesh_provenance.glb").write_bytes(b"provenance")
        (out_dir / "observations.npz").write_bytes(b"observations")
        return {"observed_fraction": 0.75}

    def fake_uncertainty(provenance_dir, out_dir):
        seen["uncertainty_input"] = provenance_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "mesh_uncertainty.glb").write_bytes(b"uncertainty")
        return {}

    monkeypatch.setattr(provenance, "provenance_run", fake_provenance)
    monkeypatch.setattr(uncertainty, "uncertainty_run", fake_uncertainty)

    result = service.run_reconstruction_pipeline(capture_dir, artifact_dir)

    assert seen["provenance_mesh"] == artifact_dir / "model.glb"
    assert seen["uncertainty_input"] == artifact_dir / "provenance"
    assert result["glb_path"] == str(
        artifact_dir / "uncertainty" / "mesh_uncertainty.glb"
    )
    assert result["observed_fraction"] == 0.75
    assert "provenance" in result["stage_timings_s"]
    assert "uncertainty" in result["stage_timings_s"]


def test_malformed_gyro_is_contract_valid_and_explicitly_not_a_vlm_decision(tmp_path):
    app = service.create_app(tmp_path, pipeline_runner=_fake_pipeline)
    with TestClient(app) as client:
        created = client.post("/v1/sessions", json={"order_code": "PO-1"})
        session_id = created.json()["session_id"]
        response = client.post(
            f"/v1/sessions/{session_id}/frames",
            files={
                "frame": ("frame.jpg", b"jpeg-data", "image/jpeg"),
                "gyro": ("blob", "{not-json", "application/json"),
            },
        )

        assert response.status_code == 422
        _validate_contract_response("uploadFrame", "422", response.json())
        assert response.json()["code"] == "vlm_veto"
        assert (
            "Malformed frame upload (not a VLM decision)" in response.json()["message"]
        )


def test_empty_frame_is_rejected_with_contract_error(tmp_path):
    app = service.create_app(tmp_path, pipeline_runner=_fake_pipeline)
    with TestClient(app) as client:
        created = client.post("/v1/sessions", json={"order_code": "PO-1"})
        session_id = created.json()["session_id"]
        response = client.post(
            f"/v1/sessions/{session_id}/frames",
            files={
                "frame": ("frame.jpg", b"", "image/jpeg"),
                "gyro": (
                    "blob",
                    json.dumps({"alpha": 0, "beta": 0, "gamma": 0}),
                    "application/json",
                ),
            },
        )

        assert response.status_code == 422
        _validate_contract_response("uploadFrame", "422", response.json())
        assert response.json()["code"] == "frame_rejected_exposure"


def test_upload_race_after_finish_returns_422_not_500(tmp_path, monkeypatch):
    app = service.create_app(tmp_path, pipeline_runner=_fake_pipeline)
    original_add_frame = app.state.store.add_frame

    def lost_race(*args, **kwargs):
        raise service.SessionAlreadyFinishedError("session_already_finished")

    monkeypatch.setattr(app.state.store, "add_frame", lost_race)
    with TestClient(app) as client:
        created = client.post("/v1/sessions", json={"order_code": "PO-1"})
        session_id = created.json()["session_id"]
        response = client.post(
            f"/v1/sessions/{session_id}/frames",
            files={
                "frame": ("frame.jpg", b"jpeg-data", "image/jpeg"),
                "gyro": (
                    "blob",
                    json.dumps({"alpha": 0, "beta": 0, "gamma": 0}),
                    "application/json",
                ),
            },
        )

    monkeypatch.setattr(app.state.store, "add_frame", original_add_frame)
    assert response.status_code == 422
    assert response.json()["code"] == "session_already_finished"


def test_finish_race_returns_409_not_500(tmp_path, monkeypatch):
    app = service.create_app(tmp_path, pipeline_runner=_fake_pipeline)
    record = app.state.store.create("PO-1")
    record["frame_count"] = service.MIN_FINISH_FRAMES
    app.state.store.save(record)

    def lost_race(*args, **kwargs):
        raise service.SessionAlreadyFinishedError("session_already_finished")

    monkeypatch.setattr(app.state.store, "mark_finished", lost_race)
    with TestClient(app) as client:
        response = client.post(f"/v1/sessions/{record['session_id']}/finish")

    assert response.status_code == 409
    assert response.json()["code"] == "session_already_finished"


def test_unexpected_pipeline_exception_becomes_terminal_failure(tmp_path):
    class UnexpectedPipelineError(Exception):
        pass

    def failing_pipeline(capture_dir: Path, artifact_dir: Path) -> dict:
        raise UnexpectedPipelineError("synthetic unexpected model failure")

    app = service.create_app(tmp_path, pipeline_runner=failing_pipeline)
    with TestClient(app) as client:
        created = client.post("/v1/sessions", json={"order_code": "PO-1"})
        session_id = created.json()["session_id"]
        _upload_three_frames(client, session_id)
        assert client.post(f"/v1/sessions/{session_id}/finish").status_code == 202

        result = client.get(f"/v1/sessions/{session_id}/reconstruction")
        assert result.status_code == 200
        assert result.json()["status"] == "failed"
        assert "synthetic unexpected model failure" in result.json()["warnings"][0]


def test_model_404_uses_error_detail_envelope(tmp_path):
    app = service.create_app(tmp_path, pipeline_runner=_fake_pipeline)
    with TestClient(app) as client:
        response = client.get("/v1/sessions/does-not-exist/model.glb")

    assert response.status_code == 404
    assert response.json() == {
        "code": "session_not_found",
        "message": "No session with that id.",
    }


def _finished_session(client: TestClient) -> str:
    created = client.post("/v1/sessions", json={"order_code": "PO-1"})
    session_id = created.json()["session_id"]
    _upload_three_frames(client, session_id)
    client.post(f"/v1/sessions/{session_id}/finish")
    return session_id


def test_glb_url_defaults_to_request_url_for_unchanged(tmp_path):
    """R-15 regression guard: an ordinary direct caller (local dev, CI, every existing test)
    must see byte-identical behavior to before the Lightning-proxy fix."""
    app = service.create_app(tmp_path, pipeline_runner=_fake_pipeline)
    with TestClient(app) as client:
        session_id = _finished_session(client)
        body = client.get(f"/v1/sessions/{session_id}/reconstruction").json()

    assert body["glb_url"] == f"http://testserver/v1/sessions/{session_id}/model.glb"


def test_glb_url_uses_x_forwarded_host_when_present(tmp_path):
    """R-15: Lightning's own deployment proxy was confirmed (2026-10-07) to present the
    container with Host: localhost -- X-Forwarded-Host, when a proxy sends it, is the fix."""
    app = service.create_app(tmp_path, pipeline_runner=_fake_pipeline)
    with TestClient(app) as client:
        session_id = _finished_session(client)
        body = client.get(
            f"/v1/sessions/{session_id}/reconstruction",
            headers={
                "x-forwarded-host": "8000-dep-example.cloudspaces.litng.ai",
                "x-forwarded-proto": "https",
            },
        ).json()

    assert body["glb_url"] == (
        f"https://8000-dep-example.cloudspaces.litng.ai/v1/sessions/{session_id}/model.glb"
    )


def test_glb_url_forwarded_proto_defaults_to_the_request_scheme(tmp_path):
    app = service.create_app(tmp_path, pipeline_runner=_fake_pipeline)
    with TestClient(app) as client:
        session_id = _finished_session(client)
        body = client.get(
            f"/v1/sessions/{session_id}/reconstruction",
            headers={"x-forwarded-host": "example.cloudspaces.litng.ai"},
        ).json()

    assert body["glb_url"].startswith("http://example.cloudspaces.litng.ai/")


def test_glb_url_env_override_wins_over_everything(tmp_path, monkeypatch):
    """The operator-set override (R-15's deploy workflow) takes priority even over a forwarded
    host, so a known-good public URL is never second-guessed by a proxy header."""
    monkeypatch.setenv(service.PUBLIC_BASE_URL_ENV, "https://recon.proofshape.example/")
    app = service.create_app(tmp_path, pipeline_runner=_fake_pipeline)
    with TestClient(app) as client:
        session_id = _finished_session(client)
        body = client.get(
            f"/v1/sessions/{session_id}/reconstruction",
            headers={"x-forwarded-host": "should-be-ignored.example"},
        ).json()

    assert body["glb_url"] == (
        f"https://recon.proofshape.example/v1/sessions/{session_id}/model.glb"
    )

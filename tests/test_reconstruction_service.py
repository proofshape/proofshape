"""R-14 tests for the file-backed reconstruction HTTP service."""

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from jsonschema import Draft7Validator, RefResolver

from recon import service

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO_ROOT / "contracts" / "openapi.yaml"


@pytest.fixture(autouse=True)
def _no_gemini_key(monkeypatch):
    # create_app's default supervisor reads GEMINI_API_KEY. A developer's real key must not
    # turn these tests into paid API calls, or let a finished check add vlm_feedback (A-11)
    # to responses the tests compare exactly.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


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


def test_vlm_never_blocks_upload_and_timeout_evidence_survives_session_save(tmp_path):
    import threading
    import time

    from recon.vlm_check import VlmSupervisor

    entered, release = threading.Event(), threading.Event()

    def blocked(frame, reference):
        entered.set()
        assert release.wait(5)
        raise TimeoutError("private-api-key")

    supervisor = VlmSupervisor(inspector=blocked, join_timeout_s=0.02)
    app = service.create_app(
        tmp_path, pipeline_runner=_fake_pipeline, vlm_supervisor=supervisor
    )
    try:
        with TestClient(app) as client:
            session_id = client.post(
                "/v1/sessions", json={"order_code": "PO-1"}
            ).json()["session_id"]
            started = time.monotonic()
            _upload_three_frames(client, session_id)
            assert time.monotonic() - started < 1
            assert entered.wait(
                1
            )  # API is still blocked after every upload returned 202.
            assert client.post(f"/v1/sessions/{session_id}/finish").status_code == 202
            session_dir = tmp_path / session_id
            report = json.loads((session_dir / "vlm_checks.json").read_text())
            assert report["sampled_indices"] == [0, 2]
            assert all(c["reason"] == "join_timeout" for c in report["checks"])
            record = app.state.store.load(session_id)
            assert record["reconstruction"]["status"] == "complete"
            assert "vlm" not in record and "vlm_checks" not in record
            assert "private-api-key" not in json.dumps(report)
    finally:
        release.set()
        supervisor.close()


def test_vlm_join_is_written_even_if_reconstruction_fails_without_key(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def broken(*args):
        raise RuntimeError("pipeline failed")

    app = service.create_app(tmp_path, pipeline_runner=broken)
    with TestClient(app) as client:
        sid = client.post("/v1/sessions", json={"order_code": "PO-1"}).json()[
            "session_id"
        ]
        _upload_three_frames(client, sid)
        client.post(f"/v1/sessions/{sid}/finish")
        report = json.loads((tmp_path / sid / "vlm_checks.json").read_text())
        assert report["sampled_indices"] == [0, 2]
        assert all(c["reason"] == "missing_api_key" for c in report["checks"])
        assert app.state.store.load(sid)["reconstruction"]["status"] == "failed"


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


def test_cors_allows_any_browser_origin(tmp_path):
    # C-13: capture/ is served from a different origin than this service; D-042 left the exact
    # access-control story to whichever story first connected them. The wildcard is deliberate
    # (see create_app's own comment) -- confirm it actually answers a cross-origin request
    # rather than relying on FastAPI's default (none) going unnoticed.
    app = service.create_app(
        tmp_path, pipeline_runner=_fake_pipeline, known_order_codes={"PO-1"}
    )
    with TestClient(app) as client:
        preflight = client.options(
            "/v1/sessions",
            headers={
                "Origin": "https://example.lightning.ai",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "*"

        created = client.post(
            "/v1/sessions",
            json={"order_code": "PO-1"},
            headers={"Origin": "https://example.lightning.ai"},
        )
        assert created.status_code == 201
        assert created.headers["access-control-allow-origin"] == "*"


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
        assert response.json()["code"] == "malformed_request"
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
    assert recovered["reconstruction"]["failure"]["code"] == "internal_error"
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


def test_pipeline_maps_fuse_run_failures_to_a_named_fusion_error_not_a_generic_one(
    tmp_path, monkeypatch
):
    # R-20: fuse_run can now raise FileNotFoundError/ValueError (a run.json missing or
    # predating min_confidence). Without the pipeline's own except clause, that would propagate
    # past run_reconstruction_pipeline uncaught and get caught by run_session's generic
    # terminal-exception handler instead, which reports an unrelated "Reconstruction failed: "
    # message -- this confirms the pipeline maps it explicitly before that happens.
    from recon import board_pose, metric_alignment, reconstruction_runner, tsdf_fusion

    capture_dir = tmp_path / "capture"
    artifact_dir = tmp_path / "artifacts"
    capture_dir.mkdir()

    monkeypatch.setattr(
        board_pose,
        "estimate_board_poses",
        lambda capture, out: {"posed_count": 3, "failure_counts": {}},
    )
    monkeypatch.setattr(
        reconstruction_runner,
        "run_on_capture",
        lambda *args, **kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(
        metric_alignment, "load_and_align", lambda *args, **kwargs: {"scale": 1.0}
    )
    monkeypatch.setattr(
        metric_alignment, "write_alignment", lambda *args, **kwargs: None
    )

    def fail_fuse_run(*args, **kwargs):
        raise FileNotFoundError("run.json does not exist")

    monkeypatch.setattr(tsdf_fusion, "fuse_run", fail_fuse_run)

    with pytest.raises(service.ReconstructionFailure) as excinfo:
        service.run_reconstruction_pipeline(capture_dir, artifact_dir)

    assert excinfo.value.code == "metric_alignment_failed"
    assert "Fusion failed" in excinfo.value.message
    assert "run.json does not exist" in excinfo.value.message


@pytest.mark.parametrize(
    ("env_value", "expected_device"), [(None, "cuda"), ("mps", "mps")]
)
def test_pipeline_reconstructs_on_the_device_the_environment_names(
    tmp_path, monkeypatch, env_value, expected_device
):
    # C-13 / D-049: before this, the pipeline never passed a device, so the backend always got
    # cuda and the service could not run on the M2 Max. Unset must stay cuda (the container).
    from recon import board_pose, reconstruction_runner

    if env_value is None:
        monkeypatch.delenv(reconstruction_runner.DEVICE_ENV, raising=False)
    else:
        monkeypatch.setenv(reconstruction_runner.DEVICE_ENV, env_value)

    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    monkeypatch.setattr(
        board_pose,
        "estimate_board_poses",
        lambda capture, out: {"posed_count": 3, "failure_counts": {}},
    )
    seen = {}

    def fake_run_on_capture(*args, device_type="unset", **kwargs):
        seen["device_type"] = device_type
        # A refusal ends the pipeline right after this stage, so no later stage needs faking.
        return {"status": "refused", "reason": "test stops here"}

    monkeypatch.setattr(reconstruction_runner, "run_on_capture", fake_run_on_capture)

    with pytest.raises(service.ReconstructionFailure):
        service.run_reconstruction_pipeline(capture_dir, tmp_path / "artifacts")

    assert seen["device_type"] == expected_device


def test_missing_focal_length_is_reported_as_such_not_as_insufficient_baseline():
    # C-13: the phone was told "insufficient_baseline ... found 0" (advice: move further between
    # shots) when the real cause was frames with no EXIF focal length.
    failure = service._classify_board_failure(
        {"posed_count": 0, "failure_counts": {"intrinsics_unavailable": 19}}
    )

    assert failure is not None
    assert failure.code == "metric_alignment_failed"
    assert "focal length" in failure.message
    assert "EXIF" in failure.message


def test_board_calibration_is_skipped_when_frames_had_a_focal_length(
    tmp_path, monkeypatch
):
    from recon import camera_intrinsics

    def fail_calibration(*args, **kwargs):
        raise AssertionError(
            "calibration must not run for frames R-02 could already pose"
        )

    monkeypatch.setattr(camera_intrinsics, "load_board_session", fail_calibration)
    report = {"posed_count": 30, "failure_counts": {}}

    board_dir, returned = service._pose_board_with_session_intrinsics(
        tmp_path / "capture", tmp_path, tmp_path / "board_pose", report
    )

    assert board_dir == tmp_path / "board_pose"
    assert returned is report


def test_failed_board_calibration_is_a_named_failure_not_a_crash(tmp_path, monkeypatch):
    from recon import camera_intrinsics

    monkeypatch.setattr(camera_intrinsics, "load_board_session", lambda path: object())

    def fail(session):
        raise camera_intrinsics.CalibrationFailure("too few non-collinear views")

    monkeypatch.setattr(camera_intrinsics, "calibrate_from_board", fail)

    with pytest.raises(service.ReconstructionFailure) as excinfo:
        service._pose_board_with_session_intrinsics(
            tmp_path / "capture",
            tmp_path,
            tmp_path / "board_pose",
            {"posed_count": 0, "failure_counts": {"intrinsics_unavailable": 4}},
        )

    assert excinfo.value.code == "metric_alignment_failed"
    assert "no EXIF focal length" in excinfo.value.message
    assert "too few non-collinear views" in excinfo.value.message


def test_pipeline_reads_the_calibrated_board_poses_downstream(tmp_path, monkeypatch):
    # After calibration, R-02's poses live in a new folder (R-02 never overwrites its outputs);
    # the backend must be handed that folder's poses, not the uncalibrated ones.
    from recon import board_pose, reconstruction_runner

    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    artifact_dir = tmp_path / "artifacts"
    calibrated_dir = artifact_dir / service.CALIBRATED_BOARD_DIR_NAME
    monkeypatch.setattr(
        board_pose,
        "estimate_board_poses",
        lambda capture, out: {
            "posed_count": 0,
            "failure_counts": {"intrinsics_unavailable": 5},
        },
    )
    monkeypatch.setattr(
        service,
        "_pose_board_with_session_intrinsics",
        lambda capture, artifacts, board_dir, report: (
            calibrated_dir,
            {"posed_count": 5, "failure_counts": {}},
        ),
    )
    seen = {}

    def fake_run_on_capture(*args, board_pose_path=None, **kwargs):
        seen["board_pose_path"] = board_pose_path
        return {"status": "refused", "reason": "test stops here"}

    monkeypatch.setattr(reconstruction_runner, "run_on_capture", fake_run_on_capture)

    with pytest.raises(service.ReconstructionFailure):
        service.run_reconstruction_pipeline(capture_dir, artifact_dir)

    assert seen["board_pose_path"] == calibrated_dir / "board_poses.npz"


def test_malformed_gyro_returns_malformed_request_not_a_vlm_code(tmp_path):
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
        assert response.json()["code"] == "malformed_request"
        assert "gyro part is invalid" in response.json()["message"]


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
        assert response.json()["code"] == "malformed_request"


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


def test_mark_finished_raises_the_typed_error_not_a_plain_runtime_error(tmp_path):
    # Regression (A-06 review): mark_finished used to raise a plain RuntimeError on a losing
    # race, which `except SessionAlreadyFinishedError` never caught -- it escaped as an
    # uncaught 500 instead of the structured 409 every other race path returns.
    store = service.SessionStore(tmp_path)
    record = store.create("PO-1")
    store.mark_finished(record["session_id"])

    with pytest.raises(service.SessionAlreadyFinishedError):
        store.mark_finished(record["session_id"])


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
        warning = result.json()["warnings"][0]
        assert warning.startswith("internal_error:")
        # #186: the crash's text goes to the operator's log, never to the supplier.
        assert "synthetic unexpected model failure" not in warning


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


# A-11: the upload response carries the newest checked VLM result (contract v1.1).


def _no_key_app(tmp_path):
    from recon.vlm_check import VlmSupervisor

    # No key, so uploads only ever write `unchecked` results; tests add checked ones by hand.
    return service.create_app(
        tmp_path, pipeline_runner=_fake_pipeline, vlm_supervisor=VlmSupervisor()
    )


def _write_checked(
    app, session_id, frame_index, hint="Turn the part so the hole faces you."
):
    from types import SimpleNamespace

    from recon.vlm_check import VlmSupervisor, atomic_json, parse_response

    observation = {
        "same_part": None if frame_index == 0 else True,
        "hands_occluding": False,
        "screen_photo": False,
        "lighting_adequate": True,
        "hint": hint,
    }
    response = SimpleNamespace(
        text=json.dumps(observation),
        model_version="test-model",
        candidates=[SimpleNamespace(finish_reason="STOP")],
        prompt_feedback=None,
        usage_metadata=None,
    )
    check = parse_response(frame_index, response)
    assert check.status == "checked"
    session_dir = app.state.store.session_dir(session_id)
    atomic_json(VlmSupervisor.result_path(session_dir, frame_index), check.model_dump())


def _upload(client, session_id):
    return client.post(
        f"/v1/sessions/{session_id}/frames",
        files={
            "frame": ("frame.jpg", b"jpeg-data", "image/jpeg"),
            "gyro": (
                "blob",
                json.dumps({"alpha": 1, "beta": 2, "gamma": 3}),
                "application/json",
            ),
        },
    )


def test_upload_has_no_vlm_feedback_until_a_check_has_completed(tmp_path):
    app = _no_key_app(tmp_path)
    with TestClient(app) as client:
        session_id = client.post("/v1/sessions", json={"order_code": "PO-1"}).json()[
            "session_id"
        ]

        body = _upload(client, session_id).json()

        # Frame 0's own check is only `unchecked` (no key), so there's nothing to show.
        assert "vlm_feedback" not in body
        _validate_contract_response("uploadFrame", "202", body)


def test_upload_carries_the_newest_checked_result_and_skips_unchecked(tmp_path):
    from recon.vlm_check import VlmSupervisor, atomic_json, unchecked

    app = _no_key_app(tmp_path)
    with TestClient(app) as client:
        session_id = client.post("/v1/sessions", json={"order_code": "PO-1"}).json()[
            "session_id"
        ]
        _upload(client, session_id)
        _write_checked(app, session_id, 5, hint="Move left so the hole is visible.")
        session_dir = app.state.store.session_dir(session_id)
        atomic_json(
            VlmSupervisor.result_path(session_dir, 10),
            unchecked(10, "api_error").model_dump(),
        )

        body = _upload(client, session_id).json()

        assert body["vlm_feedback"] == {
            "frame_index": 5,
            "hint": "Move left so the hole is visible.",
            "same_part": True,
            "hands_occluding": False,
            "screen_photo": False,
            "lighting_adequate": True,
        }
        _validate_contract_response("uploadFrame", "202", body)


def test_reference_frame_feedback_keeps_same_part_present_and_null(tmp_path):
    app = _no_key_app(tmp_path)
    with TestClient(app) as client:
        session_id = client.post("/v1/sessions", json={"order_code": "PO-1"}).json()[
            "session_id"
        ]
        _upload(client, session_id)
        _write_checked(app, session_id, 0)

        body = _upload(client, session_id).json()

        # Frame 0 is the reference: same_part must be sent as null, not dropped.
        assert "same_part" in body["vlm_feedback"]
        assert body["vlm_feedback"]["same_part"] is None
        _validate_contract_response("uploadFrame", "202", body)


def test_crash_is_logged_for_the_operator(tmp_path, caplog):
    def failing_pipeline(capture_dir: Path, artifact_dir: Path) -> dict:
        raise RuntimeError("secret internal path /srv/models")

    app = service.create_app(tmp_path, pipeline_runner=failing_pipeline)
    with TestClient(app) as client, caplog.at_level("ERROR", logger="recon.service"):
        session_id = client.post("/v1/sessions", json={"order_code": "PO-1"}).json()[
            "session_id"
        ]
        _upload_three_frames(client, session_id)
        client.post(f"/v1/sessions/{session_id}/finish")
        result = client.get(f"/v1/sessions/{session_id}/reconstruction").json()

    assert result["warnings"][0].startswith("internal_error:")
    assert "/srv/models" not in result["warnings"][0]
    assert "/srv/models" in caplog.text

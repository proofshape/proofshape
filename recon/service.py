"""R-14: file-backed HTTP service for the frozen ProofShape reconstruction contract."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

MIN_FINISH_FRAMES = 3

ErrorCode = Literal[
    "board_not_detected",
    "board_partially_occluded",
    "insufficient_baseline",
    "metric_alignment_failed",
    "scale_disagreement",
    "no_reference_detected",
    "frame_rejected_blur",
    "frame_rejected_exposure",
    "frame_rejected_duplicate",
    "vlm_veto",
    "session_not_found",
    "session_already_finished",
    "too_few_frames",
    "order_code_unknown",
    "no_cad_uploaded",
]


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str
    frame_indices: list[int] | None = None


class CreateSessionRequest(BaseModel):
    order_code: str


class SessionResponse(BaseModel):
    session_id: str
    order_code: str
    status: Literal["capturing", "finished"]


class UploadFrameResult(BaseModel):
    accepted: bool
    frame_index: int


class ScaleAgreement(BaseModel):
    sources_compared: list[str]
    flagged: bool


class Region(BaseModel):
    name: str
    provenance: Literal["observed", "unobserved", "inferred"]
    deviation_mm: float | None = None
    contributes_to_verdict: bool


class VerdictResult(BaseModel):
    verdict: Literal["Pass", "Fail", "Rescan", "Unverifiable"]
    declared_at: Literal["intake", "post_reconstruction"]
    orientation_ambiguous: bool
    regions: list[Region]
    reason_codes: list[str] | None = None


class ReconstructionResult(BaseModel):
    status: Literal["pending", "complete", "failed"]
    glb_url: str | None = None
    reference_tier_used: (
        Literal["charuco_board", "blank_sheet", "credit_card", "none"] | None
    ) = None
    metric: bool | None = None
    observed_fraction: float | None = None
    scale_agreement: ScaleAgreement | None = None
    stage_timings_s: dict[str, float] | None = None
    warnings: list[str] | None = None


@dataclass(frozen=True)
class ReconstructionFailure(Exception):
    code: ErrorCode
    message: str
    frame_indices: list[int] | None = None

    def as_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.frame_indices is not None:
            body["frame_indices"] = self.frame_indices
        return body


PipelineRunner = Callable[[Path, Path], dict[str, Any]]


class SessionAlreadyFinishedError(RuntimeError):
    """Raised when a store mutation loses a race with finishSession."""


class SessionStore:
    """Persist sessions and frames beneath one service data directory."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    def record_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "session.json"

    def create(self, order_code: str) -> dict[str, Any]:
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        record = {
            "session_id": session_id,
            "order_code": order_code,
            "status": "capturing",
            "frame_count": 0,
            "reconstruction": {"status": "not_started"},
        }
        with self._lock:
            session_dir = self.session_dir(session_id)
            (session_dir / "frames").mkdir(parents=True, exist_ok=False)
            self._write_record(record)
        return record

    def load(self, session_id: str) -> dict[str, Any] | None:
        path = self.record_path(session_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._write_record(record)

    def add_frame(
        self, session_id: str, frame_bytes: bytes, gyro: dict[str, float]
    ) -> int:
        with self._lock:
            record = self.load(session_id)
            if record is None:
                raise KeyError(session_id)
            if record["status"] != "capturing":
                raise SessionAlreadyFinishedError("session_already_finished")

            frame_index = int(record["frame_count"])
            frame_dir = self.session_dir(session_id) / "frames"
            frame_path = frame_dir / f"frame_{frame_index:04d}.jpg"
            frame_path.write_bytes(frame_bytes)
            gyro_path = frame_dir / f"frame_{frame_index:04d}.gyro.json"
            gyro_path.write_text(
                json.dumps(gyro, sort_keys=True) + "\n", encoding="utf-8"
            )
            record["frame_count"] = frame_index + 1
            self._write_record(record)
            return frame_index

    def fail_orphaned_pending(self) -> int:
        """Fail jobs left pending by a previous process so clients do not poll forever."""
        failed = 0
        with self._lock:
            for record_path in self.root.glob("*/session.json"):
                record = json.loads(record_path.read_text(encoding="utf-8"))
                reconstruction = record.get("reconstruction", {})
                if (
                    record.get("status") == "finished"
                    and reconstruction.get("status") == "pending"
                ):
                    record["reconstruction"] = {
                        "status": "failed",
                        "failure": {
                            "code": "metric_alignment_failed",
                            "message": (
                                "Reconstruction was interrupted by a service restart; "
                                "start a new capture session."
                            ),
                        },
                    }
                    self._write_record(record)
                    failed += 1
        return failed

    def mark_finished(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            record = self.load(session_id)
            if record is None:
                raise KeyError(session_id)
            if record["status"] != "capturing":
                raise RuntimeError("session_already_finished")
            record["status"] = "finished"
            record["reconstruction"] = {"status": "pending"}
            self._write_record(record)
            return record

    def _write_record(self, record: dict[str, Any]) -> None:
        path = self.record_path(record["session_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temp_path.replace(path)


def _parse_gyro(payload: bytes) -> dict[str, float]:
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("gyro must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise TypeError("gyro must be a JSON object")

    result: dict[str, float] = {}
    for key in ("alpha", "beta", "gamma"):
        value = parsed.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"gyro.{key} must be a number")
        result[key] = float(value)
    return result


def _classify_board_failure(
    board_report: dict[str, Any],
) -> ReconstructionFailure | None:
    posed_count = int(board_report.get("posed_count", 0))
    if posed_count >= 3:
        return None

    failures = board_report.get("failure_counts", {})
    if posed_count == 0 and failures.get("board_not_detected", 0):
        return ReconstructionFailure(
            "board_not_detected",
            "The ChArUco reference board was not detected in enough frames "
            "to reconstruct metrically.",
        )
    if failures.get("board_partially_occluded", 0):
        return ReconstructionFailure(
            "board_partially_occluded",
            "The ChArUco reference board was partially occluded in too many frames.",
        )
    return ReconstructionFailure(
        "insufficient_baseline",
        f"Metric alignment needs at least 3 usable board poses; found {posed_count}.",
    )


def _map_alignment_error(exc: Exception) -> ReconstructionFailure:
    message = str(exc)
    lowered = message.lower()
    if "at least" in lowered or "collinear" in lowered or "ambiguous" in lowered:
        return ReconstructionFailure("insufficient_baseline", message)
    return ReconstructionFailure("metric_alignment_failed", message)


def run_reconstruction_pipeline(
    capture_dir: Path, artifact_dir: Path
) -> dict[str, Any]:
    """Run the existing R-02 -> backend -> R-04 -> R-07 -> R-08 chain."""
    from recon import (
        board_pose,
        mesh_cleanup,
        metric_alignment,
        provenance,
        reconstruction_runner,
        tsdf_fusion,
        uncertainty,
    )

    capture_dir = Path(capture_dir)
    artifact_dir = Path(artifact_dir)
    board_dir = artifact_dir / "board_pose"
    reconstruction_dir = artifact_dir / "reconstruction"
    fusion_dir = artifact_dir / "fusion"
    model_path = artifact_dir / "model.glb"
    provenance_dir = artifact_dir / "provenance"
    uncertainty_dir = artifact_dir / "uncertainty"

    stage_timings_s: dict[str, float] = {}

    started = time.perf_counter()
    board_report = board_pose.estimate_board_poses(capture_dir, board_dir)
    stage_timings_s["board_detection"] = time.perf_counter() - started
    failure = _classify_board_failure(board_report)
    if failure is not None:
        raise failure

    started = time.perf_counter()
    run_info = reconstruction_runner.run_on_capture(
        capture_dir,
        reconstruction_dir,
        board_pose_path=board_dir / "board_poses.npz",
    )
    stage_timings_s["reconstruction"] = time.perf_counter() - started
    if run_info.get("status") == "refused":
        raise ReconstructionFailure(
            "metric_alignment_failed",
            "Reconstruction backend refused the session: "
            f"{run_info.get('reason', 'unknown reason')}",
        )

    started = time.perf_counter()
    try:
        alignment = metric_alignment.load_and_align(reconstruction_dir, board_dir)
        alignment_path = reconstruction_dir / "metric_alignment.json"
        metric_alignment.write_alignment(alignment_path, alignment)
    except (FileNotFoundError, ValueError) as exc:
        raise _map_alignment_error(exc) from exc
    stage_timings_s["metric_alignment"] = time.perf_counter() - started

    started = time.perf_counter()
    tsdf_fusion.fuse_run(reconstruction_dir, alignment_path, fusion_dir)
    stage_timings_s["fusion"] = time.perf_counter() - started

    started = time.perf_counter()
    mesh_cleanup.cleanup_run(fusion_dir / "mesh.ply", model_path)
    stage_timings_s["export"] = time.perf_counter() - started

    started = time.perf_counter()
    provenance_report = provenance.provenance_run(
        model_path,
        reconstruction_dir,
        alignment_path,
        provenance_dir,
    )
    stage_timings_s["provenance"] = time.perf_counter() - started

    started = time.perf_counter()
    uncertainty.uncertainty_run(provenance_dir, uncertainty_dir)
    stage_timings_s["uncertainty"] = time.perf_counter() - started
    final_model_path = uncertainty_dir / "mesh_uncertainty.glb"

    return {
        "glb_path": str(final_model_path),
        "reference_tier_used": "charuco_board",
        "metric": True,
        "observed_fraction": provenance_report.get("observed_fraction"),
        "stage_timings_s": stage_timings_s,
        "warnings": [],
    }


def create_app(
    data_dir: Path | None = None,
    pipeline_runner: PipelineRunner = run_reconstruction_pipeline,
    known_order_codes: set[str] | None = None,
) -> FastAPI:
    root = Path(
        data_dir or os.environ.get("PROOFSHAPE_DATA_DIR", "proofshape-data/sessions")
    )
    store = SessionStore(root)
    app = FastAPI(title="ProofShape reconstruction service", version="1.0.0")
    app.state.store = store
    app.state.orphaned_pending_failed = store.fail_orphaned_pending()

    def error(status_code: int, detail: ErrorDetail) -> JSONResponse:
        return JSONResponse(
            status_code=status_code, content=detail.model_dump(exclude_none=True)
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        if request.method == "POST" and request.url.path.endswith("/frames"):
            return error(
                422,
                ErrorDetail(
                    code="vlm_veto",
                    message=(
                        "Invalid frame upload. Both frame and gyro multipart file "
                        "parts are required."
                    ),
                ),
            )
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.post(
        "/v1/sessions",
        status_code=201,
        response_model=SessionResponse,
        responses={400: {"model": ErrorDetail}},
        operation_id="createSession",
    )
    def create_session(body: CreateSessionRequest):
        order_code = body.order_code.strip()
        if not order_code or (
            known_order_codes is not None and order_code not in known_order_codes
        ):
            return error(
                400,
                ErrorDetail(
                    code="order_code_unknown", message="The order code is unknown."
                ),
            )
        record = store.create(order_code)
        return SessionResponse(**record)

    @app.post(
        "/v1/sessions/{session_id}/frames",
        status_code=202,
        response_model=UploadFrameResult,
        responses={422: {"model": ErrorDetail}},
        operation_id="uploadFrame",
    )
    async def upload_frame(
        session_id: str,
        frame: Annotated[UploadFile, File()],
        gyro: Annotated[UploadFile, File()],
    ):
        record = store.load(session_id)
        if record is None:
            return error(
                422,
                ErrorDetail(
                    code="session_not_found", message="No session with that id."
                ),
            )
        if record["status"] != "capturing":
            return error(
                422,
                ErrorDetail(
                    code="session_already_finished",
                    message="The session is already finished.",
                ),
            )
        frame_bytes = await frame.read()
        if not frame_bytes:
            return error(
                422,
                ErrorDetail(
                    code="frame_rejected_exposure",
                    message="The uploaded frame is empty.",
                ),
            )
        try:
            gyro_value = _parse_gyro(await gyro.read())
        except (TypeError, ValueError) as exc:
            # F-03 froze the upload 422 envelope before defining a request-validation
            # ErrorCode. Keep the response contract-valid for v1 and make the semantic
            # debt explicit; D-031 requires a sprint-boundary contract change to add a
            # dedicated malformed_request code.
            return error(
                422,
                ErrorDetail(
                    code="vlm_veto",
                    message=f"Malformed frame upload (not a VLM decision): {exc}",
                ),
            )
        try:
            frame_index = store.add_frame(session_id, frame_bytes, gyro_value)
        except SessionAlreadyFinishedError:
            return error(
                422,
                ErrorDetail(
                    code="session_already_finished",
                    message="The session is already finished.",
                ),
            )
        return UploadFrameResult(accepted=True, frame_index=frame_index)

    def run_session(session_id: str) -> None:
        record = store.load(session_id)
        if record is None:
            return
        session_dir = store.session_dir(session_id)
        try:
            result = pipeline_runner(session_dir / "frames", session_dir / "artifacts")
            glb_path = Path(result["glb_path"])
            if not glb_path.exists():
                raise FileNotFoundError(
                    f"pipeline reported {glb_path}, but no GLB exists there"
                )
            record["reconstruction"] = {
                "status": "complete",
                "reference_tier_used": result.get("reference_tier_used"),
                "metric": result.get("metric"),
                "observed_fraction": result.get("observed_fraction"),
                "scale_agreement": result.get("scale_agreement"),
                "stage_timings_s": result.get("stage_timings_s"),
                "warnings": result.get("warnings", []),
                "glb_path": str(glb_path),
            }
        except ReconstructionFailure as exc:
            record["reconstruction"] = {"status": "failed", "failure": exc.as_dict()}
        except Exception as exc:  # noqa: BLE001 - terminal job boundary
            # Background reconstruction must never leave a persisted session at
            # status=pending. Unexpected numpy/cv2/model exceptions are converted
            # into a terminal failure here; the original exception text is preserved.
            record["reconstruction"] = {
                "status": "failed",
                "failure": {
                    "code": "metric_alignment_failed",
                    "message": f"Reconstruction failed: {exc}",
                },
            }
        store.save(record)

    @app.post(
        "/v1/sessions/{session_id}/finish",
        status_code=202,
        response_model=SessionResponse,
        responses={409: {"model": ErrorDetail}},
        operation_id="finishSession",
    )
    def finish_session(session_id: str, background_tasks: BackgroundTasks):
        record = store.load(session_id)
        if record is None:
            return error(
                409,
                ErrorDetail(
                    code="session_not_found", message="No session with that id."
                ),
            )
        if record["status"] != "capturing":
            return error(
                409,
                ErrorDetail(
                    code="session_already_finished",
                    message="The session is already finished.",
                ),
            )
        if int(record["frame_count"]) < MIN_FINISH_FRAMES:
            return error(
                409,
                ErrorDetail(
                    code="too_few_frames",
                    message=(
                        f"At least {MIN_FINISH_FRAMES} accepted frames are required "
                        "before finishing."
                    ),
                ),
            )
        try:
            record = store.mark_finished(session_id)
        except SessionAlreadyFinishedError:
            return error(
                409,
                ErrorDetail(
                    code="session_already_finished",
                    message="The session is already finished.",
                ),
            )
        background_tasks.add_task(run_session, session_id)
        return SessionResponse(**record)

    @app.get(
        "/v1/sessions/{session_id}/reconstruction",
        response_model=ReconstructionResult,
        response_model_exclude_none=True,
        responses={404: {"model": ErrorDetail}},
        operation_id="getReconstruction",
    )
    def get_reconstruction(session_id: str, request: Request):
        record = store.load(session_id)
        if record is None or record["status"] != "finished":
            return error(
                404,
                ErrorDetail(
                    code="session_not_found",
                    message=(
                        "No session with that id, or the session has not been "
                        "finished yet."
                    ),
                ),
            )
        reconstruction = record["reconstruction"]
        if reconstruction["status"] == "pending":
            return ReconstructionResult(status="pending")
        if reconstruction["status"] == "failed":
            failure = reconstruction.get("failure", {})
            code = failure.get("code", "metric_alignment_failed")
            message = failure.get("message", "Reconstruction failed.")
            return ReconstructionResult(
                status="failed", warnings=[f"{code}: {message}"]
            )

        response = {
            key: value
            for key, value in reconstruction.items()
            if key
            in {
                "status",
                "reference_tier_used",
                "metric",
                "observed_fraction",
                "scale_agreement",
                "stage_timings_s",
                "warnings",
            }
            and value is not None
        }
        response["glb_url"] = str(request.url_for("get_model", session_id=session_id))
        return ReconstructionResult(**response)

    @app.get(
        "/v1/sessions/{session_id}/verdict",
        response_model=VerdictResult,
        responses={404: {"model": ErrorDetail}},
        operation_id="getVerdict",
    )
    def get_verdict(session_id: str):
        record = store.load(session_id)
        if record is None:
            return error(
                404,
                ErrorDetail(
                    code="session_not_found", message="No session with that id."
                ),
            )
        return error(
            404,
            ErrorDetail(
                code="no_cad_uploaded",
                message="No CAD was supplied for this session — no verdict exists.",
            ),
        )

    @app.get(
        "/v1/sessions/{session_id}/model.glb",
        name="get_model",
        include_in_schema=False,
    )
    def get_model(session_id: str):
        record = store.load(session_id)
        if record is None:
            return error(
                404,
                ErrorDetail(
                    code="session_not_found", message="No session with that id."
                ),
            )
        reconstruction = record.get("reconstruction", {})
        if reconstruction.get("status") != "complete":
            return error(
                404,
                ErrorDetail(
                    code="session_not_found",
                    message="No completed reconstruction exists for this session.",
                ),
            )
        path = Path(reconstruction["glb_path"])
        if not path.exists():
            return error(
                404,
                ErrorDetail(
                    code="session_not_found",
                    message="The reconstruction artifact is unavailable.",
                ),
            )
        return FileResponse(path, media_type="model/gltf-binary", filename="model.glb")

    return app


app = create_app()

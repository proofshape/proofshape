"""R-14: file-backed HTTP service for the frozen ProofShape reconstruction contract."""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
import uuid
import weakref
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

# R-31: import the process guard before agent_tools -> provenance -> Open3D. The module applies
# the macOS-only OMP_NUM_THREADS=1 guard as an import side effect; Linux is unchanged.
from recon import runtime_env as _runtime_env  # noqa: F401
from recon.order_store import OrderStore
from recon.spec_parser import (
    GeminiSpecParser,
    SpecParseResult,
    parse_spec_response,
    unparsed,
)
from recon.vlm_check import VlmSupervisor

MIN_FINISH_FRAMES = 3

# R-15: on Lightning's deployment proxy, the connection the container actually sees has
# Host: localhost (confirmed on a real deployment, 2026-10-07) -- uvicorn's --forwarded-allow-ips
# trusts X-Forwarded-Proto but has no equivalent for the host, so request.url_for alone builds a
# glb_url nobody outside the container can reach. Layered, most-specific-first, so an unmodified
# local/CI/in-process caller (none of these three ever apply) keeps today's exact behavior.
PUBLIC_BASE_URL_ENV = "PROOFSHAPE_PUBLIC_BASE_URL"

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
    "cad_unreadable",
    "verdict_not_available",
    # I-07: a critical spec feature's tolerance_mm was missing or null at verdict time.
    "tolerance_missing",
    "buyer_auth_required",
    # Contract v1.1 (A-11, D-056).
    "malformed_request",
    "internal_error",
]

# A crash's exception text can name paths or internals, so the supplier gets a generic message
# and the operator gets the detail here (#186).
logger = logging.getLogger(__name__)
INTERNAL_ERROR_MESSAGE = (
    "The service hit an internal error while reconstructing. This is not a problem with your "
    "photos; please try a new capture session."
)


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str
    frame_indices: list[int] | None = None


class CreateSessionRequest(BaseModel):
    order_code: str


class SpecParseRequest(BaseModel):
    prose: str = Field(min_length=1)


class SessionResponse(BaseModel):
    session_id: str
    order_code: str
    status: Literal["capturing", "finished"]


class IntakeCheckResult(BaseModel):
    """I-09: one feature's Unverifiable-at-intake check, board tier only (D-073/D-077)."""

    feature_name: str
    tolerance_mm: float | None
    status: Literal["ok", "unverifiable", "refused"]
    reason: str | None = None


class OrderResponse(BaseModel):
    order_code: str
    created_at: str
    has_cad: bool
    cad_units: Literal["mm", "assumed_mm"] | None = None
    cad_bounding_box_mm: list[list[float]] | None = None
    spec: dict[str, Any]
    placement_note: str | None = None
    buyer_reference: str | None = None
    intake_check: list[IntakeCheckResult] = Field(default_factory=list)
    verdict: VerdictResult | None = None


class VlmFeedback(BaseModel):
    """A-11: the newest checked VLM result, as the supplier sees it. It describes an earlier
    sampled frame (frame_index), not necessarily this upload, because checks are asynchronous.
    Model version, token counts and unchecked reasons stay server-side."""

    frame_index: int
    # Matches the contract's VlmFeedback schema (minLength 1 / maxLength 240) and
    # FrameObservation.hint (recon/vlm_check.py), the field this is always built from.
    hint: str = Field(min_length=1, max_length=240)
    same_part: bool | None  # null for frame 0, the reference
    hands_occluding: bool
    screen_photo: bool
    lighting_adequate: bool

    @field_validator("hint")
    @classmethod
    def one_line_hint(cls, value: str) -> str:
        if not value.strip() or "\n" in value or "\r" in value:
            raise ValueError("hint must be one nonblank line")
        return value


class UploadFrameResult(BaseModel):
    accepted: bool
    frame_index: int
    # Absent, never null, when there's no checked result yet (contract v1.1). A field-level
    # rule rather than the route's exclude_none, which would also drop same_part: null.
    vlm_feedback: VlmFeedback | None = Field(
        default=None, exclude_if=lambda v: v is None
    )


class AgentMessageResult(BaseModel):
    """A-15 / contract v1.9: the latest persisted capture-agent turn for this session."""

    finished: bool
    fallback_reason: Literal["model_error", "timeout", "loop_detected"] | None
    message: str | None


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
    narrative: str | None = None


def verdict_result_with_narrative(payload: dict[str, Any]) -> VerdictResult:
    """A-09/D-082: attaches the deterministic narrative to a `VerdictResult` payload about to
    be served, from either `getVerdict` (session-level, both the intake-shortcut and
    post-reconstruction returns) or `getOrder`/`createOrder` (order-level, I-10's own intake
    verdict) -- every place this project serves a `VerdictResult` gets the same field, round 6,
    @TabeenRaoof, PR #391."""
    from recon.report_narrative import VerdictInput, generate_narrative

    # generate_narrative is the v1, template-only entry point -- never generate_llm_narrative,
    # which is kept as untested-in-production groundwork only (D-082).
    narrative = generate_narrative(VerdictInput.model_validate(payload)).text
    return VerdictResult(**payload, narrative=narrative)


OrderResponse.model_rebuild()


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

    def update_reconstruction(
        self, session_id: str, reconstruction: dict[str, Any]
    ) -> None:
        """Publish the terminal reconstruction without replacing concurrent session fields."""
        with self._lock:
            record = self.load(session_id)
            if record is not None:
                record["reconstruction"] = reconstruction
                self._write_record(record)

    def update_ambiguity(self, session_id: str, ambiguity: dict[str, Any]) -> None:
        """Publish measurement status without overwriting guidance or reconstruction."""
        with self._lock:
            record = self.load(session_id)
            if record is not None:
                record["ambiguity"] = ambiguity
                self._write_record(record)

    def update_deviation_glb(
        self, session_id: str, expected_path: str, output_path: str
    ) -> None:
        """Switch only the GLB path, preserving the latest record and reconstruction fields."""
        with self._lock:
            record = self.load(session_id)
            if record is None:
                raise KeyError(session_id)
            reconstruction = record.get("reconstruction", {})
            if (
                reconstruction.get("status") != "complete"
                or reconstruction.get("glb_path") != expected_path
            ):
                raise ValueError("reconstruction changed before deviation publication")
            reconstruction["glb_path"] = output_path
            self._write_record(record)

    def update_agent_message(
        self,
        session_id: str,
        *,
        fallback_reason: str | None,
        message: str | None,
    ) -> dict[str, Any] | None:
        """Merge only the phone-facing agent result into the latest session record.

        A-15 runs the model turn outside the store lock so uploads stay fast. Reloading under the
        lock here prevents a turn that started from an older snapshot from overwriting a concurrent
        frame_count update or a Finish transition.
        """
        with self._lock:
            record = self.load(session_id)
            if record is None:
                return None
            record["agent_message"] = {
                "finished": record.get("status") == "finished",
                "fallback_reason": fallback_reason,
                "message": message,
            }
            self._write_record(record)
            return record

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
                            "code": "internal_error",
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
                raise SessionAlreadyFinishedError("session_already_finished")
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
    # C-13: without this case, frames that had a board but no focal length fell through to
    # insufficient_baseline ("found 0"), whose advice -- move further between shots -- is wrong.
    # The frozen contract has no intrinsics code (D-031), so it reports as metric_alignment_failed
    # with the real reason. _pose_board_with_session_intrinsics normally removes this case first.
    if failures.get("intrinsics_unavailable", 0):
        return ReconstructionFailure(
            "metric_alignment_failed",
            "No camera focal length was available to compute board poses: the frames carry no "
            f"EXIF focal length, and only {posed_count} frame(s) could be posed without it.",
        )
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


CALIBRATED_BOARD_DIR_NAME = "board_pose_calibrated"
BOARD_CALIBRATION_RECORD = "board_calibration.json"

# R-44: whether fusion first moves each photo onto its own ChArUco corners. Off until the story's
# gate (pose, error and coverage on the ten calipered parts, set before the measurement) is met
# and both other members agree in a numbered decision. Turning it on is a reviewed change to this
# line, not a configuration switch, so the served path can't change without that decision.
CORNER_REANCHORING_ENABLED = False


def _pose_board_with_session_intrinsics(
    capture_dir: Path,
    artifact_dir: Path,
    board_dir: Path,
    board_report: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Give frames with no EXIF focal length a camera model calibrated from the board itself.

    C-13's first real-phone sessions failed with 0 poses although the board was found in every
    frame. The capture app draws each frame from the live camera preview onto a canvas, and a
    canvas JPEG carries no EXIF at all, so R-02 had no focal length to turn detected corners into
    a pose (every frame: intrinsics_unavailable). The golden capture never hit this: those are
    Camera-app photos with EXIF. R-03 already calibrates intrinsics from the corners R-02 records,
    and needs no initial guess for a planar board, so calibrate from this session's own corners
    and run R-02 again with that K for every frame.

    Returns the board-pose folder downstream stages must read (a new one when calibration ran,
    because R-02 refuses to overwrite its outputs) and that folder's report.
    """
    from recon import board_pose, camera_intrinsics

    if not board_report.get("failure_counts", {}).get("intrinsics_unavailable", 0):
        return board_dir, board_report

    try:
        calibration = camera_intrinsics.calibrate_from_board(
            camera_intrinsics.load_board_session(board_dir)
        )
    except camera_intrinsics.CalibrationFailure as exc:
        raise ReconstructionFailure(
            "metric_alignment_failed",
            "The frames carry no EXIF focal length, and calibrating the camera from the "
            f"reference board failed: {exc}",
        ) from exc

    calibrated_dir = artifact_dir / CALIBRATED_BOARD_DIR_NAME
    calibrated_report = board_pose.estimate_board_poses(
        capture_dir, calibrated_dir, intrinsic_override=calibration.intrinsic
    )
    # Kept beside the two board-pose folders so a reviewer can see which K the metric result
    # actually rests on, and how well it fit, without rerunning anything.
    record = {
        "reason": "frames carry no EXIF 35 mm equivalent focal length",
        "source": calibration.source,
        "reprojection_rms_px": calibration.reprojection_rms_px,
        "calibration_view_count": len(calibration.per_view_frame_names),
        "intrinsic": calibration.intrinsic.tolist(),
        "distortion": calibration.distortion.ravel().tolist(),
        "posed_count_before": int(board_report.get("posed_count", 0)),
        "posed_count_after": int(calibrated_report.get("posed_count", 0)),
    }
    (artifact_dir / BOARD_CALIBRATION_RECORD).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return calibrated_dir, calibrated_report


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
        completion,
        disagreement,
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
    started = time.perf_counter()
    posed_board_dir, board_report = _pose_board_with_session_intrinsics(
        capture_dir, artifact_dir, board_dir, board_report
    )
    if posed_board_dir != board_dir:
        stage_timings_s["board_calibration"] = time.perf_counter() - started
        board_dir = posed_board_dir
    failure = _classify_board_failure(board_report)
    if failure is not None:
        raise failure

    started = time.perf_counter()
    # C-13 / D-049: without an explicit device this defaulted to cuda, so the service could not
    # reconstruct on the M2 Max at all. Resolved per session, not at import, so the variable is
    # read from the environment uvicorn was actually started with.
    run_info = reconstruction_runner.run_on_capture(
        capture_dir,
        reconstruction_dir,
        device_type=reconstruction_runner.resolve_device(),
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
    # R-38: load_and_align already rejects per-frame outliers before this point, but the
    # live service never checked the *final* residual against any threshold at all -- found on
    # review (PR #358): the CLI's --max-residual-mm gate only existed in metric_alignment.main,
    # which this pipeline never calls. A session whose alignment is still this bad once rejection
    # has already done what it can is exactly the case the gate exists to stop before fusion.
    if alignment.rms_residual_mm > metric_alignment.DEFAULT_MAX_RESIDUAL_MM:
        raise ReconstructionFailure(
            "metric_alignment_failed",
            f"Board-anchored alignment residual {alignment.rms_residual_mm:.3f} mm exceeds "
            f"{metric_alignment.DEFAULT_MAX_RESIDUAL_MM} mm after outlier-frame rejection.",
        )
    stage_timings_s["metric_alignment"] = time.perf_counter() - started

    started = time.perf_counter()
    try:
        # board_dir is the folder load_and_align read above, so R-44's per-photo corners (when
        # enabled) are the same corners the global alignment was solved from.
        tsdf_fusion.fuse_run(
            reconstruction_dir,
            alignment_path,
            fusion_dir,
            corner_reanchoring=CORNER_REANCHORING_ENABLED,
            board_pose_dir=board_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        # R-20: fuse_run reads its confidence threshold from reconstruction_dir's own run.json
        # by default and raises rather than silently guessing one if it's missing or predates
        # that field (an R-01 run from before R-20). The frozen contract (D-032) has no
        # dedicated error code for this -- adding one needs D-031's two-member sprint-boundary
        # process, not a unilateral change here -- so this reuses metric_alignment_failed, the
        # closest existing "reconstruction-stage precondition failed" code, but names the real
        # cause explicitly rather than falling through to run_session's generic terminal-
        # exception handler, which would report it as an unrelated "Reconstruction failed".
        raise ReconstructionFailure(
            "metric_alignment_failed", f"Fusion failed: {exc}"
        ) from exc
    stage_timings_s["fusion"] = time.perf_counter() - started

    started = time.perf_counter()
    mesh_cleanup.cleanup_run(fusion_dir / "mesh.ply", model_path)
    stage_timings_s["export"] = time.perf_counter() - started

    started = time.perf_counter()
    provenance.provenance_run(
        model_path,
        reconstruction_dir,
        alignment_path,
        provenance_dir,
        fusion_dir=fusion_dir,
    )
    stage_timings_s["provenance"] = time.perf_counter() - started

    started = time.perf_counter()
    uncertainty_report = uncertainty.uncertainty_run(provenance_dir, uncertainty_dir)
    stage_timings_s["uncertainty"] = time.perf_counter() - started
    final_model_path = uncertainty_dir / "mesh_uncertainty.glb"

    # A-04: shape completion only reads the files above and writes inside completion/, so the
    # served GLB is the same with it on or off. It never raises; a configured runner that fails
    # adds one warning, and a host with no runner just records that (D-058).
    warnings: list[str] = []
    if not completion.completion_disabled():
        started = time.perf_counter()
        outcome = completion.run_completion_stage(
            completion.PipelineArtifacts(
                capture_dir=capture_dir,
                board_pose_dir=board_dir,
                reconstruction_dir=reconstruction_dir,
                fusion_dir=fusion_dir,
                provenance_dir=provenance_dir,
                uncertainty_dir=uncertainty_dir,
            ),
            artifact_dir / "completion",
            completion.resolve_runner_command(),
        )
        stage_timings_s["completion"] = time.perf_counter() - started
        if outcome.warning is not None:
            warnings.append(outcome.warning)

        # A-05: the disagreement map, from the samples just written. Guidance only (D-005, D-006):
        # it writes inside completion/disagreement/ and nothing here reads it back. It never
        # raises and adds no warning; with no runner it just records `unavailable`.
        started = time.perf_counter()
        disagreement.disagreement_stage(
            artifact_dir / "completion", fusion_dir=fusion_dir
        )
        stage_timings_s["disagreement"] = time.perf_counter() - started

    return {
        "glb_path": str(final_model_path),
        "reference_tier_used": "charuco_board",
        "metric": True,
        # R-25: report coverage from the same post-downgrade labels encoded in the GLB we serve.
        "observed_fraction": uncertainty_report.get("observed_fraction"),
        "stage_timings_s": stage_timings_s,
        "warnings": warnings,
    }


def public_model_url(request: Request, session_id: str) -> str:
    """The absolute, externally-reachable URL for a session's finished GLB.

    Most specific first: an operator-set override, then a proxy-forwarded original host, then
    falling back to request.url_for's own scheme+host (correct for a direct local/CI caller,
    wrong behind Lightning's proxy -- see PUBLIC_BASE_URL_ENV above).
    """
    override = os.environ.get(PUBLIC_BASE_URL_ENV, "").strip()
    if override:
        return f"{override.rstrip('/')}/v1/sessions/{session_id}/model.glb"

    forwarded_host = request.headers.get("x-forwarded-host", "").strip()
    if forwarded_host:
        scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
        return f"{scheme}://{forwarded_host}/v1/sessions/{session_id}/model.glb"

    return str(request.url_for("get_model", session_id=session_id))


def create_app(
    data_dir: Path | None = None,
    pipeline_runner: PipelineRunner = run_reconstruction_pipeline,
    known_order_codes: set[str] | None = None,
    vlm_supervisor: VlmSupervisor | None = None,
    orders_dir: Path | None = None,
    buyer_key: str | None = None,
    spec_parser: Callable[[str], Any] | None = None,
    agent_model_caller: Callable[[Any], Any] | None = None,
    agent_timeout_s: float | None = None,
) -> FastAPI:
    # R-31: keep agent_tools lazy so the runtime guard above has already run before its
    # provenance dependency can import Open3D.
    from recon import agent_tools

    root = Path(
        data_dir or os.environ.get("PROOFSHAPE_DATA_DIR", "proofshape-data/sessions")
    )
    store = SessionStore(root)
    orders_root = Path(
        orders_dir or os.environ.get("PROOFSHAPE_ORDERS_DIR", "proofshape-data/orders")
    )
    order_store = OrderStore(orders_root)
    configured_buyer_key = (
        buyer_key
        if buyer_key is not None
        else os.environ.get("PROOFSHAPE_BUYER_KEY", "")
    )
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    supervisor = vlm_supervisor or VlmSupervisor(gemini_key)
    configured_spec_parser = spec_parser
    if configured_spec_parser is None and gemini_key:
        configured_spec_parser = GeminiSpecParser(gemini_key)

    # A-15: the phone-facing agent uses A-07's existing loop. The model caller is injectable so
    # service tests never need a live provider. With no key, deliberately enter A-07's
    # deterministic fallback path rather than silently disabling the assistant.
    from recon.agent_loop import DEFAULT_TIMEOUT_S as AGENT_DEFAULT_TIMEOUT_S
    from recon.agent_loop import run_agent_turn
    from recon.agent_model import GeminiAgentCaller

    configured_agent_timeout_s = (
        AGENT_DEFAULT_TIMEOUT_S if agent_timeout_s is None else float(agent_timeout_s)
    )
    if configured_agent_timeout_s <= 0:
        raise ValueError("agent_timeout_s must be positive")
    configured_agent_model_caller = agent_model_caller
    if configured_agent_model_caller is None and gemini_key:
        configured_agent_model_caller = GeminiAgentCaller(gemini_key)

    def missing_agent_model(_state: Any) -> Any:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    # A-15 review: accepted frame uploads must return promptly and stay accepted even if the
    # optional capture agent fails. One background turn per accepted upload is serialized per
    # session. The map holds locks weakly: active/waiting turns keep their lock alive, and once no
    # turn references a finished/abandoned session its entry disappears instead of growing for the
    # life of a long-running deployment.
    agent_turn_locks_guard = threading.Lock()
    agent_turn_locks: weakref.WeakValueDictionary[str, threading.Lock] = (
        weakref.WeakValueDictionary()
    )

    def agent_turn_lock(session_id: str) -> threading.Lock:
        with agent_turn_locks_guard:
            lock = agent_turn_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                agent_turn_locks[session_id] = lock
            return lock

    def run_agent_after_upload(session_id: str) -> None:
        scheduled_reconstructions: list[str] = []
        lock = agent_turn_lock(session_id)
        try:
            with lock:
                record = store.load(session_id)
                if record is None or record.get("status") != "capturing":
                    return
                caller = configured_agent_model_caller or missing_agent_model
                agent_result = run_agent_turn(
                    record,
                    store,
                    session_id,
                    scheduled_reconstructions.append,
                    caller,
                    min_frames=MIN_FINISH_FRAMES,
                    timeout_s=configured_agent_timeout_s,
                )
                store.update_agent_message(
                    session_id,
                    fallback_reason=agent_result.fallback_reason,
                    message=agent_result.message,
                )
        except Exception:
            # Optional guidance must not fail an accepted upload or strand a finished session.
            logger.exception("Capture agent turn failed for session %s", session_id)
        finally:
            # The finish tool has already committed the session state. Even if message
            # persistence fails, launch its scheduled reconstruction outside the agent lock.
            for scheduled_session_id in scheduled_reconstructions:
                run_session(scheduled_session_id)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            supervisor.close()

    app = FastAPI(
        title="ProofShape reconstruction service", version="1.3.0", lifespan=lifespan
    )
    # C-13: capture/ is served from a different origin than this service (D-042 left this
    # decision to whichever story first connected them). A wildcard origin is deliberate, not an
    # oversight: CORS only restricts *browser* JS -- a non-browser caller (curl, smoke_r14.py)
    # ignores it entirely -- and the deployed service's real access controls are Lightning's own
    # bearer token (checked at its proxy, never in this app) and the known-order-code check
    # below, neither of which CORS can add to or take away from. No cookies/credentials are
    # used, so allow_credentials stays False, which is what makes a wildcard origin valid here.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.state.vlm_supervisor = supervisor
    app.state.store = store
    app.state.order_store = order_store
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
                    code="malformed_request",
                    message=(
                        "Invalid frame upload. Both frame and gyro multipart file "
                        "parts are required."
                    ),
                ),
            )
        if request.method == "POST" and request.url.path.endswith("/v1/sessions"):
            # R-29 (#186's validation half) made this a contract-shaped 400; v1.1 (D-056) gives it
            # its own code, since a body that can't be read is an app bug, not an unknown order
            # code. createSession's 400 already covers "the request is malformed".
            return error(
                400,
                ErrorDetail(
                    code="malformed_request", message="The request body is malformed."
                ),
            )
        if request.method == "POST" and request.url.path.endswith("/v1/orders"):
            return error(
                422,
                ErrorDetail(
                    code="malformed_request",
                    message="The order request is malformed; a JSON spec part is required.",
                ),
            )
        if request.method == "POST" and request.url.path.endswith("/v1/spec/parse"):
            return error(
                422,
                ErrorDetail(
                    code="malformed_request",
                    message="The spec-parser request needs non-empty buyer prose.",
                ),
            )
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    def order_response(record: dict[str, Any]) -> OrderResponse:
        cad = record.get("cad")
        # I-09: computed fresh from the stored spec on every read, never persisted into the
        # record itself -- so a future change to k/sigma_floor (or G-10 adding sheet/card
        # floors) is reflected on the next read of an existing order, not frozen at creation.
        from inspection.intake_check import (
            check_intake_spec,
            intake_unverifiable_verdict,
        )

        intake_verdict = (
            intake_unverifiable_verdict(record["spec"]) if record["has_cad"] else None
        )
        return OrderResponse(
            order_code=record["order_code"],
            created_at=record["created_at"],
            has_cad=bool(record["has_cad"]),
            cad_units=None if cad is None else cad.get("units"),
            cad_bounding_box_mm=None if cad is None else cad.get("bounding_box_mm"),
            spec=record["spec"],
            placement_note=record.get("placement_note"),
            buyer_reference=record.get("buyer_reference"),
            intake_check=[
                IntakeCheckResult(**result)
                for result in check_intake_spec(record["spec"])
            ],
            verdict=(
                None
                if intake_verdict is None
                else verdict_result_with_narrative(intake_verdict)
            ),
        )

    def buyer_key_authorized(value: str | None) -> bool:
        return bool(
            configured_buyer_key
            and value is not None
            and secrets.compare_digest(value, configured_buyer_key)
        )

    def spec_parse_response(result: SpecParseResult) -> JSONResponse:
        # C-16: the contract omits top-level fields that do not apply, but a nested
        # feature.tolerance_mm=null is meaningful A-03 evidence and must survive.
        content = {
            key: value
            for key, value in result.model_dump(mode="json").items()
            if value is not None
        }
        return JSONResponse(content=content)

    @app.post(
        "/v1/spec/parse",
        response_model=SpecParseResult,
        responses={
            401: {"model": ErrorDetail},
            422: {"model": ErrorDetail},
        },
        operation_id="parseSpec",
    )
    def parse_spec(
        body: SpecParseRequest,
        x_proofshape_buyer_key: Annotated[str | None, Header()] = None,
    ):
        if not buyer_key_authorized(x_proofshape_buyer_key):
            return error(
                401,
                ErrorDetail(
                    code="buyer_auth_required",
                    message="A valid buyer key is required to parse a spec.",
                ),
            )
        prose = body.prose.strip()
        if not prose:
            return error(
                422,
                ErrorDetail(
                    code="malformed_request",
                    message="The buyer prose must not be blank.",
                ),
            )
        if configured_spec_parser is None:
            return spec_parse_response(unparsed("missing_api_key"))
        try:
            response = configured_spec_parser(prose)
        except Exception:
            logger.exception("Spec parser call failed")
            return spec_parse_response(unparsed("model_error"))
        return spec_parse_response(parse_spec_response(response))

    @app.post(
        "/v1/orders",
        status_code=201,
        response_model=OrderResponse,
        response_model_exclude_none=True,
        responses={
            401: {"model": ErrorDetail},
            422: {"model": ErrorDetail},
        },
        operation_id="createOrder",
    )
    def create_order(
        spec: Annotated[UploadFile, File()],
        cad: Annotated[UploadFile | None, File()] = None,
        placement_note: Annotated[str | None, Form()] = None,
        buyer_reference: Annotated[str | None, Form()] = None,
        x_proofshape_buyer_key: Annotated[str | None, Header()] = None,
    ):
        if not buyer_key_authorized(x_proofshape_buyer_key):
            return error(
                401,
                ErrorDetail(
                    code="buyer_auth_required",
                    message="A valid buyer key is required to create an order.",
                ),
            )

        try:
            raw_spec = spec.file.read()
            parsed_spec = json.loads(raw_spec.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return error(
                422,
                ErrorDetail(
                    code="malformed_request",
                    message="The spec part must contain a JSON object.",
                ),
            )
        if not isinstance(parsed_spec, dict):
            return error(
                422,
                ErrorDetail(
                    code="malformed_request",
                    message="The spec part must contain a JSON object.",
                ),
            )

        def normalized_optional_text(value: str | None) -> str | None:
            if value is None:
                return None
            stripped = value.strip()
            return stripped or None

        cad_name = None
        cad_source_path = None
        cad_units = None
        cad_bbox_mm = None
        if cad is not None and cad.filename:
            from inspection.cad import load_mesh

            cad_name = Path(cad.filename).name
            cad_bytes = cad.file.read()
            if not cad_bytes:
                return error(
                    422,
                    ErrorDetail(
                        code="cad_unreadable",
                        message="The uploaded CAD file is empty or unreadable.",
                    ),
                )
            suffix = Path(cad_name).suffix.lower()
            if suffix not in {".stl", ".obj", ".glb"}:
                return error(
                    422,
                    ErrorDetail(
                        code="cad_unreadable",
                        message="The CAD file must be STL, OBJ, or GLB.",
                    ),
                )

            validate_dir = orders_root / ".validate"
            validate_dir.mkdir(parents=True, exist_ok=True)
            cad_source_path = validate_dir / f"{uuid.uuid4().hex}{suffix}"
            cad_source_path.write_bytes(cad_bytes)
            try:
                vertices, _faces, cad_units = load_mesh(cad_source_path)
                mins = vertices.min(axis=0).astype(float).tolist()
                maxs = vertices.max(axis=0).astype(float).tolist()
                cad_bbox_mm = [mins, maxs]
            except (
                AttributeError,
                IndexError,
                KeyError,
                OSError,
                TypeError,
                ValueError,
            ):
                cad_source_path.unlink(missing_ok=True)
                return error(
                    422,
                    ErrorDetail(
                        code="cad_unreadable",
                        message="The CAD file could not be read.",
                    ),
                )

        try:
            record = order_store.create(
                created_at=datetime.now(UTC).isoformat(),
                spec=parsed_spec,
                placement_note=normalized_optional_text(placement_note),
                buyer_reference=normalized_optional_text(buyer_reference),
                cad_name=cad_name,
                cad_source_path=cad_source_path,
                cad_units=cad_units,
                cad_bbox_mm=cad_bbox_mm,
            )
        finally:
            if cad_source_path is not None:
                cad_source_path.unlink(missing_ok=True)
        return order_response(record)

    @app.get(
        "/v1/orders/{order_code}",
        response_model=OrderResponse,
        response_model_exclude_none=True,
        responses={404: {"model": ErrorDetail}},
        operation_id="getOrder",
    )
    def get_order(order_code: str):
        record = order_store.load(order_code)
        if record is None:
            return error(
                404,
                ErrorDetail(
                    code="order_code_unknown", message="The order code is unknown."
                ),
            )
        return order_response(record)

    @app.post(
        "/v1/sessions",
        status_code=201,
        response_model=SessionResponse,
        responses={400: {"model": ErrorDetail}},
        operation_id="createSession",
    )
    def create_session(body: CreateSessionRequest):
        order_code = body.order_code.strip()
        known = (
            order_code in known_order_codes
            if known_order_codes is not None
            else order_store.exists(order_code)
        )
        if not order_code or not known:
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
        background_tasks: BackgroundTasks,
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
                    code="malformed_request",
                    message="The uploaded frame is empty.",
                ),
            )
        try:
            gyro_value = _parse_gyro(await gyro.read())
        except (TypeError, ValueError) as exc:
            return error(
                422,
                ErrorDetail(
                    code="malformed_request",
                    message=f"Malformed frame upload: the gyro part is invalid ({exc}).",
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
        # No Google call (or capacity wait) occurs on this request path. The saved files
        # are the source of truth, including after a restart or upload/finish race.
        session_dir = store.session_dir(session_id)
        supervisor.submit(session_dir, frame_index)
        # A-11: attach the newest result already on disk. Reading files never waits on the
        # VLM, so this upload is no slower; the hint may describe an earlier frame.
        checked = supervisor.latest_checked(session_dir)
        feedback = None
        if checked is not None:
            feedback = VlmFeedback(
                frame_index=checked.frame_index, **checked.observation.model_dump()
            )

        # A-15 option (a): the accepted upload returns immediately. The agent turn starts after
        # the response is sent, is serialized per session, and persists only its latest
        # phone-facing result. GET /agent-message remains read-only.
        background_tasks.add_task(run_agent_after_upload, session_id)

        return UploadFrameResult(
            accepted=True, frame_index=frame_index, vlm_feedback=feedback
        )

    @app.get(
        "/v1/sessions/{session_id}/agent-message",
        response_model=AgentMessageResult,
        responses={404: {"model": ErrorDetail}},
        operation_id="getAgentMessage",
    )
    def get_agent_message(session_id: str):
        record = store.load(session_id)
        if record is None:
            return error(
                404,
                ErrorDetail(
                    code="session_not_found", message="No session with that id."
                ),
            )
        message = record.get("agent_message")
        if message is None:
            # Normally the phone asks only after an accepted upload, but a direct caller may ask
            # earlier. This is a valid empty state, not a fake "session not found".
            return AgentMessageResult(
                finished=record.get("status") == "finished",
                fallback_reason=None,
                message=None,
            )
        return AgentMessageResult(
            finished=record.get("status") == "finished",
            fallback_reason=message.get("fallback_reason"),
            message=message.get("message"),
        )

    def run_session(session_id: str) -> None:
        record = store.load(session_id)
        if record is None:
            return
        session_dir = store.session_dir(session_id)
        try:
            supervisor.schedule_finish(session_dir, int(record["frame_count"]))
            result = pipeline_runner(session_dir / "frames", session_dir / "artifacts")
            glb_path = Path(result["glb_path"])
            if not glb_path.exists():
                raise FileNotFoundError(
                    f"pipeline reported {glb_path}, but no GLB exists there"
                )
            reconstruction = {
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
            reconstruction = {"status": "failed", "failure": exc.as_dict()}
        except Exception:  # terminal job boundary, logged below
            # Background reconstruction must never leave a persisted session at
            # status=pending. Unexpected numpy/cv2/model exceptions become a terminal
            # internal_error; the exception text is logged for the operator, not shown to
            # the supplier (#186).
            logger.exception("Reconstruction crashed for session %s", session_id)
            reconstruction = {
                "status": "failed",
                "failure": {
                    "code": "internal_error",
                    "message": INTERNAL_ERROR_MESSAGE,
                },
            }
        # Join even when reconstruction fails. Publish only the terminal result under the
        # store lock: a guidance turn may have updated session.json during the pipeline.
        supervisor.join(session_dir, int(record["frame_count"]))
        store.update_reconstruction(session_id, reconstruction)
        run_post_reconstruction_measurement(session_id)

    def run_post_reconstruction_measurement(session_id: str) -> None:
        """I-07: once reconstruction completes, run I-05's registration/ambiguity check and
        I-06's deviation computation against the order's CAD (if any) -- both are otherwise
        dormant, called by nothing in the live pipeline. Best-effort: any failure here is
        caught and recorded as `record["ambiguity"]["error"]`, never raised. Reconstruction
        itself already succeeded; this is a second, independent pass `getVerdict` treats as
        Rescan if it didn't complete, not a reconstruction failure.
        """
        record = store.load(session_id)
        if record is None:
            return

        reconstruction = record.get("reconstruction", {})
        if reconstruction.get("status") != "complete":
            return
        order = order_store.load(record["order_code"])
        if order is None or not order.get("has_cad"):
            return

        from inspection import registration
        from inspection.cad import load_mesh
        from inspection.deviation import publish_deviation_glb

        from recon.mesh_cleanup import read_glb
        from recon.part_segmentation import BoardSurface
        from recon.provenance import colors_to_labels

        try:
            session_dir = store.session_dir(session_id)
            cad_path = order_store.cad_path(record["order_code"])
            if cad_path is None:
                # PR #379 review (@TabeenRaoof): order.has_cad=True but no CAD file on disk is
                # an inconsistent order, not a transient timing gap -- get_verdict's own earlier
                # has_cad check won't 404 for it, so silently returning here (never writing
                # `ambiguity`) left get_verdict answering "measurement has not finished yet"
                # forever, for a measurement that will now never run. Recorded as an error, the
                # same as every other best-effort failure in this function.
                store.update_ambiguity(
                    session_id, {"error": "registration_or_deviation_failed"}
                )
                return
            cad_vertices, cad_faces, cad_units = load_mesh(cad_path)
            cad_mesh = (cad_vertices, cad_faces, cad_units)

            glb_path = Path(reconstruction["glb_path"])
            model_vertices, model_faces, model_colors = read_glb(glb_path)
            model_labels = colors_to_labels(model_colors)
            model = (model_vertices, model_faces, model_labels)

            board_surface_report = json.loads(
                (session_dir / "artifacts" / "fusion" / "tsdf_fusion.json").read_text(
                    encoding="utf-8"
                )
            )["board_surface"]
            board_surface = BoardSurface(**board_surface_report)

            candidates = registration.global_register(model, cad_mesh)
            ambiguity = registration.check_ambiguity(candidates, model, cad_mesh)
            publish_deviation_glb(
                store,
                session_id,
                cad_mesh,
                ambiguity.best.transformation,
                board_surface,
            )
        except Exception:
            logger.exception(
                "Post-reconstruction measurement failed for session %s", session_id
            )
            store.update_ambiguity(
                session_id, {"error": "registration_or_deviation_failed"}
            )
            return

        store.update_ambiguity(
            session_id,
            {
                "orientation_ambiguous": ambiguity.orientation_ambiguous,
                "reason_codes": list(ambiguity.reason_codes),
            },
        )

    @app.post(
        "/v1/sessions/{session_id}/finish",
        status_code=202,
        response_model=SessionResponse,
        responses={409: {"model": ErrorDetail}},
        operation_id="finishSession",
    )
    def finish_session(session_id: str, background_tasks: BackgroundTasks):
        result = agent_tools.finish_session(
            store,
            session_id,
            lambda sid: background_tasks.add_task(run_session, sid),
            min_frames=MIN_FINISH_FRAMES,
        )
        if not result.ok:
            return error(
                409, ErrorDetail(code=result.error_code, message=result.error_message)
            )
        return SessionResponse(**result.record)

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
            code = failure.get("code", "internal_error")
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
        response["glb_url"] = public_model_url(request, session_id)
        return ReconstructionResult(**response)

    @app.get(
        "/v1/sessions/{session_id}/verdict",
        response_model=VerdictResult,
        responses={404: {"model": ErrorDetail}, 422: {"model": ErrorDetail}},
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
        order = order_store.load(record["order_code"])
        if order is None or not order.get("has_cad"):
            return error(
                404,
                ErrorDetail(
                    code="no_cad_uploaded",
                    message="No CAD was supplied for this session — no verdict exists.",
                ),
            )
        from inspection.intake_check import intake_unverifiable_verdict

        intake_verdict = intake_unverifiable_verdict(order["spec"])
        if intake_verdict is not None:
            return verdict_result_with_narrative(intake_verdict)

        reconstruction = record.get("reconstruction", {})
        if reconstruction.get("status") != "complete":
            return error(
                404,
                ErrorDetail(
                    code="verdict_not_available",
                    message=(
                        "A CAD is stored for this order, but reconstruction has not "
                        "completed yet."
                    ),
                ),
            )

        import numpy as np
        from inspection.deviation import (
            DEVIATION_MEASURED_ATTRIBUTE,
            DEVIATION_MM_ATTRIBUTE,
        )
        from inspection.verdict import (
            NoJudgedFeaturesError,
            RefusedToleranceError,
            compute_verdict,
        )

        from recon.mesh_cleanup import read_glb, read_glb_extra_attributes
        from recon.provenance import colors_to_labels
        from recon.uncertainty import read_sigma_npz

        session_dir = store.session_dir(session_id)
        glb_path = Path(reconstruction["glb_path"])
        vertices, _faces, colors = read_glb(glb_path)
        labels = colors_to_labels(colors)
        extra = read_glb_extra_attributes(
            glb_path, [DEVIATION_MM_ATTRIBUTE, DEVIATION_MEASURED_ATTRIBUTE]
        )
        # I-07: a session whose post-reconstruction measurement (registration + deviation)
        # never completed has no deviation attributes on its served GLB yet -- treated as
        # registration_failed below, same as an explicit error.
        deviation_mm = extra.get(DEVIATION_MM_ATTRIBUTE, np.zeros(len(vertices)))
        deviation_measured = extra.get(
            DEVIATION_MEASURED_ATTRIBUTE, np.zeros(len(vertices))
        )

        ambiguity_record = record.get("ambiguity")
        if ambiguity_record is None:
            # PR #379 review (@TabeenRaoof): run_session marks reconstruction "complete" and
            # saves the record *before* calling run_post_reconstruction_measurement (so a crash
            # in the latter can never leave the former unpersisted) -- a request landing in that
            # window previously fell through to registration_failed below and returned a
            # Rescan that looked final, for a part that was never actually measured. Measurement
            # genuinely not having run yet is distinct from it having run and recorded an error;
            # only the latter is `registration_failed`.
            return error(
                404,
                ErrorDetail(
                    code="verdict_not_available",
                    message=(
                        "Reconstruction is complete, but post-reconstruction measurement "
                        "has not finished yet."
                    ),
                ),
            )
        registration_failed = "error" in ambiguity_record
        orientation_ambiguous = bool(
            not registration_failed and ambiguity_record.get("orientation_ambiguous")
        )
        ambiguity_reason_codes = (
            [] if registration_failed else ambiguity_record.get("reason_codes", [])
        )

        sigma_npz_path = session_dir / "artifacts" / "uncertainty" / "sigma_mm.npz"
        sigma_mm = (
            read_sigma_npz(sigma_npz_path)["sigma_mm"]
            if sigma_npz_path.exists()
            else np.full(len(vertices), np.nan)
        )

        # D-073: sigma_floor only covers out-of-plane (height) comparisons -- compute_verdict
        # needs each vertex's height above the board to tell those apart from in-plane (wall)
        # ones, so it needs the same board plane TSDF fusion already fit for this session
        # (written during reconstruction itself, independent of whether the post-reconstruction
        # measurement step below later succeeded).
        board_surface_report = json.loads(
            (session_dir / "artifacts" / "fusion" / "tsdf_fusion.json").read_text(
                encoding="utf-8"
            )
        )["board_surface"]
        board_plane = (
            board_surface_report["slope_x"],
            board_surface_report["slope_y"],
            board_surface_report["z_at_origin_mm"],
        )

        try:
            result = compute_verdict(
                deviation_mm,
                deviation_measured,
                labels,
                sigma_mm,
                vertices,
                board_plane,
                order["spec"],
                orientation_ambiguous=orientation_ambiguous,
                ambiguity_reason_codes=ambiguity_reason_codes,
                registration_failed=registration_failed,
            )
        except RefusedToleranceError as exc:
            return error(422, ErrorDetail(code="tolerance_missing", message=str(exc)))
        except NoJudgedFeaturesError as exc:
            # PR #379 review (@TabeenRaoof, @mbj1994): reused, not a new contract error code --
            # there is as little a verdict to serve here as when measurement hasn't run yet.
            return error(
                404, ErrorDetail(code="verdict_not_available", message=str(exc))
            )

        # A-02/D-084: vlm_checks.json always exists once reconstruction is "complete" --
        # VlmSupervisor.join (recon/vlm_check.py) writes it unconditionally from run_session,
        # even on a failed reconstruction, before this endpoint's own "complete" check above
        # could ever let a request reach this point. A missing file is still read as None,
        # not crashed on (review, @TabeenRaoof, PR #396): apply_vlm_veto treats that the same
        # as unchecked evidence, never as a clean session.
        from recon.vlm_veto import apply_vlm_veto

        try:
            vlm_checks_report = json.loads(
                (session_dir / "vlm_checks.json").read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            vlm_checks_report = None
        result = apply_vlm_veto(result, vlm_checks_report)

        return verdict_result_with_narrative(
            {**result, "declared_at": "post_reconstruction"}
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

"""R-12: COLMAP reconstruction backend using ChArUco board camera positions as priors.

R-02 stores camera-from-board extrinsics in millimetres:

    x_cam = R @ x_board + t

The camera centre in board coordinates is therefore:

    C_board = -R.T @ t

COLMAP's pose-prior pipeline consumes camera positions.  R-12 converts the
R-02 centres from millimetres to metres before giving them to COLMAP.

These positions are reconstruction priors and alignment evidence only.
They are not object-accuracy or tolerance measurements.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from recon.frames import find_capture_frames

MIN_POSITION_PRIORS = 3


def camera_center_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    """Return camera centre from one camera-from-world [R|t] matrix."""
    extrinsic = np.asarray(extrinsic, dtype=np.float64)

    if extrinsic.shape != (3, 4):
        raise ValueError(f"extrinsic must have shape (3,4), got {extrinsic.shape}")

    if not np.isfinite(extrinsic).all():
        raise ValueError("extrinsic contains non-finite values")

    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]

    return -(rotation.T @ translation)


def load_board_position_priors(
    board_pose_path: Path,
) -> dict[str, np.ndarray]:
    """Load usable R-02 poses as COLMAP Cartesian camera-position priors.

    Returned positions are in metres. R-02 stores millimetres.
    """
    board_pose_path = Path(board_pose_path)

    if not board_pose_path.is_file():
        raise FileNotFoundError(f"Board pose file not found: {board_pose_path}")

    with np.load(board_pose_path) as board:
        required = ("frame_names", "pose_found", "extrinsic")

        for key in required:
            if key not in board:
                raise ValueError(f"{board_pose_path} is missing required array {key!r}")

        frame_names = [str(name) for name in board["frame_names"]]
        pose_found = np.asarray(board["pose_found"], dtype=bool)
        extrinsics = np.asarray(board["extrinsic"], dtype=np.float64)

    frame_count = len(frame_names)

    if pose_found.shape != (frame_count,):
        raise ValueError("pose_found must contain one value per frame")

    if extrinsics.shape != (frame_count, 3, 4):
        raise ValueError(
            f"extrinsic must have shape ({frame_count},3,4), got {extrinsics.shape}"
        )

    if len(set(frame_names)) != frame_count:
        raise ValueError("Board pose frame names must be unique")

    priors: dict[str, np.ndarray] = {}

    for index, frame_name in enumerate(frame_names):
        if not pose_found[index]:
            continue

        center_mm = camera_center_from_extrinsic(extrinsics[index])

        # COLMAP's pose-prior workflow conventionally operates in metres.
        priors[frame_name] = center_mm / 1000.0

    if len(priors) < MIN_POSITION_PRIORS:
        raise ValueError(
            "COLMAP needs at least "
            f"{MIN_POSITION_PRIORS} usable board position priors; "
            f"found {len(priors)}"
        )

    return priors


def load_pycolmap() -> Any:
    """Import PyCOLMAP lazily so normal CI does not require the GPU package."""
    try:
        import pycolmap
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyCOLMAP is not installed. R-12 requires the pinned COLMAP GPU environment."
        ) from exc

    return pycolmap


def write_board_position_priors(
    database: Any,
    database_images: list[Any],
    position_priors_m: dict[str, np.ndarray],
    pycolmap_module: Any,
) -> tuple[int, list[str]]:
    """Replace all imported EXIF/GPS priors with R-02 board-frame positions.

    Feature extraction may automatically import phone GPS metadata as WGS84
    priors. Mixing those coordinates with the board-frame Cartesian priors
    would be invalid, so R-12 removes all existing priors first.
    """
    database.clear_pose_priors()

    written = 0
    missing: list[str] = []

    for image in database_images:
        position = position_priors_m.get(image.name)

        if position is None:
            missing.append(image.name)
            continue

        prior = pycolmap_module.PosePrior(
            corr_data_id=image.data_id,
            position=np.asarray(position, dtype=np.float64),
            coordinate_system=(pycolmap_module.PosePriorCoordinateSystem.CARTESIAN),
        )

        database.write_pose_prior(prior)
        written += 1

    if written < MIN_POSITION_PRIORS:
        raise ValueError(
            "COLMAP needs at least "
            f"{MIN_POSITION_PRIORS} board priors among the capture frames; "
            f"found {written}"
        )

    return written, missing


def choose_best_reconstruction(
    models: dict[int, Any],
) -> tuple[int, Any] | None:
    """Choose the strongest COLMAP model deterministically.

    Prefer the model registering the most images, then the most sparse
    points, then the lower model id as a deterministic tie-break.
    """
    if not models:
        return None

    best_model_id: int | None = None
    best_reconstruction: Any = None
    best_score: tuple[int, int, int] | None = None

    for model_id, reconstruction in models.items():
        score = (
            int(reconstruction.num_reg_images()),
            int(reconstruction.num_points3D()),
            -int(model_id),
        )

        if best_score is None or score > best_score:
            best_score = score
            best_model_id = int(model_id)
            best_reconstruction = reconstruction

    return best_model_id, best_reconstruction


def run_sparse_registration(
    capture_dir: Path,
    board_pose_path: Path,
    workspace: Path,
    device_type: str = "cuda",
) -> dict[str, Any]:
    """Run COLMAP SfM with R-02 board-position priors.

    This stage deliberately stops at sparse registration. Dense depth and
    the R-01 four-file contract are added by the next R-12 stage.
    """
    if device_type != "cuda":
        raise ValueError(
            "R-12 has currently only been validated on CUDA; use --device cuda."
        )

    capture_dir = Path(capture_dir)
    workspace = Path(workspace)

    frame_paths = find_capture_frames(capture_dir)
    frame_names = [path.name for path in frame_paths]

    position_priors_m = load_board_position_priors(board_pose_path)

    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError(
            f"{workspace} already contains files; refusing to mix COLMAP runs."
        )

    workspace.mkdir(parents=True, exist_ok=True)

    database_path = workspace / "database.db"
    sparse_path = workspace / "sparse"
    sparse_path.mkdir()

    pycolmap = load_pycolmap()

    timings: dict[str, float] = {}
    total_started = time.perf_counter()

    started = time.perf_counter()

    pycolmap.extract_features(
        database_path=database_path,
        image_path=capture_dir,
        image_names=frame_names,
        camera_mode=pycolmap.CameraMode.SINGLE,
        device=pycolmap.Device.auto,
    )

    timings["feature_extraction_s"] = time.perf_counter() - started

    started = time.perf_counter()

    pycolmap.match_sequential(
        database_path=database_path,
        device=pycolmap.Device.auto,
    )

    timings["matching_s"] = time.perf_counter() - started

    started = time.perf_counter()

    with pycolmap.Database.open(database_path) as database:
        database_images = database.read_all_images()

        prior_count, frames_without_prior = write_board_position_priors(
            database,
            database_images,
            position_priors_m,
            pycolmap,
        )

        stored_prior_count = len(database.read_all_pose_priors())

    timings["prior_setup_s"] = time.perf_counter() - started

    if stored_prior_count != prior_count:
        raise RuntimeError(
            "COLMAP pose-prior database verification failed: "
            f"wrote {prior_count}, read back {stored_prior_count}"
        )

    options = pycolmap.IncrementalPipelineOptions()
    options.use_prior_position = True
    options.use_robust_loss_on_prior_position = True

    started = time.perf_counter()

    models = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=capture_dir,
        output_path=sparse_path,
        options=options,
    )

    timings["mapping_s"] = time.perf_counter() - started
    timings["total_s"] = time.perf_counter() - total_started

    selected = choose_best_reconstruction(models)

    if selected is None:
        return {
            "status": "refused",
            "reason": "colmap_registered_no_model",
            "frame_count": len(frame_names),
            "registered_frame_count": 0,
            "sparse_point_count": 0,
            "board_prior_count": prior_count,
            "frames_without_board_prior": frames_without_prior,
            "timings": timings,
            "database_path": database_path,
            "sparse_path": sparse_path,
            "reconstruction": None,
        }

    model_id, reconstruction = selected

    return {
        "status": "success",
        "reason": None,
        "model_id": model_id,
        "frame_count": len(frame_names),
        "registered_frame_count": int(reconstruction.num_reg_images()),
        "sparse_point_count": int(reconstruction.num_points3D()),
        "board_prior_count": prior_count,
        "frames_without_board_prior": frames_without_prior,
        "timings": timings,
        "database_path": database_path,
        "sparse_path": sparse_path,
        "reconstruction": reconstruction,
    }

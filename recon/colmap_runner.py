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

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from recon.frames import find_capture_frames
from recon.vggt_runner import write_run_outputs

MIN_POSITION_PRIORS = 3
DEFAULT_MAX_IMAGE_SIZE = 1600


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


def default_board_pose_path(capture_dir: Path) -> Path:
    """Return the conventional R-02 board-pose path for one capture."""
    return (
        Path("fixtures/data/board_poses") / Path(capture_dir).name / "board_poses.npz"
    )


def depth_validity_mask(depth: np.ndarray) -> np.ndarray:
    """Return geometric-depth validity as float32 0/1 values.

    This is not a learned or probabilistic confidence estimate.
    """
    depth = np.asarray(depth)
    return (np.isfinite(depth) & (depth > 0)).astype(np.float32)


def load_dense_contract(
    dense_workspace: Path,
    frame_names: list[str],
    pycolmap_module: Any | None = None,
) -> dict[str, Any]:
    """Load dense depth, poses and intrinsics in the R-01 convention."""
    pycolmap = pycolmap_module or load_pycolmap()
    dense_workspace = Path(dense_workspace)

    reconstruction = pycolmap.Reconstruction(dense_workspace / "sparse")

    images_by_name = {image.name: image for _, image in reconstruction.images.items()}

    missing = [name for name in frame_names if name not in images_by_name]

    if missing:
        raise RuntimeError(
            "COLMAP dense model is missing registered input frames: "
            + ", ".join(missing)
        )

    depth_dir = dense_workspace / "stereo" / "depth_maps"

    depth_maps = []
    extrinsics = []
    intrinsics = []
    camera_sizes_hw = []

    expected_depth_hw = None

    for frame_name in frame_names:
        image = images_by_name[frame_name]

        if not bool(image.has_pose):
            raise RuntimeError(f"COLMAP image {frame_name!r} has no camera pose")

        camera = reconstruction.cameras[image.camera_id]

        extrinsic = np.asarray(
            image.cam_from_world().matrix(),
            dtype=np.float32,
        )

        intrinsic = np.asarray(
            camera.calibration_matrix(),
            dtype=np.float32,
        )

        if extrinsic.shape != (3, 4):
            raise RuntimeError(
                f"COLMAP pose for {frame_name!r} must be (3,4), got {extrinsic.shape}"
            )

        if intrinsic.shape != (3, 3):
            raise RuntimeError(
                f"COLMAP intrinsic for {frame_name!r} must be (3,3), "
                f"got {intrinsic.shape}"
            )

        depth_path = depth_dir / f"{frame_name}.geometric.bin"

        if not depth_path.is_file():
            raise RuntimeError(
                f"COLMAP did not produce a geometric depth map for {frame_name!r}"
            )

        depth_map = pycolmap.DepthMap()
        depth_map.read(depth_path)

        depth = np.asarray(
            depth_map.to_array(),
            dtype=np.float32,
        )

        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]

        if depth.ndim != 2:
            raise RuntimeError(
                f"COLMAP depth for {frame_name!r} must be (H,W), got {depth.shape}"
            )

        depth_hw = (
            int(depth.shape[0]),
            int(depth.shape[1]),
        )

        if expected_depth_hw is None:
            expected_depth_hw = depth_hw
        elif depth_hw != expected_depth_hw:
            raise RuntimeError(
                "All COLMAP depth maps must share one size; "
                f"expected {expected_depth_hw}, "
                f"got {depth_hw} for {frame_name!r}"
            )

        depth_maps.append(depth)
        extrinsics.append(extrinsic)
        intrinsics.append(intrinsic)

        camera_sizes_hw.append([int(camera.height), int(camera.width)])

    if not depth_maps:
        raise RuntimeError("COLMAP produced no dense depth maps")

    depth = np.stack(depth_maps, axis=0)

    return {
        "extrinsic": np.stack(
            extrinsics,
            axis=0,
        ),
        "intrinsic": np.stack(
            intrinsics,
            axis=0,
        ),
        "depth": depth,
        "depth_conf": depth_validity_mask(depth),
        "camera_sizes_hw": camera_sizes_hw,
    }


def point_cloud_from_reconstruction(
    reconstruction: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert fused COLMAP points to ProofShape xyz + RGB arrays."""
    point_count = len(reconstruction.points3D)

    if point_count == 0:
        raise RuntimeError("COLMAP stereo fusion produced no 3D points")

    points = np.empty(
        (point_count, 3),
        dtype=np.float32,
    )

    colors = np.empty(
        (point_count, 3),
        dtype=np.uint8,
    )

    kept = 0

    for point in reconstruction.points3D.values():
        xyz = np.asarray(
            point.xyz,
            dtype=np.float32,
        )

        color = np.asarray(
            point.color,
            dtype=np.uint8,
        )

        if xyz.shape != (3,) or color.shape != (3,):
            raise RuntimeError("COLMAP fused Point3D has an unexpected shape")

        if not np.isfinite(xyz).all():
            continue

        points[kept] = xyz
        colors[kept] = color
        kept += 1

    if kept == 0:
        raise RuntimeError("COLMAP stereo fusion produced no finite 3D points")

    return points[:kept], colors[:kept]


def run_colmap_model(
    capture_dir: Path,
    board_pose_path: Path,
    device_type: str = "cuda",
    max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
) -> dict[str, Any]:
    """Run sparse + dense COLMAP and normalize to the R-01 contract."""
    if device_type != "cuda":
        raise ValueError("R-12 has only been validated on CUDA; use --device cuda.")

    if max_image_size <= 0:
        raise ValueError("max_image_size must be positive")

    capture_dir = Path(capture_dir)

    frame_paths = find_capture_frames(capture_dir)
    frame_names = [path.name for path in frame_paths]

    pycolmap = load_pycolmap()

    with tempfile.TemporaryDirectory(prefix="proofshape-colmap-") as temp_dir:
        workspace = Path(temp_dir) / "workspace"

        sparse = run_sparse_registration(
            capture_dir=capture_dir,
            board_pose_path=board_pose_path,
            workspace=workspace,
            device_type=device_type,
        )

        timings = dict(sparse["timings"])

        sparse_total = timings.pop(
            "total_s",
            None,
        )

        if sparse_total is not None:
            timings["sparse_total_s"] = float(sparse_total)

        common = {
            "frame_names": frame_names,
            "frame_count": len(frame_names),
            "registered_frame_count": int(sparse["registered_frame_count"]),
            "sparse_point_count": int(sparse["sparse_point_count"]),
            "board_prior_count": int(sparse["board_prior_count"]),
            "frames_without_board_prior": list(sparse["frames_without_board_prior"]),
            "timings": timings,
            "pycolmap_version": str(pycolmap.__version__),
        }

        if sparse["status"] != "success":
            return {
                **common,
                "status": "refused",
                "reason": sparse["reason"],
            }

        # The common R-01 output contract requires depth and pose
        # rows for every input frame. Never fabricate missing ones.
        if sparse["registered_frame_count"] != len(frame_names):
            return {
                **common,
                "status": "refused",
                "reason": ("colmap_did_not_register_all_input_frames"),
            }

        dense_workspace = workspace / "dense"

        sparse_model = workspace / "sparse" / str(sparse["model_id"])

        undistort_options = pycolmap.UndistortCameraOptions()

        undistort_options.max_image_size = max_image_size

        started = time.perf_counter()

        pycolmap.undistort_images(
            output_path=dense_workspace,
            input_path=sparse_model,
            image_path=capture_dir,
            undistort_options=undistort_options,
        )

        timings["undistort_s"] = time.perf_counter() - started

        patch_options = pycolmap.PatchMatchOptions()

        patch_options.max_image_size = max_image_size

        patch_options.gpu_index = "0"

        started = time.perf_counter()

        pycolmap.patch_match_stereo(
            workspace_path=dense_workspace,
            options=patch_options,
        )

        timings["patch_match_s"] = time.perf_counter() - started

        fusion_options = pycolmap.StereoFusionOptions()

        fusion_options.max_image_size = max_image_size

        started = time.perf_counter()

        fused_reconstruction = pycolmap.stereo_fusion(
            output_path=(dense_workspace / "fused.ply"),
            workspace_path=dense_workspace,
            input_type="geometric",
            options=fusion_options,
            output_type="ply",
        )

        timings["fusion_s"] = time.perf_counter() - started

        started = time.perf_counter()

        dense = load_dense_contract(
            dense_workspace,
            frame_names,
            pycolmap_module=pycolmap,
        )

        points, point_colors = point_cloud_from_reconstruction(fused_reconstruction)

        timings["extract_dense_s"] = time.perf_counter() - started

    return {
        **common,
        **dense,
        "status": "success",
        "reason": None,
        "model_id": int(sparse["model_id"]),
        "points": points,
        "point_colors": point_colors,
        "max_image_size": int(max_image_size),
    }


def _write_refusal_run(
    out_dir: Path,
    run_info: dict[str, Any],
) -> None:
    """Persist an explicit refusal without fabricating geometry."""
    out_dir = Path(out_dir)

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already has files in it. "
            "Pick a new --out-dir or delete it deliberately."
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out_dir / "run.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            run_info,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def run_on_capture(
    capture_dir: Path,
    out_dir: Path,
    board_pose_path: Path | None = None,
    device_type: str = "cuda",
    min_confidence: float = 0.0,
    max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
) -> dict[str, Any]:
    """Run R-12 and write R-01-compatible outputs on success."""
    if min_confidence != 0.0:
        raise ValueError(
            "COLMAP does not expose VGGT-style learned depth "
            "confidence. Leave --min-confidence at 0.0; "
            "R-12 stores a binary depth-validity mask instead."
        )

    total_started = time.perf_counter()

    capture_dir = Path(capture_dir)
    out_dir = Path(out_dir)

    if board_pose_path is None:
        board_pose_path = default_board_pose_path(capture_dir)
    else:
        board_pose_path = Path(board_pose_path)

    result = run_colmap_model(
        capture_dir=capture_dir,
        board_pose_path=board_pose_path,
        device_type=device_type,
        max_image_size=max_image_size,
    )

    timings = result["timings"]

    run_info = {
        "story": "R-12",
        "backend": "colmap",
        "status": result["status"],
        "reason": result["reason"],
        "capture_dir": str(capture_dir),
        "board_pose_path": str(board_pose_path),
        "frame_count": int(result["frame_count"]),
        "frame_names": list(result["frame_names"]),
        "registered_frame_count": int(result["registered_frame_count"]),
        "sparse_point_count": int(result["sparse_point_count"]),
        "board_prior_count": int(result["board_prior_count"]),
        "frames_without_board_prior": list(result["frames_without_board_prior"]),
        "device": device_type,
        "dtype": "float32",
        "pycolmap_version": (result["pycolmap_version"]),
        "depth_conf_semantics": (
            "binary validity mask: 1 for finite positive "
            "COLMAP geometric depth, 0 otherwise; "
            "not learned or probabilistic confidence"
        ),
        "scale": (
            "board-position-prior constrained reconstruction; "
            "not accepted as metric inspection evidence by "
            "itself; R-04 remains authoritative"
        ),
        "timings": timings,
    }

    if result["status"] == "refused":
        run_info["point_count"] = 0

        timings["total_s"] = time.perf_counter() - total_started

        _write_refusal_run(
            out_dir,
            run_info,
        )

        return run_info

    depth = result["depth"]

    _, height, width = depth.shape

    run_info.update(
        {
            "model_id": int(result["model_id"]),
            "processed_image_size_hw": [
                int(height),
                int(width),
            ],
            "colmap_camera_sizes_hw": (result["camera_sizes_hw"]),
            "pixel_frame": ("COLMAP undistorted image pixel frame"),
            "pose_convention": ("camera-from-world 3x4 from COLMAP"),
            "max_image_size": int(result["max_image_size"]),
            "point_cloud_source": (
                "COLMAP geometric stereo fusion, rewritten "
                "to ProofShape binary little-endian xyz+rgb PLY"
            ),
            "point_count": int(result["points"].shape[0]),
            "min_confidence": 0.0,
        }
    )

    started = time.perf_counter()

    write_run_outputs(
        out_dir,
        result["frame_names"],
        result["extrinsic"],
        result["intrinsic"],
        result["depth"],
        result["depth_conf"],
        result["points"],
        result["point_colors"],
        run_info,
    )

    timings["write_s"] = time.perf_counter() - started

    timings["total_s"] = time.perf_counter() - total_started

    # Refresh run.json now that final write and wall-clock
    # timings are known.
    with open(
        out_dir / "run.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            run_info,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    return run_info

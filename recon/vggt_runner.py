"""R-01: run VGGT on a golden-capture folder and write poses, depth and a point cloud to disk.

The module is split in two on purpose. Everything that does not need PyTorch or a GPU (finding
frames, choosing a dtype, unprojecting depth, writing the output files) is plain NumPy and is
unit-tested on any machine, including CI. Only ``run_vggt_model`` touches PyTorch and the VGGT
package; it imports them lazily so importing this module never requires a GPU environment.

Output layout (one directory per run) is the shape R-11 (MASt3R) and R-12 (COLMAP) must also
emit, so R-04 can consume any backend the config flag selects (D-009):

    poses.npz   frame_names (S,), extrinsic (S,3,4) camera-from-world, intrinsic (S,3,3)
    depth.npz   depth (S,H,W), depth_conf (S,H,W)
    points.ply  binary little-endian point cloud with per-point RGB, in the world frame
    run.json    timings, device, dtype, versions and the frame list

Poses are in VGGT's own arbitrary, unit-less frame. They are NOT metric — R-04 (metric
alignment against the ChArUco board) is what fixes scale, so nothing here may be reported as a
measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

FRAME_SUFFIXES = (".jpg", ".jpeg")
DEFAULT_MODEL_ID = "facebook/VGGT-1B"

# Compute capability 8.0 is Ampere, the first NVIDIA generation with fast bf16. The Lightning
# Studio's Tesla T4 is 7.5, where bf16 is emulated and slow, so it must use fp16 (D-011 records
# the same lesson for Metal, where fp16 also wins).
_MIN_BF16_COMPUTE_CAPABILITY = (8, 0)


def find_capture_frames(capture_dir: Path) -> list[Path]:
    """Return the capture's JPEG frames in a stable, sorted order.

    Sorting matters for reproducibility: VGGT treats the first frame as the reference frame, so a
    filesystem-dependent order would give a different world frame from run to run.
    """
    capture_dir = Path(capture_dir)
    if not capture_dir.is_dir():
        raise FileNotFoundError(
            f"Capture folder not found: {capture_dir}. "
            "Run `bash fixtures/download_golden_capture.sh` first."
        )

    frames: list[Path] = []
    for path in sorted(capture_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES:
            frames.append(path)

    if not frames:
        raise FileNotFoundError(f"No .jpg/.jpeg frames found in {capture_dir}.")
    return frames


def choose_dtype_name(
    device_type: str, compute_capability: tuple[int, int] | None
) -> str:
    """Pick ``"bfloat16"`` or ``"float16"`` for the device.

    Reconstruction code often auto-selects bf16 on anything that looks like a modern GPU; that
    default is wrong on a T4 and on Apple Silicon, so we choose explicitly and record the choice.
    """
    if device_type == "cuda":
        if compute_capability is None:
            raise ValueError(
                "A CUDA device needs a compute capability to choose a dtype."
            )
        if compute_capability >= _MIN_BF16_COMPUTE_CAPABILITY:
            return "bfloat16"
        return "float16"
    if device_type == "mps":
        return "float16"
    raise ValueError(
        f"Unsupported device type {device_type!r}: R-01 targets cuda (and mps)."
    )


def unproject_depth_to_world(
    depth: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    """Lift every depth pixel to a 3D point in the world frame.

    ``depth`` is (S,H,W); ``extrinsic`` is (S,3,4) camera-from-world in the OpenCV convention
    (x_cam = R @ x_world + t); ``intrinsic`` is (S,3,3). Returns (S,H,W,3).

    We unproject the depth maps ourselves instead of using VGGT's separate point-map head because
    depth + camera is the representation R-04/R-07 will fuse, so the cloud we save is exactly what
    those stories would reconstruct from the saved depth and poses.
    """
    frame_count, height, width = depth.shape
    if extrinsic.shape != (frame_count, 3, 4):
        raise ValueError(
            f"extrinsic must be ({frame_count},3,4), got {extrinsic.shape}"
        )
    if intrinsic.shape != (frame_count, 3, 3):
        raise ValueError(
            f"intrinsic must be ({frame_count},3,3), got {intrinsic.shape}"
        )

    # Pixel centres in homogeneous coordinates, shared by every frame.
    u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack([u_grid, v_grid, np.ones_like(u_grid)], axis=-1).astype(
        np.float64
    )
    pixels = pixels.reshape(-1, 3)

    world_points = np.empty((frame_count, height, width, 3), dtype=np.float32)
    for frame_index in range(frame_count):
        rotation = extrinsic[frame_index, :, :3].astype(np.float64)
        translation = extrinsic[frame_index, :, 3].astype(np.float64)
        inverse_intrinsic = np.linalg.inv(intrinsic[frame_index].astype(np.float64))

        rays = pixels @ inverse_intrinsic.T
        camera_points = rays * depth[frame_index].reshape(-1, 1).astype(np.float64)
        # Invert x_cam = R x_world + t. For a rotation, R^-1 = R^T, and with row vectors
        # (x_cam - t) @ R is the same as R^T @ (x_cam - t).
        points = (camera_points - translation) @ rotation
        world_points[frame_index] = points.reshape(height, width, 3)

    return world_points


def select_point_cloud(
    world_points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten per-pixel points to a cloud, dropping non-finite and low-confidence points.

    ``min_confidence`` is an absolute threshold on VGGT's depth confidence. The default the CLI
    passes is 0.0 (keep everything finite) because no threshold has been measured yet; picking
    one is R-04+ work once there is ground truth to judge it against.
    """
    if (
        world_points.shape[:3] != confidence.shape
        or world_points.shape[:3] != colors.shape[:3]
    ):
        raise ValueError(
            "world_points, colors and confidence must share (S,H,W) dimensions"
        )

    flat_points = world_points.reshape(-1, 3)
    flat_colors = colors.reshape(-1, 3)
    flat_confidence = confidence.reshape(-1)

    keep = np.isfinite(flat_points).all(axis=1)
    keep &= np.isfinite(flat_confidence)
    keep &= flat_confidence >= min_confidence
    return flat_points[keep], flat_colors[keep]


def write_ply(path: Path, points: np.ndarray, colors_uint8: np.ndarray) -> None:
    """Write a binary little-endian PLY with float32 xyz and uint8 rgb."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape}")
    if colors_uint8.shape != points.shape or colors_uint8.dtype != np.uint8:
        raise ValueError("colors must be a uint8 array with the same shape as points")

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(points.shape[0], dtype=vertex_dtype)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors_uint8[:, 0]
    vertices["green"] = colors_uint8[:, 1]
    vertices["blue"] = colors_uint8[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices.tobytes())


def write_run_outputs(
    out_dir: Path,
    frame_names: list[str],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    depth: np.ndarray,
    depth_confidence: np.ndarray,
    points: np.ndarray,
    colors_uint8: np.ndarray,
    run_info: dict[str, Any],
) -> None:
    """Write the four run files. Refuses to overwrite, so two runs can never be mixed."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already has files in it. Pick a new --out-dir or delete it deliberately; "
            "overwriting silently would let outputs from two runs get mixed."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "poses.npz",
        frame_names=np.array(frame_names),
        extrinsic=extrinsic.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
    )
    np.savez_compressed(
        out_dir / "depth.npz",
        depth=depth.astype(np.float32),
        depth_conf=depth_confidence.astype(np.float32),
    )
    write_ply(out_dir / "points.ply", points, colors_uint8)

    with open(out_dir / "run.json", "w", encoding="utf-8") as handle:
        json.dump(run_info, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_vggt_model(
    frame_paths: list[Path], model_id: str, device_type: str
) -> dict[str, Any]:
    """Run VGGT on the frames and return NumPy arrays plus per-stage wall-clock timings.

    PyTorch and VGGT are imported here, not at module level, so the rest of this file stays
    importable (and testable) without them. Install them with ``requirements-gpu.txt``.
    """
    try:
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing dependency {exc.name!r}. Install the GPU stack with "
            "`pip install -r requirements-gpu.txt` (see recon/README.md)."
        ) from exc

    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but PyTorch cannot see a GPU.")

    compute_capability = None
    device_name = device_type
    if device_type == "cuda":
        compute_capability = tuple(torch.cuda.get_device_capability(0))
        device_name = torch.cuda.get_device_name(0)
    dtype_name = choose_dtype_name(device_type, compute_capability)
    dtype = getattr(torch, dtype_name)

    def synchronise() -> None:
        # CUDA kernels run asynchronously; without this the clock would stop before the GPU
        # finished and the recorded time would be too small.
        if device_type == "cuda":
            torch.cuda.synchronize()

    timings: dict[str, float] = {}

    started = time.perf_counter()
    model = VGGT.from_pretrained(model_id).to(device_type).eval()
    synchronise()
    timings["model_load_s"] = time.perf_counter() - started

    started = time.perf_counter()
    images = load_and_preprocess_images([str(path) for path in frame_paths]).to(
        device_type
    )
    synchronise()
    timings["preprocess_s"] = time.perf_counter() - started

    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=dtype):
        predictions = model(images)
    synchronise()
    timings["inference_s"] = time.perf_counter() - started

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )

    def to_numpy(tensor: Any) -> np.ndarray:
        # Drop the batch dimension (we always run one sequence) and leave autocast precision.
        return tensor.squeeze(0).float().cpu().numpy()

    depth = to_numpy(predictions["depth"])
    if depth.ndim == 4:
        depth = depth[..., 0]

    return {
        "extrinsic": to_numpy(extrinsic),
        "intrinsic": to_numpy(intrinsic),
        "depth": depth,
        "depth_conf": to_numpy(predictions["depth_conf"]),
        "images": to_numpy(images).transpose(0, 2, 3, 1),
        "timings": timings,
        "device": device_name,
        "dtype": dtype_name,
        "torch_version": str(torch.__version__),
    }


def run_on_capture(
    capture_dir: Path,
    out_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    device_type: str = "cuda",
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    """Run the whole R-01 pipeline on one capture folder and return the run.json contents."""
    total_started = time.perf_counter()
    frame_paths = find_capture_frames(capture_dir)

    result = run_vggt_model(frame_paths, model_id, device_type)

    started = time.perf_counter()
    world_points = unproject_depth_to_world(
        result["depth"], result["extrinsic"], result["intrinsic"]
    )
    colors_uint8 = np.clip(result["images"] * 255.0, 0, 255).astype(np.uint8)
    points, point_colors = select_point_cloud(
        world_points, colors_uint8, result["depth_conf"], min_confidence
    )
    result["timings"]["unproject_s"] = time.perf_counter() - started

    frame_names = [path.name for path in frame_paths]
    _, height, width = result["depth"].shape
    run_info: dict[str, Any] = {
        "story": "R-01",
        "backend": "vggt",
        "model_id": model_id,
        "capture_dir": str(capture_dir),
        "frame_count": len(frame_names),
        "frame_names": frame_names,
        "processed_image_size_hw": [int(height), int(width)],
        "device": result["device"],
        "dtype": result["dtype"],
        "torch_version": result["torch_version"],
        "min_confidence": min_confidence,
        "point_count": int(points.shape[0]),
        "scale": "arbitrary (VGGT frame, not metric; R-04 fixes scale)",
        "timings": result["timings"],
    }
    started = time.perf_counter()
    write_run_outputs(
        out_dir,
        frame_names,
        result["extrinsic"],
        result["intrinsic"],
        result["depth"],
        result["depth_conf"],
        points,
        point_colors,
        run_info,
    )
    run_info["timings"]["write_s"] = time.perf_counter() - started
    run_info["timings"]["total_s"] = time.perf_counter() - total_started

    # Rewrite run.json so the final two timings (which could not be known until now) are on disk.
    with open(Path(out_dir) / "run.json", "w", encoding="utf-8") as handle:
        json.dump(run_info, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return run_info


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VGGT on a golden-capture folder (R-01)."
    )
    parser.add_argument(
        "capture_dir", type=Path, help="folder of JPEG frames, e.g. .../s-04"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output folder (default: fixtures/data/recon_runs/<capture folder name>)",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", choices=("cuda", "mps"), default="cuda")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="drop cloud points whose depth confidence is below this (default keeps all)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    out_dir = args.out_dir or Path("fixtures/data/recon_runs") / args.capture_dir.name
    try:
        run_info = run_on_capture(
            args.capture_dir, out_dir, args.model_id, args.device, args.min_confidence
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"R-01 RUN: FAIL - {exc}", file=sys.stderr)
        return 1

    print(
        f"R-01 RUN: PASS - {run_info['frame_count']} frames, {run_info['point_count']} points"
    )
    for stage, seconds in run_info["timings"].items():
        print(f"  {stage}: {seconds:.2f}")
    print(f"Outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

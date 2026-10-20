"""R-01: run VGGT on a golden-capture folder and write poses, depth and a point cloud to disk.

The module is split in two on purpose. Everything that does not need PyTorch or a GPU (finding
frames, choosing a dtype, unprojecting depth, writing the output files) is plain NumPy and is
unit-tested on any machine, including CI. Only ``run_vggt_model`` touches PyTorch and the VGGT
package; it imports them lazily so importing this module never requires a GPU environment.

Output layout (one directory per run) is the shape R-11 (MASt3R) and R-12 (COLMAP) must also
emit, so R-04 can consume any backend the config flag selects (D-009):

    poses.npz   frame_names (S,), extrinsic (S,3,4) camera-from-world, intrinsic (S,3,3) in
                processed-image pixels (see run.json processed_image_size_hw)
    depth.npz   depth (S,H,W), depth_conf (S,H,W)
    points.ply  binary little-endian point cloud with per-point RGB, in the world frame
    run.json    timings, device, dtype, versions and the frame list

Poses are in VGGT's own arbitrary, unit-less frame. They are NOT metric — R-04 (metric
alignment against the ChArUco board) is what fixes scale, so nothing here may be reported as a
measurement.

The intrinsics are NOT in full-resolution photo pixels. VGGT sees each frame resized (e.g.
5712x4284 -> 518x392), so its focal length is roughly 400 px where R-02's board-derived K is
roughly 4,300 px. The two axes are also scaled slightly differently (518/5712 in x, 392/4284 in
y), because VGGT rounds the height to a multiple of 14. Anything comparing this K with R-02/R-03's
must rescale each axis separately using processed_image_size_hw, or it gets a silent unit bug.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from recon.frames import find_capture_frames
from recon.timing import format_timing_table

DEFAULT_MODEL_ID = "facebook/VGGT-1B"

# D-045: the golden-set sweep (recon/confidence_sweep.py) found that almost all of the achievable
# stray-point reduction is captured the instant the threshold clears VGGT's exact confidence
# floor of 1.0 -- the median pixel sits exactly at that floor on every golden object. Pushing the
# threshold any higher past that buys little further cleanup while steadily costing part
# coverage, so 1.0001 (strictly above the floor, not a larger round number) is the chosen
# default. See D-045 for the full evidence and stories/R/R-20.md's notes for the per-object table.
DEFAULT_MIN_CONFIDENCE = 1.0001

# Compute capability 8.0 is Ampere, the first NVIDIA generation with fast bf16. The Lightning
# Studio's Tesla T4 is 7.5, where bf16 is emulated and slow, so it must use fp16 (D-011 records
# the same lesson for Metal, where fp16 also wins).
_MIN_BF16_COMPUTE_CAPABILITY = (8, 0)

# find_capture_frames now lives in recon/frames.py (issue #43); imported here so
# `vggt_runner.find_capture_frames` keeps working for existing callers and tests.


VGGT_TARGET_WIDTH_PX = 518
VGGT_MULTIPLE_PX = 14


def describe_vggt_crop_preprocess(frame_paths: list[Path]) -> list[dict[str, Any]]:
    """Describe VGGT pinned default crop preprocessing in full-resolution pixel terms.

    R-23 needs the exact map from R-02 stored-image pixels to VGGT processed pixels. The
    pinned loader first resizes width to 518, rounds height to a multiple of 14, then
    centre-crops height to at most 518. We record that geometry with the run so downstream
    code never has to infer it from a single final HxW pair.

    EXIF orientation is deliberately ignored here, matching R-01/R-02 stored-pixel convention.
    Mixed-shape batches that VGGT later pads are not representable by this simple map; R-23
    detects that by comparing each recorded final size with the actual depth image size and
    refuses rather than guessing.
    """
    descriptions: list[dict[str, Any]] = []
    for path in frame_paths:
        with Image.open(path) as image:
            original_width, original_height = image.size
        if original_width <= 0 or original_height <= 0:
            raise ValueError(
                f"Invalid image dimensions for {path}: {original_width}x{original_height}"
            )

        resized_width = VGGT_TARGET_WIDTH_PX
        resized_height = int(
            round(original_height * (resized_width / original_width) / VGGT_MULTIPLE_PX)
            * VGGT_MULTIPLE_PX
        )
        resized_height = max(VGGT_MULTIPLE_PX, resized_height)

        crop_top = max(0, (resized_height - VGGT_TARGET_WIDTH_PX) // 2)
        crop_left = 0
        final_height = min(resized_height, VGGT_TARGET_WIDTH_PX)
        final_width = resized_width

        descriptions.append(
            {
                "frame_name": path.name,
                "original_size_hw": [int(original_height), int(original_width)],
                "resized_size_hw": [int(resized_height), int(resized_width)],
                "crop_top_px": int(crop_top),
                "crop_left_px": int(crop_left),
                "processed_size_hw": [int(final_height), int(final_width)],
                "scale_x": float(resized_width / original_width),
                "scale_y": float(resized_height / original_height),
            }
        )
    return descriptions

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


def frames_chw_to_hwc(images: np.ndarray) -> np.ndarray:
    """Turn VGGT's preprocessed frames, (S,3,H,W), into (S,H,W,3) for colouring the cloud.

    This is deliberately separate from the batch-squeezing used for the model outputs. The
    predictions carry a leading batch axis of 1, but the preprocessed frames do not, so the same
    squeeze(0) would silently drop the frame axis of a one-frame capture and the transpose would
    then fail.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"images must be (S,3,H,W), got {images.shape}")
    return images.transpose(0, 2, 3, 1)


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


def confidence_keep_mask(confidence: np.ndarray, min_confidence: float) -> np.ndarray:
    """Return a boolean mask, same shape as ``confidence``: True where a pixel is trusted.

    A pixel is kept when its confidence is finite and at least ``min_confidence``. This is the
    one predicate the whole pipeline uses to decide "do we trust this pixel's depth" --
    ``select_point_cloud`` below applies it when building R-01's point cloud, R-07's
    ``recon.tsdf_fusion.fuse_run`` applies it again, on the same ``depth_conf`` array, when
    deciding which pixels may update the TSDF, and R-09's ``recon.provenance.provenance_run``
    applies it a third time when deciding which pixels may count as a frame observing a vertex
    (R-24). Sharing one function means a pixel dropped from the cloud is dropped from the fused
    surface and from the observation count too, rather than three call sites independently
    reimplementing "finite and above threshold" and risking them drifting apart (the same
    reasoning as the PLY read/write pair sharing one fixed layout, above).
    """
    confidence = np.asarray(confidence)
    return np.isfinite(confidence) & (confidence >= min_confidence)


def resolve_min_confidence(
    reconstruction_run_dir: Path, min_confidence: float | None
) -> tuple[float, str]:
    """Decide which confidence threshold a downstream stage applies to a run's ``depth_conf``.

    Returns ``(threshold, source)``. ``min_confidence=None`` reads the threshold the run itself
    was built with -- ``run.json``'s own ``min_confidence``, which every backend writes -- and
    reports the source as ``"run.json"``. That is the default for both R-07's fusion and R-09's
    provenance, so each agrees with ``points.ply`` (and with each other) by construction, not by
    two callers happening to pass the same number. An explicit value overrides it, reported as
    ``"override"`` (e.g. to apply D-045's threshold to a run recorded before it existed).

    Shared rather than inlined in each caller for the same reason ``confidence_keep_mask`` is:
    R-20 wrote this logic inside ``fuse_run``, and R-24 needed the identical rule in
    ``provenance_run`` -- two copies of "where does the threshold come from" is exactly how
    fusion and provenance could quietly start trusting different pixels.

    A missing ``run.json``, or one that predates the field, is a genuine ambiguity, not a reason
    to assume 0.0: silently trusting every pixel would let a stage accept depth the cloud itself
    excluded. Both are explicit errors, and the caller must pass ``min_confidence`` to proceed.
    """
    if min_confidence is not None:
        return float(min_confidence), "override"

    run_json_path = Path(reconstruction_run_dir) / "run.json"
    if not run_json_path.exists():
        raise FileNotFoundError(
            f"{run_json_path} does not exist, so there is no record of which confidence "
            "threshold points.ply was built with. Pass min_confidence explicitly if this run "
            "predates that field."
        )
    run_json = json.loads(run_json_path.read_text(encoding="utf-8"))
    if "min_confidence" not in run_json:
        raise ValueError(
            f"{run_json_path} has no 'min_confidence' field (an older run, before R-20). "
            "Pass min_confidence explicitly rather than silently assuming 0.0."
        )
    return float(run_json["min_confidence"]), "run.json"


def select_point_cloud(
    world_points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten per-pixel points to a cloud, dropping non-finite and low-confidence points.

    ``min_confidence`` is an absolute threshold on VGGT's depth confidence -- see
    ``DEFAULT_MIN_CONFIDENCE``'s docstring for how that default was chosen (D-045).
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
    keep &= confidence_keep_mask(flat_confidence, min_confidence)
    return flat_points[keep], flat_colors[keep]


# The only PLY layout this codebase writes or reads: float32 xyz, uint8 rgb, binary
# little-endian. Shared by write_ply and read_ply so the two can never drift apart.
PLY_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)

PLY_HEADER_TEMPLATE = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "element vertex {vertex_count}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "property uchar red\n"
    "property uchar green\n"
    "property uchar blue\n"
    "end_header\n"
)


def write_ply(path: Path, points: np.ndarray, colors_uint8: np.ndarray) -> None:
    """Write a binary little-endian PLY with float32 xyz and uint8 rgb."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape}")
    if colors_uint8.shape != points.shape or colors_uint8.dtype != np.uint8:
        raise ValueError("colors must be a uint8 array with the same shape as points")

    vertices = np.empty(points.shape[0], dtype=PLY_VERTEX_DTYPE)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors_uint8[:, 0]
    vertices["green"] = colors_uint8[:, 1]
    vertices["blue"] = colors_uint8[:, 2]

    header = PLY_HEADER_TEMPLATE.format(vertex_count=points.shape[0])
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices.tobytes())


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a PLY written by write_ply back into (points (N,3) float32, colors (N,3) uint8).

    Only understands write_ply's own fixed layout (binary little-endian, float32 xyz, uint8
    rgb) -- this is a reader for that one format, not a general PLY parser -- and raises rather
    than guessing if a file doesn't match it.
    """
    path = Path(path)
    raw = path.read_bytes()
    header, separator, payload = raw.partition(b"end_header\n")
    if not separator:
        raise ValueError(
            f"{path} has no 'end_header' line; not a PLY this reader understands"
        )

    header_text = header.decode("ascii", errors="replace")
    if not header_text.startswith("ply\n"):
        raise ValueError(f"{path} does not start with a 'ply' magic line")
    if "format binary_little_endian 1.0" not in header_text:
        raise ValueError(
            f"{path} is not binary_little_endian 1.0, which is all this reader supports"
        )

    vertex_count = None
    for line in header_text.splitlines():
        if line.startswith("element vertex "):
            vertex_count = int(line.removeprefix("element vertex "))
            break
    if vertex_count is None:
        raise ValueError(f"{path} header has no 'element vertex' line")

    expected_bytes = vertex_count * PLY_VERTEX_DTYPE.itemsize
    if len(payload) != expected_bytes:
        raise ValueError(
            f"{path} payload is {len(payload)} bytes, expected {expected_bytes} for "
            f"{vertex_count} vertices"
        )

    vertices = np.frombuffer(payload, dtype=PLY_VERTEX_DTYPE)
    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(
        np.float32
    )
    colors = np.column_stack(
        [vertices["red"], vertices["green"], vertices["blue"]]
    ).astype(np.uint8)
    return points, colors


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

    preprocess_geometry = describe_vggt_crop_preprocess(frame_paths)

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
        # Not to_numpy: `images` has no batch axis, so squeeze(0) would drop the frame axis of
        # a single-frame capture.
        "images": frames_chw_to_hwc(images.float().cpu().numpy()),
        "timings": timings,
        "device": device_name,
        "dtype": dtype_name,
        "torch_version": str(torch.__version__),
        "preprocess_geometry": preprocess_geometry,
    }


def run_on_capture(
    capture_dir: Path,
    out_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    device_type: str = "cuda",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> dict[str, Any]:
    """Run the whole R-01 pipeline on one capture folder and return the run.json contents."""
    total_started = time.perf_counter()
    find_frames_started = time.perf_counter()
    frame_paths = find_capture_frames(capture_dir)
    find_frames_s = time.perf_counter() - find_frames_started

    result = run_vggt_model(frame_paths, model_id, device_type)
    # Prepended, not appended -- find_capture_frames runs before run_vggt_model's own internal
    # stages, and the table should read in the order stages actually happened.
    result["timings"] = {"find_frames_s": find_frames_s, **result["timings"]}

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
    preprocess_geometry = result.get("preprocess_geometry")
    if preprocess_geometry is not None:
        for description in preprocess_geometry:
            if description["processed_size_hw"] != [int(height), int(width)]:
                description["supported_for_metric_alignment"] = False
                description["unsupported_reason"] = (
                    "VGGT batch output size differs from this frame's resize/crop size; "
                    "the loader likely padded a mixed-shape batch."
                )
            else:
                description["supported_for_metric_alignment"] = True

    run_info: dict[str, Any] = {
        "story": "R-01",
        "backend": "vggt",
        "model_id": model_id,
        "capture_dir": str(capture_dir),
        "frame_count": len(frame_names),
        "frame_names": frame_names,
        "processed_image_size_hw": [int(height), int(width)],
        "preprocess_geometry": preprocess_geometry,
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
        default=DEFAULT_MIN_CONFIDENCE,
        help=(
            "drop cloud points whose depth confidence is below this "
            f"(default {DEFAULT_MIN_CONFIDENCE}, D-045: excludes VGGT's exact confidence "
            "floor of 1.0; pass 0.0 to keep every finite point)"
        ),
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
    print(format_timing_table(run_info["timings"]))
    print(f"Outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

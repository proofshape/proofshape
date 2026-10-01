"""R-11: run MASt3R behind the reconstruction backend flag.

The output contract deliberately matches recon.vggt_runner so downstream code does not need
to know which learned backend produced a run:

    poses.npz   frame_names, camera-from-world extrinsics, intrinsics
    depth.npz   per-frame depth and confidence
    points.ply  point cloud with RGB
    run.json    backend, timings, versions, frame list and processing metadata

MASt3R returns camera-to-world 4x4 poses. They are inverted here to R-01's
camera-from-world convention. Dense depth is z-depth in the processed camera frame, so the
saved PLY is rebuilt from the saved depth, poses and intrinsics with R-01's unprojection
helper. The four artifacts therefore describe the same geometry.

JPEG EXIF orientation is deliberately ignored, matching R-01/R-02. A portrait tag remains a
camera roll instead of rotating the pixel buffer underneath the calibration.

Nothing produced here is treated as metric inspection evidence. R-04 establishes metric scale.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from recon.timing import format_timing_table
from recon.vggt_runner import (
    find_capture_frames,
    select_point_cloud,
    unproject_depth_to_world,
    write_run_outputs,
)

DEFAULT_MODEL_ID = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
MAST3R_UPSTREAM_COMMIT = "f5209afc300cec36239a7ac992263f36847bbba0"
DEFAULT_MAST3R_ROOT = Path.home() / "third_party" / "mast3r"
DEFAULT_IMAGE_SIZE = 512
# Golden captures are ordered phone sweeps. A short sequential graph avoids an O(N^2)
# complete graph while still connecting every frame to several neighbours. The chosen graph
# is recorded in run.json and can be overridden for experiments.
DEFAULT_SCENE_GRAPH = "swin-3-noncyclic"
DEFAULT_COARSE_ITERS = 300
DEFAULT_FINE_ITERS = 300


def camera_to_world_to_extrinsic(cams2world: np.ndarray) -> np.ndarray:
    """Convert MASt3R camera-to-world poses to R-01 camera-from-world extrinsics."""
    cams2world = np.asarray(cams2world, dtype=np.float64)
    if cams2world.ndim != 3 or cams2world.shape[1:] != (4, 4):
        raise ValueError(f"cams2world must have shape (S,4,4), got {cams2world.shape}")
    if not np.isfinite(cams2world).all():
        raise ValueError("cams2world contains non-finite values")

    extrinsics: list[np.ndarray] = []
    for pose in cams2world:
        try:
            world_to_camera = np.linalg.inv(pose)
        except np.linalg.LinAlgError as exc:
            raise ValueError("MASt3R returned a singular camera pose") from exc
        extrinsics.append(world_to_camera[:3, :])
    return np.stack(extrinsics, axis=0).astype(np.float32)


def reshape_dense_outputs(
    depthmaps: Sequence[np.ndarray],
    confidences: Sequence[np.ndarray],
    images: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack flattened MASt3R depth with HxW confidence and RGB arrays."""
    if not (len(depthmaps) == len(confidences) == len(images)):
        raise ValueError(
            "depth, confidence and image sequences must have the same frame count"
        )
    if not depthmaps:
        raise ValueError("MASt3R returned no dense frames")

    depth_out: list[np.ndarray] = []
    confidence_out: list[np.ndarray] = []
    image_out: list[np.ndarray] = []
    expected_hw: tuple[int, int] | None = None

    for frame_index, (depth, confidence, image) in enumerate(
        zip(depthmaps, confidences, images, strict=True)
    ):
        depth = np.asarray(depth, dtype=np.float32)
        confidence = np.asarray(confidence, dtype=np.float32)
        image = np.asarray(image, dtype=np.float32)

        if confidence.ndim != 2:
            raise ValueError(
                f"frame {frame_index}: confidence must be (H,W), got {confidence.shape}"
            )
        height, width = confidence.shape
        if expected_hw is None:
            expected_hw = (height, width)
        elif expected_hw != (height, width):
            raise ValueError(
                "all MASt3R frames must share one processed image size; "
                f"expected {expected_hw}, got {(height, width)}"
            )

        if depth.size != height * width:
            raise ValueError(
                f"frame {frame_index}: depth has {depth.size} values but confidence is "
                f"{height}x{width}"
            )
        depth = depth.reshape(height, width)
        if image.shape != (height, width, 3):
            raise ValueError(
                f"frame {frame_index}: image must be ({height},{width},3), "
                f"got {image.shape}"
            )

        depth_out.append(depth)
        confidence_out.append(confidence)
        image_out.append(image)

    return (
        np.stack(depth_out, axis=0),
        np.stack(confidence_out, axis=0),
        np.stack(image_out, axis=0),
    )


def open_stored_rgb(path: Path) -> Any:
    """Open RGB pixels as stored, deliberately ignoring the EXIF orientation tag."""
    from PIL import Image

    with Image.open(path) as opened:
        return opened.convert("RGB")


def _resize_long_edge(image: Any, long_edge_size: int) -> Any:
    """Resize like the pinned DUSt3R loader, without applying EXIF transpose."""
    from PIL import Image

    longest = max(image.size)
    if longest > long_edge_size:
        resample = Image.Resampling.LANCZOS
    else:
        resample = Image.Resampling.BICUBIC
    new_size = tuple(round(side * long_edge_size / longest) for side in image.size)
    return image.resize(new_size, resample)


def load_images_stored_orientation(
    frame_paths: Sequence[Path],
    image_size: int = DEFAULT_IMAGE_SIZE,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Build MASt3R image dictionaries while deliberately ignoring EXIF orientation."""
    try:
        import torchvision.transforms as tvf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing dependency {exc.name!r}; run bash scripts/bootstrap_mast3r.sh."
        ) from exc

    normalise = tvf.Compose(
        [
            tvf.ToTensor(),
            tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    images: list[dict[str, Any]] = []
    for path in frame_paths:
        image = open_stored_rgb(path)
        original_width, original_height = image.size
        image = _resize_long_edge(image, image_size)
        width, height = image.size
        centre_x, centre_y = width // 2, height // 2
        half_width = ((2 * centre_x) // 16) * 8
        half_height = ((2 * centre_y) // 16) * 8
        if width == height:
            half_height = 3 * half_width // 4
        image = image.crop(
            (
                centre_x - half_width,
                centre_y - half_height,
                centre_x + half_width,
                centre_y + half_height,
            )
        )
        processed_width, processed_height = image.size
        if verbose:
            print(
                f" - adding {path.name} with stored resolution "
                f"{original_width}x{original_height} -> "
                f"{processed_width}x{processed_height}"
            )
        images.append(
            {
                "img": normalise(image)[None],
                "true_shape": np.int32([[processed_height, processed_width]]),
                "idx": len(images),
                "instance": str(len(images)),
            }
        )

    if not images:
        raise ValueError("At least one frame is required")
    return images


def _validate_mast3r_checkout(root: Path) -> str:
    root = Path(root).expanduser().resolve()
    if not (root / "mast3r").is_dir() or not (root / "dust3r").is_dir():
        raise RuntimeError(
            f"MASt3R checkout not found at {root}. "
            "Run bash scripts/bootstrap_mast3r.sh or pass --mast3r-root."
        )

    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Could not verify MASt3R checkout at {root}: {exc}"
        ) from exc

    commit = completed.stdout.strip()
    if commit != MAST3R_UPSTREAM_COMMIT:
        raise RuntimeError(
            "MASt3R checkout is not the pinned R-11 revision: "
            f"expected {MAST3R_UPSTREAM_COMMIT}, found {commit}. "
            "Run bash scripts/bootstrap_mast3r.sh."
        )
    return commit


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def run_mast3r_model(
    frame_paths: list[Path],
    model_id: str,
    device_type: str,
    mast3r_root: Path,
    scene_graph: str,
    coarse_iters: int,
    fine_iters: int,
    image_size: int,
) -> dict[str, Any]:
    """Run MASt3R sparse global alignment and normalize outputs to R-01 conventions."""
    if device_type != "cuda":
        raise ValueError("R-11 has only been validated on CUDA; use --device cuda.")
    if len(frame_paths) < 2:
        raise ValueError(
            "MASt3R global alignment requires at least two capture frames."
        )
    if image_size != DEFAULT_IMAGE_SIZE:
        raise ValueError(
            f"R-11 is pinned to MASt3R image size {DEFAULT_IMAGE_SIZE}; "
            f"got {image_size}."
        )
    if coarse_iters < 0 or fine_iters < 0:
        raise ValueError("MASt3R optimization iteration counts cannot be negative.")

    root = Path(mast3r_root).expanduser().resolve()
    upstream_commit = _validate_mast3r_checkout(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        import torch
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
        from mast3r.image_pairs import make_pairs
        from mast3r.model import AsymmetricMASt3R
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing MASt3R dependency {exc.name!r}. "
            "Run bash scripts/bootstrap_mast3r.sh."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but PyTorch cannot see a GPU.")

    def synchronise() -> None:
        torch.cuda.synchronize()

    timings: dict[str, float] = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    model = AsymmetricMASt3R.from_pretrained(model_id).to(device_type).eval()
    synchronise()
    timings["model_load_s"] = time.perf_counter() - started

    started = time.perf_counter()
    images = load_images_stored_orientation(
        frame_paths,
        image_size=image_size,
        verbose=False,
    )
    pairs = make_pairs(
        images,
        scene_graph=scene_graph,
        prefilter=None,
        symmetrize=True,
    )
    if not pairs:
        raise ValueError(
            f"Scene graph {scene_graph!r} produced no image pairs "
            f"for {len(images)} frames."
        )
    synchronise()
    timings["preprocess_and_pair_s"] = time.perf_counter() - started

    filelist = [str(path) for path in frame_paths]
    with tempfile.TemporaryDirectory(prefix="proofshape-mast3r-") as cache_dir:
        started = time.perf_counter()
        scene = sparse_global_alignment(
            filelist,
            pairs,
            cache_dir,
            model,
            lr1=0.07,
            niter1=coarse_iters,
            lr2=0.01,
            niter2=fine_iters,
            device=device_type,
            opt_depth=fine_iters > 0,
            shared_intrinsics=False,
            matching_conf_thr=0.0,
            kinematic_mode="mst",
        )
        synchronise()
        timings["alignment_s"] = time.perf_counter() - started

        started = time.perf_counter()
        cams2world = _to_numpy(scene.get_im_poses())
        intrinsics = np.stack(
            [_to_numpy(value) for value in scene.intrinsics],
            axis=0,
        )
        _, depthmaps, confidences = scene.get_dense_pts3d(clean_depth=False)
        depthmaps = [_to_numpy(value) for value in depthmaps]
        confidences = [_to_numpy(value) for value in confidences]
        scene_images = [np.asarray(value, dtype=np.float32) for value in scene.imgs]
        synchronise()
        timings["extract_dense_s"] = time.perf_counter() - started

    extrinsic = camera_to_world_to_extrinsic(cams2world)
    depth, depth_confidence, rgb_images = reshape_dense_outputs(
        depthmaps,
        confidences,
        scene_images,
    )
    if intrinsics.shape != (len(frame_paths), 3, 3):
        raise ValueError(
            f"MASt3R intrinsics must be ({len(frame_paths)},3,3), "
            f"got {intrinsics.shape}"
        )

    return {
        "extrinsic": extrinsic,
        "intrinsic": intrinsics.astype(np.float32),
        "depth": depth,
        "depth_conf": depth_confidence,
        "images": rgb_images,
        "timings": timings,
        "device": torch.cuda.get_device_name(0),
        "dtype": "float32",
        "torch_version": str(torch.__version__),
        "mast3r_upstream_commit": upstream_commit,
        "pair_count": len(pairs),
        "peak_gpu_memory_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
    }


def run_on_capture(
    capture_dir: Path,
    out_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    device_type: str = "cuda",
    min_confidence: float = 0.0,
    mast3r_root: Path = DEFAULT_MAST3R_ROOT,
    scene_graph: str = DEFAULT_SCENE_GRAPH,
    coarse_iters: int = DEFAULT_COARSE_ITERS,
    fine_iters: int = DEFAULT_FINE_ITERS,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> dict[str, Any]:
    """Run R-11 and write the same four-file output contract as R-01."""
    total_started = time.perf_counter()
    find_frames_started = time.perf_counter()
    frame_paths = find_capture_frames(capture_dir)
    find_frames_s = time.perf_counter() - find_frames_started

    result = run_mast3r_model(
        frame_paths,
        model_id,
        device_type,
        mast3r_root,
        scene_graph,
        coarse_iters,
        fine_iters,
        image_size,
    )
    # Prepended, not appended -- find_capture_frames runs before run_mast3r_model's own internal
    # stages, and the table should read in the order stages actually happened.
    result["timings"] = {"find_frames_s": find_frames_s, **result["timings"]}

    started = time.perf_counter()
    world_points = unproject_depth_to_world(
        result["depth"],
        result["extrinsic"],
        result["intrinsic"],
    )
    colors_uint8 = np.clip(result["images"] * 255.0, 0, 255).astype(np.uint8)
    points, point_colors = select_point_cloud(
        world_points,
        colors_uint8,
        result["depth_conf"],
        min_confidence,
    )
    result["timings"]["unproject_s"] = time.perf_counter() - started

    frame_names = [path.name for path in frame_paths]
    _, height, width = result["depth"].shape
    run_info: dict[str, Any] = {
        "story": "R-11",
        "backend": "mast3r",
        "model_id": model_id,
        "mast3r_upstream_commit": result["mast3r_upstream_commit"],
        "capture_dir": str(capture_dir),
        "frame_count": len(frame_names),
        "frame_names": frame_names,
        "processed_image_size_hw": [int(height), int(width)],
        "pixel_frame": (
            "stored sensor orientation; EXIF rotation ignored to match R-01/R-02"
        ),
        "pose_convention": (
            "camera-from-world 3x4 (inverted from MASt3R camera-to-world)"
        ),
        "device": result["device"],
        "dtype": result["dtype"],
        "torch_version": result["torch_version"],
        "scene_graph": scene_graph,
        "pair_count": result["pair_count"],
        "coarse_iters": coarse_iters,
        "fine_iters": fine_iters,
        "min_confidence": min_confidence,
        "point_count": int(points.shape[0]),
        "peak_gpu_memory_gib": result["peak_gpu_memory_gib"],
        "scale": "arbitrary reconstruction frame, not metric; R-04 fixes scale",
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

    # The writer cannot know write_s/total_s until it returns, so refresh run.json once.
    with open(Path(out_dir) / "run.json", "w", encoding="utf-8") as handle:
        json.dump(run_info, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return run_info


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run MASt3R on a golden capture using the R-01 output contract (R-11)."
        )
    )
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output folder (default: fixtures/data/recon_runs/<capture folder name>)",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--mast3r-root",
        type=Path,
        default=Path(os.environ.get("MAST3R_ROOT", DEFAULT_MAST3R_ROOT)),
    )
    parser.add_argument("--scene-graph", default=DEFAULT_SCENE_GRAPH)
    parser.add_argument("--coarse-iters", type=int, default=DEFAULT_COARSE_ITERS)
    parser.add_argument("--fine-iters", type=int, default=DEFAULT_FINE_ITERS)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help=(
            "drop points below this MASt3R confidence "
            "(default keeps every finite point)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    out_dir = args.out_dir or Path("fixtures/data/recon_runs") / args.capture_dir.name
    try:
        run_info = run_on_capture(
            args.capture_dir,
            out_dir,
            args.model_id,
            args.device,
            args.min_confidence,
            args.mast3r_root,
            args.scene_graph,
            args.coarse_iters,
            args.fine_iters,
            args.image_size,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"R-11 RUN: FAIL - {exc}", file=sys.stderr)
        return 1

    print(
        f"R-11 RUN: PASS - {run_info['frame_count']} frames, "
        f"{run_info['point_count']} points, "
        f"{run_info['pair_count']} directed pairs"
    )
    print(format_timing_table(run_info["timings"]))
    print(f"  peak_gpu_memory_gib: {run_info['peak_gpu_memory_gib']:.2f}")
    print(f"Outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

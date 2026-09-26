# recon

The reconstruction engine (backend lane) — board detection, per-frame camera pose, VGGT
reconstruction (MASt3R and COLMAP behind a config flag), TSDF fusion, provenance, and the
FastAPI endpoint that `capture/` talks to.

This directory is registered as the `recon` Python package: from the repository root,
`pip install -e .` installs it in editable mode.

## R-01 · run VGGT on a golden capture

On the shared Lightning GPU Studio (D-036):

```bash
bash scripts/bootstrap_dev_env.sh
pip install -r requirements-gpu.txt           # VGGT, pinned upstream commit
export HF_HOME=<Teamspace Drive folder>/hf    # keep model weights out of git and Studio-local disk
bash fixtures/download_golden_capture.sh
python -m recon.vggt_runner fixtures/data/golden_capture/s-04
```

Writes `poses.npz`, `depth.npz`, `points.ply` and `run.json` (per-stage timings) to
`fixtures/data/recon_runs/s-04/`. Poses are in VGGT's own arbitrary scale, not metric.

## R-02 · board detection and per-frame camera pose

Runs anywhere — laptop, Studio or CI; no GPU or PyTorch needed:

```bash
bash scripts/bootstrap_dev_env.sh
bash fixtures/download_golden_capture.sh
python -m recon.board_pose fixtures/data/golden_capture/s-04
```

Writes `board_poses.npz` (camera-from-board extrinsics in **millimetres**, NaN where a frame has
no pose; intrinsics; every detected ChArUco corner for R-03) and `board_poses.json` (per-frame
status, failure reason and reprojection error) to `fixtures/data/board_poses/s-04/`. Frames are
read in their stored sensor orientation, ignoring EXIF rotation, to match R-01's VGGT loader.
Until R-03 lands, intrinsics come from the EXIF 35 mm-equivalent focal length; that's
approximate, and the source is recorded on every frame.

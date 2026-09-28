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
`fixtures/data/recon_runs/s-04/`. Poses are in VGGT's own arbitrary scale, not metric, and the
intrinsics are in VGGT's resized-image pixels (`processed_image_size_hw` in `run.json`), not the
full-resolution photo pixels R-02 uses.

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

## R-03 · camera intrinsics from the board

R-03 consumes R-02's saved ChArUco corner observations; it does not detect the board again:

~~~bash
python -m recon.camera_intrinsics fixtures/data/board_poses/s-04
~~~

By default it writes `intrinsics.npz` and `intrinsics.json` under
`fixtures/data/camera_intrinsics/s-04/`. The primary path calibrates one session-level camera
matrix and distortion vector from the board views and reports overall plus per-view reprojection
RMS in pixels.

If board calibration cannot be solved, the command explicitly falls back to R-01's VGGT
intrinsics from `fixtures/data/recon_runs/<sample>/`. VGGT's K is in resized-image pixels, so
R-03 rescales x and y independently to R-02's original stored-sensor pixel frame before taking
the session median. The output always records `source` and, for fallback, `fallback_reason`;
there is no silent substitution. Reprojection error is a calibration self-consistency measure,
not a metric-accuracy claim.

## R-04 · metric alignment

Align R-01's arbitrary-scale camera centres to R-02's metric board-frame camera centres:

```bash
python -m recon.metric_alignment \
  fixtures/data/recon_runs/s-04 fixtures/data/board_poses/s-04
```

The command pairs poses by frame name, solves a 3D similarity transform, and writes
`metric_alignment.json` beside the reconstruction output. The report includes the scale,
rotation, translation in millimetres, and per-frame camera-centre residuals. Residuals are not
an object-accuracy or tolerance result; known-size validation requires an independent measured
reference.

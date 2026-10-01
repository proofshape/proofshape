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


## R-11 · MASt3R behind the config flag

R-11 keeps VGGT as the D-009 default and adds MASt3R as an alternative selected by one
configuration value. Both backends take the same capture folder and write the same four files:
`poses.npz`, `depth.npz`, `points.ply` and `run.json`.

On the shared Lightning T4:

```bash
bash scripts/bootstrap_mast3r.sh
export PROOFSHAPE_RECON_BACKEND=mast3r
python -m recon.reconstruction_runner \
  fixtures/data/golden_capture/s-04 \
  --out-dir fixtures/data/recon_runs/s-04-mast3r
```

Or select it explicitly with `--backend mast3r`. With no flag or environment variable,
`recon.reconstruction_runner` still chooses VGGT.

MASt3R is pinned to upstream commit
`f5209afc300cec36239a7ac992263f36847bbba0`. The adapter intentionally reads stored JPEG
pixels without applying EXIF orientation, matching R-01/R-02. MASt3R's camera-to-world poses
are inverted into R-01's camera-from-world convention, and the PLY is rebuilt from the saved
depth, intrinsics and poses so all four artifacts describe the same geometry.

The MASt3R code is CC BY-NC-SA 4.0 and its checkpoint documentation also asks users to review
the training-data licences. Treat this as an academic backend unless those restrictions are
cleared for another use. Its output is not accepted as metric inspection evidence by itself;
R-04 still establishes board-frame scale.

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

## R-05 · scale agreement check

`recon.scale_agreement.check_scale_agreement` compares the three independent scale sources from
D-007 -- the nominal printed board pitch, an optional caliper reading, and an optional CAD
registration residual (`inspect/`, not built yet) -- and flags the session if any two disagree
by more than about 2%. All three are converted to a scale ratio relative to the board pitch
before comparing, since they start out as different kinds of quantities; see D-039 for the full
reasoning. Board pitch alone is never accepted -- at least one of the other two sources must also
be supplied.

## R-13 · per-stage latency table

Every backend's `run_on_capture` (`vggt_runner`, `mast3r_runner`, and `colmap_runner` once R-12
lands) returns a `timings` dict already embedded in `run.json`; `recon.timing.format_timing_table`
renders it as one table -- stage, seconds, share of total -- shared by every runner's own CLI
and by the unified `recon.reconstruction_runner` entry point, so none of them has to format it
separately. Frame discovery (`find_capture_frames`) is its own timed stage now, listed first.
This feeds the M2 gate in `docs/sprint-plan.md` ("the latency table decides interactive vs
batch") -- there's no numeric target anywhere for that gate, so this table is what a human reads
to judge "nearly free," not a pass/fail check this code makes itself.

Still open: COLMAP isn't on `main` yet (R-12), so the table above has only been produced for
VGGT and MASt3R so far; recording all three on the same capture needs R-12 merged and one real
run on the shared Lightning GPU.

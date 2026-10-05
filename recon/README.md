# recon

The reconstruction engine (backend lane) — board detection, per-frame camera pose, VGGT
reconstruction (MASt3R and COLMAP behind a config flag), TSDF fusion, provenance, and the
FastAPI endpoint that `capture/` talks to.

This directory is registered as the `recon` Python package: from the repository root,
`pip install -e .` installs it in editable mode.


## R-14 · reconstruction HTTP service

`recon.service` implements the frozen F-03 `/v1` operations and stores each session below
`PROOFSHAPE_DATA_DIR` (default `proofshape-data/sessions`). Frame uploads use multipart form data;
the `gyro` part is a JSON blob exactly as D-041 specifies.

Run the API locally with:

```bash
uvicorn recon.service:app --host 0.0.0.0 --port 8000
```

`finishSession` hands the saved frames to the existing R-02 → selected reconstruction backend →
R-04 → R-07 → R-08 chain in a FastAPI background task. The resulting GLB is served from the
URL returned by `getReconstruction`. Order-code validation is injectable because buyer/order
storage does not exist yet; the default service accepts any non-empty code rather than inventing
a buyer database.

The frozen `ReconstructionResult` schema has no structured failure object for `status: failed`.
R-14 therefore keeps the R-02/R-04 `ErrorDetail` code and message in the persisted session record
and returns a contract-valid failed result with the same code/message in `warnings`; changing the
wire schema would require the D-031 contract-change process. Optional metadata such as
`observed_fraction` and `scale_agreement` is omitted when the pipeline has no measured value
rather than fabricated.

## R-15 · container and deploy (runbook)

The repository-root `Dockerfile` packages the R-14 service on the CUDA PyTorch build the Lightning
Studio was validated on. It contains the VGGT backend only (D-009's default): MASt3R and COLMAP are
not installed in the image, so setting `PROOFSHAPE_RECON_BACKEND` to either fails the session.
`scripts/container_entrypoint.sh` downloads the VGGT-1B weights into `HF_HOME` at start-up, so the
weights are never baked into the public image. It then serves on port 8000 with exactly one uvicorn
worker. The reasons for one worker and one replica, plus the rest of the deploy policy, are in
D-042.

Every merge to `main` that touches the image publishes `ghcr.io/proofshape/proofshape-recon:main`
(`.github/workflows/recon-image.yml`). That starts no GPU. Starting the service is a separate
manual step.

### One-time setup (a repository admin, once)

1. **Lightning credentials.** From the Lightning account that owns the team's teamspace, get the
   user id and an API key, then:
   ```bash
   gh secret set LIGHTNING_USER_ID
   gh secret set LIGHTNING_API_KEY
   gh variable set LIGHTNING_TEAMSPACE --body "<owner>/<teamspace>"
   ```
2. **Endpoint token.** Generate it, store it as a secret, and put the same value in the team's
   password manager. GitHub never shows a secret again, so the password manager is how members
   get it for testing.
   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
   gh secret set PROOFSHAPE_ENDPOINT_TOKEN
   ```
3. **Make the image public**, once, after the first `recon-image` run on `main`. Go to GitHub →
   `proofshape` organisation → Packages → `proofshape-recon` → Package settings → Change
   visibility → Public. Lightning pulls it without registry credentials. Check that it worked
   while logged out:
   ```bash
   docker logout ghcr.io && docker buildx imagetools inspect ghcr.io/proofshape/proofshape-recon:main
   ```

### Each time you need the service

```bash
gh workflow run deploy-recon.yml --ref main -f action=start
gh run watch "$(gh run list --workflow deploy-recon.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
```

The run ends with `R-15 DEPLOY: READY at https://…`, also shown in the run's summary, along with
the image digest and commit it deployed. The first start of a fresh replica downloads the model
weights before it reports ready, so it takes several minutes. Then check it, with the token from
the password manager:

```bash
export URL=https://…                         # from the run summary
export PROOFSHAPE_ENDPOINT_TOKEN=…            # from the password manager, never typed on a command line you share

# TLS and auth: -v shows the TLS handshake; without the token this must be refused (401/403) ...
curl -v -o /dev/null -w '%{http_code}\n' -X POST "$URL/v1/sessions" \
  -H 'Content-Type: application/json' -d '{"order_code": "R15-CHECK"}'
# ... and with it, 201 Created.
curl -s -w '\n%{http_code}\n' -X POST "$URL/v1/sessions" \
  -H "Authorization: Bearer $PROOFSHAPE_ENDPOINT_TOKEN" \
  -H 'Content-Type: application/json' -d '{"order_code": "R15-CHECK"}'

# End to end: real golden frames in, GLB out, over HTTPS.
bash fixtures/download_golden_capture.sh
python scripts/smoke_r14.py fixtures/data/golden_capture/s-04 --base-url "$URL"
```

The smoke check passes only if the GLB link the service returns is also `https://`. That catches
the TLS-terminating proxy downgrading it.

**Stop it when you're done.** It is billed while it runs. It is created with a four-hour
`--max-runtime` as a backstop, but don't rely on that:

```bash
gh workflow run deploy-recon.yml --ref main -f action=stop
```

If a start fails, read the replica's logs with `lightning deployment logs proofshape-recon
--teamspace <owner>/<teamspace>` (needs `pip install lightning-sdk` and `lightning login`).

### Build and run it locally

```bash
docker build --platform linux/amd64 -t proofshape-recon:local .
docker run --rm -p 8000:8000 -e PROOFSHAPE_PREFETCH_MODEL=0 proofshape-recon:local
```

Without an NVIDIA GPU this proves only that the service answers (`POST /v1/sessions` → 201):
reconstruction itself fails with an explicit "cannot see a GPU" message. Day-to-day development
stays native, per D-012.

### Known limits

- Sessions live on the container's disk. Stopping or restarting the deployment loses them.
- One replica, one worker (D-042). Two sessions finishing at once would reconstruct concurrently
  on the one T4. That case is untested, so run one session at a time.
- VGGT only, as above.

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

## R-12 · COLMAP behind the config flag

R-12 adds COLMAP as a third reconstruction backend while keeping VGGT as the D-009 default.
It takes the same capture folder and writes the same four-file contract:
`poses.npz`, `depth.npz`, `points.ply` and `run.json`.

On the shared Lightning T4 / CUDA 12.8 environment:

```bash
python -m pip install -r requirements-colmap.txt

python -m recon.reconstruction_runner \
  fixtures/data/golden_capture/s-04 \
  --backend colmap \
  --device cuda \
  --out-dir fixtures/data/recon_runs/s-04-colmap
```

R-02 must already have produced the board-pose file for the capture. For `s-04` that is
`fixtures/data/board_poses/s-04/board_poses.npz`.

COLMAP may import phone GPS metadata during feature extraction. R-12 deliberately clears those
priors and replaces them with R-02 board-derived Cartesian camera-position priors, so GPS and
board coordinate systems are never mixed.

The dense path uses COLMAP image undistortion, PatchMatch stereo and geometric stereo fusion.
If COLMAP cannot register every input frame needed by the common ProofShape output contract,
R-12 records an explicit refusal instead of fabricating missing geometry.

COLMAP geometric depth does not provide VGGT-style learned confidence. For the shared file
contract, `depth_conf` is therefore a binary validity mask: `1.0` means finite positive
geometric depth and `0.0` means invalid or missing depth. It is not a probability, learned
confidence score, or measured uncertainty; that meaning is also recorded in `run.json`.

The board positions are reconstruction priors, not an object-accuracy measurement. COLMAP
output is not accepted as metric inspection evidence by itself; R-04 remains the authoritative
metric-alignment stage.

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

# fixtures

The golden capture set — several objects photographed on a printed reference board — lives
outside git, in cloud storage. This directory holds the download/verification metadata and
reference calibration assets, never the captured photographs themselves (see `.gitignore` and
`AGENTS.md`).

**`charuco_board.pdf`** is the calibration target itself, not captured data — small,
deterministic, and exactly what R-02/R-03 need to match. Print at 100% scale, never "fit to
page." Parameters (`DICT_5X5_250`, 8×6 squares, 20 mm/15 mm, non-legacy pattern) are recorded
in `docs/decisions.md` D-034.

## F-06 golden capture

The five photo sets are stored in the team's shared Google Drive and are described by
`golden_capture_manifest.json`. The manifest records the Drive folder IDs, expected frame
counts, aggregate byte counts, object/lighting notes, reference-CAD IDs, and a `content_sha256`
per sample.

Run:

```bash
bash fixtures/download_golden_capture.sh
```

The script downloads `s-01` through `s-05`, verifies the expected JPEG count for each sample,
and then verifies each sample's `content_sha256` against the manifest — a combined hash over
every downloaded file's own SHA-256, keyed by filename. That second check exists because the
JPEG count alone doesn't catch a Drive file that gets replaced or corrupted without changing
how many files are in the folder: two developers could otherwise end up reconstructing against
different actual bytes from the same commit, with nothing erroring. After both checks pass, the
script writes `SHA256SUMS.local` beside the downloaded data for manual diffing. The downloaded
JPEGs remain ignored by git. Hashing uses `sha256sum` where available and falls back to the
BSD/macOS-native `shasum -a 256` — the documented local dev path is native Apple Silicon
(D-010/D-012), which doesn't ship coreutils by default.

`golden_capture_manifest.sha256` is the committed SHA-256 of the versioned inventory manifest
(including the per-sample `content_sha256` values). It detects accidental changes to the
expected Drive fixture inventory. The per-image `SHA256SUMS.local` generated after download is
intentionally local because the photographs themselves remain outside git.

A reference CAD model for `s-04` is committed under `reference_cad/`. It is the explicit
exception to the global `*.stl` ignore rule required by F-06.

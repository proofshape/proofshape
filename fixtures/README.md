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
counts, aggregate byte counts, object/lighting notes, and reference-CAD IDs.

Run:

```bash
bash fixtures/download_golden_capture.sh
```

The script downloads `s-01` through `s-05`, verifies the expected JPEG count for each sample,
and writes `SHA256SUMS.local` beside the downloaded data. The downloaded JPEGs remain ignored
by git.

`golden_capture_manifest.sha256` is the committed SHA-256 of the versioned inventory manifest.
It detects accidental changes to the expected Drive fixture inventory. The per-image
`SHA256SUMS.local` generated after download is intentionally local because the photographs
themselves remain outside git.

A reference CAD model for `s-04` is committed under `reference_cad/`. It is the explicit
exception to the global `*.stl` ignore rule required by F-06.

# fixtures

The golden capture set — several objects photographed on a printed reference board — lives
outside git, in cloud storage. This directory holds the download/verification metadata and
reference calibration assets, never the captured photographs themselves (see `.gitignore` and
`AGENTS.md`).

**`charuco_board.pdf`** is the calibration target itself, not captured data — small,
deterministic, and exactly what R-02/R-03 need to match. Print at 100% scale, never "fit to
page." Parameters (`DICT_5X5_250`, 8×6 squares, 20 mm/15 mm; generated with calib.io's Legacy
box unchecked, which is OpenCV's `setLegacyPattern(True)` layout) are recorded in
`docs/decisions.md` D-034.

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

Reference CAD models for `s-04` and `s-05` are committed under `reference_cad/` (both already
listed in the manifest's `reference_cad` block above). They are the explicit exception to the
global `*.stl` ignore rule required by F-06; R-22 added `s-05`'s file to let its reconstructed
size be checked against CAD the same way R-17 did for `s-04`.

## Caliper readings

The ground-truth stories (`stories/G/`) record caliper readings as one file per story,
`fixtures/caliper_<story-id>.json` (for example `caliper_g-06.json`). The files sit at the top
level of `fixtures/` because `.gitignore` only lets top-level JSON through. They hold raw readings
only: no tolerance, no pass or fail, nothing inferred from CAD. The measured print is the truth,
and judging it is G-04's job.

Top level:

| Field | Meaning |
|---|---|
| `version` | `1` |
| `story` | The story that took the readings, e.g. `"G-06"` |
| `measured_by` | The person's `@github-handle` |
| `date` | `YYYY-MM-DD` |
| `instrument` | An object with the three fields below |
| `instrument.type` | e.g. `"digital caliper"` |
| `instrument.resolution_mm` | The smallest step it displays |
| `instrument.accuracy_spec` | The maker's stated accuracy, or `null` when it states none. Resolution is not accuracy. |
| `features` | An array of objects, one per measured feature, with the fields below |

Each object in `features`:

| Field | Meaning |
|---|---|
| `object` | `s-04`, a specimen ID, `board`, or a cube ID |
| `kind` | `edge`, `height`, `board_span` or `cube_axis` |
| `cad_vertices` | For an `edge`, the CAD vertex pair it runs between, as `[[x, y], [x, y]]` in the STL's millimetres. `null` otherwise |
| `nominal_mm` | The size it should be, in millimetres |
| `nominal_source` | Where `nominal_mm` comes from: `"cad"`, `"D-034"` (the board's 20 mm squares) or `"cube"` |
| `square_count` | For a `board_span`, how many squares it spans |
| `where` | How it was read, e.g. `"mid-height, across the parallel faces"` or `"corner to corner"` |
| `orientation_on_board` | For a square footprint, which way this wall faced on the board. `null` otherwise |
| `readings_mm` | The 10 independent placements, in millimetres |
| `mean_mm` | Mean of `readings_mm` |
| `min_mm` | Smallest reading |
| `max_mm` | Largest reading |
| `std_mm` | Sample standard deviation of `readings_mm` |

`tests/test_caliper_files.py` (G-06) checks every `fixtures/caliper_*.json` against this format,
including that the summary figures match the readings they summarise.

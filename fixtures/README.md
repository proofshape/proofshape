# fixtures

The golden capture set — several objects photographed on a printed reference board — lives
outside git, in cloud storage. This directory holds the download/verification metadata and
reference calibration assets, never the captured photographs themselves (see `.gitignore` and
`AGENTS.md`).

**`charuco_board.pdf`** is the calibration target itself, not captured data — small,
deterministic, and exactly what R-02/R-03 need to match. C-17 also commits
`order_sheet_PO-48213-A.pdf`, the deterministic browser-generator sample used by
`tests/test_order_sheet.py`; its embedded board page must remain at the same 1:1 scale. Print at
100% scale, never "fit to page." Parameters (`DICT_5X5_250`, 8×6 squares, 20 mm/15 mm; generated with calib.io's Legacy
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

`reference_cad/` also holds G-01's `g01_calibration_cube_30mm.stl` and G-02's six
`g02_*.stl` specimen files, for the ground-truth lane's own σ_floor measurement -- not part of
F-06's golden capture set, same exception to the `*.stl` ignore rule.

It also holds G-05's ten `g05_*.stl` defect-ladder parts: a control and its 2 / 4 / 8 mm
shifted-hole, pocket and bump variants, written by `scripts/make_g05_defect_ladder.py`. They
are planted-defect parts for M4 and the proposal's §12 test, not σ_floor specimens, and
deliberately not prisms; `stories/G/G-05.md` has the design and why.

G-11's thin-walled part, `g11_01_thin_wall_pentagon_r22x25mm.stl`, is `g02_04` hollowed into an
open-top tray with 1.2 mm walls. It is written by `scripts/make_g11_thin_wall.py`. G-11's glossy
and dark parts are `g02_04` itself, printed in another filament, so they have no file of their
own. `stories/G/G-11.md` has the design.

## G-08 specimen capture

The six G-02 specimens (`g02_01` to `g02_06`) were photographed for G-04's accuracy runs, and G-09
added G-05's two defect parts. The photos are on the same Google Drive as the golden set, and
`specimen_capture_manifest.json` describes them in the golden manifest's own format: Drive folder
IDs, frame counts, aggregate byte counts and a `content_sha256` per sample. Its `reference_cad`
block points at each part's CAD under `reference_cad/`.

- **Capture:** 25–30 landscape stills each, 2026-11-15, on the same G-06-calipered board sheet
  and table as the golden set, under lighting condition A. iPhone 16 main camera, 5712 × 4284,
  the same camera the golden set used.
- **Checks:** the board was posed in every frame by R-02's own detection, and no frame was
  softer than the golden set's softest. One frame taken on the ultra-wide lens was dropped
  before the hashes were computed: a second lens would break the single-camera calibration.
- **G-09 added G-05's two printed parts** to the same manifest: the control `g05_00` and the
  4 mm hole shift `g05_02` (D-069), 28 stills each, under the same setup. `g05_00` was shot on
  2026-11-18. `g05_02` was re-shot on 2026-11-20, because the first set had the plate top face
  down, a mirror image of the CAD that no rigid pose can match (#407; G-09 has the details).
  Their `reference_cad` entries point at G-05's STLs. Every frame was posed by R-02, and the
  softest scored 284 against the golden set's softest of 40. The real downloader fetched both
  from Drive and verified their hashes. These parts aren't prisms, so G-04's script (which lists
  `g02_*` by name) doesn't pick them up.

The same script downloads either set; point it at this manifest and give it its own output
directory:

```bash
GOLDEN_CAPTURE_MANIFEST=fixtures/specimen_capture_manifest.json \
  bash fixtures/download_golden_capture.sh fixtures/data/specimen_capture
```

`specimen_capture_manifest.sha256` locks the manifest the same way the golden one does.
`.gitignore` now lets any top-level `fixtures/*.sha256` be committed without a force-add, while
checksum files inside `fixtures/data/` stay ignored.

If gdown stops partway with "Failed to retrieve file url" (it did after about 100 files on the
day these were checked, while Drive still served the same files on its plain download URL), wait
and run the script again, ideally later or from another network. It never swaps in a sample that
failed its count or hash check, so a retry is always safe.

## G-11 robustness capture

G-11's three printed robustness parts were photographed the same way as G-08's specimens, and
`robustness_capture_manifest.json` describes them in the same format, locked by
`robustness_capture_manifest.sha256`. They have their own Drive folder (`g-11`), beside G-08's.

| Sample | Part | Frames | CAD |
|---|---|---|---|
| `g11_glossy` | silver silk PLA pentagon | 30 | `g02_04_pentagon_r22x25mm.stl` |
| `g11_dark` | black PLA Basic pentagon | 30 | `g02_04_pentagon_r22x25mm.stl` |
| `g11_thin` | grey open-top tray, 1.2 mm walls | 29 | `g11_01_thin_wall_pentagon_r22x25mm.stl` |

- **Capture:** 2026-11-17, on the same board sheet and table as the golden set and G-08, under
  lighting condition A, with the iPhone 16 main camera at 5712 × 4284. The grey `g02_04` in
  `specimen_capture_manifest.json` is the control for all three.
- **Checks:** the board was posed in every frame by R-02's own detection, and no frame was softer
  than the golden set's softest. No part has a label on a face the camera sees: the glossy part
  and the tray were reshot after a pen label was found on a side face.

```bash
GOLDEN_CAPTURE_MANIFEST=fixtures/robustness_capture_manifest.json \
  bash fixtures/download_golden_capture.sh fixtures/data/robustness_capture
```

The manifest's `stills` list holds the two single-image controls that A-01's live VLM check and
A-02's veto testing share. Both are locked by their file's sha256. The downloader fetches
`captures` only, so these come down by hand, and you check them against the hash.

| Still | File | Where it is |
|---|---|---|
| `g11_screen_photo` | `screen_photo.jpeg`, a phone photo of a laptop screen showing s-04's `IMG_4098.jpeg` | Drive `vlm_checks/`, by `drive_file_id` |
| `g11_wrong_part` | `IMG_4161.jpeg`, the red s-05 part | the golden set's `s-05` sample, so there's no second copy |

`scripts/check_vlm.py` takes them as `--screen-photo` and `--wrong-part`. Any capture folder can
be the first argument, and the wrong part only has to be a different part from that capture's.

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
| `object` | `s-04`, a specimen ID, `board`, a cube ID, a G-05 part ID such as `g05-00`, or a G-11 sample ID such as `g11_thin` |
| `kind` | `edge`, `height`, `board_span`, `cube_axis`, `wall` (G-05: the web between an outside face and a hole wall, outside jaws; G-11: the thin tray's wall, read across both its faces) or `hole_width` (G-05: across a hole, inside jaws) |
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

Write `mean_mm` and `std_mm` to four decimal places. A caliper's spread is often about 0.01 mm,
and at that size three decimals can't tell the sample standard deviation from the population one.

`tests/test_caliper_files.py` (G-06) checks every `fixtures/caliper_*.json` against this format,
including that the summary figures match the readings they summarise and that every reading is a
whole number of the instrument's displayed steps.

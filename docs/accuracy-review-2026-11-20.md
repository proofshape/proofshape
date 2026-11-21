# Accuracy review: ten calipered parts on `main`, 2026-11-20

P-41. Measured by @TabeenRaoof on the M2 Max (`mps`), on `main` at `607450a`, with
`GEMINI_API_KEY` unset. The tables in sections 2 and 3 were then produced again from this
story's branch at `89d0de4`, which differs from `607450a` in story files only. Everything here
is one capture per part. Nothing in this document is a decision, and no threshold changes
because of it.

**How to read the sources.** Sections 2 and 3 come from `scripts/measure_part_accuracy.py`,
which is in this pull request with its tests, and can be re-run (section 8). Sections 4 and 5
come from a scratch harness that re-fused saved depth outside the repo. They are screening
results, and R-44 and I-15 have to reproduce them in repo code before anything rests on them.

## 1. Summary

- **Top faces are measured well. Walls are not.** On ten printed parts, each compared with its
  own CAD, top-face vertices sit within about 0.2 mm of the CAD (median absolute deviation).
  Wall vertices read 0.1 to 2.1 mm inside it (median per part), and single readings reach
  5.6 mm.
- **A good part first Passes at 3.7 to 6 mm**, with orientation ambiguity set aside. Six of the
  ten are symmetric shapes and flag ambiguous, as D-061 intends, so in the service they Rescan.
- **Most of the wall error is photos disagreeing sideways.** After R-23's alignment, a photo's
  own board corners land a median 2.2 mm from where the board says they are. R-30 corrects each
  photo's height and tilt, and leaves about 1.6 mm in the board plane. A sideways shift doesn't
  move a flat top face. It moves every wall and edge.
- **Re-anchoring each photo to its own corners removes most of it, in a scratch test.** The
  typical worst reading on a part goes from 4.3 to 1.9 mm, and seven parts Pass at 3.7 mm
  (three today). This is R-44.
- **Stray board surface misleads the CAD alignment.** On six parts, 12 to 44% of the Observed
  model is surface a few millimetres above the board and away from the part. With R-44's
  correction on and nothing else changed, the G-05 control comes out 180° off with no ambiguity
  flag and reads 9.8 mm. This is I-15, and R-44 can't go on in the served path without it.
- **The 4 mm hole shift still reads like its control** in every variant. The control's own
  12 mm hole is bridged over (up to +5.6 mm). None of this changes #407.
- **These made no difference:** dropping pixels at depth edges, letting board pixels vote
  "empty space" (two settings), and a 1 mm grid on its own.

## 2. What a good part reads on `main`

**Method.** Each capture went through `run_reconstruction_pipeline`, then `global_register`,
`check_ambiguity` and `compute_deviation` against the part's own CAD, the same calls in the same
order as the service. A part against its own CAD has no defect, so every deviation is
reconstruction and measurement error. The script then tries v1's verdict rule at a range of
tolerances for one non-critical `face` feature, with orientation ambiguity left out. That is
what the measurement supports, not a service verdict.

**Reference.** Caliper means are within 0.2 mm of CAD nominal on every part except `g02_03`
(edges 0.5 to 1.0 mm short) and s-04 (0.1 to 0.4 mm over). At the error sizes below, nominal CAD
is the reference.

| Part | Observed vertices | Unmeasured (outside the CAD outline) | Stray (over 5 mm from the part) | Top face, 95% within | Wall, median | Wall, 95% within | Worst reading | First Pass at | Flags ambiguous | Alignment fitness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| s-04, triangle | 1,602 | 40% | 3% | 0.68 | −0.77 | 3.55 | 4.48 | 4.5 | no | 0.98 |
| s-05, 30 × 30 × 60 | 2,247 | 24% | 0% | 1.56 | −2.09 | 3.92 | 4.80 | 5 | yes | 0.99 |
| `g05_00`, control | 1,586 | 58% | 24% | 1.46 | −0.67 | 3.46 | 5.58 | 6 | no | 0.72 |
| `g05_02`, own CAD | 2,111 | 69% | 32% | 1.00 | −0.14 | 3.02 | 5.01 | 5.5 | no | 0.73 |
| `g02_01`, square 20 | 567 | 72% | 44% | 2.51 | −1.73 | 3.70 | 4.14 | 4.5 | yes | 0.37 |
| `g02_02`, rect 50 × 20 | 1,354 | 41% | 1% | 0.54 | −0.52 | 0.94 | 1.70 | 3.7 | yes | 0.99 |
| `g02_03`, triangle | 1,188 | 69% | 17% | 0.86 | −0.32 | 1.05 | 2.01 | 3.7 | no | 0.77 |
| `g02_04`, pentagon | 1,807 | 58% | 38% | 1.60 | −1.52 | 3.28 | 3.74 | 4 | yes | 0.55 |
| `g02_05`, hexagon | 2,481 | 40% | 12% | 1.88 | −1.85 | 4.32 | 5.33 | 5.5 | yes | 0.86 |
| `g02_06`, rect 70 × 45 | 3,619 | 22% | 2% | 1.05 | −0.62 | 2.85 | 3.19 | 3.7 | yes | 1.00 |

Distances are in millimetres. Negative means inside the CAD: the part reads small.

- **"Top" and "wall" here are the CAD face a vertex sits on**, by that face's normal. The verdict
  splits differently, by height: the top tenth of measured vertices use the height floor, and
  the rest use the in-plane floor. On these parts 35 to 530 vertices that sit on a top face fall
  in the in-plane pool.
- **Every part's wall median is negative**, and so is the 95th percentile of every part's wall
  except the two G-05 plates. D-086 measures a vertex only inside the CAD's outline, so the
  outer half of each wall is not in these figures. D-086 records that bias.
- **"Unmeasured" is Observed surface the verdict never sees**: 22 to 72% of it.
- **Run to run.** A second full run of the ten parts, through the committed script, gave the
  same figures as the first to every digit printed here.

**Three saved phone sessions of s-04** (the capture app's 480 × 640 frames), same method: worst
readings 4.20, 0.83 and 0.97 mm, first Pass at 4.5, 3.7 and 3.7 mm. In the last two only 19% and
31% of the Observed surface is inside the outline and measured. Two more saved sessions fail the
CAD alignment outright (`sess_20201ffd675a`, 13 frames, and `sess_cdf9822969a4`, whose stored
record shows the same failure). No photograph was opened for this review, so what was on the
board in those two is not verified.

## 3. Why walls are noisy

### 3.1 Photos disagree sideways

Each photo's detected ChArUco corners were lifted through that photo's own depth, as R-23 does,
and compared with their printed positions.

| Capture | Photos measured | After the global alignment: median (95th percentile) | After R-30: in the board plane | After R-30: height | Shift at the board's centre: median (worst photo) | A rigid fit to the photo's own corners leaves |
|---|---:|---:|---:|---:|---:|---:|
| s-04 | 29 of 30 | 1.98 (2.80) | 1.62 | 0.07 | 1.54 (2.75) | 0.48 |
| s-05 | 29 of 29 | 1.99 (3.47) | 1.60 | 0.07 | 1.53 (4.00) | 0.56 |
| `g05_00` | 28 of 28 | 2.76 (4.57) | 2.32 | 0.09 | 2.20 (3.49) | 0.89 |
| `g05_02` | 27 of 28 | 1.49 (2.57) | 1.06 | 0.25 | 0.90 (3.94) | 0.61 |
| `g02_01` | 29 of 29 | 3.72 (5.47) | 3.00 | 0.06 | 2.91 (4.95) | 0.81 |
| `g02_02` | 29 of 30 | 1.47 (2.94) | 1.06 | 0.15 | 1.01 (2.45) | 0.54 |
| `g02_03` | 25 of 25 | 1.80 (3.56) | 1.31 | 0.11 | 1.24 (3.14) | 0.57 |
| `g02_04` | 27 of 27 | 2.32 (4.02) | 1.87 | 0.17 | 1.84 (3.54) | 0.80 |
| `g02_05` | 29 of 29 | 3.28 (4.27) | 2.60 | 0.18 | 2.54 (3.73) | 0.67 |
| `g02_06` | 29 of 29 | 1.63 (7.53) | 1.24 | 0.17 | 1.14 (2.85) | 0.63 |

Millimetres, per capture, as medians over its photos. A photo with fewer than eight corners is
left out (one on s-04). So is a photo R-38 rejected (one each on `g05_02` and `g02_02`), because
it never reaches fusion.

- **After R-23's global alignment**, a photo's corners sit a median 2.23 mm from true over the
  281 photos of the ten parts (95th percentile 4.35 mm).
  `metric_alignment.json` already records a per-photo figure of this kind as
  `per_frame_residual_mm`.
- **After R-30**, the height part is gone (under 0.3 mm on every capture) and the in-plane part
  remains: 1.1 to 3.0 mm per capture, and 1.60 mm as the median over all 281 photos. At the
  centre of the board that is a shift of 0.9 to 2.9 mm in a capture's typical photo, and up to
  5.0 mm in the worst.
- **A rigid fit of each photo to its own corners would leave 0.5 to 0.9 mm** per capture
  (0.65 mm over all photos).
- **Phone sessions start closer.** The three s-04 phone sessions read 1.1 to 1.3 mm after the
  global alignment and 0.9 to 1.2 mm in the board plane after R-30. A rigid fit would leave 0.3
  to 0.4 mm.

On the part itself (s-04 and s-05, scratch harness): one photo's top-face pixels agree with each
other to 0.25–0.29 mm, but photos differ from each other by 0.66–0.73 mm (standard deviation).
For walls, photos differ by 0.8–1.1 mm, with single photos −4.2 to +2.0 mm off.

**This fits the two measured floors.** D-073 puts heights at 0.86 mm and D-078 puts in-plane
edges at 5.278 mm. R-30's own notes scope it to the frame's board plane, three of a pose's six
degrees of freedom. The other three are the sideways ones.

### 3.2 Board surface kept as part

The "stray" column in section 2 is Observed surface more than 5 mm from the registered CAD. On
the six parts where it is 12% or more, it sits a median 3 to 4 mm above the board, just over the
cut (2.0 to 2.75 mm). #292 describes this for s-02. It is on six of these ten parts.

It is outside the CAD's outline, so I-14 keeps it out of the verdict. It still reaches the pose
search, which seeds every candidate from the mean of all Observed model points
(`_board_pose_from_yaw`):

- alignment fitness is 0.37 to 0.77 on five parts;
- on `main`'s meshes, s-05 and the hexagon sit about 10° and 16° from the pose found on the
  re-anchored mesh, beyond what their symmetry allows;
- section 4's G-05 control.

### 3.3 Limits nothing moved

- **A 12 mm through-hole is bridged.** The G-05 control reads +5.1 to +5.6 mm over its own hole
  in every variant with a correct pose.
- **Resolution.** The depth maps are 392 × 518 pixels, and one pixel covers about 0.73 mm at the
  part. The grid is 2 mm and the truncation 8 mm.

## 4. What was tried, offline (scratch harness)

Each variant re-fused the same saved depth with one change, then ran the repo's own cleanup,
provenance, uncertainty, registration, deviation and verdict code. The unchanged variant
reproduces section 2 exactly.

| Variant | Wall, 95% within (median of parts) | Top face, 95% within | Worst reading (median of parts / worst part) | Parts that Pass at 3.7 mm |
|---|---:|---:|---:|---:|
| `main` | 3.37 | 1.25 | 4.31 / 5.58 | 3 |
| **Rigid re-anchoring to each photo's corners (R-44), nothing else changed** | **1.38** | **0.61** | **1.88 / 9.80** | **7** |
| The same, CAD alignment started from `main`'s pose | 1.46 | 0.60 | 2.00 / 5.27 | 8 |
| The same, CAD alignment fitted on the upper half of the model | 1.44 | 0.59 | 1.99 / 5.57 | 8 |
| `main`, CAD alignment fitted on the upper half of the model | 3.16 | 1.24 | 4.40 / 5.56 | 3 |
| Re-anchoring that also rescales each photo's depth | 1.39 | 0.60 | 1.84 / 9.36 | 7 |
| Drop pixels at depth edges | 3.52 | 1.13 | 4.75 / 6.54 | 3 |
| Board pixels vote "empty", 8 mm in front of themselves | 3.58 | 1.15 | 4.78 / 6.77 | 3 |
| Board pixels vote "empty", 3 mm in front of themselves | 3.57 | 1.37 | 4.78 / 7.08 | 4 |

**Rigid re-anchoring, part by part (worst reading, mm):** s-04 4.48 → 2.76. s-05 4.80 → 2.07.
`g02_02` 1.70 → 1.08. `g02_03` 2.01 → 1.69. `g02_04` 3.74 → 1.46. `g02_05` 5.33 → 1.49. `g02_06`
3.19 → 1.16. `g05_02` 5.01 → 5.26. **`g05_00` 5.58 → 9.80. `g02_01` 4.14 → 5.42.**

- **The two that get worse are CAD alignment, not reconstruction.** `g05_00` lands 180° off,
  fitness 0.61, not flagged. `g02_01` lands at fitness 0.29. Started from `main`'s pose, the
  same two meshes read 5.22 and 2.51 mm. In six other variants the control came out 90° off,
  unflagged.
- **A check that doesn't use the per-vertex rule.** Wall-to-wall width at mid-height on the three
  clean boxes, against the caliper mean: s-05 reads 26.7 and 25.8 mm on `main` and 28.7 and
  28.6 mm re-anchored (caliper 29.97). Over six widths (s-05, `g02_02`, `g02_06`) the mean
  absolute error goes from 2.3 to 1.0 mm.
- **The gate binds, not the data.** The parts that Pass re-anchored have worst readings of 1.1
  to 2.8 mm. They Pass at exactly 3.7 mm because Pass needs tolerance ÷ 0.70 to reach the
  in-plane floor, 5.278 mm.
- **Rigid is enough.** Rescaling each photo's depth as well gives the same wall figures, adds
  stray surface on `g05_00` (24% → 34%), and would need a per-photo depth factor carried into
  provenance.
- **Side effect.** With the rigid correction the stray share rises on `g02_04` (38% → 48%) and
  `g02_06` (2% → 12%).
- **A finer grid on its own does not move the walls.** At 1 mm, without re-anchoring: 3.55 →
  3.31, 3.92 → 3.85 and 4.32 → 4.29 mm on s-04, s-05 and `g02_05`. With re-anchoring, 1 mm
  against 2 mm is mixed (1.89 → 2.12, 1.74 → 1.47, 0.98 → 0.86). Fusion goes from 0.3 s to 2.4 s.
  R-43 measures this properly, against caliper edges.
- **Weighting each pixel by how head-on it sees the surface** was tried only on top of the
  re-anchoring, the edge filter and the empty-space votes together: 1.50 mm against 1.44 without
  it.
- **Upper-half alignment** is one rule at one setting. On `main`'s meshes it lifts the median
  fitness from 0.81 to 0.99. On one variant it landed the control 19° off, and flagged it.
- **Phone sessions gain little**, because they start closer (section 3.1). With rigid
  re-anchoring the first goes 4.20 → 3.88 mm and first Passes at 4 mm, not 4.5. The other two
  stay under 1 mm.

## 5. The planted hole shift (#407)

The re-shot `g05_02` against the control's CAD, in every variant with a correct pose:

- the strip where the shifted part has material and the CAD has a hole reads +1.2 to +1.9 mm
  (median), against +0.9 to +3.0 on the control;
- over the whole CAD hole the control reads up to +5.6 mm and the shifted part up to +5.4;
- both first Pass at 5.5 to 6 mm.

The control's own hole is bridged, so the defect has no clean signal. The re-anchoring does not
change that.

## 6. The verdict, as it stands

- **The two floors were measured on a different quantity from the one the verdict uses.** D-073
  (five heights) and D-078 (eleven edge lengths) come from `cad_size_check` on R-01's raw cloud.
  The verdict thresholds each fused-mesh vertex's distance from the registered CAD. Section 2
  measures that quantity directly.
- **Every-vertex rules turn on one vertex.** Pass needs every vertex inside the tolerance, and
  Fail needs one outside. Per-vertex sigma is capped at 3.0 mm, so 0.70 × 3.0 = 2.1 mm bounds a
  Pass tolerance from below wherever a vertex sits near the cap.
- **Symmetric parts get no verdict.** Six of ten flag ambiguous. For a CAD that maps onto
  itself under the other pose, both poses give the same deviations.

## 7. What it calls for

**Filed with this review:**

- **R-44** (claimed, @TabeenRaoof): rigid per-photo corner re-anchoring in `fuse_run`.
- **I-15** (claimed, @TabeenRaoof): the CAD alignment starts from the part, not from stray
  surface. R-44's default waits on it, so it is worked first.
- **G-13** (Blocked on R-44 and I-15, filed on 2026-11-21): measure the floors and the outline
  margin on the verdict's own quantity, over the ten calipered parts on the pipeline as served.
  It proposes amendments to D-073, D-078 and D-086, and the constants change only if both other
  members approve. What prompted it: on the re-anchored meshes a 2 mm margin gives two-sided
  wall readings (medians −0.5 to +0.3 mm), roughly doubles the measured surface on several
  parts, and keeps worst readings at or under 2.7 mm on seven of the eight prisms.

**Proposals for the team, not filed:**

- **Bin by CAD face, not by height.** It needs no contract change, unlike a height / in-plane
  field on spec features.
- **Symmetric CADs**: whether a pose that maps the CAD onto itself should count as ambiguous.

**Unchanged:** X-03's planted defect is still #407's choice. The hole shift is a measured
negative here too.

## 8. Re-run it

From the repo root, with the weights in place (`scripts/bootstrap_dev_env.sh`) and
`PYTHONPATH=.:inspect` so the checkout's own code is what runs. The captures come from the
download script: the golden set by default, and the G-02 and G-05 parts with
`GOLDEN_CAPTURE_MANIFEST=fixtures/specimen_capture_manifest.json` (`fixtures/README.md`).

```
export PYTHONPATH=.:inspect OMP_NUM_THREADS=1 PROOFSHAPE_RECON_DEVICE=mps   # or cuda
python scripts/measure_part_accuracy.py run fixtures/data/golden_capture/s-04 \
    --cad fixtures/reference_cad/s04_triangular_prism_60-26-65x30mm.stl --out runs/s-04
python scripts/measure_part_accuracy.py report  runs/s-04 runs/s-05     # section 2
python scripts/measure_part_accuracy.py corners runs/s-04 runs/s-05     # section 3.1
```

`run` takes about a minute per part on the M2 Max. Each capture is run against its own CAD in
`fixtures/reference_cad/`:

- s-04: `s04_triangular_prism_60-26-65x30mm.stl`; s-05: `s05_rect_prism_30x30x60mm.stl`;
- `g05_00`: `g05_00_control_50x32x20mm.stl`; `g05_02`: `g05_02_hole_shift_4mm.stl`;
- `g02_01` to `g02_06`: the file that starts with the same name.

**A cheaper first check needs no run at all:** any finished session's
`reconstruction/metric_alignment.json` already lists `per_frame_residual_mm`, the per-photo
corner residual after the global alignment.

## 9. Limits

- One capture per part: ten printed prisms and plates, one filament, one phone, one host.
- Sections 4 and 5 are from a scratch harness, with no tests. R-44 and I-15 reproduce them.
- The "stray" distance (5 mm) and the top / wall bins are this review's reporting choices.
- "Parts that Pass at 3.7 mm" is one non-critical `face` feature with ambiguity set aside.
- No photograph was opened. Two phone sessions' contents are unverified.
- The capture app and the hosted service were not run for this review.

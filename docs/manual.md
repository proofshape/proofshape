# ProofShape — operating manual

How to use ProofShape: what you need, how to set it up, how to capture a part, and how to read
what comes back. Written for the two people who actually touch the system — the **buyer** setting
up an order, and the **supplier** holding the phone — plus anyone on the team who needs the
physical procedure.

**This is a living draft.** The system is mostly not built yet, and this manual says so section by
section rather than describing a product that does not exist. Every section carries a status:

| Status | Meaning |
|---|---|
| **Works now** | You can do this today, exactly as written. |
| **Planned** | Designed and specified, not built. The story that will build it is named. |
| **Undecided** | Genuinely open. Do not assume a behaviour here. |

As of this draft, the only thing that genuinely **works now** is printing the reference sheet
(§3). Everything the phone and the service do is still ahead of us — the repository holds the
frozen interface, the CI, and the skeleton, but no capture app and no reconstruction engine yet.
Rather than wait until it all works, this manual is being written alongside the build, and each
section flips to **Works now** as its story lands.

---

## 1 · What ProofShape does, and what it does not

**Status: Works now (as a statement of scope — nothing to run yet).**

ProofShape turns a smartphone into a **coarse 3D inspection tool** for manufactured parts. A
supplier places a part on a printed reference sheet, photographs it with on-screen guidance, and
gets back a metric 3D model in the browser. If the buyer uploaded a CAD file, the system aligns
the model to it and returns a **Pass / Fail / Rescan / Unverifiable** report with a deviation
heatmap, so gross defects are caught before the part ships.

**Read this part carefully, because it is the whole basis on which the report can be trusted:**

- **This is coarse geometric inspection — several millimetres on hand-sized parts.** It
  complements CMM inspection. It does not replace it, and it never will (D-003).
- **No tolerance figure in this manual is a measured result.** The numbers that would let you
  decide whether ProofShape is precise enough for a given feature do not exist yet; they come from
  the calibration study (milestone M3, October). Until then, treat every figure anywhere in this
  project as a planning estimate.
- **Geometry the camera did not properly see can never pass or fail a part.** It may only trigger
  a Rescan. This is the hard rule (D-005), and it is enforced structurally — in the data format
  itself, with a test that fails the build if a report ever violates it, not merely as a policy.
- **No CAD file, no verdict** (D-004). Without a design to compare against, ProofShape returns a
  measured model and nothing more. That is a legitimate output, just not an inspection.

---

## 2 · What you need

**Status: Works now.**

| Item | Notes |
|---|---|
| **Printed ChArUco reference sheet** | One sheet of A4 or US Letter. See §3 — printing it correctly matters more than it sounds. |
| **The part** | Rigid. Roughly hand-sized — comparable to, or a little larger than, the printed pattern. See §3.4 on parts larger than the board. |
| **A smartphone** | With a working camera and gyroscope. No app install — capture runs in the browser over HTTPS. |
| **Digital calipers** *(recommended)* | For verifying the printed square size, and for the optional typed scale check. Roughly $25. |
| **The CAD file** *(buyer, optional)* | STL/OBJ/GLB. Required only if you want a verdict rather than just a model. |
| **A flat, evenly lit surface** | Enough room to walk around the part, or to turn the sheet. |

---

## 3 · Printing the reference sheet

**Status: Works now.**

Photographs recover shape but never size. Something of known size has to be in frame, or no
dimensional claim can be made at all. That object is the reference sheet, and the ChArUco board is
the best of the three tiers (§6) because it does more than supply scale: its corners are each
individually identifiable, so a **partly covered board still works**, and it pins down camera
orientation on glossy, featureless parts where nothing else would match.

### 3.1 The file

`fixtures/charuco_board.pdf` in this repository is the board. Print that file — do not regenerate
one casually, because a board with different parameters will not be detected by code built against
these:

| Parameter | Value |
|---|---|
| Dictionary | `DICT_5X5_250` |
| Layout | 8 columns × 6 rows of squares |
| Square (checker) size | 20 mm |
| Marker size | 15 mm |
| Pattern size | 160 mm × 120 mm |
| calib.io "ChArUco Legacy" checkbox | **Off** (unchecked) |

Recorded as D-034. If it is ever lost, it regenerates from those parameters at
[calib.io's pattern generator](https://calib.io/pages/camera-calibration-pattern-generator) (free
for educational use). Leave calib.io's "ChArUco Legacy" box unchecked, as it was for the golden
capture print. Watch out: the board that setting produces is the one OpenCV constructs with
`setLegacyPattern(True)`, the opposite of what the names suggest (D-034's 2026-09-25
correction). With the wrong OpenCV flag, the markers are found but no corners are, so every frame
fails even though the dictionary and geometry look right. `recon/board_pose.py` sets the flag
correctly, and a test checks it against `fixtures/charuco_board.pdf`.

### 3.2 Printing it

**The single most important setting: print at 100% scale.**

1. Open the PDF and choose **Print**.
2. Paper size: **A4 or US Letter**. Either works.
3. Orientation: **Landscape** — the pattern is wider than it is tall.
4. Scaling: select **Scale: 100%**. **Do not** use "Scale to Fit," "Fit to Page," or "Shrink to
   Fit." On macOS the print dialog defaults to "Scale to Fit," which silently enlarges the pattern
   (about 135% on Letter) — the squares come out near 27 mm instead of 20 mm, and every
   measurement downstream is wrong by that factor while looking perfectly fine.
5. Print single-sided, one copy.

Do not trim the page. The **paper's own outline is part of the measurement** — it is one of the
three independent scale checks (§7), because the PDF says what the squares *should* be and the
sheet outline reveals what the printer actually did.

### 3.3 Verify it before you shoot anything

**Measure two or three squares with calipers.** They should read close to 20 mm.

This is not belt-and-braces. It is the reference number the scale-agreement check compares
against (D-007), and home and office printers do rescale pages silently. A board that is 3% off
produces a model that is 3% off, confidently and with no warning anywhere.

If the measurement is off by more than a hair, reprint — check the scaling setting first, then
the printer's own "fit to printable area" setting, which some drivers apply on top of whatever the
application asked for.

### 3.4 Margins, and parts larger than the board

The printed pattern (160 × 120 mm) sits inside a much larger sheet, with wide white margins. That
is deliberate, and the margins are doing three jobs:

1. Keeping the paper's outline clearly separated from the pattern, so the outline scale check
   works.
2. Reserving space for the per-order QR code and orientation arrow (**Planned** — see §4).
3. Giving a part that is larger than the pattern somewhere to sit without hanging off the sheet.

**If the part is larger than the pattern but still roughly sheet-sized, that is fine.** The system
is designed for partial occlusion — as long as some markers stay visible in a frame, the pose can
still be computed, and frames where too little board is visible get rejected with a reason rather
than silently accepted (§8).

**If the part is dramatically larger than the whole sheet, ProofShape is the wrong tool for it.**
The stated scope is hand-sized parts. Supporting genuinely large parts would need a physically
larger board (bigger paper, or larger squares at the cost of marker headroom), which is not a
decision the team has made. **Undecided.**

---

## 4 · Setting up an order — the buyer

**Status: Planned.** Built by the capture-lane stories (C-13 buyer setup, C-14 board PDF with
order code) and the inspection stories.

The intended flow:

1. **Upload the CAD** (STL/OBJ/GLB).
2. **Enter what matters** — the tolerance, and the critical features by name ("the four mounting
   holes", "the sealing face"). **Planned:** you will also be able to write this in plain English
   and have it converted into the structured specification (A-03, prose-to-spec parser).
3. **Intake pre-check runs.** Any feature whose tolerance is tighter than the method's own noise
   floor for the required reference tier is flagged **Unverifiable before any photograph is
   taken** — so it can be routed to a CMM knowingly, rather than discovered through endless
   Rescans in the field.
4. **Generate the capture link and the printable sheet** for that purchase order. The sheet
   carries a per-order QR code and an orientation arrow in the margin, binding the capture to the
   order.

**Undecided:** when the per-order QR code and orientation arrow actually get added to the printed
sheet. Margin space is reserved for them; the golden-capture set does not need them, since there
is no real purchase order to bind to yet (D-034).

---

## 5 · Capturing a part — the supplier

**Status: Planned.** Built by the capture-lane stories (C-01 through C-15) and the reconstruction
engine (R-01 onward).

1. **Open the capture link** on the phone. No install. You will be asked once for camera
   permission, and once for motion/gyroscope permission — iOS requires that prompt be triggered by
   a tap, over HTTPS.
2. **Place the part on the printed sheet**, ideally near the middle of the pattern, with as much of
   the board still visible around it as the part allows.
3. **Follow the on-screen guidance.** The guaranteed path is **batch mode** (D-002): a fixed orbit
   of roughly 12–16 positions — eight around the part at mid height, the rest higher, including
   one straight down. An arrow on screen points to the next position.
4. **Frames are checked as you shoot**, on the phone, in milliseconds: blur, exposure, and
   duplicate detection. A rejected frame tells you *why* rather than just refusing.
5. **Finish the session.** Reconstruction runs once, on the whole set.

**Planned, and part of what makes this more than a photo app:**
- **An LLM agent runs the session** (A-06, A-07) — it decides what happens next, interprets why a
  frame failed, rephrases it for a non-expert, and adapts if you are clearly struggling. It calls
  the geometric scorer as a tool; it never computes geometry itself. If it errors, times out, or
  loops, capture falls straight back to the deterministic fixed orbit. The session always
  completes.
- **A vision-language model checks sampled frames** (A-01, A-02) — roughly seven calls a session,
  not one per frame. It catches things geometry cannot: the wrong part, hands in shot, a
  photograph of a screen. **It can veto a Pass**, downgrading the session to Rescan, and the
  report names the reason.

---

## 6 · Reference tiers

**Status: Planned** (tier detection). The tier-1 board itself — §3 — **works now**.

Whichever reference is in frame is auto-detected, and the report records which one was used.

| Tier | Reference | What it gives | Error bars |
|---|---|---|---|
| **1** | **ChArUco board** (§3) | Metric camera pose for *every* frame, scale, part orientation, order binding | Narrowest |
| **2** | **Blank A4 / Letter sheet** | Scale from the sheet outline | Wider |
| **3** | **Credit card**, face down | Scale from a small rectangle | Widest |
| — | **Nothing detected** | A model with **no dimensional claims**, and the report says so | — |

**Plain grid paper is rejected on purpose.** Printer scaling plus self-similar cells produce a
confidently wrong scale with no warning — worse than no scale at all.

---

## 7 · Scale integrity

**Status: Planned** (the check). The printed-pitch measurement — §3.3 — **works now**.

Scale has **three independent sources that must agree** (D-007):

1. **The printed board pitch** — what the PDF specifies, checked against what the printer actually
   produced.
2. **An optional caliper reading** of one square, typed in by the supplier. This is **untrusted
   input** and is never the only source.
3. **The CAD registration residual** — the scale error when the model is fitted to the design.

Any two disagreeing by more than about 2% flags the session. That 2% is a rule the team set, not a
measured value.

---

## 8 · Reading the result

**Status: Planned.** The data format is frozen (F-03, `contracts/openapi.yaml`); the code that
produces it is not built.

### 8.1 The model

A 3D model in the browser, coloured by **what was actually seen**:

| Label | Meaning |
|---|---|
| **Observed** | Seen in at least two frames with an adequate viewing angle between them. This is the only geometry that can produce a Pass or a Fail. |
| **Unobserved** | Not properly seen. Drawn grey. Can only ever trigger a Rescan. |
| **Inferred** | Filled in by generative shape completion (**Planned**, A-04). Drawn distinctly, excluded from every deviation calculation, and **never** contributes to a verdict. |

### 8.2 The four verdicts

There are exactly four. No fifth value can be added without a recorded decision, because the hard
rule's guarantees are stated in terms of this closed set.

| Verdict | What it means | What to do |
|---|---|---|
| **Pass** | Every critical region was properly observed, and no observed deviation exceeded the tolerance. | Ship it. |
| **Fail** | An **observed** deviation exceeded the tolerance by a margin larger than the measurement's own uncertainty at that location. | The part is out of spec at a place the camera genuinely saw. |
| **Rescan** | The measurement is not trustworthy *yet* — a critical region was unobserved or too uncertain, the alignment was ambiguous, or the only deviation found lies in a region that was never properly seen. | Shoot it again, covering the named region. Not a judgement on the part. |
| **Unverifiable** | Declared **at intake, before any photograph** — the tolerance asked for is below what this method can resolve for the required reference tier. | Route to a CMM. ProofShape cannot answer this question, and says so up front rather than wasting your time. |

**The distinction that matters most: Fail and Rescan are not the same.** A Fail is a statement
about the part. A Rescan is a statement about the measurement. ProofShape will never convert an
unobserved region into either a Pass or a Fail — if it could not see it, it says so.

### 8.3 The report

One page, shareable with the buyer: the verdict, the deviation heatmap, the model, **which
reference sheet was used**, which regions were observed, and any limits declared at intake.

**Planned:** an LLM writes the narrative summary (A-09).

---

## 9 · When something goes wrong

**Status: Planned** (the runtime behaviour). The error codes themselves are **frozen** in
`contracts/openapi.yaml`, so these are the real names you will see.

| Code | What happened | What to do |
|---|---|---|
| `board_not_detected` | No reference board found in the frame. | Get more of the sheet in shot; check lighting and glare. |
| `board_partially_occluded` | Too little of the board visible to compute a pose. | Move the part toward the centre of the pattern, or shoot from a higher angle. |
| `insufficient_baseline` | Frames too similar to triangulate — you photographed from nearly the same spot. | Move further between shots; follow the orbit. |
| `metric_alignment_failed` | The model could not be locked to the board's metric frame. | Usually a knock-on from poor board visibility across the session. Reshoot. |
| `scale_disagreement` | The independent scale sources disagreed (§7). | Check the printed square size with calipers. Suspect a rescaled printout first. |
| `no_reference_detected` | No reference of any tier in frame. | You get a model with no dimensional claims. Add a reference sheet and reshoot. |
| `frame_rejected_blur` | Motion blur. | Hold steadier; pause before each shot. |
| `frame_rejected_exposure` | Blown highlights or crushed shadows. | Diffuse the light; avoid direct glare on glossy parts. |
| `frame_rejected_duplicate` | Effectively the same frame again. | Move to the next orbit position. |
| `vlm_veto` | The semantic check rejected the session — wrong part, hands in shot, a photo of a screen. | The report names the reason. Fix it and reshoot. |
| `too_few_frames` | Not enough accepted frames to reconstruct. | Keep going around the part. |
| `order_code_unknown` | The order code was not recognised. | Check the code printed on the sheet. |
| `no_cad_uploaded` | A verdict was requested for an order with no CAD. | Expected — no CAD, no verdict (D-004). The model is still valid. |

---

## 10 · What ProofShape will not tell you

**Status: Works now (as a statement of limits).**

- **Anything about surface it did not see.** Not a Pass, not a Fail — only a Rescan.
- **Anything a generative model guessed.** Completed surface is labelled Inferred, drawn
  distinctly, and excluded from every deviation calculation. The model's guess directs the camera;
  it never judges the component.
- **Metrology-grade measurements.** Several millimetres on hand-sized parts, complementing a CMM,
  never replacing one.
- **Any dimensional claim at all, without a reference in frame.**
- **Tolerances below its own noise floor.** Those are declared Unverifiable at intake, on purpose.

---

## Where else to look

| You want | Read |
|---|---|
| What a term means | [`docs/glossary.md`](glossary.md) |
| Why something works this way | [`docs/decisions.md`](decisions.md) |
| The full technical proposal | [`submissions/`](../submissions/) |
| The exact API shapes | [`contracts/openapi.yaml`](../contracts/openapi.yaml) |
| What is built so far | [`stories/README.md`](../stories/README.md), [`docs/progress-log.md`](progress-log.md) |

---

## Maintaining this manual

Flip a section from **Planned** to **Works now** in the same pull request that makes it true — not
afterwards. A manual that describes more than the system does is worse than no manual, because it
is the one document a reader has no way to check against reality.

When a section's behaviour is set by a decision, cite the decision number rather than restating its
reasoning — the reasoning belongs in `docs/decisions.md` and drifts if copied.

**Never write a measured figure here that has not been measured.** Accuracy, tolerance, noise
floor, latency: if the number does not exist yet, say it does not exist yet and name the milestone
that will produce it.

# ProofShape — operating manual

How to use ProofShape: what you need, how to set it up, how to capture a part, and how to read
what comes back. Written for the two people who actually touch the system — the **buyer** setting
up an order, and the **supplier** holding the phone — plus anyone on the team who needs the
physical procedure.

**This is a living draft.** Parts of the system are built and parts are not, and this manual says
which, section by section, rather than describing a product that does not exist. Every section
carries a status:

| Status | Meaning |
|---|---|
| **Works now** | You can do this today, on `main`, exactly as written. |
| **In review** | Built in an open pull request, which is named. Not on `main` yet, so not usable yet. It flips to **Works now** in the pull request that merges it. |
| **Planned** | Designed and specified, not built. The story that will build it is named. |
| **Undecided** | Genuinely open. Do not assume a behaviour here. |

**Where things stand (checked against `main` on 2026-10-17).** The supplier side mostly works: you
can print the board, open the capture app in a phone browser, shoot a guided orbit with on-phone
frame checks, finish the session, and look at a 3D model coloured by what the camera actually saw
(§5, §8.1). The reconstruction service that turns photos into that model is built and has been
deployed on a cloud GPU (§0). What is **not** on `main` yet is the link between the two: today the
phone app talks only to a mock service that returns a canned model, and pointing it at the real
service is **In review** (C-13, PR #146). Nothing on the buyer side — CAD upload, tolerances,
verdicts, the report — is built yet (§4, §8.2, §8.3).

---

## 0 · Trying it today (team)

**Status: Works now**, with the limits stated in each line.

This section is for the team, not for a buyer or supplier. Setup details live in each part's own
README; this is only which one to reach for.

| You want | Do this | Where |
|---|---|---|
| The capture app on a real phone | `npm run dev:device` in `capture/`, then open the printed `https://<LAN-IP>:<port>` on a phone on the same Wi-Fi and accept the self-signed certificate once. It answers from a **mock service** with a canned model; enter the order code `PO-48213-A`. | [`capture/README.md`](../capture/README.md) |
| Real photos through the real reconstruction service | `uvicorn recon.service:app` on a machine with a CUDA GPU, then drive a capture folder through it with `scripts/smoke_r14.py`. | [`recon/README.md`](../recon/README.md), R-14 |
| The service on the shared cloud GPU | R-15's runbook: start, check TLS and the token, smoke-test, **stop**. Stopping is the only thing that ends billing. | [`recon/README.md`](../recon/README.md), R-15 |
| A real phone session against the real service | **In review** (C-13, PR #146): the dev server proxies `/v1` to the service running natively on the M2 Max (D-049). | PR #146 |

---

## 1 · What ProofShape does, and what it does not

**Status: Works now (as a statement of scope).**

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
  the calibration study (milestone M3), which has not run yet. Until then, treat every figure
  anywhere in this project as a planning estimate.
- **Geometry the camera did not properly see can never pass or fail a part.** It may only trigger
  a Rescan. This is the hard rule (D-005), and it is enforced structurally — every vertex of the
  model carries an Observed/Unobserved label (§8.1), and the data format itself forbids an
  unobserved region from contributing to a verdict, with a test that fails the build if a report
  ever violates it.
- **No CAD file, no verdict** (D-004). Without a design to compare against, ProofShape returns a
  measured model and nothing more. That is a legitimate output, just not an inspection.

---

## 2 · What you need

**Status: Works now.**

| Item | Notes |
|---|---|
| **Printed ChArUco reference sheet** | One sheet of A4 or US Letter. See §3 — printing it correctly matters more than it sounds. |
| **The part** | Rigid and opaque. Roughly hand-sized — comparable to, or a little larger than, the printed pattern. See §3.4 on parts larger than the board, and §10 on transparent parts. |
| **A smartphone** | With a working camera and gyroscope. No app install — capture runs in the browser over HTTPS. So far it has been checked on iPhones in Safari only; Android is untested (**Planned**, C-18). |
| **Digital calipers** *(recommended)* | For verifying the printed square size, and for the optional typed scale check (§7). Roughly $25. |
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

**If the part is larger than the pattern but still roughly sheet-sized, that is fine for pose.**
The board's corners are individually identifiable, so as long as enough of them stay visible in a
frame, that frame's camera pose can still be computed. A frame where the board can't be found
simply doesn't count, and a session with too few usable board views fails with a named reason
(§9) rather than silently producing a model. There is no on-phone check for board visibility yet;
the phone only checks blur, exposure and duplicates (§5).

**But keep as much of the board visible around the part as you can.** Segmentation measures the
real board surface from the board points *around* the part (R-21), and in the first real phone
sessions most failures happened exactly there (§9). A part that hides most of the pattern makes
that more likely.

**If the part is dramatically larger than the whole sheet, ProofShape is the wrong tool for it.**
The stated scope is hand-sized parts. Supporting genuinely large parts would need a physically
larger board (bigger paper, or larger squares at the cost of marker headroom), which is not a
decision the team has made. **Undecided.**

---

## 4 · Setting up an order — the buyer

**Status: Planned.** Built by the capture-lane stories (C-16 buyer setup, C-17 board PDF with
order code) and the inspection stories (I-01 through I-10).

The intended flow:

1. **Upload the CAD** (STL/OBJ/GLB).
2. **Enter what matters** — the tolerance, and the critical features by name ("the four mounting
   holes", "the sealing face"). **Planned:** you will also be able to write this in plain English
   and have it converted into the structured specification (A-03, prose-to-spec parser).
3. **Intake pre-check runs** (I-09). Any feature whose tolerance is tighter than the method's own
   noise floor for the required reference tier is flagged **Unverifiable before any photograph is
   taken** (I-10) — so it can be routed to a CMM knowingly, rather than discovered through endless
   Rescans in the field.
4. **Generate the capture link and the printable sheet** for that purchase order. The sheet
   carries a per-order QR code and an orientation arrow in the margin, binding the capture to the
   order.

**What exists today instead.** There is no buyer side and no order storage. The supplier types an
order code by hand (§5). The mock service accepts only `PO-48213-A`; the real reconstruction
service accepts **any** non-empty code, because there is no buyer database to check it against
(`recon/README.md`, R-14). A capture link that pre-fills the order code (`?order=`) is **In review**
(C-13, PR #146).

**Undecided:** when the per-order QR code and orientation arrow actually get added to the printed
sheet. Margin space is reserved for them; the golden-capture set does not need them, since there
is no real purchase order to bind to yet (D-034).

---

## 5 · Capturing a part — the supplier

**Status: Works now**, against the mock service (§0). Against the real reconstruction service:
**In review** (C-13, PR #146). Built by C-01 through C-12.

1. **Open the capture page** on the phone. No install. Enter the order code and tap **Start**.
   The phone then asks for camera permission and for motion/gyroscope permission, both from that
   one tap — iOS only grants them from a direct tap, over HTTPS (C-02, C-03, C-04). If you deny
   one and change your mind, iOS remembers the answer per site: clear the site's data in
   Settings → Safari → Advanced → Website Data, then reload.
2. **Place the part on the printed sheet**, near the middle of the pattern, with as much of the
   board still visible around it as the part allows (§3.4). Hold the phone in **portrait**; the
   guidance assumes it.
3. **Shoot the guided orbit.** The camera preview stays live (C-05). Tap **Capture frame** for
   each shot. The guaranteed path is **batch mode** (D-002): a fixed orbit of **13 positions** —
   eight around the part at 45°, four higher at 70°, and one straight down (C-07). Your first
   accepted frame sets which way is "around"; after that, an arrow points toward the next
   uncovered position and a counter shows **Covered: n/13**.
4. **Each frame is checked on the phone before it is uploaded** (C-06, C-09): blur, exposure
   (clipped highlights or shadows) and duplicates (the phone barely moved *and* the picture barely
   changed). A rejected frame shows the reason and its code (§9), nothing is uploaded, and the
   counter does not move. An accepted frame is uploaded with its gyro reading, and its orbit
   position only counts as covered once the service accepts it.
5. **Tap Finish.** The camera turns off (the phone's green camera indicator goes away), and
   reconstruction runs once, on the whole set. The page waits for the result and shows it (§8.1).
   The real service needs at least 3 accepted frames; the full orbit is what the guidance is for.

**Known limits of capture today:**
- **Frames are captured at the preview's size, 480 × 640 on the iPhones tried so far**, because
  the app asks the browser for no particular resolution (`capture/src/camera.ts`). That is well
  below what the camera can do; reconstruction works at that size, but detail is lost.
- **Frames carry no EXIF focal length**, because they are taken from the live preview rather than
  as photos. On `main`, the real service can't pose such frames and fails with a misleading
  `insufficient_baseline` (§9). Calibrating the camera from the board itself instead is
  **In review** (D-050, PR #146).
- **Next-view scoring** — ranking where one more photo would see the most unobserved surface —
  exists on `main` (C-08, `recon/next_view.py`), but only as a command-line tool run on a
  finished model. Nothing calls it yet: not the service, and not the arrow, which follows the
  fixed orbit.

**Planned, and part of what makes this more than a photo app:**
- **An LLM agent runs the session** (A-06, A-07, with A-08's deterministic fallback) — it decides
  what happens next, interprets why a frame failed, rephrases it for a non-expert, and adapts if
  you are clearly struggling. It calls the geometric scorer as a tool; it never computes geometry
  itself. If it errors, times out, or loops, capture falls straight back to the deterministic
  fixed orbit. The session always completes.
- **A vision-language model checks sampled frames** (A-01, A-02) — roughly seven calls a session,
  not one per frame. It catches things geometry cannot: the wrong part, hands in shot, a
  photograph of a screen. **It can veto a Pass**, downgrading the session to Rescan, and the
  report names the reason.

---

## 6 · Reference tiers

**Status: Planned** (tiers 2 and 3, and tier detection). Tier 1, the ChArUco board (§3), **works
now** and is the only tier the service handles.

Whichever reference is in frame is auto-detected, and the report records which one was used.

| Tier | Reference | What it gives | Error bars |
|---|---|---|---|
| **1** | **ChArUco board** (§3) | Metric camera pose for *every* frame, scale, part orientation, order binding | Narrowest |
| **2** | **Blank A4 / Letter sheet** | Scale from the sheet outline | Wider |
| **3** | **Credit card**, face down | Scale from a small rectangle | Widest |
| — | **Nothing detected** | A model with **no dimensional claims**, and the report says so | — |

**What happens today without a board:** the session fails with `board_not_detected` (§9) rather
than returning a model with no dimensional claims. When a result *is* non-metric, the result screen
already says "No dimensional claims" (C-10).

**Plain grid paper is rejected on purpose.** Printer scaling plus self-similar cells produce a
confidently wrong scale with no warning — worse than no scale at all.

---

## 7 · Scale integrity

**Status: Planned** (in the service). The comparison itself exists (R-05, D-039), but nothing feeds
it yet: there is no caliper entry in the app and no CAD registration, so results report no scale
check. The printed-pitch measurement — §3.3 — **works now**.

Scale has **three independent sources that must agree** (D-007):

1. **The printed board pitch** — what the PDF specifies, checked against what the printer actually
   produced.
2. **An optional caliper reading** of one square, typed in by the supplier. This is **untrusted
   input** and is never the only source.
3. **The CAD registration residual** — the scale error when the model is fitted to the design.

Any two disagreeing by more than about 2% flags the session. That 2% is a rule the team set, not a
measured value.

**A known scale error today, measured, not yet fixed.** Using the board itself as a ruler inside
the reconstruction showed every golden object coming out **4–9% small** (R-22). The cause is in how
the model is aligned to the board, and the fix is R-23. Until it lands, **do not read dimensions
off a model**, even a metric one.

---

## 8 · Reading the result

### 8.1 The model

**Status: Works now** (C-10, C-11, C-12), against the mock service; with a real reconstruction,
**In review** (C-13, PR #146).

When reconstruction finishes, the page shows a short summary — the reference tier used, whether the
result is metric, the **observed fraction**, and any warnings — and then the model, in a 3D viewer
you can orbit with one finger and zoom with two. A **Download GLB** link is always there, so the
raw model stays retrievable even if the viewer can't render it. If the result is not metric, the
summary says "No dimensional claims".

The model is coloured by **what was actually seen**:

| Colour | Label | Meaning |
|---|---|---|
| **Blue** | **Observed** | Seen in at least two frames whose viewpoints are at least 2° apart, through pixels the reconstruction model is confident about (R-24, D-045), and not hidden behind other surface in those frames: its position agrees with the depth each frame actually measured (R-09). The spread of that depth across the frames (σ) must also be 3 mm or less, or it is downgraded to Unobserved (R-10). This is the only geometry that can produce a Pass or a Fail. |
| **Slate grey** | **Unobserved** | Not properly seen. Can only ever trigger a Rescan. |
| **Orange** | **Inferred** | Reserved: filled in by generative shape completion (**Planned**, A-04). Will be excluded from every deviation calculation and will **never** contribute to a verdict. Nothing produces it yet. |
| **Magenta** | **Unknown** | A colour the viewer did not expect. Should never appear; if it does, the viewer also shows a warning. |

The 2° and 3 mm limits are the code's current defaults, chosen by the team, not measured values —
the calibration study will set real ones.

**Expect a lot of slate grey on today's models.** Fusion still leaves wall-like artifact surfaces
around the part (R-18, R-20). They come out Unobserved, which is the hard rule working, but they
make the observed fraction low: about 10–40% on the golden objects (R-24). The part itself is
usually the blue region.

### 8.2 The four verdicts

**Status: Planned** (I-07 verdict logic, I-08 unobserved-geometry rule, I-10 Unverifiable path).
The four values are frozen in the contract; today a verdict request always answers
`no_cad_uploaded`, because there is no way to upload a CAD yet.

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

**Status: Planned** (C-14 heatmap rendering, C-15 report page).

One page, shareable with the buyer: the verdict, the deviation heatmap, the model, **which
reference sheet was used**, which regions were observed, and any limits declared at intake.

**Planned:** an LLM writes the narrative summary (A-09).

---

## 9 · When something goes wrong

**Status:** each row says whether you can actually get that code today. The codes themselves are
**frozen** in `contracts/openapi.yaml`, so these are the real names you will see.

**On the phone, before anything uploads (works now):**

| Code | What happened | What to do |
|---|---|---|
| `frame_rejected_blur` | Motion blur or soft focus. | Hold steadier; pause before each shot. |
| `frame_rejected_exposure` | Too many pixels are pure black or pure white. The service also uses this code for an empty upload. | Diffuse the light; avoid direct glare on glossy parts. |
| `frame_rejected_duplicate` | You barely moved and the picture barely changed. | Move to the next orbit position. |

**From the service, when you start, upload or finish (works now):**

| Code | What happened | What to do |
|---|---|---|
| `order_code_unknown` | The order code was not recognised. Today the mock service accepts only `PO-48213-A`, and the real service rejects only an empty or blank code, because there is no buyer database to check against yet (§4). | Check the code. |
| `too_few_frames` | Not enough accepted frames to reconstruct. The real service needs at least 3. | Keep going around the part. |
| `session_not_found` | The service has no session with that id, for example because it was restarted without its saved sessions. | Start a new session. |
| `session_already_finished` | The session has already been finished, so it takes no more frames and can't be finished again. | The first Finish still stands, and its reconstruction runs. To add frames, start a new session. |
| `vlm_veto` | **Today this means a malformed upload, not a semantic check** — the VLM is not built (A-01, A-02). The contract has no dedicated code for a malformed request yet, and adding one goes through the contract-change process (D-031). Once the VLM exists, this will mean the semantic check rejected the session: wrong part, hands in shot, a photo of a screen. | Reshoot; if it repeats, tell the team, since it points at an app bug. |

**From the reconstruction, after you finish (works now):**

| Code | What happened | What to do |
|---|---|---|
| `board_not_detected` | No reference board found in enough frames. | Get more of the sheet in shot; check lighting and glare. |
| `board_partially_occluded` | Too little of the board visible to compute poses. | Move the part toward the centre of the pattern, or shoot from a higher angle. |
| `insufficient_baseline` | Fewer than 3 frames could be posed from the board. On `main` this is also what frames with **no EXIF focal length** produce — every phone-app frame (§5) — even though the board was found. That case gets its own message and a board-calibration fallback in PR #146 (D-050). | Move further between shots and follow the orbit. If the board was clearly visible, suspect the EXIF case. |
| `metric_alignment_failed` | The model could not be locked to the board's metric frame. **If the message starts "Fusion failed:"**, the board surface around the part could not be measured: too few board points near it, or the fitted board tilted more than 2° (R-21). It was the most common failure in the first real phone sessions (C-13, PR #146), and why phone frames hit it more than the golden photos is not yet known. | Lay the sheet flat, keep as much board visible around the part as possible (§3.4), and shoot the full orbit. |
| `no_cad_uploaded` | A verdict was requested for an order with no CAD. Today that is every order (§8.2). | Expected — no CAD, no verdict (D-004). The model is still valid. |

**Not produced yet** (frozen in the contract, nothing raises them):

| Code | Will mean | What to do |
|---|---|---|
| `scale_disagreement` | The independent scale sources disagreed (§7). | Check the printed square size with calipers. Suspect a rescaled printout first. |
| `no_reference_detected` | No reference of any tier in frame (§6). | You get a model with no dimensional claims. Add a reference sheet and reshoot. |

**Every failure names its reason.** If something fails without one, that is a bug — report it
rather than reshooting blindly.

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

**And, today specifically:**

- **Trustworthy dimensions.** Models currently come out 4–9% small (§7, R-22) until R-23 lands.
- **A clean surface.** Models still carry fusion artifacts around the part (§8.1). They are labelled
  Unobserved, so they cannot affect a verdict, but they are there.
- **Transparent parts.** A clear glass object was not reconstructed, and the board showed through
  it (C-13's phone sessions, PR #146). This is a known weakness of image-based reconstruction, not
  a bug to fix.

---

## Where else to look

| You want | Read |
|---|---|
| How to run the app or the service | [`capture/README.md`](../capture/README.md), [`recon/README.md`](../recon/README.md) |
| What a term means | [`docs/glossary.md`](glossary.md) |
| Why something works this way | [`docs/decisions.md`](decisions.md) |
| The full technical proposal | [`submissions/`](../submissions/) |
| The exact API shapes | [`contracts/openapi.yaml`](../contracts/openapi.yaml) |
| What is built so far | [`stories/README.md`](../stories/README.md), [`docs/progress-log.md`](progress-log.md) |

---

## Maintaining this manual

Flip a section from **Planned** or **In review** to **Works now** in the same pull request that
makes it true — not afterwards. A manual that describes more than the system does is worse than no
manual, because it is the one document a reader has no way to check against reality.

When a section's behaviour is set by a decision, cite the decision number rather than restating its
reasoning — the reasoning belongs in `docs/decisions.md` and drifts if copied.

**Never write a measured figure here that has not been measured.** Accuracy, tolerance, noise
floor, latency: if the number does not exist yet, say it does not exist yet and name the milestone
that will produce it.

*Last checked against `main` on 2026-10-17. The catch-up pass that started on 2026-10-16 found
nothing here updated since 2026-09-25, while most of the capture app and the reconstruction
engine had landed.*

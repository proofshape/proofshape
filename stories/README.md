# ProofShape Story Backlog

**CS595 Capstone · Fall 2026**
Companion to `../docs/sprint-plan.md`. The plan says when and how much; this says what, in pieces one person can finish and another can review.

---

## Quick primer — read this once

**A story is one pull request.** One or two work sessions, a few hundred lines at most. If it
grows past that, split it.

**The ID says where the work lives, not who does it.**

| Prefix | Area |
|---|---|
| **F** | Foundations — repo, CI, environment, fixtures |
| **R** | Reconstruction engine — the backend |
| **C** | Capture app — the phone front end |
| **I** | Inspection — CAD registration, verdict logic |
| **G** | Ground truth — printing, calipers, calibration |
| **A** | Generative AI — agent, completion, VLM, spec parser |
| **X** | Cross-cutting — integration points, demo |
| **P** | Planning — proposal, sprint plan, backlog, process rules, documents, reviews and reports |

The number is just the order we wrote them down. **`R-04` is not more important than `R-12`, and
it is not "step 4"** — read the **Depends on** line to know what actually has to come first.

**Working a story, start to finish:**

1. Find one marked **Ready** with Owner `_unclaimed_` in the tables below. Ready means nothing
   is blocking it; an owner means someone has it.
2. **Claim it in two places, in one commit, before writing any code** (D-037):
   - in the story file, set **State: Claimed** and put your GitHub handle on the **Owner** line;
   - in this index, set that story's **State** cell to `Claimed` and its **Owner** cell to the
     same handle, written exactly the same way.

   The owner is always the **person** who will answer for the work, as an `@github-handle`
   (two people: `@a; @b`). **If an AI assistant makes the claim for you, it writes your handle,
   never its own name.** It asks you if it can't tell whose handle to use; `gh api user --jq
   .login` gives the account the GitHub CLI is logged in as. **Get the claim onto `main` right
   away**, as its own small pull request (see `chore/claim-r-01` for the pattern) — a claim that
   only exists on an unmerged branch is invisible to everyone else, which is exactly what "so
   nobody duplicates your work" is supposed to prevent. Then run
   `python3 scripts/check_story_states.py`: it fails if the two places disagree, if a Claimed
   story has no owner, or if the owner isn't a person's handle.
3. Branch as `<area>/<id>-short-name`, for example `recon/R-04-metric-alignment`.
4. Build it. Write notes in the story file as you go — what surprised you, what you chose and
   why. That section is for the next person, including future you.
5. Open a pull request. Either of the other two reviews it.
6. **Push one more commit to the PR — before merging, not after (D-038):** set **State: Done**
   in the story file and its index row here, fill in the **Completion record** (who, date, PR
   number, and **actual hours against the estimate**, never the estimate copied in), and run
   `python3 scripts/check_story_states.py --fix`, committing whatever it changes. **Push it
   before the approval you intend to merge on, not after** — `main` dismisses a stale approval
   on every new commit (D-013), so this commit needs to be part of what gets reviewed, not
   tacked on afterward. Only merge once it's in and approved, with nothing pushed after it. The
   actuals are how our estimates stop being wrong; the state change is how the index stays
   trustworthy for the next person picking work.
7. After merging, append an entry to `docs/progress-log.md`.

**If you get stuck, say so.** Someone joining to unblock you is expected and gets noted. Two
people quietly working the same story is the thing to avoid.

---

## How this works

**Backend first.** The reconstruction engine — photos in, metric 3D model out — is built before the other layers. Once it works we have certainty about the hardest part of the project, and everything downstream can be built against something real instead of a guess.

**Lanes are places, not people.** `/recon`, `/capture` and `/inspect` are directories in the repository. Any member can pick up any story in any of them. This is what lets all three of us push on the engine at once.

**One person per story.** Nobody works on a story someone else has claimed. If you are stuck, ask — a second person joining to unblock you is fine and expected, and gets noted in the record. What is not fine is two people quietly working the same thing.

**One person writes, either of the other two reviews.** No fixed reviewer per area. Whoever has capacity picks it up.

**Each story has its own file** under this folder. That file is the source of truth for its
detail: acceptance criteria, working notes, decisions taken, and the completion record. This
index keeps one row per story so it stays scannable, and so two people editing different
stories do not collide in the same file. Each row repeats two facts from its story file, the
**State** and the **Owner**, so anyone can see what's free and who holds what without opening
every file. The two copies must always match, and `scripts/check_story_states.py` checks that
they do.

### Story states

| State | Meaning |
|---|---|
| **Blocked** | A dependency is not finished. Do not start it. Owner is `_unclaimed_`. |
| **Ready** | Dependencies are done and nobody has claimed it. Anyone can take it. Owner is `_unclaimed_`. |
| **Claimed** | Someone has put their `@github-handle` on it, in the story file and in this index, before starting. |
| **In review** | Pull request open, waiting on one approval. |
| **Done** | Merged, and the completion record below is filled in. P stories have one exception here, stated in `AGENTS.md`'s definition of done (D-052). |

### Completion record

Fill this into the story when it merges. Actual hours matter: they are how our estimates get less wrong.

```
Completed: @member · 2026-09-08 · PR #12 · 4.5 h actual (est 4)
Notes: forced fp16 — bf16 runs at half speed on Apple Silicon.
```

### Definition of done

Applies to every story, unchanged from the plan: merged to `main`, runs on a second member's machine from a clean checkout, has one automated test or one written manual check, reviewed and approved by one other member, and demonstrated at the sprint boundary.

Acceptance criteria are per story. The definition of done is not, apart from two exceptions for
P stories (the planning lane), set by D-052 and stated in `AGENTS.md`'s definition of done.

---

## Phase 1 · Foundations

Twenty-three hours. Everything else is blocked on some part of this, so it goes first and it goes fast. **F-01, F-05 and F-06 have no dependencies — those are the three to start on day one, one each.**

Full detail in each story file.

| ID | Story | Est | Depends on | State | Owner |
|---|---|---|---|---|---|
| [F-01](F/F-01.md) | Public repository with branch protection | 2 h | nothing | Done | @TabeenRaoof |
| [F-02](F/F-02.md) | Repository skeleton and module boundaries | 3 h | F-01 | Done | @dalwalyk |
| [F-03](F/F-03.md) | Interface contract v1, frozen | 3 h | F-01 | Done | @TabeenRaoof |
| [F-04](F/F-04.md) | CI skeleton | 3 h | F-02 | Done | @dalwalyk |
| [F-05](F/F-05.md) | Shared GPU workspace provisioned, three-way access | 4 h | nothing | Done | @mbj1994 |
| [F-06](F/F-06.md) | Golden capture set | 5 h | nothing | Done | @mbj1994; @TabeenRaoof |
| [F-07](F/F-07.md) | Reproducible dev environment | 3 h | F-02, F-05 | Done | @dalwalyk |

---

## Phase 2 · The reconstruction engine

Seventy hours, and the reason we are doing backend first. Exit criterion for the whole phase: **the golden capture goes in, and a metric, segmented mesh with per-vertex provenance comes out, reproducibly, with per-stage timings recorded.**

**Parallelism.** R-01, R-02, R-11, R-12 and R-15 have no dependency on each other, so three people can work simultaneously from the start. The chain narrows at R-04, which needs both R-01 and R-02 — expect that to be the bottleneck and plan to have two people free when it lands.

Full detail in each story file.

| ID | Story | Est | Depends on | State | Owner |
|---|---|---|---|---|---|
| [R-01](R/R-01.md) | Run the reconstruction model on the golden capture | 5 h | F-05, F-06, F-07 | Done | @dalwalyk |
| [R-02](R/R-02.md) | Board detection and per-frame camera pose | 5 h | F-06, F-07 | Done | @TabeenRaoof |
| [R-03](R/R-03.md) | Camera intrinsics from the board | 4 h | R-02 | Done | @mbj1994 |
| [R-04](R/R-04.md) | Metric alignment | 6 h | R-01, R-02 | Done | @dalwalyk |
| [R-05](R/R-05.md) | Scale agreement check | 4 h | R-04 | Done | @dalwalyk |
| [R-06](R/R-06.md) | Segment the part from the board | 4 h | R-04 | Done | @dalwalyk |
| [R-07](R/R-07.md) | TSDF fusion to a mesh | 6 h | R-04, R-06 | Done | @dalwalyk |
| [R-08](R/R-08.md) | Mesh cleanup and GLB export | 4 h | R-07 | Done | @dalwalyk |
| [R-09](R/R-09.md) | Per-vertex provenance | 5 h | R-07 | Done | @dalwalyk |
| [R-10](R/R-10.md) | Per-vertex uncertainty | 5 h | R-09 | Done | @dalwalyk |
| [R-11](R/R-11.md) | MASt3R behind the config flag | 5 h | R-01 | Done | @mbj1994 |
| [R-12](R/R-12.md) | COLMAP behind the config flag | 5 h | R-01 | Done | @mbj1994 |
| [R-13](R/R-13.md) | Per-stage latency instrumentation | 3 h | R-07 | Done | @TabeenRaoof |
| [R-14](R/R-14.md) | Reconstruction endpoint | 5 h | F-03, R-08 | Done | @mbj1994 |
| [R-15](R/R-15.md) | Container image and deploy | 4 h | R-14, F-05 | Done | @TabeenRaoof |
| [R-16](R/R-16.md) | Known-size tolerance validation | 2 h | R-04, G-03, G-04 | Blocked | _unclaimed_ |
| [R-17](R/R-17.md) | Visually verify part segmentation on the golden set | 1 h | R-06 | Done | @TabeenRaoof |
| [R-18](R/R-18.md) | Produce and check TSDF-fused meshes on the golden set | 1 h | R-07 | Done | @dalwalyk |
| [R-19](R/R-19.md) | Report per-vertex sigma distribution on the golden set | 1 h | R-10 | Done | @mbj1994 |
| [R-20](R/R-20.md) | Use VGGT's confidence in the point cloud and in fusion | 4 h | R-01, R-07 | Done | @TabeenRaoof |
| [R-21](R/R-21.md) | Measure the height cut from the real board surface | 3 h | R-06 | Done | @TabeenRaoof |
| [R-22](R/R-22.md) | Find why parts reconstruct smaller than their CAD | 3 h | R-03, R-04 | Done | @TabeenRaoof |
| [R-23](R/R-23.md) | Anchor R-04's metric scale on the board in the scene | 5 h | R-04, R-22 | Done | @mbj1994; @TabeenRaoof |
| [R-24](R/R-24.md) | Use VGGT's confidence in provenance's observation count | 3 h | R-20 | Done | @TabeenRaoof |

---

## Thin capture thread

The one slice of Phase 3 written up early, on purpose — the sprint plan calls it out separately
from the rest of the capture app because it's meant to run **alongside Phases 1–2, in spare
time**, not wait for them. It doesn't touch the chain (`R-01` → `R-15`) at all. Three sprints'
worth of allocated capture hours (S0, S1, S2) have already passed with nothing built here yet,
so this is behind schedule, not early.

| ID | Story | Est | Depends on | State | Owner |
|---|---|---|---|---|---|
| [C-01](C/C-01.md) | Minimal upload page | 3 h | F-02, F-03 | Done | @TabeenRaoof |
| [C-02](C/C-02.md) | Camera permissions on a real iPhone | 2 h | C-01 | Done | @TabeenRaoof |
| [C-03](C/C-03.md) | Gyro permission flow | 2 h | C-01 | Done | @dalwalyk |
| [C-04](C/C-04.md) | App shell | 3 h | C-01, C-02, C-03 | Done | @dalwalyk |
| [C-05](C/C-05.md) | Live camera | 3 h | C-04 | Done | @dalwalyk |
| [C-06](C/C-06.md) | Frame gate | 3 h | nothing | Done | @dalwalyk |
| [C-07](C/C-07.md) | Gyro overlay and arrow | 3 h | C-03 | Done | @dalwalyk |
| [C-08](C/C-08.md) | Next-view scoring | 4 h | R-09, R-10 | Done | @TabeenRaoof |
| [C-09](C/C-09.md) | Capture loop | 4 h | C-05, C-06, C-07 | Done | @dalwalyk |
| [C-10](C/C-10.md) | Reconstruction result and the mock service | 3 h | C-01 | Done | @mbj1994 |
| [C-11](C/C-11.md) | Model viewer | 3 h | C-10 | Done | @mbj1994 |
| [C-12](C/C-12.md) | Observed/unobserved colouring | 2 h | C-11 | Done | @mbj1994 |
| [C-13](C/C-13.md) | Connect capture to the deployed service | 3 h | C-09, C-11, R-15 | Done | @dalwalyk; @TabeenRaoof |

---

## Ground truth

The second slice of Phase 3 written up early, because σ_floor is the next step on the critical
path: "provenance and σ → σ_floor measured → verdict logic → accuracy runs → freeze"
(`docs/sprint-plan.md` §3). Provenance and σ are done (R-09, R-10). Every tolerance the verdict
uses waits on this lane, and so does R-16. Printing is on the SFBU lab printer (G-01).

**The measured print is truth, not the CAD.** FDM drifts by a few tenths of a millimetre
(proposal §12), so every comparison is against caliper readings of the actual printed part, never
against the CAD's nominal sizes.

**How every story here measures:**
- **Method first.** Write the method down before measuring, then report what was measured,
  including a negative result. No tolerance and no `k` without a measurement behind it (D-003;
  R-04's reverted tolerance is the precedent).
- **Calipers, PR #35's protocol, matched to `measure_prism`.** Name each feature by its CAD vertex
  pair. Read at mid-height, away from the first-layer flare, because `measure_prism` fits edges
  from the 0.3–0.7 h band. Read across parallel faces for rectangles and corner to corner for
  triangles. For a square footprint, record which way each wall faced on the board. Take 10
  placements and give the mean, min, max and standard deviation. State the resolution, and say
  "no accuracy spec" where the instrument has none.
- **Where readings go.** One file per story, `fixtures/caliper_<id>.json`, in the format in
  `fixtures/README.md`. `tests/test_caliper_files.py` (G-06) checks every such file; a readings
  file that lands before G-06 is checked against that format by hand, and the test covers it
  from then on. Photographs are never committed.
- **Where code goes.** `recon/cad_size_check.py` is extended rather than `inspect/` started, since
  D-030 leaves `inspect/`'s import name to I-01.
- **Naming.** Say "the board reference", "the sheet reference", "the card reference", never
  "tier 1/2/3": *Tier* also means how essential a feature is (D-001), and the two collide. The
  glossary's Reference sheet entry gives both names.

Full detail in each story file.

| ID | Story | Est | Depends on | State | Owner |
|---|---|---|---|---|---|
| [G-01](G/G-01.md) | Book SFBU lab printer time and characterise the printer with a calibration cube | 3 h | nothing | Claimed | @dalwalyk |
| [G-02](G/G-02.md) | Specimen batch: 5–8 prisms the size check can measure | 4 h | G-01 | Blocked | _unclaimed_ |
| [G-03](G/G-03.md) | Caliper every specimen | 3 h | G-02 | Blocked | _unclaimed_ |
| [G-04](G/G-04.md) | Error and σ_floor for the board reference, measured against caliper truth | 6 h | G-03, G-06, G-08, R-23 | Blocked | _unclaimed_ |
| [G-05](G/G-05.md) | Defect ladder at 2 / 4 / 8 mm | 4 h | G-01 | Blocked | _unclaimed_ |
| [G-06](G/G-06.md) | Caliper the golden-set parts and the printed board | 1.5 h | nothing | Done | @TabeenRaoof |
| [G-07](G/G-07.md) | Reliability curve: does per-vertex σ predict measured error | 6 h | G-04 | Blocked | _unclaimed_ |
| [G-08](G/G-08.md) | Photograph the specimen batch as a fixture set | 4 h | G-02, G-06 | Blocked | _unclaimed_ |
| [G-09](G/G-09.md) | Photograph the defect ladder | 1.5 h | G-05, G-08 | Blocked | _unclaimed_ |

---

## Inspection

The third slice of Phase 3 written up early. The sprint plan's S4 "Inspection" sprint starts on
2026-10-19 and ends on Oct 30 with two gates: I3, "verdict proven end to end", and M4, "planted
4 mm hole shift → Fail; unobserved region → Rescan; substituted part caught on scale or shape".
Both need registration first. These five don't wait on ground truth: the sprint plan built the synthetic mesh generator
(§4 item 3) so that registration and verdict tests run on every pull request with no GPU and no
fixtures.

**The method is settled, not open** (proposal §7.1, D-008): FPFH + RANSAC global registration,
then robust point-to-plane ICP with a Tukey loss; scale fixed to 1 in the fit, the free-scale
residual computed separately as the substitution check; a runner-up within 10% of the best marks
the session orientation-ambiguous. Open3D 0.18 (already pinned) has all of it.

**No tolerance until G-04.** Nothing here decides whether a deviation passes. Every threshold (σ,
patch size, voxel size, Tukey `k`, the 10%) is a declared parameter with a stated default, and
the verdict (I-06..I-10) waits for σ_floor and the tolerance schema (D-032). Issue #157 already
binds I-07 to `max(σ, σ_floor)`.

**Where the code goes.** D-053 settles the import name D-030 left open: the directory stays
`inspect/`, the importable package is `inspection`, at `inspect/inspection/`. I-01 puts it in
place.

These five total 19 h against roughly 15 h implied by the plan's S1 and S2 inspection bullets once
printing and the calibration cube moved to G-01/G-02; the extra is densifying the tiny fixture
CADs and the packaging decision.

Full detail in each story file.

| ID | Story | Est | Depends on | State | Owner |
|---|---|---|---|---|---|
| [I-01](I/I-01.md) | Synthetic mesh generator, and the `inspection` package | 4 h | nothing | Done | @dalwalyk |
| [I-02](I/I-02.md) | CAD loading: ASCII and binary STL, and GLB, coordinates as stored | 3 h | nothing | Done | @mbj1994 |
| [I-03](I/I-03.md) | Global registration: candidate poses, scale fixed, free-scale residual | 5 h | I-01, I-02 | Claimed | @mbj1994 |
| [I-04](I/I-04.md) | Robust point-to-plane ICP with a Tukey loss | 4 h | I-03 | Blocked | _unclaimed_ |
| [I-05](I/I-05.md) | Orientation ambiguity: refine every candidate, flag a runner-up within 10% | 3 h | I-04 | Blocked | _unclaimed_ |
| [I-11](I/I-11.md) | CAD loading: OBJ, extending I-02's loader | 2 h | I-02 | Done | @dalwalyk |

---

## Generative AI

The fourth slice of Phase 3 written up early, because it is overdue. The course requires about a
third of the project to be generative AI, and those components are never cut (AGENTS.md, D-017).
VLM supervision was due in S2 (its M2 gate, "a screen photo is rejected by the VLM", was missed)
and shape completion in S3; none of it is built. These are the first stories that can start
without the verdict, plus the agent's tools (A-06) and its deterministic fallback (A-08), which
the proposal says is built first. A-02 (veto), A-03 (spec parser), A-05 (disagreement map), A-07
(agent loop) and A-09 (narrative) stay titles-only until the verdict, A-04 or A-06/A-08 exist.

**What counts.** The test is "if you run it twice, can it give two different answers?"
(AGENTS.md). A VLM's reply and sampled completions both can; the reconstruction backbone can't
(D-018).

**Hard rules.** A model's guess never passes or fails a part (D-005). σ comes from depth residuals,
and the spread across completions is guidance only (D-006, issue #157), so completed surface is
labelled Inferred and never touches the verdict mesh. No cost, latency, accuracy or K is promised:
each is measured or a declared parameter.

**What the proposal already decides.** The VLM is Gemini Flash-Lite, version pinned, called on
sampled frames, not every frame (§6.5, §8.3); its key lives in Actions secrets and a gitignored
local env file, and never reaches the phone app. Completion is diffusion-based, K samples, with
coverage-only guidance as the fallback (§6.6, §9); no model is named, hence the A-10 spike.

The A-10..A-13 on the unmerged `chore/proposal-revisions-2026-09-05` branch are superseded, as
that branch's G-06 and C-09..C-15 already were.

Full detail in each story file.

| ID | Story | Est | Depends on | State | Owner |
|---|---|---|---|---|---|
| [A-01](A/A-01.md) | VLM frame supervision: sample frames, call a pinned Gemini Flash-Lite model, record structured results | 7 h | nothing | Done | @mbj1994 |
| [A-10](A/A-10.md) | Spike: choose the shape-completion model | 4 h | nothing | Ready | _unclaimed_ |
| [A-04](A/A-04.md) | Shape completion stage: K samples stored, Inferred label defined, verdict mesh untouched | 8 h | A-10 | Blocked | _unclaimed_ |
| [A-11](A/A-11.md) | Show the VLM hint to the supplier: contract field and capture message (D-031) | 4 h | nothing | In review | @TabeenRaoof |
| [A-06](A/A-06.md) | Agent tool definitions: the five capture tools as typed, deterministic server functions | 5 h | nothing | Done | @dalwalyk |
| [A-08](A/A-08.md) | Deterministic capture policy: the agent's fallback, so every session completes | 5 h | A-06 | Done | @dalwalyk |

---

## Review follow-ups

Fixes for the code bugs P-27's review of every Done story found on 2026-10-23. Each bug is a
GitHub issue, verified against `main` with file and line links (#178–#201). Fixes that touch the
same code are bundled into one story, and nothing that waits on a decision is bundled with a fix
that is ready. These ten total 31 h, planning estimates like every other row.

**Start with R-25.** It is the hard-rule fix: surface no camera saw can be labelled Observed
today (#178). R-26 (flat parts cut away by segmentation) and R-27 (inward-facing hole fills, the
Poisson fallback) change what the supplier gets back.

**Waiting on a decision, no story yet:**
- #181 changes D-039's scale-agreement formula.
- #200 (Python 3.12 vs the open3d pin) touches D-047.
- The internal-failure error code in #186 is a contract change; A-11's D-031 request carries
  it (`internal_error`, D-056).

**Closing issues.** In the pull request, write the keyword before each number
(`Fixes #189, fixes #191`). GitHub closes only the first of a comma list, which is how PR #45 left
two of its three issues open.

Issues per story:
- R-25 #178, #185, #179
- R-26 #180
- R-27 #183, #184
- R-28 #182, #188
- R-29 #187, plus the validation half of #186
- C-19 #192, #193
- C-20 #189, #190, #191
- F-08 #196, #197, #198
- I-12 #199
- P-28 #194, #195, #201

Full detail in each story file.

| ID | Story | Est | Depends on | State | Owner |
|---|---|---|---|---|---|
| [R-25](R/R-25.md) | Provenance occlusion: facing test, σ window, `observed_fraction` from final labels | 5 h | R-09, R-10, R-14 | Claimed | @mbj1994 |
| [R-26](R/R-26.md) | Segmentation keeps flat parts; the height override bypasses the layer search | 3 h | R-21 | Ready | _unclaimed_ |
| [R-27](R/R-27.md) | Mesh cleanup and the Poisson fallback: winding, components, size limit, oriented normals | 5 h | R-07, R-08 | Ready | _unclaimed_ |
| [R-28](R/R-28.md) | Portrait-safe VGGT intrinsics rescale and correct board metadata | 2 h | R-02, R-03, R-22, R-23 | Claimed | @dalwalyk |
| [R-29](R/R-29.md) | Contract-shaped validation errors and an Inferred-aware hard-rule test | 2 h | F-03, R-14 | Ready | _unclaimed_ |
| [C-19](C/C-19.md) | The viewer says when a model has no provenance; the mock matches the service | 3 h | C-10, C-12, C-13 | Ready | _unclaimed_ |
| [C-20](C/C-20.md) | Capture gate and orbit maths: heading wrap, roll, gate order | 3 h | C-06, C-07, C-09 | Ready | _unclaimed_ |
| [F-08](F/F-08.md) | Fixture and image safety: downloader verification, case-insensitive photo ignores, image path filters | 3 h | F-06, R-15 | Ready | _unclaimed_ |
| [I-12](I/I-12.md) | Synthetic-generator tests that pin noise, seed and the board frame | 2 h | I-01 | Done | @dalwalyk |
| [P-28](P/P-28.md) | Process tooling: checker exit codes, the decision-number snippet, a portable deck build | 3 h | nothing | Ready | _unclaimed_ |

---

## The chain that sets the date

Total work is 93 hours, and at 27 hours a week across three people that looks like three and a half weeks. It is not, because much of this work cannot be done in parallel.

The longest run of dependent stories is:

> F-02 → F-07 → R-01 → R-04 → R-06 → R-07 → R-08 → R-14 → R-15

That is about 42 hours of work that can only happen one story after another. Only one person can be advancing it at a time, so it moves at one person's pace, roughly 9 hours a week. **That is about five weeks, not three and a half.** Adding a third person to the project does not shorten it.

Two consequences worth acting on:

**The chain is always somebody's top priority.** If you are holding a chain story and you get stuck or busy, hand it off the same day rather than letting it sit. Everything else in the backlog can wait; this cannot. This is the single most important scheduling rule under a shared backlog.

**The September 25 milestone is tight.** A working endpoint needs about 40 hours of that chain, which is roughly four and a half weeks from a September 3 start. That lands in early October, not late September. The milestone is reachable if the chain never stalls, and it is the first thing to slip if it does. We are leaving the date as it stands for now and will revisit at the second sprint boundary with real velocity numbers rather than estimates.

The off-chain stories — R-02, R-03, R-05, R-09, R-10, R-11, R-12, R-13, R-16, R-17, R-18, R-19 — are where the other two people work while the chain advances. There is enough of that work to keep everyone busy, which is the point of laying the backlog out this way.

**Since 2026-10-18 the chain that sets the date runs through ground truth.** Everything above
describes Phases 1–2 as written in September: the 93 hours and the F-02 → R-15 chain, which is
now done. The sprint plan's critical path (§3) continues "provenance and σ → σ_floor measured →
verdict logic → accuracy runs → freeze", and provenance and σ are done. The new serial run is:

> G-01 → G-02 → G-08 → G-04 → verdict logic (I-07)

G-01 3 h, G-02 4 h, G-08 4 h and G-04 6 h are about 17 hours one after another. Three things
join it without lengthening it if they're done in time: G-06 (calipering the board sheet, 1.5 h)
must finish before G-08 shoots on that sheet, G-03 runs beside G-08, and G-04 also waits on R-23
(#152). G-05, G-07 and G-09 are off the chain. The rule above applies unchanged: whoever holds a
chain story hands it off the same day if stuck.

---

## Planning lane

The work that keeps the project steerable but belongs to no single story: the proposal and the
sprint plan, the decisions and process rules, writing and splitting stories, keeping documents
such as the manual in step with `main`, audits and reconciliations across stories, and the class
reviews and reports. Until D-052 none of it had a story, so none of it had actual hours.

**The boundary.** Anything done inside a story's own pull request — its claim, its `Done` state
and completion record, its progress-log entry, the follow-up stories it files — stays in that
story's hours. P is for what no single story owns, plus work on someone else's story after it
merged (a late close-out, a reconciliation, a record correction), since its owner's actual never
included it.

**Past work.** Most of these stories record work done before the lane existed. They are `Done`,
with their evidence in the notes, and `<n> h actual` until the owner fills in a real figure
(D-033). Where no estimate existed before the work, Est is `—` rather than a number made up
afterwards; the class reviews use the sprint plan's 2.5 h per review. Full rules in D-052.

Full detail in each story file.

| ID | Story | Est | Depends on | State | Owner |
|---|---|---|---|---|---|
| [P-01](P/P-01.md) | Proposal and proposal deck | — | nothing | Done | @TabeenRaoof; @dalwalyk; @mbj1994 |
| [P-02](P/P-02.md) | Sprint plan | — | nothing | Done | @TabeenRaoof |
| [P-03](P/P-03.md) | Initial backlog | — | nothing | Done | @TabeenRaoof |
| [P-04](P/P-04.md) | Decisions log and working agreements | — | nothing | Done | @TabeenRaoof |
| [P-05](P/P-05.md) | Make the definition of done checkable: PR template and state checker | — | nothing | Done | @TabeenRaoof |
| [P-06](P/P-06.md) | Check every branch before taking a decision number | — | nothing | Done | @TabeenRaoof |
| [P-07](P/P-07.md) | Tests are written during implementation | — | nothing | Done | @TabeenRaoof |
| [P-08](P/P-08.md) | Agent entry point and measured hours; restore decisions a merge dropped | — | nothing | Done | @TabeenRaoof |
| [P-09](P/P-09.md) | Claims shown in the backlog index and enforced | — | nothing | Done | @TabeenRaoof |
| [P-10](P/P-10.md) | Set Done before merge, not after | — | nothing | Done | @TabeenRaoof |
| [P-11](P/P-11.md) | Story files in one subfolder per lane | — | nothing | Done | @TabeenRaoof |
| [P-12](P/P-12.md) | Write up the thin capture thread | — | nothing | Done | @TabeenRaoof |
| [P-13](P/P-13.md) | Reconciliations and record corrections on others' stories | — | nothing | Done | @TabeenRaoof |
| [P-14](P/P-14.md) | Close-out audit of every R story | — | nothing | Done | @dalwalyk |
| [P-15](P/P-15.md) | Operating manual, first version | — | nothing | Done | @TabeenRaoof |
| [P-16](P/P-16.md) | README brought out of skeleton state | — | nothing | Done | @TabeenRaoof |
| [P-17](P/P-17.md) | Operating manual refresh | — | nothing | Done | @TabeenRaoof |
| [P-18](P/P-18.md) | 18 Sept progress review: deck and generator | 2.5 h | nothing | Done | @TabeenRaoof |
| [P-19](P/P-19.md) | 2 Oct progress review | 2.5 h | nothing | Done | @TabeenRaoof |
| [P-20](P/P-20.md) | Midterm presentation deck | — | nothing | Done | @TabeenRaoof |
| [P-21](P/P-21.md) | Midterm report | — | nothing | Done | @TabeenRaoof; @dalwalyk; @mbj1994 |
| [P-22](P/P-22.md) | Planning lane: D-052, the checker change, the P stories | 3 h | nothing | Done | @TabeenRaoof |
| [P-23](P/P-23.md) | Refresh the sprint plan and point the manual at D-052 | 2 h | P-17 | Done | @mbj1994 |
| [P-24](P/P-24.md) | Write up the ground-truth lane | — | nothing | Done | @TabeenRaoof |
| [P-25](P/P-25.md) | Write up the inspection lane | — | nothing | Done | @TabeenRaoof |
| [P-26](P/P-26.md) | Write up the first generative-AI stories | — | nothing | Done | @TabeenRaoof |
| [P-27](P/P-27.md) | Review every Done story; file its code bugs and write up the follow-up stories | — | nothing | Done | @TabeenRaoof |

---

## Phase 3 and beyond · titles only

Not written out yet, on purpose. We elaborate one sprint ahead at the boundary call, because most of these would be wrong if written in September.

The thin capture thread (C-01–C-13), ground truth (G-01–G-09), the first half of inspection
(I-01–I-05) and the first generative-AI stories (A-01, A-04, A-06, A-08, A-10, A-11) are the
exceptions — all written up above, not here: the capture thread is meant to run early, ground
truth sits on the critical path, inspection's sprint has started, and generative AI is overdue
from S2/S3.

**Capture app**
C-14 heatmap rendering · C-15 report page · C-16 buyer setup · C-17 board PDF with order code · C-18 device testing

**Inspection**
I-06 deviation computation · I-07 verdict logic (#157: `max(σ, σ_floor)`) · I-08 unobserved-geometry rule · I-09 intake pre-check · I-10 unverifiable path

**Ground truth**
G-10 σ_floor for the sheet and card references · G-11 robustness set (glossy, dark and thin parts, a screen photo, the wrong part)

**Generative AI**
A-02 VLM verdict veto · A-03 prose-to-spec parser · A-05 disagreement map · A-07 agent loop · A-09 report narrative

**Cross-cutting**
X-01 integration point I1 · X-02 I2 · X-03 I3 · X-04 I4 · X-05 recorded backup demo

---

*Estimates are planning numbers. Record actuals on completion and re-estimate at each sprint boundary.*

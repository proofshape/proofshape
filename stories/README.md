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
   .login` gives the account the GitHub CLI is logged in as. Push the claim right away, on your
   story branch or as a small PR, so the rest of the team can see it. Then run
   `python3 scripts/check_story_states.py`: it fails if the two places disagree, if a Claimed
   story has no owner, or if the owner isn't a person's handle.
3. Branch as `<area>/<id>-short-name`, for example `recon/R-04-metric-alignment`.
4. Build it. Write notes in the story file as you go — what surprised you, what you chose and
   why. That section is for the next person, including future you.
5. Open a pull request. Either of the other two reviews it.
6. On merge, fill in the **Completion record**: who, date, PR number, and **actual hours against
   the estimate**. The actuals are how our estimates stop being wrong.

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
| **Done** | Merged, and the completion record below is filled in. |

### Completion record

Fill this into the story when it merges. Actual hours matter: they are how our estimates get less wrong.

```
Completed: @member · 2026-09-08 · PR #12 · 4.5 h actual (est 4)
Notes: forced fp16 — bf16 runs at half speed on Apple Silicon.
```

### Definition of done

Applies to every story, unchanged from the plan: merged to `main`, runs on a second member's machine from a clean checkout, has one automated test or one written manual check, reviewed and approved by one other member, and demonstrated at the sprint boundary.

Acceptance criteria are per story. The definition of done is not.

---

## Phase 1 · Foundations

Twenty-three hours. Everything else is blocked on some part of this, so it goes first and it goes fast. **F-01, F-05 and F-06 have no dependencies — those are the three to start on day one, one each.**

Full detail in each story file.

| ID | Story | Est | Depends on | State | Owner |
|---|---|---|---|---|---|
| [F-01](F-01.md) | Public repository with branch protection | 2 h | nothing | Done | @TabeenRaoof |
| [F-02](F-02.md) | Repository skeleton and module boundaries | 3 h | F-01 | Done | @dalwalyk |
| [F-03](F-03.md) | Interface contract v1, frozen | 3 h | F-01 | Done | @TabeenRaoof |
| [F-04](F-04.md) | CI skeleton | 3 h | F-02 | Done | @dalwalyk |
| [F-05](F-05.md) | Shared GPU workspace provisioned, three-way access | 4 h | nothing | Done | @mbj1994 |
| [F-06](F-06.md) | Golden capture set | 5 h | nothing | Done | @mbj1994; @TabeenRaoof |
| [F-07](F-07.md) | Reproducible dev environment | 3 h | F-02, F-05 | Done | @dalwalyk |
---

## Phase 2 · The reconstruction engine

Seventy hours, and the reason we are doing backend first. Exit criterion for the whole phase: **the golden capture goes in, and a metric, segmented mesh with per-vertex provenance comes out, reproducibly, with per-stage timings recorded.**

**Parallelism.** R-01, R-02, R-11, R-12 and R-15 have no dependency on each other, so three people can work simultaneously from the start. The chain narrows at R-04, which needs both R-01 and R-02 — expect that to be the bottleneck and plan to have two people free when it lands.

Full detail in each story file.

| ID | Story | Est | Depends on | State | Owner |
|---|---|---|---|---|---|
| [R-01](R-01.md) | Run the reconstruction model on the golden capture | 5 h | F-05, F-06, F-07 | Claimed | @yashidalwala |
| [R-02](R-02.md) | Board detection and per-frame camera pose | 5 h | F-06, F-07 | Done | @TabeenRaoof |
| [R-03](R-03.md) | Camera intrinsics from the board | 4 h | R-02 | Ready | _unclaimed_ |
| [R-04](R-04.md) | Metric alignment | 6 h | R-01, R-02 | Blocked | _unclaimed_ |
| [R-05](R-05.md) | Scale agreement check | 4 h | R-04 | Blocked | _unclaimed_ |
| [R-06](R-06.md) | Segment the part from the board | 4 h | R-04 | Blocked | _unclaimed_ |
| [R-07](R-07.md) | TSDF fusion to a mesh | 6 h | R-04, R-06 | Blocked | _unclaimed_ |
| [R-08](R-08.md) | Mesh cleanup and GLB export | 4 h | R-07 | Blocked | _unclaimed_ |
| [R-09](R-09.md) | Per-vertex provenance | 5 h | R-07 | Blocked | _unclaimed_ |
| [R-10](R-10.md) | Per-vertex uncertainty | 5 h | R-09 | Blocked | _unclaimed_ |
| [R-11](R-11.md) | MASt3R behind the config flag | 5 h | R-01 | Blocked | _unclaimed_ |
| [R-12](R-12.md) | COLMAP behind the config flag | 5 h | R-01 | Blocked | _unclaimed_ |
| [R-13](R-13.md) | Per-stage latency instrumentation | 3 h | R-07 | Blocked | _unclaimed_ |
| [R-14](R-14.md) | Reconstruction endpoint | 5 h | F-03, R-08 | Blocked | _unclaimed_ |
| [R-15](R-15.md) | Container image and deploy | 4 h | R-14, F-05 | Blocked | _unclaimed_ |
---

## The chain that sets the date

Total work is 93 hours, and at 27 hours a week across three people that looks like three and a half weeks. It is not, because much of this work cannot be done in parallel.

The longest run of dependent stories is:

> F-02 → F-07 → R-01 → R-04 → R-06 → R-07 → R-08 → R-14 → R-15

That is about 42 hours of work that can only happen one story after another. Only one person can be advancing it at a time, so it moves at one person's pace, roughly 9 hours a week. **That is about five weeks, not three and a half.** Adding a third person to the project does not shorten it.

Two consequences worth acting on:

**The chain is always somebody's top priority.** If you are holding a chain story and you get stuck or busy, hand it off the same day rather than letting it sit. Everything else in the backlog can wait; this cannot. This is the single most important scheduling rule under a shared backlog.

**The September 25 milestone is tight.** A working endpoint needs about 40 hours of that chain, which is roughly four and a half weeks from a September 3 start. That lands in early October, not late September. The milestone is reachable if the chain never stalls, and it is the first thing to slip if it does. We are leaving the date as it stands for now and will revisit at the second sprint boundary with real velocity numbers rather than estimates.

The off-chain stories — R-02, R-03, R-05, R-09, R-10, R-11, R-12, R-13 — are where the other two people work while the chain advances. There is enough of that work to keep everyone busy, which is the point of laying the backlog out this way.

---

## Phase 3 and beyond · titles only

Not written out yet, on purpose. We elaborate one sprint ahead at the boundary call, because most of these would be wrong if written in September.

**Thin capture thread** (runs alongside Phases 1–2, in spare time)
C-01 minimal upload page · C-02 camera permissions on a real iPhone · C-03 gyro permission flow

**Capture app**
C-04 app shell · C-05 live camera · C-06 frame gate · C-07 gyro overlay and arrow · C-08 next-view scoring · C-09 model viewer · C-10 provenance colouring · C-11 heatmap rendering · C-12 report page · C-13 buyer setup · C-14 board PDF with order code · C-15 device testing

**Inspection**
I-01 synthetic mesh generator · I-02 CAD loading · I-03 global registration · I-04 robust ICP · I-05 symmetry ambiguity check · I-06 deviation computation · I-07 verdict logic · I-08 unobserved-geometry rule · I-09 intake pre-check · I-10 unverifiable path

**Ground truth**
G-01 printer characterisation · G-02 specimen batch · G-03 caliper measurement · G-04 noise floor per tier · G-05 defect ladder

**Generative AI**
A-01 VLM frame supervision · A-02 VLM verdict veto · A-03 prose-to-spec parser · A-04 shape completion integration · A-05 disagreement map · A-06 agent tool definitions · A-07 agent loop · A-08 deterministic fallback · A-09 report narrative

**Cross-cutting**
X-01 integration point I1 · X-02 I2 · X-03 I3 · X-04 I4 · X-05 recorded backup demo

---

*Estimates are planning numbers. Record actuals on completion and re-estimate at each sprint boundary.*

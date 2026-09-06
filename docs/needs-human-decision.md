# Needs a human decision

Items deliberately **not** decided by an assistant. Each says what was found, why it is here, and
what the decision is. Nothing in this file has been acted on beyond what each entry states.

Created 2026-09-06 during the proposal revision pass.

---

## Move to private notes

Content removed from the public repository during this pass, preserved verbatim so a human can
relocate it to the local playbook or private notes. **These are the exact original quotes.**

### 1 · From §9 of the proposal — removed

> **The requirement is met if the reconstruction backbone counts.** That remains the open question with the instructor (§16) and is worth 28 hours on its own. If it does not count, the share is 24.9% and closing the rest would mean removing inspection scope rather than adding hours.

> **Why the extra hour a week all went to AI.** The team raised capacity from 8 to 9 hours per person per week, which is 39 additional person-hours. Spending only the agent's 30 there would have *lowered* the share to 30.4%, because the denominator grew faster than the numerator. Every one of the 39 new hours is therefore generative-AI work: the agent, plus expanding VLM supervision from sampled frames to per-frame with a local fallback.

### 2 · From D-016 in the decisions log — removed, decision retained

> *Why:* adding capacity without spending it on generative AI *lowers* the share, because the denominator grows faster than the numerator. Spending only the agent's 30 hours across 39 new hours would have produced 30.4%, worse than building it inside the old budget.

**Why these were removed.** All three describe sizing components to reach a ratio rather than
choosing them on merit. Component sizing is now on merit, so the reasoning no longer governs — but
it is genuinely useful planning history and should not simply be lost. The capacity decision itself
(9 hours per person per week) is unchanged and remains public.

**Decision needed:** where these go. The local playbook outside the repository is the obvious home.

---

## AI-assisted development is publicly disclosed, and the policy is unknown

`AGENTS.md` and `CLAUDE.md` sit at the repository root and document in detail that this project is
built with AI assistance. The instructor's AI-use disclosure policy is **§16 question 2 and is
still unanswered.**

**Decision needed before the presentation:** keep them public and disclose proactively, or move them
to a private companion repository until the policy is known. Proactive disclosure is usually the
stronger position, but that is a judgement about this course, not one an assistant should make.

### Passages in those files that could read badly to an instructor

Listed, not changed. Each is defensible; each could also be read uncharitably.

1. **`AGENTS.md`, "The generative-AI requirement" section** — "The course requires roughly one third
   of the project to be generative AI. Five components carry that… **These are never cut**… If a
   schedule conversation proposes dropping one, flag it rather than agreeing." Reads as engineering
   the project around a grading criterion. The underlying practice is sound, and after this revision
   the components are protected because they are load-bearing rather than because they are counted,
   but the wording still frames it as a requirement to be satisfied.
2. **`AGENTS.md`, refusal list** — "Cutting a generative-AI component to save schedule." Same issue
   in one line.
3. **`AGENTS.md`, "The test for whether something counts"** — the run-it-twice test. Honest and
   defensible, and it argues *against* over-claiming, but it is visibly a test for a grading
   boundary.

**Suggested rewording if these stay public:** protect the components on the grounds stated in D-017
as amended — they are structurally load-bearing — and drop the references to the share entirely.
Not done here because it edits the substance of `AGENTS.md`, which this pass was told not to touch.

---

## Should graded coursework stay public?

`submissions/` holds the proposal and the presented deck. Some programs restrict public posting of
graded work, and some treat it as fine or encourage it.

**Decision needed:** confirm the course's position. **Never commit instructor feedback, grades or
review comments to this folder** regardless of the answer.

---

## Scan results: nothing found

Checked and clean, recorded so it is not re-checked blindly:

- **Remarks about teammates' availability, health or performance** — none in `docs/progress-log.md`
  or `stories/`.
- **Quotes attributed to the instructor** — none.
- **Grades or review feedback** — none.
- **Frank assessments of the course** — none.
- **Content about making the generative share look larger** — none outside the three quotes above,
  all of which are listed and removed.

---

## Also worth a human eye

- **`docs/archive/team-proposal-v2.md`** names the instructor four times and references a different
  course code. The name has been genericized to "the instructor" per the hygiene pass. The file is a
  superseded draft kept for its reasoning; whether a public repository should carry it at all is a
  judgement call.
- **`O-002`, the presentation window,** is inherited and unverified but now anchors both Gantt
  charts and every sprint boundary. A caveat has been added to the milestone table and the sprint
  plan. Confirming the real date is still outstanding.

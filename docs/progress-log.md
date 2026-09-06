# Progress log

Running chronological journal of what has been decided, built and shipped. Complements — does
not replace — the per-story files in `stories/`, which stay the source of truth for a story's
own detail, and `sprint-plan.md`, which is the schedule.

**This file is the history.** One entry per story completed, pull request reviewed, or
measurement run. Newest at the bottom. Never delete an old entry — if something turned out to
be wrong, append a correction rather than editing the original, because the mistake is usually
more instructive than the fix.

**Why it earns its keep here:** the entries since the last class presentation *are* the "what we
did" slide. Keep this current and the biweekly review takes minutes to prepare instead of an
evening.

---

## Entry template

```
## YYYY-MM-DD — <ID> <short title> (<built | reviewed | merged | measured>)

**Who:** @member
**What changed:** one or two sentences.
**Command / how to reproduce:** the exact command, if there was one.
**Result:** numbers if there are numbers; "works" is not a result.
**Concluded:** what we now believe that we did not before.
**Next:** what this unblocks, or what it forces us to decide.
```

Keep entries short. Three or four lines is a good entry. If it needs more, most of it probably
belongs in the story file instead.

---

## 2026-09-05 — project scaffolding created (built)

**Who:** all three
**What changed:** `AGENTS.md`, `CLAUDE.md`, `docs/decisions.md`, `docs/glossary.md`,
`docs/story-template.md` and this log. The story backlog was split from one file into
`stories/<ID>.md` with `stories/README.md` kept as the index.
**Concluded:** twenty-two stories are written out for foundations and the reconstruction engine,
totalling 93 hours, of which about 42 sit on a serial chain that no amount of parallelism
shortens.
**Next:** F-01, F-05 and F-06 are Ready and have no dependencies — one each, starting day one.

## 2026-09-05 — proposal presented; repository restructured (shipped)

**Who:** all three
**What changed:** the proposal and its deck were presented to the class. Afterwards the folder
was reorganised: coursework deliverables into `submissions/` dated by hand-in, working documents
into `docs/`, the story index into `stories/README.md`, plus a `README.md` and `.gitignore`.
**Concluded:** the presentation happened on 5 September, three days earlier than the M0 date the
plan assumed. Whether the M0 spike ran before it is not recorded here — someone should note it.
**Next:** initialise the git repository and push, which is story F-01.

## 2026-09-05 — F-01: repository created, protected, and moved to an org (built)

**Who:** @TabeenRaoof
**What changed:** repo created public, MIT licence, eight branch protection rules applied, then
transferred from the personal account to the new `proofshape` organisation.
**Result:** both gates proven, not just configured — a merge without approval was refused
("base branch policy prohibits the merge") and a direct push to `main` was rejected (GH006).
All eight rules verified intact after the transfer.
**Concluded:** `enforce_admins` means nobody can bypass, including the owner, so with one
collaborator **no pull request can merge at all**. Correct behaviour, but it makes adding the
other two members the hard blocker on all further work.
**Next:** add collaborators, then PR #1 can be approved. F-02 and F-03 become Ready on merge.

## 2026-09-06 — proposal revision pass: sizing on merit, hours moved to risk (revised)

**Who:** @TabeenRaoof
**What changed:** a revision pass across the proposal, decisions log, sprint plan and stories.

- **Component sizing is now on merit, not to reach a share.** The generative-AI proportion is not
  being audited, so components are sized by what they do. The LLM capture agent drops from ~30 h to
  ~12 h of orchestration, on the grounds the proposal itself already argued: a language model
  choosing a direction is worse than the geometric scorer, so most of what an "agent" might do is
  work we should not want it doing.
- **A real contradiction was fixed.** Batch was the guaranteed path, completion drove only
  interactive guidance, and interactive mode was cuttable — so cutting it stranded completion.
  Batch now includes one completion-driven recapture round, which is the single decision batch mode
  retains. D-002 amended; D-005 and D-006 untouched.
- **Freed hours went to risk, not features:** the reliability curve that justifies k and sigma_floor
  (6 h), specimen breadth across glossy, dark and thin-walled finishes (4 h), session integrity
  promoted from Tier 3 to Tier 2 (5 h), and placement notes drafted by the spec parser (3 h).
- **The ratio arithmetic left the public repository.** Two §9 passages and D-016's original
  reasoning are quoted verbatim in `docs/needs-human-decision.md` for relocation to private notes.
- **The pod is now the canonical environment** (D-010 inverted); local Apple Silicon development is
  optional and its fp16 finding is a documented gotcha rather than a project-wide decision.

**Decisions amended:** D-002, D-010, D-011, D-012, D-016, D-017, and O-001 updated.
**Concluded:** the cut list was still ordered so that cutting interactive mode broke a Tier 2
component. Reserve remains ~7% of capacity, which is thin, and nothing in this pass improved it —
every freed hour was allocated.
**Next:** the items in `docs/needs-human-decision.md` need a human, particularly whether
`AGENTS.md` and `CLAUDE.md` stay public while the AI-use policy is unanswered.

## 2026-09-06 (later) — module layout completed; tests-with-code made a rule (built)

**Who:** @TabeenRaoof
**What changed:** added `ai/`, `service/` and `common/` with README stubs, and a test convention for
every module. Made "unit tests ship in the same pull request as the code" a standing rule rather
than a style preference.
**Concluded:** thirteen A-prefix stories had no directory to live in. The layout had been written
from the system diagram and the story prefixes were written later from the work, so nobody noticed
until asked directly. Prompts and API-key handling would have scattered across three modules.
**Also:** the tests rule now appears in four places — the standing rules, the definition of done,
the story template and the refusal list — because a rule that lives only in prose is a preference.
**Decisions added:** D-025 (module layout, AI grouped by technology), D-026 (tests in the same PR).
**Next:** F-02 creates these directories properly with their package structure.

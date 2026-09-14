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

## 2026-09-11 — F-02: repository skeleton and module boundaries (built)

**Who:** @dalwalyk
**What changed:** `/recon`, `/capture`, `/inspect`, `/contracts`, `/fixtures`, `/scripts` created,
each with a `README.md`. Root `pyproject.toml` registers `recon` as an editable package;
`inspect` deliberately left unregistered (D-030 — it collides with the standard library
`inspect` module). Root README gained a "Running each part" section.
**Command / how to reproduce:** `pip install -e .` from a clean venv.
**Result:** installs cleanly; `import recon` works; `import inspect` still resolves to the
standard library, confirmed in a scratch venv.
**Concluded:** the `inspect/` directory name (fixed by D-020) and a top-level Python import of
the same name can't both exist safely — packaging has to route around it, not the directory.
**Next:** F-04 (CI skeleton) and F-07 (reproducible dev environment) become Ready on merge.

## 2026-09-14 — F-02 marked Done; found a merged-but-corrupted story file (build)

**Who:** @TabeenRaoof
**What changed:** PR #6 merged, but `stories/F-02.md` was left at `State: In review` — the third
time this exact gap has hit us — and worse, the file was actually **corrupted**: it had two
`State:`/`Owner:` blocks stacked back to back (`In review`/`@dalwalyk` immediately followed by
`Ready`/`_unclaimed_`, the pre-merge content from `main`). Fixed: de-duplicated the header, set
`State: Done`, filled the completion record (PR #6, merged 2026-09-14, reviewed by
@TabeenRaoof — actual hours left blank, real data). Flipped `stories/README.md` and `F-02` to
`Done`, and `F-04` (its only remaining dependency) to `Ready`. `F-07` stays `Blocked` — it also
needs F-05, which is only `Ready`, not `Done`.
**Result:** verified by reading `git show <merge-commit>^1:stories/F-02.md` and
`^2:stories/F-02.md` directly — both parents made a single-line edit to the same `State:` and
`Owner:` lines, and the merge silently kept both instead of conflicting.
**Concluded:** GitHub's `mergeable: MERGEABLE` only means no conflict *markers* were left — it
does not mean the merged content is *correct*. A clean two-parent line-merge can still silently
duplicate a small, adjacent block instead of flagging it as a conflict, especially when both
sides touch only a line or two. We checked `mergeable` before every approval in this session and
treated it as sufficient; it isn't. This is a new failure mode, distinct from the state-drift
bug `scripts/check_story_states.py` (still unmerged, on PR #5) already guards against — that
script would have caught the *un-updated* state, but not this *duplicated* one, since the file's
first `State:` line was still technically correct.
**Next:** worth a follow-up: `check_story_states.py`'s regex only reads the *first* `**State:**`
match in a file. If a duplicate block like this recurs, the script would silently read the first
(possibly stale) one and miss the corruption entirely. Consider having it also flag a file with
more than one `**State:**` line, once PR #5 merges.

## 2026-09-14 — AGENTS_START_HERE.md added; actual-hours rule tightened (built)

**Who:** @TabeenRaoof
**What changed:** new `AGENTS_START_HERE.md` — a 17-step, numbered order of operations naming
which file governs each step, rather than restating any of them. `CLAUDE.md` (auto-loaded every
Claude Code session in this repo) and `README.md`'s front door now both point to it first.
`AGENTS.md`'s definition of done and refusal list were also tightened: the actual-hours field in
a completion record must be a real, measured number — copying the estimate in, or leaving it
implied, is now explicitly listed as something to refuse. Recorded as D-033.
**Command / how to reproduce:** n/a — documentation only; `pytest tests/` and
`python3 scripts/check_story_states.py` still pass, confirming this touched no code path.
**Result:** caught in passing while reviewing this: `stories/F-01.md`'s completion record has
carried the unfilled `<n> h actual` placeholder since the PR merged on 2026-09-11. Left as-is
rather than guessed — the real number isn't known, and guessing it is exactly what D-033 now
says not to do.
**Concluded:** a rule that exists only in `AGENTS.md`'s prose gets read once at onboarding and
then forgotten under real work — the same lesson D-025 already drew for the state/index gap.
Making the entry point itself carry the checklist, and wiring the one file every session
actually auto-loads (`CLAUDE.md`) to point at it, is the mechanical fix rather than another
paragraph nobody re-reads.
**Next:** whoever knows F-01's real hours can fill them in directly; otherwise it stays `<n>`
until someone does.

## 2026-09-14 — AGENTS_START_HERE.md trimmed before review (built)

**Who:** @TabeenRaoof
**What changed:** the entry above described `AGENTS_START_HERE.md`'s 17 steps as "naming which
file governs each step, rather than restating any of them" — true for about half the steps, not
all of it. Steps 6–9 (before starting a story), 10–11 (while working) and 14–17 (on merge) had
paraphrased `AGENTS.md`'s own sections instead of just pointing at them. Collapsed each of those
three groups into a single pointer line, kept only the genuinely new content (model-file
detection, the pre-PR check list, and one explicit callout on the actual-hours rule, since that
gap is the whole reason this file exists). Prompted by the user asking, directly, whether a
fourth instruction file was worth it given how many already exist — a fair question this repo's
own history (D-025) had already answered once.
**Result:** file is now pointers-plus-five-new-facts rather than a partial second copy of
`AGENTS.md`. Confirmed no code path touched: `pytest tests/` and
`python3 scripts/check_story_states.py` unchanged.
**Concluded:** a startup checklist earns its place by adding facts that exist nowhere else
(what to check for, what to run, in what order) — not by re-explaining rules a different file
already owns. Worth checking any future addition to this file against that before it's merged
in, not after.

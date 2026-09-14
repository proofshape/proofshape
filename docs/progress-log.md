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

# scripts

One-command helpers. Each is runnable on its own; none is part of the shipped system.

| Script | What it does |
|---|---|
| `check_story_states.py` | Reads the dependency graph in `stories/`, flags any story whose file and index disagree, and reports (or with `--fix`, corrects) stories stuck `Blocked` when every dependency is `Done`. Run after any merge that sets a story to `Done` — see `AGENTS.md`. |
| `build_review_deck.js` | Builds the biweekly progress-review deck as a `.pptx` into `submissions/`. See the header comment for why its `package.json` lives here rather than at the repository root. |
| `build_midterm_deck.js` | Builds the CS595 midterm presentation deck (P-20) as a `.pptx`. Applies an optional theme pass (#201) when `PPTX_APPLY_THEME` points at a theming tool (`PPTX_APPLY_THEME=/path/to/apply_theme.js node build_midterm_deck.js ...`); otherwise skips it with a one-line message, and the deck is unaffected either way. This is a personal tool, not something every team member has — `PPTX_APPLY_THEME` may also need `NODE_PATH` set to `scripts/node_modules` so the tool can find `pptxgenjs`. |
| `build_midterm_report.js` | Builds the CS595 midterm report (P-21) as a `.docx`. |
| `smoke_r14.py` | Runs real golden frames through the reconstruction HTTP flow: in-process (R-14), or against a deployed URL with `--base-url` (R-15). |
| `deploy_recon.py` | Starts or stops the deployed reconstruction service on Lightning. Run by `.github/workflows/deploy-recon.yml` on `main` only; see `recon/README.md`, R-15. |
| `container_entrypoint.sh` | **The one exception to "not shipped":** the R-15 image's entrypoint. It fetches the model weights, then serves `recon.service`. |
| `setup_completion_env.sh` | A-12: builds LaS-Comp's own Python 3.10 / torch 2.4 environment on a CUDA host (Colab or Lightning), at D-057's pins, and prints the `PROOFSHAPE_COMPLETION_RUNNER` line to use. Never touches the host's own torch. See `recon/README.md`, A-12. |
| `lascomp_runner.py` | A-12: the real model behind A-04's completion contract. Runs inside that environment, launched by the service (or by hand on a prepared completion folder). Also `--check` (setup's last step) and `--ensure-encoder-entry`. |
| `measure_wall_share.py` | R-33: classifies a served `mesh_provenance.glb`'s faces into part / board skin / volume boundary / wall by centroid, the same diagnostic #231 and #258 ran by hand, so it can be rerun after a pipeline change instead of redone by hand. Needs the part's own x/y footprint supplied (`--part-x-mm`, `--part-y-mm`); nothing records it. |

## Node helpers

```
cd scripts && npm install
node build_review_deck.js ../submissions/YYYY-MM-DD-progress-review-deck.pptx
```

**Do not move `package.json` to the repository root.** `.github/workflows/typescript-ci.yml`
detects a TypeScript project with `[ -f package.json ]` from the workspace root, so a root
manifest switches on its `tsc`/eslint job — which then fails, because there is no `tsconfig.json`
and no eslint config. Whoever starts C-01 needs to add both in the *same* pull request as the
root manifest.

`tests/test_generator_scripts.py` smoke-tests all three `build_*.js` scripts by running each one
as a subprocess (the same pattern `tests/test_download_golden_capture.py` uses for its bash
script) and checking it exits zero and writes a real, nonempty file (#201). It's a `pytest`
test, not a separate CI job: `.github/workflows/python-ci.yml`'s `pytest` job now installs
`scripts/node_modules` before running tests, so this is enforced in CI, not just locally. Running
`pytest` locally without having run `npm install` here first skips these tests rather than
failing them.

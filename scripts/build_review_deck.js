// Builds the biweekly progress-review deck as a .pptx.
//
//   cd scripts && npm install        # once
//   node build_review_deck.js ../submissions/YYYY-MM-DD-progress-review-deck.pptx
//
// The manifest lives in scripts/, NOT at the repository root, and that is deliberate:
// .github/workflows/typescript-ci.yml runs `[ -f package.json ]` from the workspace root,
// so a root manifest would switch on its tsc/eslint job — which would then fail, because
// there is no tsconfig and no eslint config. Whoever starts C-01 must add tsconfig.json
// and an eslint config in the SAME pull request as the root package.json, or CI breaks
// for everyone.
//
// Deliberately kept as a script rather than hand-built slides: the sprint plan budgets
// 18 person-hours for seven reviews, and that only holds if a review is assembly rather
// than invention (D-015). Edit the content blocks below for the next one.
//
// Every figure here was read from the repository, not recalled — see build notes in the PR.

const PptxGenJS = require("pptxgenjs");
const pres = new PptxGenJS();

pres.layout = "LAYOUT_16x9";           // 13.33 x 7.5 in
pres.author = "ProofShape team";
pres.title = "ProofShape — Progress Review, 18 Sept 2026";

// ── palette ────────────────────────────────────────────────────────────────
const INK = "1B1B1F";
const MUTED = "6B7280";
const ACCENT = "16697A";   // teal — headings, rules
const ACCENT2 = "489FB5";  // lighter teal — secondary bars
const GOOD = "15803D";
const WARN = "B45309";
const BAD = "B91C1C";
const PAPER = "FFFFFF";
const BAND = "F1F5F7";

const W = 13.33, H = 7.5;
const M = 0.62;            // page margin

// ── helpers ────────────────────────────────────────────────────────────────
function slide(opts = {}) {
  const s = pres.addSlide();
  s.background = { color: opts.bg || PAPER };
  return s;
}

function header(s, title, kicker) {
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: M, y: 0.34, w: W - 2 * M, h: 0.26,
      fontSize: 11, bold: true, color: ACCENT, charSpacing: 1.6,
      fontFace: "Helvetica",
    });
  }
  s.addText(title, {
    x: M, y: kicker ? 0.62 : 0.5, w: W - 2 * M, h: 0.62,
    fontSize: 30, bold: true, color: INK, fontFace: "Helvetica",
  });
  s.addShape(pres.ShapeType.rect, {
    x: M, y: kicker ? 1.3 : 1.18, w: 1.5, h: 0.045, fill: { color: ACCENT },
  });
}

function footer(s, n) {
  s.addText("ProofShape · CS595 · 18 Sept 2026", {
    x: M, y: H - 0.5, w: 6, h: 0.3, fontSize: 9.5, color: MUTED, fontFace: "Helvetica",
  });
  if (n) s.addText(String(n), {
    x: W - M - 0.6, y: H - 0.5, w: 0.6, h: 0.3, fontSize: 9.5,
    color: MUTED, align: "right", fontFace: "Helvetica",
  });
}

function bullets(s, items, o = {}) {
  s.addText(
    items.map((t) => {
      if (typeof t === "string") return { text: t, options: { bullet: { code: "2022" }, breakLine: true } };
      return { text: t.text, options: { bullet: t.sub ? { indent: 20, code: "25AA" } : { code: "2022" }, indentLevel: t.sub ? 1 : 0, color: t.color || INK, bold: t.bold || false, breakLine: true } };
    }),
    {
      x: o.x || M, y: o.y || 1.75, w: o.w || W - 2 * M, h: o.h || 4.6,
      fontSize: o.fontSize || 16.5, color: INK, lineSpacingMultiple: 1.35, fontFace: "Helvetica",
    }
  );
}

function statCard(s, x, y, w, big, label, color) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h: 1.32, fill: { color: BAND }, line: { color: BAND },
    rectRadius: 0.06,
  });
  s.addText(big, { x, y: y + 0.12, w, h: 0.66, fontSize: 34, bold: true, color: color || ACCENT, align: "center", fontFace: "Helvetica" });
  s.addText(label, { x: x + 0.1, y: y + 0.78, w: w - 0.2, h: 0.44, fontSize: 11.5, color: MUTED, align: "center", fontFace: "Helvetica" });
}

function table(s, rows, o = {}) {
  s.addTable(rows, {
    x: o.x || M, y: o.y || 1.8, w: o.w || W - 2 * M,
    colW: o.colW,
    fontSize: o.fontSize || 13,
    color: INK, fontFace: "Helvetica",
    border: { type: "solid", pt: 0.5, color: "DDE3E8" },
    fill: { color: PAPER },
    rowH: o.rowH || 0.34,
    valign: "middle",
  });
}
function th(t) { return { text: t, options: { bold: true, color: PAPER, fill: { color: ACCENT }, fontSize: 12.5 } }; }
function td(t, opts = {}) { return { text: t, options: { fontSize: 12.5, ...opts } }; }

// Slide assets live beside the deck they belong to, so a dated submission stays
// self-contained. Resolved from __dirname so the script runs from any directory.
const path = require("path");
const A = path.join(__dirname, "..", "submissions", "assets", "2026-09-18") + path.sep;

// Screenshot with a thin border and a caption underneath.
function shot(s, file, o) {
  s.addShape(pres.ShapeType.rect, {
    x: o.x - 0.035, y: o.y - 0.035, w: o.w + 0.07, h: o.h + 0.07,
    fill: { color: "E6ECF0" }, line: { color: "D3DCE3", width: 0.75 },
  });
  s.addImage({ path: A + file, x: o.x, y: o.y, w: o.w, h: o.h });
  if (o.caption) {
    s.addText(o.caption, {
      x: o.x - 0.035, y: o.y + o.h + 0.09, w: o.w + 0.07, h: 0.42,
      fontSize: o.capSize || 10.5, color: MUTED, align: o.capAlign || "left",
      fontFace: "Helvetica", valign: "top",
    });
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1 · Title
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  s.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: 0.28, h: H, fill: { color: ACCENT } });
  s.addText("ProofShape", { x: 1.1, y: 2.0, w: 11, h: 1.0, fontSize: 52, bold: true, color: INK, fontFace: "Helvetica" });
  s.addText("Turning a smartphone into a coarse 3D inspection tool for manufactured parts",
    { x: 1.1, y: 3.0, w: 10.6, h: 0.5, fontSize: 17, color: MUTED, fontFace: "Helvetica" });
  s.addShape(pres.ShapeType.rect, { x: 1.1, y: 3.72, w: 1.5, h: 0.045, fill: { color: ACCENT } });
  s.addText("Biweekly Progress Review · 18 September 2026", { x: 1.1, y: 4.0, w: 10.6, h: 0.4, fontSize: 19, bold: true, color: ACCENT, fontFace: "Helvetica" });
  s.addText("CS595 Capstone · San Francisco Bay University · Fall 2026", { x: 1.1, y: 4.5, w: 10.6, h: 0.35, fontSize: 14, color: MUTED, fontFace: "Helvetica" });
  s.addText("@TabeenRaoof  ·  @dalwalyk  ·  @mbj1994          github.com/proofshape/proofshape",
    { x: 1.1, y: 5.35, w: 11, h: 0.35, fontSize: 13, color: INK, fontFace: "Helvetica" });
}

// ═══════════════════════════════════════════════════════════════════════════
// 2 · The four questions
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  header(s, "Every review answers the same four questions", "our standing template");
  s.addText("Fixed template, so preparing a review is assembly rather than invention — that is what keeps seven reviews at ~18 person-hours instead of eating the schedule reserve.",
    { x: M, y: 1.62, w: W - 2 * M, h: 0.5, fontSize: 13.5, color: MUTED, italic: true, fontFace: "Helvetica" });

  const qs = [
    ["1", "What we said we would do last time", "slide 6"],
    ["2", "What works now", "slides 7–10"],
    ["3", "What slipped, and why", "slide 13"],
    ["4", "What we will have by the next review", "slide 14"],
  ];
  let y = 2.35;
  qs.forEach(([n, q, where]) => {
    s.addShape(pres.ShapeType.roundRect, { x: M, y, w: W - 2 * M, h: 0.82, fill: { color: BAND }, line: { color: BAND }, rectRadius: 0.05 });
    s.addText(n, { x: M + 0.18, y: y + 0.13, w: 0.55, h: 0.56, fontSize: 26, bold: true, color: ACCENT, align: "center", fontFace: "Helvetica" });
    s.addText(q, { x: M + 0.85, y: y + 0.2, w: 8.2, h: 0.42, fontSize: 19, bold: true, color: INK, fontFace: "Helvetica" });
    s.addText(where, { x: W - M - 2.2, y: y + 0.24, w: 2.0, h: 0.36, fontSize: 12.5, color: MUTED, align: "right", fontFace: "Helvetica" });
    y += 0.98;
  });
  footer(s, 2);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3 · How we planned it — stories
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  header(s, "We planned the work as stories, not as a to-do list", "how we planned it · 1 of 3");
  bullets(s, [
    { text: "One story = one pull request = one or two work sessions.", bold: true },
    { text: "If it grows past that, it gets split — review quality collapses on large diffs.", sub: true },
    { text: "22 stories written out in full.", bold: true },
    { text: "7 foundations, 15 reconstruction engine. Later phases are titles only, elaborated one sprint ahead — most detail written in September would be wrong by November.", sub: true },
    { text: "The ID says where the work lives, not its priority.", bold: true },
    { text: "F foundations · R reconstruction · C capture · I inspection · A generative AI. R-04 is not \"step 4\" — the Depends-on line is the only ordering that matters.", sub: true },
    { text: "Every story carries a state, and a script keeps them honest.", bold: true },
    { text: "Blocked → Ready → Claimed → In review → Done.", sub: true },
  ], { y: 1.72, w: 6.35, fontSize: 14 });
  shot(s, "story-file-anatomy.jpg", {
    x: 7.35, y: 1.78, w: 5.36, h: 3.43,
    caption: "Every story is a file with the same shape: dependencies, state, owner, testable acceptance criteria, working notes, and a completion record that takes real hours — not the estimate copied across.",
  });
  footer(s, 3);
}

// ═══════════════════════════════════════════════════════════════════════════
// 4 · How we work — GitHub
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  header(s, "Nothing reaches main except through a reviewed pull request", "how we planned it · 2 of 3");

  s.addText("Eight branch-protection rules on main — and we proved each one by trying to break it, not by trusting the checkbox.",
    { x: M, y: 1.6, w: W - 2 * M, h: 0.4, fontSize: 13.5, color: MUTED, italic: true, fontFace: "Helvetica" });

  const rows = [
    [th("Gate"), th("What it enforces")],
    [td("Pull request required"), td("No direct pushes to main — proven: a direct push was rejected (GH006)")],
    [td("1 approving review"), td("Proven: a merge without approval was refused by policy")],
    [td("Stale approvals dismissed"), td("New commits invalidate an existing approval")],
    [td("Status checks required"), td("ruff + TypeScript must pass before merge is possible")],
    [td("Branch up to date"), td("No merging against a stale main")],
    [td("Force push / deletion blocked"), td("History cannot be rewritten")],
    [td("Admins cannot bypass", { bold: true, color: ACCENT }), td("Applies to the repository owner too — nobody has an override", { bold: true })],
    [td("Conversation resolution"), td("Review threads must be resolved, not ignored")],
  ];
  table(s, rows, { y: 2.12, colW: [3.4, 8.69], rowH: 0.36 });

  s.addText("Author cannot approve their own PR — a GitHub rule, not a policy choice. The reviewer's job ends at approval; the author merges.",
    { x: M, y: 5.72, w: W - 2 * M, h: 0.4, fontSize: 13, color: INK, fontFace: "Helvetica" });
  footer(s, 4);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5 · Decisions + automation
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  header(s, "Decisions are recorded with their reasoning, and rules are automated", "how we planned it · 3 of 3");

  statCard(s, M, 1.68, 2.7, "32", "decisions logged\nwith the why, not just the what");
  statCard(s, M + 2.95, 1.68, 2.7, "13", "automated tests\nrun on every pull request", GOOD);
  statCard(s, M + 5.9, 1.68, 2.7, "2", "required CI checks\nruff · TypeScript", GOOD);
  statCard(s, M + 8.85, 1.68, 2.7, "0", "rules that rely\non remembering", ACCENT2);

  bullets(s, [
    { text: "A decision without its reasoning gets reversed by whoever forgets it.", bold: true },
    { text: "D-005, the hard rule: geometry the camera did not see can never pass or fail a part — only trigger a Rescan. Enforced in the data format, with a test that fails the build if a report violates it.", sub: true },
    { text: "Where a rule could be a script, it is a script.", bold: true },
    { text: "A checklist raises the odds a person does the right thing; it does not verify they did.", sub: true },
    { text: "Tests are written during implementation — a joint obligation.", bold: true },
    { text: "Engineer and AI assistant both. Never deferred to a follow-up story.", sub: true },
  ], { y: 3.24, w: 6.35, fontSize: 13.5, h: 3.2 });
  shot(s, "dod-checklist.jpg", {
    x: 7.35, y: 3.3, w: 5.36, h: 2.46,
    caption: "The definition-of-done checklist on a live pull request — unticked boxes are the point. This one is held open because D-031 requires both other members to approve a change to the file that rule itself governs.",
  });
  footer(s, 5);
}

// ═══════════════════════════════════════════════════════════════════════════
// 6 · What we said we'd do
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  header(s, "What we said we would do", "question 1");
  bullets(s, [
    { text: "From the proposal, presented 5 September:", bold: true },
    { text: "Tier 1 — photos in, metric 3D model out. Never cut.", sub: true },
    { text: "Tier 2 — CAD alignment, deviation heatmap, Pass / Fail / Rescan / Unverifiable verdict.", sub: true },
    { text: "Tier 3 — stretch. Generative-AI components are never cut, regardless of tier.", sub: true },
    { text: "Three people × ~9 h/week × ~13 weeks ≈ 349 person-hours.", bold: true },
    { text: "Nine sprints, gates M0–M8, seven graded reviews.", sub: true },
    { text: "For this first stretch specifically: stand up the repository, freeze the interface, get CI running, provision the GPU, and shoot the golden capture set.", bold: true },
  ], { y: 1.75, fontSize: 16 });
  footer(s, 6);
}

// ═══════════════════════════════════════════════════════════════════════════
// 7 · What's done — the numbers
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  header(s, "What works now", "question 2 · the numbers");

  statCard(s, M, 1.7, 2.7, "4 / 7", "foundation stories Done\nmerged, reviewed, tested", GOOD);
  statCard(s, M + 2.95, 1.7, 2.7, "19", "pull requests opened\n15 merged · 2 closed · 2 open");
  statCard(s, M + 5.9, 1.7, 2.7, "2", "stories in flight\nGPU pod · golden capture", ACCENT2);
  statCard(s, M + 8.85, 1.7, 2.7, "13", "tests green\non every PR", GOOD);

  s.addText("Demonstrable today", { x: M, y: 3.3, w: 6, h: 0.36, fontSize: 17, bold: true, color: ACCENT, fontFace: "Helvetica" });
  bullets(s, [
    "The public repository, with all eight protection gates proven by trying to break them.",
    "CI running on every pull request — a lint failure now blocks a merge.",
    "The frozen interface contract, with 13 tests, including one that enforces the hard rule.",
    "The printed ChArUco reference board, verified at 100% scale with calipers.",
    "Three golden-capture sessions shot: two objects, two lighting conditions, 79 frames.",
  ], { y: 3.76, fontSize: 14.5, h: 2.4 });

  s.addText("No runnable reconstruction pipeline yet — that is honest, and it is what the next two weeks are for.",
    { x: M, y: 6.28, w: W - 2 * M, h: 0.4, fontSize: 13.5, bold: true, color: WARN, fontFace: "Helvetica" });
  footer(s, 7);
}

// ═══════════════════════════════════════════════════════════════════════════
// 7b · Hard evidence
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  header(s, "The parts that are real today", "question 2 · evidence, not claims");

  shot(s, "gpu-smoke-test-pass.jpg", {
    x: M, y: 1.78, w: 6.0, h: 3.05,
    caption: "GPU workspace running the shared smoke test: Tesla T4, PyTorch 2.8.0+cu128, CUDA available, GPU CHECK: PASS.",
  });
  shot(s, "golden-capture-lock.jpg", {
    x: 7.0, y: 1.78, w: 5.71, h: 3.05,
    caption: "Golden capture in progress — printed ChArUco board verified at 100% scale with calipers, part centred, whole sheet in frame.",
  });

  s.addText("Both are honestly incomplete", { x: M, y: 5.42, w: 6, h: 0.32, fontSize: 15, bold: true, color: WARN, fontFace: "Helvetica" });
  bullets(s, [
    { text: "F-05 — GPU access is proven for one member, not three. RunPod needed a payment method for the network volume, so Lightning AI was evaluated as a no-upfront-cost alternative. Which platform we standardise on is an open team decision, and the story stays Claimed until all three can connect and re-provisioning has been rehearsed." },
    { text: "F-06 — two of three objects shot, 79 frames, both lighting conditions done for the lock. Still needed: a third object and a CAD model for at least one of them." },
  ], { y: 5.74, fontSize: 12, h: 1.25 });
  footer(s, 8);
}

// ═══════════════════════════════════════════════════════════════════════════
// 8 · Story board
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  header(s, "Foundations, story by story", "question 2 · the detail");

  const rows = [
    [th("Story"), th("What it delivered"), th("State"), th("Owner"), th("PR"), th("Actual / est")],
    [td("F-01"), td("Public repo, 8 protection rules, proven"), td("Done", { color: GOOD, bold: true }), td("@TabeenRaoof"), td("#1"), td("— / 2 h")],
    [td("F-02"), td("Repo skeleton, module boundaries, packaging"), td("Done", { color: GOOD, bold: true }), td("@dalwalyk"), td("#6"), td("3 h / 3 h")],
    [td("F-03"), td("Interface contract v1, frozen + 6 tests"), td("Done", { color: GOOD, bold: true }), td("@TabeenRaoof"), td("#13"), td("4.5 h / 3 h", { color: WARN })],
    [td("F-04"), td("CI skeleton — ruff, tsc, eslint, required"), td("Done", { color: GOOD, bold: true }), td("@dalwalyk"), td("#11, #12"), td("— / 3 h")],
    [td("F-05"), td("GPU pod, three-way access"), td("Claimed", { color: ACCENT2, bold: true }), td("@mbj1994"), td("in flight"), td("— / 4 h")],
    [td("F-06"), td("Golden capture set"), td("Claimed", { color: ACCENT2, bold: true }), td("@mbj1994"), td("in flight"), td("— / 5 h")],
    [td("F-07"), td("Reproducible dev environment"), td("Blocked", { color: MUTED }), td("—"), td("—"), td("— / 3 h")],
  ];
  table(s, rows, { y: 1.82, colW: [0.95, 5.0, 1.25, 1.85, 1.45, 1.59], rowH: 0.42 });

  s.addText("Two of the four completed stories have no actual hours recorded. We have since made that a rule: the field takes a measured number or stays visibly blank — never the estimate copied across.",
    { x: M, y: 5.6, w: W - 2 * M, h: 0.55, fontSize: 13, color: WARN, fontFace: "Helvetica" });
  footer(s, 9);
}

// ═══════════════════════════════════════════════════════════════════════════
// 9 · PR / review evidence
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  header(s, "Every change was reviewed — including the ones we got wrong", "question 2 · the process working");

  shot(s, "peer-review-demands-tests.jpg", {
    x: M, y: 1.72, w: 6.0, h: 2.94,
    caption: "A reviewer blocking a merge on our own rules: \u201cit adds 134 lines of state-management logic without any automated tests, which violates the current definition of done in AGENTS.md\u201d \u2014 followed by the exact test cases required.",
  });
  shot(s, "pr-approval-gate.jpg", {
    x: 7.0, y: 1.72, w: 5.71, h: 2.94,
    caption: "The gate itself: \u201cAt least 1 approving review is required to merge this pull request.\u201d It applies to the repository owner too \u2014 nobody has an override.",
  });
  bullets(s, [
    { text: "Review also caught two defects that looked clean: a merge that silently duplicated a story's status block, and a second that silently dropped an entire decision entry. Both reported as mergeable by GitHub; both found by reading the merge, not trusting it." },
    { text: "Two pull requests were closed rather than merged — one redundant, one based on a misreading of our own approval rule. That disagreement was settled by quoting the rule, not by seniority.", color: ACCENT },
  ], { y: 5.5, fontSize: 12.5, h: 1.3 });
  footer(s, 10);
}

// ═══════════════════════════════════════════════════════════════════════════
// 10 · THE CHANGE OF PLAN
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  header(s, "Change of plan: three lanes became one shared front", "the main change since the proposal");

  // BEFORE
  s.addShape(pres.ShapeType.roundRect, { x: M, y: 1.72, w: 5.75, h: 3.0, fill: { color: "FDF2F2" }, line: { color: "F5D0D0" }, rectRadius: 0.05 });
  s.addText("BEFORE — as proposed", { x: M + 0.22, y: 1.86, w: 5.3, h: 0.34, fontSize: 13, bold: true, color: BAD, charSpacing: 1, fontFace: "Helvetica" });
  const before = [
    ["Lane 1 — Capture front end", "one owner · 97 h"],
    ["Lane 2 — Reconstruction backend", "one owner · 97 h"],
    ["Lane 3 — Inspection and truth", "one owner · 97 h"],
  ];
  let by = 2.3;
  before.forEach(([l, r]) => {
    s.addShape(pres.ShapeType.rect, { x: M + 0.22, y: by, w: 5.3, h: 0.62, fill: { color: PAPER }, line: { color: "F0DADA" } });
    s.addText(l, { x: M + 0.38, y: by + 0.06, w: 3.6, h: 0.3, fontSize: 13, bold: true, color: INK, fontFace: "Helvetica" });
    s.addText(r, { x: M + 0.38, y: by + 0.32, w: 4.9, h: 0.26, fontSize: 11, color: MUTED, fontFace: "Helvetica" });
    by += 0.72;
  });
  s.addText("Three people building three different things in parallel from day one.",
    { x: M + 0.22, y: 4.46, w: 5.3, h: 0.3, fontSize: 11.5, italic: true, color: MUTED, fontFace: "Helvetica" });

  // arrow
  s.addText("→", { x: 6.45, y: 2.9, w: 0.6, h: 0.6, fontSize: 34, bold: true, color: ACCENT, align: "center", fontFace: "Helvetica" });

  // AFTER
  s.addShape(pres.ShapeType.roundRect, { x: 7.05, y: 1.72, w: 5.66, h: 3.0, fill: { color: "EFF7F8" }, line: { color: "BFDCE2" }, rectRadius: 0.05 });
  s.addText("NOW — as we are actually working", { x: 7.27, y: 1.86, w: 5.2, h: 0.34, fontSize: 13, bold: true, color: ACCENT, charSpacing: 1, fontFace: "Helvetica" });
  const after = [
    ["Stage 1 — Foundations", "all three · repo, contract, CI, pod, fixtures"],
    ["Stage 2 — Reconstruction engine", "all three · the hardest risk, retired first"],
    ["Stage 3 — Capture · Inspection · AI", "built against something that already works"],
  ];
  let ay = 2.3;
  after.forEach(([l, r]) => {
    s.addShape(pres.ShapeType.rect, { x: 7.27, y: ay, w: 5.22, h: 0.62, fill: { color: PAPER }, line: { color: "CFE4E9" } });
    s.addText(l, { x: 7.43, y: ay + 0.06, w: 4.0, h: 0.3, fontSize: 13, bold: true, color: INK, fontFace: "Helvetica" });
    s.addText(r, { x: 7.43, y: ay + 0.32, w: 4.9, h: 0.26, fontSize: 11, color: MUTED, fontFace: "Helvetica" });
    ay += 0.72;
  });
  s.addText("Lanes are now places in the repository, not people.",
    { x: 7.27, y: 4.46, w: 5.2, h: 0.3, fontSize: 11.5, italic: true, color: MUTED, fontFace: "Helvetica" });

  s.addText("Why we changed it", { x: M, y: 4.94, w: 6, h: 0.34, fontSize: 15, bold: true, color: ACCENT, fontFace: "Helvetica" });
  bullets(s, [
    { text: "Total hours do not set the finish date — the serial chain does. About 42 of the 93 engine hours can only run one story after another, so that chain advances at one person's pace no matter how many people are free. Adding people to it changes nothing." },
    { text: "The reconstruction engine is the hardest risk in the project. Retiring it first means everything downstream is built against something real instead of a guess." },
  ], { y: 5.3, fontSize: 13, h: 1.5 });
  footer(s, 11);
}

// ═══════════════════════════════════════════════════════════════════════════
// 11 · NEW GANTT
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  header(s, "Re-planned timeline", "the same milestones, a different shape of work");

  // chart geometry
  const CX = M + 2.55;               // left edge of bar area
  const CW = W - M - CX - 0.15;      // width of bar area
  const T0 = new Date("2026-09-03").getTime();
  const T1 = new Date("2026-12-18").getTime();
  const px = (d) => CX + ((new Date(d).getTime() - T0) / (T1 - T0)) * CW;

  const TOP = 2.16;
  const ROW = 0.365;

  // month gridlines
  ["2026-09-03", "2026-10-01", "2026-11-01", "2026-12-01", "2026-12-18"].forEach((d, i) => {
    const x = px(d);
    s.addShape(pres.ShapeType.line, { x, y: TOP - 0.42, w: 0, h: 0.42 + ROW * 9 + 0.1, line: { color: "E3E9ED", width: 0.75 } });
    const lbl = ["Sep", "Oct", "Nov", "Dec", ""][i];
    if (lbl) s.addText(lbl, { x: x + 0.03, y: TOP - 0.66, w: 0.8, h: 0.24, fontSize: 10.5, color: MUTED, fontFace: "Helvetica" });
  });

  const bars = [
    { label: "Foundations", note: "all three", from: "2026-09-05", to: "2026-09-24", color: GOOD, done: true },
    { label: "Engine — parallel starts", note: "R-01 · R-02 · R-11 · R-12", from: "2026-09-22", to: "2026-10-08", color: ACCENT2 },
    { label: "Engine — the serial chain", note: "R-04→R-06→R-07→R-08→R-14→R-15", from: "2026-09-30", to: "2026-10-31", color: ACCENT, crit: true },
    { label: "Engine — off-chain work", note: "R-03 · R-05 · R-09 · R-10 · R-13", from: "2026-10-05", to: "2026-10-27", color: ACCENT2 },
    { label: "Thin capture thread", note: "C-01→C-03, in spare time", from: "2026-10-05", to: "2026-10-17", color: "94A3B8" },
    { label: "Capture app", note: "C-04→C-12", from: "2026-10-28", to: "2026-11-25", color: ACCENT2 },
    { label: "Inspection", note: "I-01→I-10", from: "2026-11-03", to: "2026-12-02", color: ACCENT2 },
    { label: "Generative AI", note: "A-01→A-09 · never cut", from: "2026-10-26", to: "2026-12-02", color: "7C3AED" },
    { label: "Integration · demo · defence", note: "I1–I4, recorded backup", from: "2026-12-02", to: "2026-12-18", color: "475569" },
  ];

  bars.forEach((b, i) => {
    const y = TOP + i * ROW;
    s.addText(b.label, { x: M, y: y + 0.015, w: 2.5, h: 0.2, fontSize: 10.5, bold: true, color: INK, fontFace: "Helvetica" });
    s.addText(b.note, { x: M, y: y + 0.185, w: 2.5, h: 0.18, fontSize: 8.5, color: MUTED, fontFace: "Helvetica" });
    const x0 = px(b.from), x1 = px(b.to);
    s.addShape(pres.ShapeType.roundRect, {
      x: x0, y: y + 0.045, w: Math.max(x1 - x0, 0.1), h: 0.245,
      fill: { color: b.color }, line: { color: b.color }, rectRadius: 0.03,
    });
    if (b.crit) s.addText("critical chain — sets the finish date", { x: x0 + 0.08, y: y + 0.06, w: 3.4, h: 0.22, fontSize: 8.5, bold: true, color: "FFFFFF", fontFace: "Helvetica" });
    if (b.done) s.addText("done", { x: x0 + 0.06, y: y + 0.06, w: 1.0, h: 0.22, fontSize: 8.5, bold: true, color: "FFFFFF", fontFace: "Helvetica" });
  });

  // milestones
  const ms = [
    ["2026-09-25", "M1"], ["2026-10-09", "M2"], ["2026-10-16", "M3"],
    ["2026-10-30", "M4"], ["2026-11-13", "M5"], ["2026-12-04", "M7"], ["2026-12-18", "M8"],
  ];
  const msY = TOP + bars.length * ROW + 0.08;
  ms.forEach(([d, l]) => {
    const x = px(d);
    s.addShape(pres.ShapeType.diamond, { x: x - 0.065, y: msY, w: 0.13, h: 0.17, fill: { color: INK }, line: { color: INK } });
    s.addText(l, { x: x - 0.28, y: msY + 0.18, w: 0.56, h: 0.2, fontSize: 8.5, color: INK, align: "center", fontFace: "Helvetica" });
  });

  // today marker
  const tx = px("2026-09-18");
  s.addShape(pres.ShapeType.line, { x: tx, y: TOP - 0.42, w: 0, h: 0.42 + ROW * 9 + 0.1, line: { color: BAD, width: 1.5, dashType: "dash" } });
  s.addText("today", { x: tx - 0.32, y: TOP - 0.68, w: 0.7, h: 0.22, fontSize: 9, bold: true, color: BAD, align: "center", fontFace: "Helvetica" });

  s.addText("M1 (25 Sept) is at risk. The chain needs ~40 hours to reach a working endpoint — about four and a half weeks from a 3 Sept start, which lands in early October, not late September. We are flagging it now rather than at the gate.",
    { x: M, y: 6.22, w: W - 2 * M, h: 0.5, fontSize: 12, color: WARN, fontFace: "Helvetica" });
  footer(s, 12);
}

// ═══════════════════════════════════════════════════════════════════════════
// 12 · What slipped
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  header(s, "What slipped, and why", "question 3");
  bullets(s, [
    { text: "No runnable reconstruction pipeline yet.", bold: true, color: WARN },
    { text: "Two weeks went into foundations and process. That was deliberate — but it means nothing reconstructs a part today.", sub: true },
    { text: "M1 (25 Sept) will not be met as scheduled.", bold: true, color: WARN },
    { text: "The critical-chain arithmetic says early October. We are saying so now, a week before the gate, rather than at it.", sub: true },
    { text: "F-03 took 4.5 hours against a 3-hour estimate.", bold: true },
    { text: "Five API operations, ten worked examples, six tests and a new contributing rule. Recorded as the real number — that is how the next estimate gets less wrong.", sub: true },
    { text: "Two completed stories have no actual hours recorded at all.", bold: true },
    { text: "Caught in review. Now a written rule: measured number, or visibly blank — never the estimate copied across, which looks identical to a real measurement.", sub: true },
    { text: "Golden capture needed three attempts.", bold: true },
    { text: "First set framed the part too small; the retake caught the phone silently switching lenses mid-session, which would have invalidated the camera calibration.", sub: true },
  ], { y: 1.68, fontSize: 13, h: 5.0 });
  footer(s, 13);
}

// ═══════════════════════════════════════════════════════════════════════════
// 13 · Next two weeks
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  header(s, "What we will have by the next review", "question 4 · by 2 October");

  const rows = [
    [th("Story"), th("What it delivers"), th("Owner")],
    [td("F-05"), td("GPU pod provisioned, all three members able to connect and run a smoke test"), td("@mbj1994")],
    [td("F-06"), td("Golden capture set complete — third object, CAD model, download script + checksum"), td("@mbj1994")],
    [td("F-07"), td("Reproducible environment — one command, same result on laptop and pod"), td("unclaimed")],
    [td("R-02"), td("ChArUco board detected, metric camera pose per frame"), td("unclaimed")],
    [td("R-01"), td("The reconstruction model run on the golden capture, end to end"), td("unclaimed")],
    [td("R-04"), td("Metric alignment — the bottleneck the whole chain waits on"), td("unclaimed")],
  ];
  table(s, rows, { y: 1.85, colW: [1.1, 8.5, 2.49], rowH: 0.44 });

  s.addText("The commitment", { x: M, y: 5.05, w: 6, h: 0.34, fontSize: 16, bold: true, color: ACCENT, fontFace: "Helvetica" });
  s.addText("By the next review we will show the golden capture going into the reconstruction model and a metric point cloud coming out — a running pipeline, not slides. If the chain stalls, that is the first thing we will say.",
    { x: M, y: 5.42, w: W - 2 * M, h: 0.7, fontSize: 15, color: INK, fontFace: "Helvetica" });
  footer(s, 14);
}

// ═══════════════════════════════════════════════════════════════════════════
// 14 · Close
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  s.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: 0.28, h: H, fill: { color: ACCENT } });
  s.addText("Where we are", { x: 1.1, y: 1.5, w: 11, h: 0.7, fontSize: 38, bold: true, color: INK, fontFace: "Helvetica" });
  s.addShape(pres.ShapeType.rect, { x: 1.1, y: 2.3, w: 1.5, h: 0.045, fill: { color: ACCENT } });
  bullets(s, [
    { text: "The scaffolding is finished and proven — repository, contract, CI, review process.", bold: true },
    { text: "All three of us are now on the reconstruction engine, because that is the risk worth retiring first." },
    { text: "The interface is frozen, so capture and inspection can be built against it without waiting." },
    { text: "M1 will slip by roughly a week. We would rather say that now than at the gate." },
  ], { x: 1.1, y: 2.75, w: 11, fontSize: 17, h: 2.6 });
  s.addText("github.com/proofshape/proofshape", { x: 1.1, y: 5.7, w: 11, h: 0.4, fontSize: 15, color: ACCENT, bold: true, fontFace: "Helvetica" });
}

const OUT = process.argv[2] || "deck.pptx";
pres.writeFile({ fileName: OUT }).then(() => console.log("wrote " + OUT));

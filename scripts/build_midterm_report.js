// ProofShape — CS595 midterm report (Word). Mirrors the P-20 midterm deck, section for section.
// Committed as the review artifact for story P-21 (D-052): the team reviews this script's diff
// and the rendered .docx in the pull request, the same pattern scripts/build_review_deck.js set
// for the biweekly review decks.
//
//   node scripts/build_midterm_report.js [out.docx]   # defaults to submissions/midterm-report-draft.docx
//
// Content is scoped to everything from the start of term up to Thursday 15 October 2026 and
// nothing after (the team's own rule for this deliverable) -- verified against main at commit
// 3a373ed (2026-10-15 21:52 PDT, the day's last commit) and, where the text cites C-13's own
// branch directly, against that branch's last commit before midnight, d296be9 (23:29 PDT).
// Figures live in submissions/assets/2026-10-16/; re-render them from the pipeline outputs named
// in each caption if a figure needs updating, following R-17's precedent for ad hoc renders.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType,
  ShadingType, AlignmentType, HeadingLevel, LevelFormat, Footer, PageNumber, TabStopType,
  BorderStyle, PageBreak, VerticalAlign,
} = require("docx");

const FIG = path.join(__dirname, "..", "submissions", "assets", "2026-10-16");
const INK = "1A2326", MUTED = "5F6E74", TEAL = "16697A", TINT = "EEF4F5", RULE = "CDD6D9";
const BODY_FONT = "Calibri", HEAD_FONT = "Cambria";
const CONTENT_W_IN = 6.5;
const DXA = (inches) => Math.round(inches * 1440);

// ── inline markup: **bold** only ───────────────────────────────────────────
function runs(text, base = {}) {
  const out = [];
  const parts = String(text).split(/(\*\*[^*]+\*\*)/g);
  for (const part of parts) {
    if (!part) continue;
    if (part.startsWith("**") && part.endsWith("**")) out.push(new TextRun({ text: part.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: part, ...base }));
  }
  return out;
}

function imageSize(file) {
  const buf = fs.readFileSync(file);
  if (buf[0] === 0x89) return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
  let i = 2;
  while (i < buf.length) {
    const marker = buf[i + 1];
    const len = buf.readUInt16BE(i + 2);
    if (marker >= 0xc0 && marker <= 0xc3) return { h: buf.readUInt16BE(i + 5), w: buf.readUInt16BE(i + 7) };
    i += 2 + len;
  }
  throw new Error("cannot size " + file);
}

// ── content ─────────────────────────────────────────────────────────────────
// Placeholders {fig:key} and {tab:key} are replaced by "Figure N" / "Table N" in order of
// appearance, so the numbering can never drift from the text.
const DOC = [];
const H1 = (title, author) => DOC.push({ type: "h1", title, author });
const H2 = (title) => DOC.push({ type: "h2", title });
const P = (text) => DOC.push({ type: "p", text });
const BULLETS = (items) => DOC.push({ type: "bullets", items });
const FIGURE = (key, file, width, caption) => DOC.push({ type: "fig", key, file, width, caption });
const TABLE = (key, caption, widths, header, rows, opts = {}) => DOC.push({ type: "table", key, caption, widths, header, rows, ...opts });
const NOTE = (text) => DOC.push({ type: "note", text });
const BREAK = () => DOC.push({ type: "break" });

// Summary
H1("Summary", "All three members");
P("ProofShape is a web application that turns a smartphone into a coarse 3D inspection tool for manufactured parts. A supplier places a part on a printed reference board, photographs it from guided positions, and gets back a 3D model in real millimetres in which every surface is labelled as either genuinely observed or not. Later in the term, that model will be compared with the buyer's CAD file to give a Pass, Fail, Rescan or Unverifiable result.");
P("In the first six weeks the team built the core of the system, Tier 1, end to end. That covers a reconstruction service with eight stages (board pose, learned 3D reconstruction, metric alignment, segmentation, fusion, mesh cleanup, provenance and uncertainty, and export), a browser-based capture app, and a containerised deployment on a cloud GPU. By 15 October the reconstruction service and the capture app were each proven independently; connecting them on a real phone was the one link not yet confirmed end to end.");
P("Testing on real photographs exposed three systematic problems that the synthetic unit tests could not: stray low-confidence points, a misplaced board plane, and a 4–9% scale error. The first two are fixed. The third has a diagnosed cause and a planned fix. Still to come are the ground-truth calibration study, Tier 2 inspection, and the five generative-AI components, which are the priority for the second half of the term.");
P("**No accuracy figure is claimed anywhere in this report.** Accuracy and tolerance come from the calibration study (milestone M3), which has not run yet. Every millimetre value below is either a measured property of a specific run, labelled as such, or a working default the team chose.");

// 1 · Introduction
H1("1. Introduction", "Mamadou");
H2("1.1 The problem");
P("Parts that ship out of specification cost a second shipment and days to weeks of delay, and they are hard to audit afterwards. Coordinate measuring machines (CMMs) are accurate but tied to a lab. A photo emailed by the supplier shows nothing quantitative in 3D. And a scan file from the supplier's own scanner is only as trustworthy as the supplier. There is no quick, phone-only, quantitative check that a buyer has reason to believe.");
P("ProofShape aims to fill that gap at the workbench, with a phone camera, a printed reference sheet and a browser: no app install and no special hardware. The output is a 3D model plus a report that says what was measured, what was not seen, and what the method cannot resolve for this part [21].");
H2("1.2 What the user does");
BULLETS([
  "**Buyer (planned).** Uploads the CAD file and the tolerances that matter, and receives a capture link and a printable reference sheet for the order.",
  "**Supplier (built).** Opens the link on a phone, places the part on the printed sheet, and photographs it from guided positions around the part. Blurry, badly exposed or duplicate frames are rejected on the phone with a reason.",
  "**Result.** The service returns a metric 3D model, coloured by what the camera actually observed. With a CAD file, it will also return a verdict and a deviation heatmap (Tier 2, not yet built).",
]);
H2("1.3 Scope and delivery tiers");
P("The project is delivered in tiers, so that a working product exists at every milestone. Tier 1 is never cut. The course also requires roughly a third of the project to be generative AI, and those components carry the same protection as Tier 1 (decision D-017).");
TABLE("tiers", "Delivery tiers, from the project proposal [21].", [1.4, 3.6, 1.5],
  ["Tier", "Deliverable", "Status (15 Oct)"],
  [
    ["1 · Core", "Phone photos → metric 3D model in the browser, with an observed/unobserved map; VLM supervision of capture", "Components built; end-to-end phone connection not yet confirmed; VLM not built"],
    ["2 · Inspection", "CAD alignment, deviation heatmap, verdict logic, condition report, plain-English inspection spec, generative shape completion, LLM capture agent", "Not started"],
    ["3 · Stretch", "Defect-aware guidance, session integrity checks, LLM report narrative", "Not started"],
  ]);
H2("1.4 Two rules that shape every design decision");
BULLETS([
  "**The hard rule (D-005).** Geometry the camera did not properly see can never pass or fail a part. It can only trigger a Rescan. A model's guess must never approve or condemn a component. This rule is built into the data format and checked by a test (Section 7).",
  "**Accuracy honesty (D-003).** This is coarse inspection, several millimetres on hand-sized parts. It complements a CMM and never replaces one. No tolerance is stated until the calibration study has measured one.",
]);

// 2 · Team
H1("2. Team Information", "Mamadou");
P("The team has three members. We work from one shared backlog of user stories: any member can take any story in any part of the code (\"lanes are places, not people\", D-020), one person owns each story, and a different member always reviews it (D-021). Table {tab:team} lists what each member built and which parts of this report they wrote. Each section heading also names its author.");
TABLE("team", "Team members, their main contributions this term, and the report sections each wrote. Story IDs are listed in Appendix A.", [1.55, 1.05, 2.9, 1.0],
  ["Member", "Student ID", "Main work this term", "Report sections"],
  [
    ["Mamadou [surname]\n@mbj1994", "[ID]", "Shared GPU workspace (F-05); golden capture set (F-06, with Tabeen); camera calibration (R-03); MASt3R and COLMAP backbones (R-11, R-12); reconstruction API service (R-14); σ baseline (R-19); result screen, mock service, 3D viewer and provenance colouring (C-10–C-12)", "Summary (joint), 1–5"],
    ["Yashi [surname]\n@dalwalyk", "[ID]", "Repository skeleton, CI and dev environment (F-02, F-04, F-07); VGGT runner (R-01); metric alignment and scale check (R-04, R-05); segmentation, fusion and mesh cleanup (R-06–R-08); provenance and uncertainty (R-09, R-10); mesh check (R-18); gyro, app shell, live camera, frame gate, orbit guidance and capture loop (C-03–C-07, C-09); connection to the service (C-13, with Tabeen)", "Summary (joint), 6–9"],
    ["Tabeen Raoof\n@TabeenRaoof", "[ID]", "Repository and branch protection, API contract (F-01, F-03); board detection and camera pose (R-02); latency instrumentation (R-13); container and cloud deployment (R-15); real-data checks and fixes (R-17, R-20–R-22, R-24); upload page and camera permission (C-01, C-02); next-view scoring (C-08, in review); the first real-phone attempt (C-13, with Yashi)", "Summary (joint), 10–14, References, Appendix A"],
  ]);
P("Capacity is about 9 hours per person per week over the semester, roughly 349 person-hours in total [21]. The midterm presentation follows the same split: Mamadou presents slides 1–5, Yashi 6–9 and Tabeen 10–15.");

// 3 · System overview and foundations
H1("3. System Overview and Foundations", "Mamadou");
P("{fig:overview} shows the whole system. The capture app runs in the phone's browser. It sends photos and gyroscope readings over HTTPS to the reconstruction service through a frozen API. The service runs eight stages, in order, on every finished session:");
BULLETS([
  "**Stage 1** detects the printed board in each photo and, where enough of its corners are found, solves where the camera was, in millimetres (Section 5).",
  "**Stage 2** runs a large learned model that reconstructs camera poses and per-pixel depth from all the photos at once (Section 4).",
  "**Stage 3** aligns that reconstruction to the board, which gives it real units (Section 5).",
  "**Stages 4–6** cut the part away from the board, fuse the depth maps into a mesh, and clean it (Section 6).",
  "**Stage 7** labels every vertex Observed or Unobserved and gives it an uncertainty (Section 7).",
  "**Stage 8** returns the result as a GLB model, with those labels stored as vertex colours [15].",
]);
FIGURE("overview", "fig_overview.png", 6.5, "ProofShape system overview. The capture app (left) talks to the reconstruction service (right) through a frozen HTTPS API, and the eight service stages run in order on every finished session. The bottom row lists the four foundations built first, so that three people could work in parallel (Section 3.1). Every component drawn here is built, but the live link between the capture app and the service had not yet been confirmed on a real phone by 15 October (Section 10.3). Tier 2 inspection and the generative-AI components are not built.");
H2("3.1 Foundations built first");
P("The first three weeks went into four pieces of groundwork whose job is to stop the three of us from blocking each other.");
BULLETS([
  "**Frozen API contract (F-03).** One OpenAPI file defines five operations: create a session, upload a frame, finish, get the reconstruction, and get the verdict. It comes with example requests and responses and contract tests, including one that fails if an unobserved region is ever marked as contributing to a verdict. The phone app and the service are each tested against this contract, which is why the app could be built against a mock service before the real one existed. Changing the contract needs both other members' approval, at a sprint boundary (D-031).",
  "**Golden capture set (F-06).** 142 real phone photos of four objects in five sets (Table {tab:golden}, {fig:golden}). Every stage in this report was tested on these photos. They live outside git, behind a download script with a checksummed manifest, so the repository never stores photographs.",
  "**Shared GPU workspace (F-05, F-07).** A Lightning AI Studio with a Tesla T4 that all three members can use (D-036) [19]. One bootstrap command sets up the same environment on an Apple Silicon laptop or on that GPU machine.",
  "**Protected repository (F-01, F-02, F-04).** A public repository in a team organisation. Nothing reaches main without a pull request and a teammate's approval; new commits dismiss stale approvals; and administrators cannot bypass the rules. CI runs linting, the Python tests and the TypeScript checks on every pull request.",
]);
TABLE("golden", "The golden capture set (F-06). The CAD files for s-04 and s-05 are committed to the repository; no photographs are.", [0.8, 2.6, 0.8, 2.3],
  ["Sample", "Object", "Frames", "Notes"],
  [
    ["s-01", "Metal padlock", "26", "Reflective; lighting condition A"],
    ["s-02", "The same padlock", "30", "Lighting condition B"],
    ["s-03", "Rounded rigid object", "27", "Smooth, low texture"],
    ["s-04", "Yellow triangular prism (printed)", "30", "Reference CAD available"],
    ["s-05", "Red rectangular prism, 30 × 30 × 60 mm (printed)", "29", "Reference CAD available"],
    ["Total", "", "142", ""],
  ]);
FIGURE("golden", "fig_golden.jpg", 4.6, "Shooting golden sample s-01: a reflective metal padlock in the middle of the printed 20 mm ChArUco reference board, with the whole sheet in frame. All five golden sets were shot this way, on a phone, before any capture app existed.");

// 4 · Backbones
H1("4. Reconstruction Backbones", "Mamadou");
P("Stage 2, learned multi-view reconstruction, is the riskiest part of the system, so we made it swappable. **VGGT** [1] is the primary backbone (R-01). It is a feed-forward transformer that takes all the photos in one pass and returns a camera pose and a depth value for every pixel. We pin it to a specific upstream commit. Its released weights are licensed for non-commercial use, which is fine for a course project and is recorded in the story.");
P("Two alternatives sit behind a single configuration value (D-009). **MASt3R** [2] (R-11) does pairwise 3D matching followed by an iterative global alignment. **COLMAP** [3], [4] (R-12) is classical structure-from-motion, which we seed with the board's camera poses as priors. All three write the same four output files (poses, depth, point cloud and a run report), so nothing downstream changes when the backbone does. We showed this by feeding MASt3R's output straight into metric alignment with no code changes, where it paired 29 of the 30 frames with their board poses.");
P("Per-stage timing (R-13) was measured for all three backbones on the same 30 golden photos (s-04) on the shared T4. {fig:latency} shows the two that completed. COLMAP registered only 28 of the 30 photos, so it **refused** the session after 80.6 s rather than invent the two missing views. We count that as a correct result: a confident wrong model is worse than a refusal.");
FIGURE("latency", "fig_latency.png", 6.3, "Measured wall-clock time per stage for the two backbones that completed, on the same 30 golden photos (s-04) on the shared Tesla T4 (R-13). VGGT's 80.2 s is dominated by loading the model; MASt3R's 265.9 s is almost all its iterative global alignment. These are single runs, not averages.");
P("The measurement also settled a design question. The proposal said an interactive capture loop, which reconstructs while the user is still shooting, would only be built if the latency table showed it was nearly free, at roughly 2 s per update. At about 80 s per session it is not, so **batch capture is the path we ship** (D-002): take all the photos, then reconstruct once.");
P("The backbone is described as a large learned model, not as generative AI (D-018). It is deterministic: the same photos always give the same output. It is counted separately from the generative-AI share (Section 14).");

// 5 · Metric scale
H1("5. Metric Scale from the Printed Board", "Mamadou");
P("VGGT's output has no units. Real millimetres come from the printed reference board: a ChArUco board [5], a checkerboard of 20 mm squares with 15 mm coded markers (dictionary DICT_5X5_250, 8 × 6 squares, 160 × 120 mm), printed at 100% scale from a PDF generated with calib.io's pattern generator [18]. Because every corner is individually identifiable, a board partly covered by the part still works. One detail mattered more than expected: the board generator's \"legacy\" setting has the opposite meaning in OpenCV [12]. With the wrong flag, markers were found but no corners, so every frame failed. A test now checks the board definition in code against the printed PDF (D-034).");
H2("5.1 Camera pose per photo (R-02)");
P("In each photo we detect the board's corners and solve the camera's pose relative to the board in millimetres, using SQPnP [6] refined by Levenberg–Marquardt. This worked on **138 of 142** golden photos. We inspected the four failures by eye: one was blurred and three were taken almost level with the table. {fig:cameras} shows the solved poses for one capture.");
FIGURE("cameras", "fig_cameras.png", 4.4, "Camera poses solved from the printed board for golden sample s-05 (R-02). Each teal dot is one of the 29 camera positions in board millimetres, and each line runs from the camera to the point on the board it was looking at. The red box marks where the part stands. The spread of positions is the orbit that the capture guidance asks for.");
H2("5.2 Lens calibration (R-03) and metric alignment (R-04, R-05)");
P("The same board corners calibrate the camera lens across a session. On s-04 this gave a reprojection error of **1.20 px** over 29 views. That is internal consistency, not accuracy. If calibration fails, the stage falls back to VGGT's predicted intrinsics and records that it did so.");
P("One similarity transform, solved with Umeyama's method [7], then maps VGGT's camera centres onto the board's camera centres. That puts the whole reconstruction in board millimetres. Per-frame residuals are reported; on s-04 the camera-centre residual was 10.0 mm RMS, which is an alignment diagnostic, not an object-accuracy figure.");
P("A separate check (R-05) compares three scale sources, each expressed as a ratio to the board pitch (D-039), and flags the session if any two disagree by more than about 2%. The three sources are the printed board pitch, an optional caliper reading, and the CAD registration residual. The 2% is a team rule, not a measured value. Nothing feeds the check yet: there is no caliper entry in the app and no CAD registration.");
H2("5.3 Finding: the scene comes out 4–9% small (R-22)");
P("The board itself is an exact ruler inside every reconstruction, so we measured it. {fig:boardscale} shows that the reconstructed board comes out **4–9% smaller** than its true size on all five golden objects. That shrinkage tracks, almost exactly, the ratio between the focal length our board poses were solved with and the focal length VGGT predicts. Because metric alignment fits camera centres, and a camera's distance from the scene scales with the focal length assumed for it, the focal mismatch turns directly into a scale error. The fix, **R-23**, anchors the scale on the board corners actually visible in the scene instead of on camera centres. R-22 filed it; it is not yet claimed.");
FIGURE("boardscale", "fig_board_scale.png", 6.3, "Using the board as a ruler inside each reconstruction (R-22). Dark bars show the size at which the 20 mm board comes out after metric alignment, as a fraction of its true size. Light bars show the ratio between the focal length used for the board poses and the focal length VGGT predicts. The two track each other on all five objects, which identifies the focal mismatch as the cause of the shrinkage.");
P("Part walls shrink somewhat more than the board does (on s-05, to 81–85% of nominal once the board's own scale error is divided out). We attribute this to VGGT's depth on surfaces seen at shallow angles, and it is still open. An earlier hand-picked measurement of a single s-04 edge in R-04 had agreed with a caliper reading to within 0.26 mm. R-22's repeatable, tested measurement supersedes it, and it is the one this report relies on.");

// 6 · Geometry
H1("6. From Photos to a Fused Mesh", "Yashi");
P("{fig:strip} follows one real object, golden sample s-05, through the geometry stages. Every image is the pipeline's real output.");
FIGURE("strip", "fig_pipeline_strip.jpg", 6.5, "The geometry chain on golden sample s-05, the red rectangular prism. (a) One of the 29 photos. (b) VGGT's per-pixel depth for all photos, converted to board millimetres, including the board and the table. (c) After segmentation: low-confidence pixels removed, and only points above the fitted board plane and inside the board's footprint kept. (d) The TSDF-fused mesh around the part before cleanup; the sheet-like surfaces around the part are fusion artifacts (Section 11).");
H2("6.1 Segmentation (R-06, R-21)");
P("Segmentation keeps only the points that stand above the board and lie inside its 160 × 120 mm footprint. The first version assumed the board sat exactly at height zero after alignment. On real data it sat 1–5 mm lower, so the cut removed the bottom of every part (Section 11). The current version (R-21, D-044) fits a plane to the real board surface around the part, using a robust least-squares fit with iterative outlier rejection. It sets the height cut from the board's own measured noise, with a 2 mm minimum. A fit with too few board points, or a board tilted more than 2°, is reported as an explicit error, never as a silent fallback. On the five golden objects the measured offsets were 0.9–5.4 mm and the tilt was under 0.8°.");
H2("6.2 TSDF fusion (R-07)");
P("Each photo's depth map is fused into a truncated signed distance field [8] of 2 mm voxels in the board frame. Every voxel is projected into every photo and compared with that photo's measured depth. This needs each camera's pose in board coordinates, which R-04 did not provide directly, so R-07 derives it from the alignment transform and checks it with a test that composes the two. The surface is extracted with marching cubes [9] (scikit-image [14]), with Poisson reconstruction [10] (Open3D [13]) as a fallback. Voxels that no photo observed are masked out, so they cannot create surface.");
H2("6.3 Cleanup and export (R-08)");
P("Cleanup keeps the largest connected piece of the mesh and fills only small holes (boundary loops of up to 8 edges); larger holes are left open on purpose. It then decimates the mesh with quadric error metrics [11] to 50,000 triangles and writes a GLB file [15] that a browser can load. The thresholds are documented, configurable defaults, not measured values.");
P("Every stage is its own module with a command-line tool, a JSON report and unit tests, so any one stage can be rerun and inspected on its own. The fused meshes still contain artifacts around the part. Section 7 shows why those artifacts cannot cause a wrong verdict, and Section 11 explains where they come from.");

// 7 · Provenance
H1("7. Provenance and Uncertainty: the Hard Rule in Code", "Yashi");
P("Stage 7 is where the hard rule (D-005) becomes code. Every mesh vertex is labelled **Observed** or **Unobserved** (R-09), and only Observed geometry will ever be allowed to pass or fail a part.");
BULLETS([
  "**Observed** means seen in at least two photos whose viewpoints are at least 2° apart (the triangulation angle), and **not hidden**: the vertex's position must agree, to within 5 mm, with the depth each of those photos actually measured. Without that occlusion check, a camera looking at the front of a part would count as having seen its back. Since R-24, a photo only counts if VGGT is confident about that pixel (the same confidence rule as the point cloud and fusion, D-045).",
  "**Uncertainty σ** is the spread of a vertex's depth across the photos that see it (R-10). A vertex with σ above 3 mm is downgraded to Unobserved. A vertex seen by fewer than two photos gets no σ at all, rather than a made-up zero.",
  "**Enforcement.** The labels travel inside the GLB as vertex colours. The frozen API contract forbids an unobserved region from contributing to a verdict, and a contract test fails the build if it ever does.",
]);
P("The 2°, 5 mm and 3 mm values are the team's working defaults. The calibration study (M3) will replace them with measured ones.");
FIGURE("provenance", "fig_provenance.jpg", 3.4, "Provenance labels on the cleaned s-05 mesh from the current pipeline (R-08 → R-09 → R-10, with R-24's confidence rule). Teal is Observed and grey is Unobserved. The Observed surface is concentrated on the part. The tall, wall-like fusion artifacts are all Unobserved, so under the hard rule they can only ever trigger a Rescan.");
P("Table {tab:provenance} gives the current figures for all five golden objects (R-24). The observed fraction is low because the fusion artifacts are labelled Unobserved, which is the rule working as intended; it will rise as the artifacts are fixed. The σ values describe the spread of depth across photos. They are not an accuracy figure.");
TABLE("provenance", "Provenance and uncertainty on the five golden objects, current pipeline (R-24 golden-set rerun). The observed fraction counts faces whose three vertices are all Observed. Median σ is taken over vertices that remain Observed after the 3 mm downgrade.", [0.9, 1.6, 1.4, 1.3, 1.3],
  ["Object", "Mesh vertices", "Observed fraction", "Median σ (mm)", "Downgraded by σ"],
  [
    ["s-01", "15,581", "9.8%", "2.10", "6.9%"],
    ["s-02", "25,013", "10.8%", "1.62", "5.5%"],
    ["s-03", "17,188", "39.2%", "2.02", "8.3%"],
    ["s-04", "31,335", "23.3%", "1.16", "3.8%"],
    ["s-05", "24,637", "13.0%", "1.75", "5.3%"],
  ]);

// 8 · Capture app
H1("8. The Capture App", "Yashi");
P("The capture app is a web page, so a supplier opens a link and installs nothing. It is written in TypeScript with Vite and no UI framework, and tested with Vitest (D-040). 11 of the 13 capture stories are merged; two are still open pull requests — C-08, next-view scoring (Section 9), and C-13, which connects the app to the real service (Section 10). Table {tab:capture} lists what each story added and how it was verified.");
P("Several behaviours can only be checked on a real phone, such as permission prompts, live video and the gyroscope. Those stories carry a written manual check on a real iPhone, in addition to their automated tests. Those device checks found real bugs that every automated test had missed. On C-02, for example, the camera preview never started playing, and the permission-denied message was present but easy to miss.");
FIGURE("mobile", "fig_mobile.jpg", 5.6, "The capture app at phone size (390 × 844 CSS pixels). (a) Capture screen: session started, live camera preview, two frames accepted and the orbit counter at 2 of 13; the small arrow at the bottom of the preview points toward the next position. (b) Result screen: the 3D viewer with its provenance legend and Download GLB link, showing our real s-05 mesh (blue = Observed, slate = Unobserved). Captured in desktop Chrome at phone size, with a golden-capture photo standing in for the camera and the mock service returning the s-05 model.");
FIGURE("desktop_capture", "fig_desktop_capture.jpg", 4.8, "The same capture screen in a desktop browser (1366 px wide), in the same state as {fig:mobile}(a). Each section heading names the story that built it. The page is deliberately plain: layout polish is planned for the device-testing story (C-18).");
FIGURE("desktop_viewer", "fig_desktop_viewer.jpg", 6.3, "The result screen in a desktop browser. The model can be orbited and zoomed with the mouse, or with one and two fingers on a phone. The raw GLB is always downloadable, even if the browser cannot render it.");
TABLE("capture", "Capture-app stories and how each was verified. \"Device check\" means a written manual check on a real iPhone, recorded in the story file.", [0.65, 2.75, 3.1],
  ["Story", "What it added", "Verified by"],
  [
    ["C-01", "Minimal upload page against the frozen contract, with a local contract stub", "Unit tests; mutation-checked"],
    ["C-02", "Camera permission over HTTPS", "Device check (iPhone 16) — found 2 bugs automated tests could not"],
    ["C-03", "Gyroscope permission flow (iOS)", "Device check (iPhone 18 Pro Max)"],
    ["C-04", "App shell: one tap creates the session and requests both permissions, respecting iOS's direct-tap rule", "Unit tests, including a regression test for the tap rule; device check"],
    ["C-05", "Live camera that stays on until the session ends; the caller owns releasing it (D-046)", "Device check over several minutes; four camera-left-on leaks found in review and fixed"],
    ["C-06", "On-phone frame gate: blur (Laplacian variance below 100), exposure (more than 5% clipped pixels), duplicate (gyro change below 2° and image difference below 5)", "Pure functions, unit tested; thresholds are unmeasured defaults"],
    ["C-07", "Orbit guidance: 13 target views (8 at 45°, 4 at 70°, 1 overhead) and a gyro-driven arrow", "Unit tests; device check walked a full orbit to 13/13"],
    ["C-09", "Capture loop: gate → JPEG → upload; an orbit slot counts only when the server accepts the frame", "Device check: all three rejection reasons, finish, camera light off"],
    ["C-10", "Result screen polling the service, and a mock service with a canned model", "Unit tests against the contract"],
    ["C-11", "3D viewer (three.js [16]) with orbit, pinch-zoom and download", "Unit tests with a mocked loader; device check in iPhone Safari"],
    ["C-12", "Provenance colouring and legend; a reserved slot for Inferred surface", "Unit tests, including a guard that fails if the server's colours change"],
    ["C-13", "Connection to the real service (Section 10)", "Two device-check attempts found a real integration gap; fix written, not yet re-verified on device; PR #146, in review"],
    ["C-08", "Next-view scoring (Section 9)", "Unit tests on synthetic meshes; PR #144, in review"],
  ]);

// 9 · Next-view scoring
H1("9. Next-View Scoring", "Yashi");
P("Once the system knows which surface is unobserved, the next question is where to point the camera. Next-view scoring (C-08, pull request #144, in review) answers it. The story was built by Tabeen; it is presented here because it completes the capture-guidance thread.");
BULLETS([
  "**Candidates.** 147 camera positions on a hemisphere centred on the part, at 25, 32.5 and 40 cm, and at most 60° from vertical. Each one looks at the part.",
  "**Visibility.** From each candidate, rays are cast with Open3D's ray-casting scene [13] (D-047) to every under-observed vertex. The whole mesh blocks the rays, so Observed surface still hides what is behind it.",
  "**Score.** A candidate's score is the number of under-observed vertices it would see. Ties go to the view that faces those surfaces most squarely.",
]);
FIGURE("nextview", "fig_nextview.png", 6.5, "Next-view scoring on C-08's synthetic test scene, scored by the real code. Each dot is a candidate camera seen from above; darker dots see more of the part's unseen face (the grey edge of the teal box). Left: the best view (red ring) looks straight at the unseen face and sees all 25 of its vertices. Right: with an Observed wall in front of that face, the same view sees none, and the best view moves round the wall and sees all 25 again.");
P("All of C-08's tests run on synthetic meshes. On a synthetic sphere of 44,702 vertices, scoring all 147 candidates took 0.33 s on a laptop, a synthetic figure only. On today's real meshes the fusion artifacts are also Unobserved, so the scorer would aim at them. It becomes useful once the mesh fixes land, and it is not yet wired into the on-screen arrow, which follows the fixed orbit.");

// 10 · Phone end to end + deployment
H1("10. End to End on a Real Phone, and Deployment", "Tabeen");
H2("10.1 The reconstruction service (R-14)");
P("The service is a FastAPI [17] application that implements the frozen contract. It stores sessions on disk, runs the eight-stage pipeline in the background when a session finishes, and turns every failure into one of the contract's named error codes rather than a generic error. Because sessions live on one machine's disk, it runs as exactly one replica, and a test pins that.");
H2("10.2 Cloud deployment (R-15)");
P("Every merge to main builds the service's container image and smoke-tests it in CI. The image is built on a PyTorch base image pinned by digest, and CI checks that the container answers. The image is then published to a container registry, again pinned by digest. A manual, main-only workflow deploys that exact image to a Tesla T4 on Lightning AI [19] behind HTTPS with token authentication.");
P("On 9 October, a second member verified the deployment by following the runbook without help:");
BULLETS([
  "TLS 1.3 with a valid certificate;",
  "a request without the token was refused (401), and with the token accepted (201);",
  "a full 30-photo golden session reconstructed remotely and returned a 1,095,640-byte model;",
  "the deployment was stopped afterwards.",
]);
P("The real deployment found three problems that no local test could:");
BULLETS([
  "The model link came back as http://localhost, because TLS ends at Lightning's proxy. Fixed, with tests.",
  "The runtime limit we relied on did not stop the GPU. One deployment ran for about 28 hours. Stopping is now an explicit runbook step, and an automatic cost guard is filed as an issue.",
  "Credits that existed but were not allocated to the workspace blocked the public endpoint for about a day.",
]);
H2("10.3 The first real-phone attempt (C-13, 15 October)");
P("For this check, the service ran on our M2 Max laptop rather than the cloud GPU (D-049). A second Lightning start never got a ready GPU, and the deployed image can only contain code that is already on main, so it could not test the branch under review. The phone loaded the app over HTTPS from a development server, which forwarded API calls to the service on the same origin.");
P("Result: two real-phone sessions, 19 and 17 frames, both failed with **zero camera poses**, even though the board was detected in every frame. Frames taken from the live camera preview carry no EXIF focal length, because they come through a canvas rather than the camera app, unlike the golden capture's Camera-app photos — so the board-pose stage had no focal length to pose with.");
H2("10.4 What the first attempt taught us");
P("The fix (D-050): when frames carry no focal length, the service calibrates the camera from the board itself, using the corners the board-detection stage already found, then re-poses every frame with that calibration. Reprocessing the first failed session's own 19 saved frames through the corrected pipeline validated the fix offline: calibration RMS 0.20 px, 18 of 19 frames posed, and the full pipeline then produced a metric result (observed fraction 0.17) in about 44 seconds. That is one reprocessed run on already-captured frames, not a live device confirming the fix end to end — getting that confirmation is the first item in Section 14's next steps.");
P("Still open (O-005): how a browser should reach the cloud deployment. Three questions remain: whether the proxy's token check blocks the browser's preflight request, whether a token inside the app's code is acceptable, and the order code. The deployed service does not check order codes, so today any non-empty code would start a paid GPU run.");

// 11 · Real data
H1("11. Testing on Real Data", "Tabeen");
P("Every stage passed its unit tests on synthetic data before it merged. When we ran the full chain on all five golden objects and looked at the results (R-17), **none passed** a visual check. That was the most useful finding of the month, because it named three specific causes, and each one became its own story with measured numbers (Table {tab:realdata}).");
BULLETS([
  "**Stray low-confidence points (R-20, R-24).** VGGT gives every pixel a confidence, and the first version kept every pixel. About three quarters of pixels sit at the model's confidence floor. Sweeping the threshold on all five objects showed that almost all of the achievable cleanup happens just above that floor, and that going higher only removes real surface. The threshold is now 1.0001 (D-045), applied to the point cloud, to fusion, and (since R-24) to provenance. {fig:confidence} shows the effect.",
  "**The base of every part was cut off (R-21).** Segmentation assumed the board sat at height zero, but it sat 1–5 mm lower, so the old cut landed 3.6–7.4 mm above the real board. On s-04 that removed a quarter of the part's 30 mm height. Segmentation now fits the real board plane (Section 6.1).",
  "**Everything comes out too small (R-22).** The whole scene is 4–9% small because of the focal-length mismatch carried through camera-centre alignment (Section 5.3). R-23 is the fix.",
]);
FIGURE("confidence", "fig_confidence.jpg", 5.2, "Effect of the confidence threshold (R-20) on golden sample s-05's segmented points. (a) Every pixel kept: 17.7% of points lie outside the part's main cluster. (b) Pixels at VGGT's confidence floor removed (threshold 1.0001, D-045): 2.5%. The part keeps its shape; the halo of stray points is gone.");
TABLE("realdata", "Real-data measurements on the five golden objects. Stray points are the share of segmented points outside the part's main cluster (R-20's sweep). Board offset is how far the real board surface sat below the assumed zero (R-21), and the cut columns compare the old fixed cut with the new measured one.", [0.75, 1.0, 1.0, 1.35, 1.2, 1.2],
  ["Object", "Stray, before", "Stray, after", "Old cut above board (mm)", "Board offset (mm)", "New cut (mm)"],
  [
    ["s-01", "19.2%", "2.5%", "4.2", "1.9", "6.0"],
    ["s-02", "18.7%", "2.5%", "3.6", "0.9", "4.5"],
    ["s-03", "11.4%", "1.8%", "4.8", "3.2", "3.5"],
    ["s-04", "29.3%", "2.0%", "7.4", "5.4", "2.5"],
    ["s-05", "17.7%", "2.5%", "5.3", "3.2", "3.5"],
  ]);
P("The new cut on s-01 and s-02 is higher than the old one on purpose. On those two, VGGT reconstructs a weaker second layer of the board grid 4–6 mm above the real surface, and the new cut is set above it.");
P("**Honest status.** The fused meshes still do not close into a recognisable part on any of the five objects. Before R-21, only 2 of 5 passed a mesh check, and those were the same padlock under two lightings (R-18). After R-20, every mesh is measurably less noisy (28–48% fewer vertices) but still not a pass. Next is the scale fix (R-23), then a rerun of the mesh check, then confirming the EXIF fix on a real device (Section 10.4).");

// 12 · Progress and process
H1("12. Progress and Engineering Process", "Tabeen");
TABLE("progress", "Progress by the numbers, as of 15 October 2026. C-08 and C-13 are each open pull requests (#144 and #146).", [1.6, 2.4, 2.5],
  ["Measure", "Value", "Notes"],
  [
    ["Stories done", "40 of 44 written (foundations, reconstruction, capture)", "C-08 and C-13 are each open pull requests; R-16 is blocked on ground truth; R-23 is filed (R-22's finding) but not yet claimed"],
    ["Pull requests", "130 merged", "Every one approved by another member"],
    ["Automated tests", "585 passing", "404 Python (1 skipped) and 181 TypeScript, on main; C-13's own branch separately reports 418 Python (1 skipped) and 203 TypeScript passing"],
    ["Decisions", "45 recorded", "Each with its reasoning; 3 more (D-048–D-050) sit in C-13's open pull request and will land when it merges"],
    ["By area", "Foundations 7/7 · Reconstruction 22/24 · Capture 11/13", "C-08 and C-13 are each open pull requests in capture; R-23 is filed but not yet claimed in reconstruction; R-16 is blocked on ground truth. Inspection, ground-truth and generative-AI stories are written one sprint ahead, so they are not counted yet"],
  ]);
P("Every story goes through the same gate. It starts as a story file with testable acceptance criteria. It is claimed by one person, on the shared main branch, so nobody duplicates work (D-037). Tests are written alongside the code, never deferred (D-029). It merges through a pull request with CI and a teammate's review. It is marked Done inside that same pull request, with the actual hours measured, never the estimate copied in (D-033, D-038). A script checks that the backlog index and the story files agree.");
P("Two habits have paid off. First, decisions are recorded with their reasoning (D-001 onward), so they are not silently reversed. Second, reviews regularly catch real defects before merge. Examples include four ways the camera could be left on (C-05), a deploy-stop command that reported success while the GPU kept billing (R-15), and an orbit reading that could cover two slots at once (C-07).");

// 13 · Timeline
H1("13. Timeline", "Tabeen");
P("{fig:timeline} compares the sprint plan with what actually happened, and Table {tab:milestones} lists the milestones.");
FIGURE("timeline", "fig_timeline.png", 6.5, "Plan versus actual, September to December. Outlined bars are the sprint plan; solid teal bars are completed work; the dashed amber box is the next step on the real-data fixes; dashed red boxes are work that was planned to have started but has not. Diamonds mark the sprint-gate milestones. (This chart is inherited from the midterm slide deck and has not yet been redrawn for this report\u2019s 15 October cutoff: it still marks Today as 16 October and shows the capture app as fully done through that date. Table 8 and the text in this section are the corrected, cutoff-accurate figures; treat the chart as approximate until it is redrawn.)");
TABLE("milestones", "Milestones from the sprint plan, with status on 15 October.", [0.55, 2.2, 1.05, 2.7],
  ["Gate", "Meaning", "Planned", "Status"],
  [
    ["M1", "Phone → upload → reconstruction → model display, nobody touching a terminal", "25 Sep", "Not yet demonstrated — already 3 weeks late. C-13 (#146), which connects the app to the service, is open; its first real-phone attempt is Section 10"],
    ["M2", "Latency table decides interactive vs batch", "9 Oct", "On time: measured 7 Oct (R-13); batch confirmed"],
    ["M3", "σ_floor measured from printed, caliper-measured parts", "16 Oct", "Not started — behind"],
    ["M4", "Inspection: planted defect → Fail; unobserved region → Rescan", "30 Oct", "Not started"],
    ["M5", "Feature freeze", "13 Nov", "—"],
    ["M6 / M7", "Tested, backup demo recorded / code freeze", "2 / 4 Dec", "—"],
    ["M8", "Final defense", "18 Dec", "—"],
  ]);
P("The slip has one main cause: we deliberately built the backend first (D-019), and its longest chain of dependent stories could only advance one story at a time. The capture app then caught up quickly, with all 13 stories done in about two and a half weeks. Ground truth and the generative-AI components are the two areas that are now behind plan.");

// 14 · Next steps
H1("14. Next Steps", "Tabeen");
TABLE("next", "The five work streams for the second half of the term, in priority order.", [1.55, 3.05, 1.0, 0.9],
  ["Stream", "What", "Stories", "Target"],
  [
    ["1 · Geometry and phone reliability", "Anchor scale on the board (R-23); confirm the EXIF fix (D-050) on a real device; capture at the camera's full resolution", "R-23 + new", "Now"],
    ["2 · Ground truth", "Print and caliper-measure test parts; measure σ_floor and the first real tolerance", "G-01–G-05, R-16", "M3 (slipping)"],
    ["3 · Tier 2 inspection", "CAD registration, deviation heatmap, the four verdicts with the hard rule built in", "I-01–I-10", "M4, 30 Oct"],
    ["4 · Generative AI", "VLM capture checking first; then the plain-English spec, shape completion, the capture agent and the report narrative", "A-01–A-09", "Start now → M5"],
    ["5 · Cloud path and product", "Browser → cloud GPU with the order code checked; hosting; Android; buyer setup and report pages", "O-005, C-14–C-18", "M5–M6"],
  ]);
P("**Generative AI is our biggest gap.** It is about a third of the course requirement, and none of the five components has started. They are never cut (D-017), so they start now, alongside inspection. VLM capture supervision comes first: it is Tier 1, and it is the only one that affects a result, and then only in the safe direction, because it can veto a Pass but never create one. Shape completion will be pointed at Rescan guidance, where its disagreement map is useful under batch capture.");
P("By the proposal's hour budget, these components are 87 of 349 person-hours (24.9%), or 115 (33.0%) if the reconstruction backbone also counts. Whether it counts is a question we still need to settle with the instructor (O-001). We plan for the case where it does not.");
P("**Risks we are watching:** GPU availability and credits on the shared workspace; the phone-session failure rate; the scale fix landing before ground truth needs it; and the schedule, now that two streams are behind.");

// ── references ──────────────────────────────────────────────────────────────
const REFS = [
  "J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “VGGT: Visual Geometry Grounded Transformer,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2025.",
  "V. Leroy, Y. Cabon, and J. Revaud, “Grounding Image Matching in 3D with MASt3R,” in Proc. European Conf. Computer Vision (ECCV), 2024.",
  "J. L. Schönberger and J.-M. Frahm, “Structure-from-Motion Revisited,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2016.",
  "J. L. Schönberger, E. Zheng, M. Pollefeys, and J.-M. Frahm, “Pixelwise View Selection for Unstructured Multi-View Stereo,” in Proc. European Conf. Computer Vision (ECCV), 2016.",
  "S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, and M. J. Marín-Jiménez, “Automatic generation and detection of highly reliable fiducial markers under occlusion,” Pattern Recognition, vol. 47, no. 6, pp. 2280–2292, 2014.",
  "G. Terzakis and M. Lourakis, “A Consistently Fast and Globally Optimal Solution to the Perspective-n-Point Problem,” in Proc. European Conf. Computer Vision (ECCV), 2020.",
  "S. Umeyama, “Least-squares estimation of transformation parameters between two point patterns,” IEEE Trans. Pattern Analysis and Machine Intelligence, vol. 13, no. 4, pp. 376–380, 1991.",
  "B. Curless and M. Levoy, “A Volumetric Method for Building Complex Models from Range Images,” in Proc. SIGGRAPH, 1996.",
  "W. E. Lorensen and H. E. Cline, “Marching Cubes: A High Resolution 3D Surface Construction Algorithm,” in Proc. SIGGRAPH, 1987.",
  "M. Kazhdan, M. Bolitho, and H. Hoppe, “Poisson Surface Reconstruction,” in Proc. Eurographics Symp. Geometry Processing, 2006.",
  "M. Garland and P. S. Heckbert, “Surface Simplification Using Quadric Error Metrics,” in Proc. SIGGRAPH, 1997.",
  "G. Bradski, “The OpenCV Library,” Dr. Dobb’s Journal of Software Tools, 2000.",
  "Q.-Y. Zhou, J. Park, and V. Koltun, “Open3D: A Modern Library for 3D Data Processing,” arXiv:1801.09847, 2018.",
  "S. van der Walt et al., “scikit-image: image processing in Python,” PeerJ, vol. 2, e453, 2014.",
  "Khronos Group, “glTF 2.0 Specification,” 2021. [Online]. Available: https://registry.khronos.org/glTF/",
  "three.js authors, “three.js — JavaScript 3D library.” [Online]. Available: https://threejs.org",
  "S. Ramírez, “FastAPI.” [Online]. Available: https://fastapi.tiangolo.com",
  "calib.io, “Camera Calibration Pattern Generator.” [Online]. Available: https://calib.io/pages/camera-calibration-pattern-generator",
  "Lightning AI, “Lightning AI Studios documentation.” [Online]. Available: https://lightning.ai/docs",
  "ProofShape team, project repository: stories, decision log and progress log, 2026. [Online]. Available: https://github.com/proofshape/proofshape",
  "ProofShape team, “ProofShape — Phone-Based 3D Capture and Inspection for Manufactured Parts,” CS595 capstone project proposal, San Francisco Bay University, Sept. 2026.",
];

// Appendix A: every story written so far (stories/README.md on main, 15 Oct 2026).
const STORIES = [
  ["F-01", "Public repository with branch protection", "@TabeenRaoof", "Done"],
  ["F-02", "Repository skeleton and module boundaries", "@dalwalyk", "Done"],
  ["F-03", "Interface contract v1, frozen", "@TabeenRaoof", "Done"],
  ["F-04", "CI skeleton", "@dalwalyk", "Done"],
  ["F-05", "Shared GPU workspace, three-way access", "@mbj1994", "Done"],
  ["F-06", "Golden capture set", "@mbj1994; @TabeenRaoof", "Done"],
  ["F-07", "Reproducible dev environment", "@dalwalyk", "Done"],
  ["R-01", "Run the reconstruction model on the golden capture", "@dalwalyk", "Done"],
  ["R-02", "Board detection and per-frame camera pose", "@TabeenRaoof", "Done"],
  ["R-03", "Camera intrinsics from the board", "@mbj1994", "Done"],
  ["R-04", "Metric alignment", "@dalwalyk", "Done"],
  ["R-05", "Scale agreement check", "@dalwalyk", "Done"],
  ["R-06", "Segment the part from the board", "@dalwalyk", "Done"],
  ["R-07", "TSDF fusion to a mesh", "@dalwalyk", "Done"],
  ["R-08", "Mesh cleanup and GLB export", "@dalwalyk", "Done"],
  ["R-09", "Per-vertex provenance", "@dalwalyk", "Done"],
  ["R-10", "Per-vertex uncertainty", "@dalwalyk", "Done"],
  ["R-11", "MASt3R behind the config flag", "@mbj1994", "Done"],
  ["R-12", "COLMAP behind the config flag", "@mbj1994", "Done"],
  ["R-13", "Per-stage latency instrumentation", "@TabeenRaoof", "Done"],
  ["R-14", "Reconstruction endpoint", "@mbj1994", "Done"],
  ["R-15", "Container image and deploy", "@TabeenRaoof", "Done"],
  ["R-16", "Known-size tolerance validation", "—", "Blocked"],
  ["R-17", "Visually verify part segmentation on the golden set", "@TabeenRaoof", "Done"],
  ["R-18", "Produce and check TSDF-fused meshes on the golden set", "@dalwalyk", "Done"],
  ["R-19", "Report per-vertex σ distribution on the golden set", "@mbj1994", "Done"],
  ["R-20", "Use VGGT's confidence in the point cloud and in fusion", "@TabeenRaoof", "Done"],
  ["R-21", "Measure the height cut from the real board surface", "@TabeenRaoof", "Done"],
  ["R-22", "Find why parts reconstruct smaller than their CAD", "@TabeenRaoof", "Done"],
  ["R-23", "Anchor metric scale on the board in the scene", "—", "Ready"],
  ["R-24", "Use VGGT's confidence in provenance's observation count", "@TabeenRaoof", "Done"],
  ["C-01", "Minimal upload page", "@TabeenRaoof", "Done"],
  ["C-02", "Camera permissions on a real iPhone", "@TabeenRaoof", "Done"],
  ["C-03", "Gyro permission flow", "@dalwalyk", "Done"],
  ["C-04", "App shell", "@dalwalyk", "Done"],
  ["C-05", "Live camera", "@dalwalyk", "Done"],
  ["C-06", "Frame gate", "@dalwalyk", "Done"],
  ["C-07", "Gyro overlay and arrow", "@dalwalyk", "Done"],
  ["C-08", "Next-view scoring", "@TabeenRaoof", "In review (#144)"],
  ["C-09", "Capture loop", "@dalwalyk", "Done"],
  ["C-10", "Reconstruction result and the mock service", "@mbj1994", "Done"],
  ["C-11", "Model viewer", "@mbj1994", "Done"],
  ["C-12", "Observed/unobserved colouring", "@mbj1994", "Done"],
  ["C-13", "Connect capture to the deployed service", "@dalwalyk; @TabeenRaoof", "In review (#146)"],
];

// ── numbering of figures and tables ─────────────────────────────────────────
const figNo = {}, tabNo = {};
for (const b of DOC) {
  if (b.type === "fig") figNo[b.key] = Object.keys(figNo).length + 1;
  if (b.type === "table") tabNo[b.key] = Object.keys(tabNo).length + 1;
}
function resolve(text) {
  return String(text)
    .replace(/\{fig:([a-z_]+)\}/g, (_, k) => { if (!figNo[k]) throw new Error("no fig " + k); return `Figure ${figNo[k]}`; })
    .replace(/\{tab:([a-z_]+)\}/g, (_, k) => { if (!tabNo[k]) throw new Error("no table " + k); return String(tabNo[k]); });
}

// ── renderers ──────────────────────────────────────────────────────────────
function caption(kind, n, text) {
  return new Paragraph({
    spacing: { before: 60, after: 220 },
    children: [new TextRun({ text: `${kind} ${n}. `, bold: true, size: 19, color: INK }), ...runs(resolve(text), { size: 19, color: MUTED })],
  });
}

function cell(text, widthIn, opts = {}) {
  const lines = String(text).split("\n");
  return new TableCell({
    width: { size: DXA(widthIn), type: WidthType.DXA },
    shading: opts.header ? { fill: TEAL, type: ShadingType.CLEAR, color: "auto" } : opts.shade ? { fill: TINT, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    verticalAlign: VerticalAlign.TOP,
    children: lines.map((line, i) => new Paragraph({
      spacing: { before: 0, after: 0, line: 252 },
      children: runs(resolve(line), { size: opts.size || 18, bold: opts.header || (opts.firstBold && i === 0), color: opts.header ? "FFFFFF" : (i > 0 ? MUTED : INK) }),
    })),
  });
}

function table(b) {
  const widths = b.widths;
  const total = widths.reduce((a, c) => a + c, 0);
  if (Math.abs(total - CONTENT_W_IN) > 0.01) throw new Error(`table ${b.key} widths sum to ${total}`);
  const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
  const rows = [new TableRow({ tableHeader: true, cantSplit: true, children: b.header.map((h, i) => cell(h, widths[i], { header: true })) })];
  b.rows.forEach((r, ri) => rows.push(new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, widths[i], { shade: ri % 2 === 1, firstBold: i === 0 && b.firstColBold !== false, size: b.size })) })));
  return [
    new Paragraph({ keepNext: true, spacing: { before: 120, after: 60 }, children: [new TextRun({ text: `Table ${tabNo[b.key]}. `, bold: true, size: 19, color: INK }), ...runs(resolve(b.caption), { size: 19, color: MUTED })] }),
    new Table({ width: { size: DXA(total), type: WidthType.DXA }, columnWidths: widths.map(DXA), borders: { top: border, bottom: border, left: border, right: border, insideHorizontal: border, insideVertical: border }, rows }),
    new Paragraph({ spacing: { before: 0, after: 160 }, children: [] }),
  ];
}

function figure(b) {
  const file = path.join(FIG, b.file);
  const { w, h } = imageSize(file);
  const widthPx = Math.round(b.width * 96);
  const heightPx = Math.round(widthPx * (h / w));
  const type = file.endsWith(".png") ? "png" : "jpg";
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 40 }, children: [new ImageRun({ type, data: fs.readFileSync(file), transformation: { width: widthPx, height: heightPx }, altText: { title: `Figure ${figNo[b.key]}`, description: resolve(b.caption).slice(0, 250), name: b.key } })] }),
    caption("Figure", figNo[b.key], b.caption),
  ];
}

function render(b) {
  switch (b.type) {
    case "h1":
      return [
        new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(b.title)] }),
        new Paragraph({ spacing: { before: 0, after: 160 }, children: [new TextRun({ text: `Author: ${b.author}`, italics: true, size: 19, color: TEAL })] }),
      ];
    case "h2":
      return [new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(b.title)] })];
    case "p":
      return [new Paragraph({ children: runs(resolve(b.text)) })];
    case "bullets":
      return b.items.map((t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 }, children: runs(resolve(t)) }));
    case "fig":
      return figure(b);
    case "table":
      return table(b);
    case "break":
      return [new Paragraph({ children: [new PageBreak()] })];
    default:
      throw new Error("unknown block " + b.type);
  }
}

// ── cover page ─────────────────────────────────────────────────────────────
function coverPage() {
  const memberRow = (name, handle) => new TableRow({
    children: [cell(name, 2.4, { size: 21, firstBold: true }), cell("Student ID: ______________", 2.3, { size: 21 }), cell(handle, 1.8, { size: 21 })],
  });
  const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
  return [
    new Paragraph({ spacing: { before: 1500, after: 120 }, children: [new TextRun({ text: "CS595 CAPSTONE · MIDTERM REPORT", bold: true, size: 22, color: TEAL, characterSpacing: 40 })] }),
    new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "ProofShape", bold: true, size: 72, font: HEAD_FONT, color: INK })] }),
    new Paragraph({ spacing: { after: 600 }, children: [new TextRun({ text: "Phone-Based 3D Capture and Inspection for Manufactured Parts", size: 32, font: HEAD_FONT, color: INK })] }),
    new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text: "Team", bold: true, size: 24, color: INK })] }),
    new Table({
      width: { size: DXA(6.5), type: WidthType.DXA }, columnWidths: [2.4, 2.3, 1.8].map(DXA),
      borders: { top: border, bottom: border, left: border, right: border, insideHorizontal: border, insideVertical: border },
      rows: [memberRow("Tabeen Raoof", "@TabeenRaoof"), memberRow("Mamadou [surname]", "@mbj1994"), memberRow("Yashi [surname]", "@dalwalyk")],
    }),
    new Paragraph({ spacing: { before: 600, after: 60 }, children: [new TextRun({ text: "San Francisco Bay University · Fall 2026", size: 22, color: INK })] }),
    new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "Instructor: [instructor name]", size: 22, color: INK })] }),
    new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "Submitted: [date], October 2026", size: 22, color: INK })] }),
    new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "Repository: github.com/proofshape/proofshape", size: 22, color: MUTED })] }),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

function contentsPage() {
  const rows = [];
  for (const b of DOC) if (b.type === "h1") rows.push([b.title, b.author]);
  rows.push(["References", "Tabeen"], ["Appendix A. Story Backlog", "Tabeen"]);
  const widths = [4.6, 1.9];
  const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
  return [
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Contents and Authorship")] }),
    new Paragraph({ spacing: { after: 160 }, children: runs("Each section of this report was written by the team member named below, and the same name appears under each section heading. All three members reviewed the whole report.") }),
    new Table({
      width: { size: DXA(6.5), type: WidthType.DXA }, columnWidths: widths.map(DXA),
      borders: { top: border, bottom: border, left: border, right: border, insideHorizontal: border, insideVertical: border },
      rows: [new TableRow({ tableHeader: true, children: [cell("Section", widths[0], { header: true }), cell("Written by", widths[1], { header: true })] }),
        ...rows.map((r, i) => new TableRow({ children: [cell(r[0], widths[0], { size: 20, shade: i % 2 === 1 }), cell(r[1], widths[1], { size: 20, shade: i % 2 === 1 })] }))],
    }),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

function referencesPage() {
  return [
    new Paragraph({ children: [new PageBreak()] }),
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("References")] }),
    new Paragraph({ spacing: { before: 0, after: 160 }, children: [new TextRun({ text: "Author: Tabeen", italics: true, size: 19, color: TEAL })] }),
    ...REFS.map((r, i) => new Paragraph({ indent: { left: 520, hanging: 520 }, spacing: { after: 100 }, children: [new TextRun({ text: `[${i + 1}]\t`, size: 20 }), new TextRun({ text: r, size: 20 })], tabStops: [{ type: TabStopType.LEFT, position: 520 }] })),
  ];
}

function appendix() {
  return [
    new Paragraph({ children: [new PageBreak()] }),
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Appendix A. Story Backlog")] }),
    new Paragraph({ spacing: { before: 0, after: 160 }, children: [new TextRun({ text: "Author: Tabeen", italics: true, size: 19, color: TEAL })] }),
    new Paragraph({ children: runs("Every story written so far, with its owner and state on 15 October 2026 (from the backlog index in the repository [20]). F = foundations, R = reconstruction engine, C = capture app. Inspection (I), ground-truth (G) and generative-AI (A) stories exist as titles and are written one sprint ahead.") }),
    ...table({ key: "backlog", caption: "The story backlog. Each story is one pull request with its own acceptance criteria, tests and completion record.", widths: [0.6, 3.55, 1.55, 0.8], header: ["ID", "Story", "Owner", "State"], rows: STORIES, size: 17 }),
  ];
}
tabNo.backlog = Object.keys(tabNo).length + 1;

// ── document ────────────────────────────────────────────────────────────────
const body = [];
for (const b of DOC) body.push(...render(b));

const doc = new Document({
  creator: "ProofShape team",
  title: "ProofShape — CS595 Midterm Report",
  description: "Midterm report for the ProofShape CS595 capstone project, Fall 2026",
  styles: {
    default: { document: { run: { font: BODY_FONT, size: 22, color: INK }, paragraph: { spacing: { after: 140, line: 276 } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: HEAD_FONT, size: 32, bold: true, color: INK }, paragraph: { spacing: { before: 360, after: 40 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: HEAD_FONT, size: 25, bold: true, color: TEAL }, paragraph: { spacing: { before: 240, after: 80 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 260 } } } }] }],
  },
  sections: [{
    properties: {
      titlePage: true,
      page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } },
    },
    footers: {
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
      default: new Footer({
        children: [new Paragraph({
          tabStops: [{ type: TabStopType.RIGHT, position: DXA(6.5) }],
          children: [new TextRun({ text: "ProofShape · CS595 Midterm Report · Fall 2026", size: 17, color: MUTED }), new TextRun({ children: ["\t", PageNumber.CURRENT], size: 17, color: MUTED })],
        })],
      }),
    },
    children: [...coverPage(), ...contentsPage(), ...body, ...referencesPage(), ...appendix()],
  }],
});

const out = process.argv[2] || path.join(__dirname, "..", "submissions", "midterm-report-draft.docx");
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes;", Object.keys(figNo).length, "figures,", Object.keys(tabNo).length, "tables");
});

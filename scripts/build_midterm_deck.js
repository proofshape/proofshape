// ProofShape — CS595 midterm presentation (15 slides, professor's structure:
// title 1 · team 1 · completed work 10 · timeline 1 · next steps 1 · references 1).
// Committed as the review artifact for story P-20 (D-052): the team reviews this script's diff
// and the rendered .pptx in the pull request, the same pattern scripts/build_review_deck.js set
// for the biweekly review decks.
//
//   node scripts/build_midterm_deck.js [out.pptx]          # full deck, defaults to
//                                                           # submissions/midterm-presentation-deck.pptx
//   node scripts/build_midterm_deck.js <out-dir> --split    # one single-slide deck per slide, for QA
//
// Content is scoped to everything from the start of term up to Thursday 15 October 2026 and
// nothing after (the team's own rule for this deliverable, D-052 / stories/P/P-20.md) — verified
// against main at commit 3a373ed (2026-10-15 21:52 PDT, the day's last commit) and, where the
// text cites C-13's own branch directly, against that branch's last commit before midnight,
// d296be9 (23:29 PDT). Figures live in submissions/assets/2026-10-16/, shared with the midterm
// report (P-21) where the same real data applies.

const path = require("path");
const fs = require("fs");
const PptxGenJS = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa6");
const { applyTheme } = require(
  "/Users/tabeenraoof/.claude/skills/synced/cb535dfd-926e-4d3f-b0be-d6153353e121_491b23fa-de91-41e9-ac36-4b59d163395f/pptx/scripts/apply_theme.js",
);

const SCRATCH = path.join(__dirname, "..");
const FIG = path.join(SCRATCH, "submissions", "assets", "2026-10-16");

// ── speaker notes, one entry per slide ───────────────────────────────────────
// ~140 words per minute, ~5 minutes per presenter. Scoped to the 15 October cutoff throughout.
const NOTES = [
  // 1 · Title
  `PRESENTER: Mamadou · 0:30

Good morning, we are ProofShape: Tabeen, Mamadou and Yashi.

ProofShape turns a phone into a coarse 3D inspection tool. A supplier puts a part on a printed reference board, photographs it from about 30 angles, and gets back a 3D model in real millimetres. Later in the term that model is checked against the buyer's CAD, and the part gets Pass, Fail, Rescan or Unverifiable. The photo is one of our real test captures.`,

  // 2 · Team
  `PRESENTER: Mamadou · 0:45

We work from one shared backlog: anyone can take any story, and a different teammate always reviews it.

I set up our cloud GPU, shot the test photos with Tabeen, calibrated the camera, added two alternative reconstruction backbones, and built the API service plus the app's result screen, 3D viewer and colouring.

Yashi built the CI and dev setup, most of the geometry chain, and most of the phone app, including the live capture loop and the app's connection to the service.

Tabeen froze the API contract and built board detection, the cloud deployment, the real-data fixes and next-view scoring, and ran our first phone attempt.

I present to slide 5, Yashi 6 to 9, Tabeen the rest.`,

  // 3 · Overview + foundations
  `PRESENTER: Mamadou · 1:15

This is the whole system. The capture app runs in the phone's browser, so there is nothing to install. It sends photos and gyroscope readings over HTTPS to our reconstruction service.

The service is eight stages. It finds the printed board in each photo, which tells us where the camera was in millimetres. A large learned model, VGGT, reconstructs 3D from all the photos at once. The board converts that into real millimetres. Then we cut the part away from the board, fuse a mesh, clean it, label every vertex as observed or not with an uncertainty, and send a labelled model back to the phone.

As of this week every box here is built. What is not yet proven is the live connection between the two: the capture app talking to this service on a real phone, which we cover in slide 10.

The bottom row is what we built first, so three people could work in parallel: a frozen API contract, 142 golden test photos, a shared Tesla T4, and a protected main branch where every pull request needs a teammate's approval.`,

  // 4 · Backbones
  `PRESENTER: Mamadou · 1:15

Learned reconstruction is the riskiest stage, so we made it swappable.

VGGT is our primary backbone: one forward pass over all the photos returns a camera pose and a depth for every pixel. I added MASt3R and COLMAP behind a single config value. All three write the same four files, so nothing downstream changes; MASt3R's output went straight into our alignment stage with no code changes.

The chart shows measured time per stage on our T4, on the same 30 photos. VGGT takes 80 seconds, almost half of it loading the model. MASt3R takes 266, nearly all in its iterative alignment. COLMAP registered only 28 of 30 photos, so it refused rather than invent two views, which we treat as a correct answer.

That 80 seconds also settled a design question: live interactive feedback is out of reach, so batch capture is what we ship.`,

  // 5 · Metric scale
  `PRESENTER: Mamadou · 1:15

VGGT's output has no units. Real millimetres come from the printed board.

First, we detect the board's coded squares in every photo and solve where the camera was. That worked on 138 of 142 photos; the four failures were one blurred shot and three taken almost level with the table. The figure shows real camera positions for one capture.

Second, we calibrate the lens from the same board, at 1.2 pixels reprojection error. That is internal consistency, not accuracy.

Third, one similarity transform lines VGGT's cameras up with the board's cameras, which puts the whole model in board millimetres.

Then we used the board as a ruler. Inside the reconstruction it comes out 4 to 9 percent small on every object, and that tracks the focal-length mismatch between VGGT and our board poses almost exactly. Because we align on camera centres, that mismatch becomes scale. The fix, R-23, anchors scale on the board itself and is next — it's filed, not yet claimed. Over to Yashi.`,

  // 6 · Photos to mesh
  `PRESENTER: Yashi · 1:15

Here is the chain on one real object, the red prism. Every image comes from our pipeline's real output.

One: one of the 29 phone photos.

Two: VGGT's depth for every pixel, converted into board millimetres, including the board and the table around it.

Three: segmentation. We fit a plane to the real board surface, drop the pixels VGGT itself isn't confident about, and keep what stands above the board inside its footprint.

Four: TSDF fusion. Each photo's depth is fused into 2-millimetre voxels and marching cubes extracts the surface; voxels no photo observed cannot create surface. Cleanup keeps the largest connected piece, fills only small holes, and exports a GLB for the browser.

Each stage is its own tested module with a command-line tool and a JSON report, so we can rerun and inspect any one. The mesh still has artifacts around the part, and the next slide shows why they can't cause a wrong verdict.`,

  // 7 · Hard rule
  `PRESENTER: Yashi · 1:15

This is the rule the project is built around: geometry the camera did not really see can never pass or fail a part. It can only trigger a Rescan.

We enforce it per vertex. A vertex is Observed only if at least two photos see it from at least two degrees apart, through pixels the model is confident about, and it passes an occlusion check, so a camera facing the front can't count as seeing the back.

Sigma is the spread of a vertex's depth across those photos; above 3 millimetres it is downgraded. That 3 is a working assumption until the calibration study gives a real number. The API contract forbids unobserved regions from counting toward a verdict, and a test enforces it.

On s-05, 13 percent of the mesh is Observed, mainly the part. Everything else, mostly fusion artifacts, can only trigger a Rescan. Median sigma is 1.2 to 2.1 millimetres across the five objects, which is a spread, not an accuracy figure.`,

  // 8 · Capture app
  `PRESENTER: Yashi · 1:20

Most of the capture thread, 11 of 13 stories, is done; the other two are open pull requests.

On the left is the capture screen, shown here in desktop Chrome at phone size with one of our golden photos as the camera feed. Each capture runs the frame gate on the phone, rejecting blur, clipped exposure and near-duplicates, then encodes a JPEG and uploads it with its gyro reading. Only when the server accepts it does that orbit slot count as covered: 2 of 13 here, with the arrow pointing to the next view.

I walked this on a real iPhone: all three rejection reasons, acceptance, finishing, and the camera light turning off.

On the right is the result screen. Mamadou's viewer colours the model by provenance: blue is Observed, slate is Unobserved. This is our real s-05 mesh, with the part in blue and the artifacts in slate.

It's Vite and TypeScript with no framework, and 203 tests.`,

  // 9 · Next-view scoring
  `PRESENTER: Yashi · 1:00

Once we know what is unseen, the next question is where to point the camera. Tabeen built next-view scoring; it's one of the two open pull requests from the last slide.

It places 147 candidate camera positions on a hemisphere around the part, 25 to 40 centimetres away and at most 60 degrees from vertical, and ray-casts from each one to every under-observed vertex. The whole mesh blocks those rays, so Observed surface can still hide what's behind it. The score is how many under-observed vertices a view would see; ties go to the most head-on view.

On the left, one face of a test part is unseen, and the best view looks straight at it. On the right we add a wall in front: that same view now sees nothing, and the best view goes round the wall and sees all 25 again.

It is tested on synthetic meshes. On today's real meshes the fusion artifacts are Unobserved too, so it would chase them; it becomes useful once the mesh fixes land. Over to Tabeen.`,

  // 10 · First real-phone attempt
  `PRESENTER: Tabeen · 1:10

This slide is about what we found, not a finished result — we have not yet gotten the whole system working end to end from a real phone, and I want to be straightforward about that rather than round it up.

First attempt: pointing the app at our deployed Lightning service hit the same chicken-and-egg problem R-15 found. The deployed image only contains code already on main, so our own CORS fix couldn't be tested there before it merged. We stopped that deployment and moved the check to our M2 Max instead, running the service locally — that's D-049.

Second attempt, on the Mac: two real sessions, 19 and 17 frames. Both failed with zero camera poses, even though the board was detected in every single frame. The reason: frames grabbed from the live camera preview carry no EXIF focal length, unlike the golden capture's Camera-app photos.

The fix, D-050: calibrate the camera from the board itself when that happens. We validated it offline, reprocessing the first failed session's own saved frames — 0.20 pixel calibration error, 18 of 19 frames posed, and a real metric result. What we have not done yet is confirm that fix on an actual live phone. That's the first item on our next-steps slide.`,

  // 11 · Real data
  `PRESENTER: Tabeen · 1:10

Every stage passed its unit tests on synthetic data. Then I ran the full chain on all five golden objects, and none passed a visual check. That gave us three concrete causes, and each became a story.

One: stray points. VGGT gives every pixel a confidence and we kept them all. Sweeping the threshold showed almost all the cleanup happens just above the model's confidence floor. On s-05 stray points drop from 17.7 to 2.5 percent, and to 2.5 percent or less on all five. Provenance now trusts only those pixels too. Merged.

Two: the bottom of every part was cut off, because we assumed the board sat exactly at zero height; it sat 1 to 5 millimetres lower. We now fit the real board plane. Merged.

Three: size, the 4 to 9 percent scale error from slide 5, plus walls that shrink a little more, which is still open.

Honestly, the meshes still don't close into a recognisable part. Next is the scale fix, then confirming the phone fix from the last slide on a real device.`,

  // 12 · Progress
  `PRESENTER: Tabeen · 0:40

The numbers, briefly. Of the 44 stories written so far, 40 are done; the rest are two open pull requests, the scale fix which is filed but not yet claimed, and a tolerance check that waits on ground truth. That's 130 merged pull requests, each approved by a teammate, 585 automated tests passing, and 45 decisions recorded with their reasoning.

Foundations are complete, the capture thread is 11 of 13, and the reconstruction engine is 22 of 24. Inspection and generative-AI stories get written one sprint ahead, so they aren't counted yet.

Every story goes through the gate on the right, ending with real measured hours, never the estimate.`,

  // 13 · Timeline
  `PRESENTER: Tabeen · 0:45

Outlined bars are the plan, solid bars are actual, and the dashed line is today, the 15th.

Foundations finished on time. The engine started two weeks late and was deployed on 9 October. M1, phone to model with nobody touching a terminal, was planned for 25 September and is still open today, about three weeks late — that's the connection slide 10 covers. M2, the latency decision, was on time.

The red rows are behind. Ground truth was planned for this week and generative AI from late September, and neither has started.`,

  // 14 · Next steps
  `PRESENTER: Tabeen · 0:50

Five streams, in order.

One: make the geometry and the phone reliable. Anchor scale on the board, confirm yesterday's EXIF fix actually works on a real device, and capture at the camera's full resolution instead of 480 by 640.

Two: ground truth. Print and caliper-measure test parts to get our noise floor and our first real tolerance. Until then we state no accuracy number.

Three: Tier 2 inspection: CAD registration, the deviation heatmap and the four verdicts.

Four is our biggest gap. Generative AI is a third of the course requirement and hasn't started. It is never cut, so it starts now, with VLM capture checking first.

Five: the browser-to-cloud path, which must check the order code server-side before anyone points the app at a paid GPU, then hosting, Android and the buyer pages.

(Optional question for the professor: does our reconstruction backbone, a large learned model but deterministic, count toward the generative-AI share? It decides whether we are planning to 25 or 33 percent.)`,

  // 15 · References
  `PRESENTER: Tabeen · 0:15, then close

These are the methods and libraries we build on, from VGGT and COLMAP to TSDF fusion and marching cubes.

To close: Tier 1 is built except for one link — the phone talking to the real service — which we've diagnosed and partly fixed but haven't confirmed live yet. Real data has told us exactly what to fix elsewhere, and the rest of the term is the inspection verdict and generative AI. Thank you, we are happy to take questions.`,
];

// ── theme ───────────────────────────────────────────────────────────────────
const THEME = {
  name: "ProofShape",
  headFontFace: "Georgia",
  bodyFontFace: "Arial",
  colors: {
    dk1: "1A2326", // ink
    lt1: "FFFFFF",
    dk2: "0E2F36", // deep teal, dark slides
    lt2: "EEF4F5", // tint panels
    accent1: "16697A", // teal: built / primary
    accent2: "D98A2B", // amber: in progress
    accent3: "C8402F", // red: the part, problems
    accent4: "5F6E74", // muted text
    accent5: "7FB3BE", // light teal
    accent6: "CDD6D9", // light grey: not started, borders
    hlink: "16697A",
    folHlink: "4E8F9C",
  },
};
const HEX = THEME.colors;

const W = 13.333;
const H = 7.5;
const M = 0.6;

// ── icons ───────────────────────────────────────────────────────────────────
const iconCache = new Map();
async function icon(name, hex) {
  const key = name + hex;
  if (iconCache.has(key)) return iconCache.get(key);
  const Comp = fa[name];
  if (!Comp) throw new Error(`no icon ${name}`);
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).resize(256, 256, { fit: "contain", background: { r: 0, g: 0, b: 0, alpha: 0 } }).png().toBuffer();
  const data = "image/png;base64," + buf.toString("base64");
  iconCache.set(key, data);
  return data;
}

// Fit an image of a known aspect ratio inside a box, centred (contain).
function fit(file, box) {
  const meta = imageSize(file);
  const ar = meta.w / meta.h;
  let w = box.w;
  let h = w / ar;
  if (h > box.h) {
    h = box.h;
    w = h * ar;
  }
  return { x: box.x + (box.w - w) / 2, y: box.y + (box.h - h) / 2, w, h };
}
function imageSize(file) {
  const buf = fs.readFileSync(file);
  if (buf[0] === 0x89) return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
  // JPEG: walk segments to the SOF marker.
  let i = 2;
  while (i < buf.length) {
    const marker = buf[i + 1];
    const len = buf.readUInt16BE(i + 2);
    if (marker >= 0xc0 && marker <= 0xc3) return { h: buf.readUInt16BE(i + 5), w: buf.readUInt16BE(i + 7) };
    i += 2 + len;
  }
  throw new Error("cannot size " + file);
}

// ── presentation factory (layouts are per-presentation) ────────────────────
function makePres() {
  const pres = new PptxGenJS();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
  pres.author = "ProofShape team — Tabeen, Mamadou, Yashi";
  pres.title = "ProofShape — CS595 Midterm Progress Review";
  pres.subject = "CS595 Capstone, Fall 2026, San Francisco Bay University";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: "PS Title Dark",
    background: { color: C.text2 },
    objects: [],
  });

  pres.defineSlideMaster({
    title: "PS Content",
    background: { color: C.background1 },
    margin: [0.5, 0.6, 0.6, 0.6],
    objects: [
      {
        placeholder: {
          options: { name: "kicker", type: "body", x: M, y: 0.3, w: 8.6, h: 0.3, fontSize: 11, bold: true, color: C.accent1, charSpacing: 1.5, margin: 0, valign: "middle" },
          text: "",
        },
      },
      {
        placeholder: {
          options: { name: "presenter", type: "body", x: 9.33, y: 0.3, w: 3.4, h: 0.3, fontSize: 11, color: C.accent4, align: "right", margin: 0, valign: "middle" },
          text: "",
        },
      },
      {
        placeholder: {
          options: { name: "title", type: "title", x: M, y: 0.62, w: W - 2 * M, h: 0.72, fontSize: 30, bold: true, color: C.text1, margin: 0, valign: "middle", align: "left" },
          text: "",
        },
      },
      { text: { text: "ProofShape · CS595 Midterm Progress Review · Fall 2026", options: { x: M, y: 7.02, w: 7, h: 0.28, fontSize: 9.5, color: C.accent4, margin: 0, valign: "middle" } } },
    ],
    slideNumber: { x: W - M - 0.6, y: 7.02, w: 0.6, h: 0.28, fontSize: 9.5, color: C.accent4, align: "right" },
  });

  return { pres, C };
}

// Text box helper: every addText gets isTextBox.
function text(s, t, o) {
  s.addText(t, { isTextBox: true, margin: 0, valign: "top", ...o });
}

function contentSlide(pres, sectionTitle, kicker, presenter, title) {
  const s = pres.addSlide({ masterName: "PS Content", sectionTitle });
  s.addText(kicker.toUpperCase(), { placeholder: "kicker" });
  s.addText(presenter, { placeholder: "presenter" });
  s.addText(title, { placeholder: "title" });
  return s;
}

function chip(s, C, label, o) {
  s.addShape("roundRect", {
    x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: o.r ?? 0.08,
    fill: { color: o.fill }, line: { color: o.line ?? o.fill, width: o.lineW ?? 0.75 },
    objectName: o.name,
  });
  text(s, label, {
    x: o.x + 0.08, y: o.y, w: o.w - 0.16, h: o.h, fontSize: o.fontSize ?? 12, bold: o.bold ?? false,
    color: o.color, align: o.align ?? "center", valign: "middle", fit: "shrink",
  });
}

function numberDot(s, C, n, x, y, d = 0.42, fill = C.accent1) {
  s.addShape("ellipse", { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  text(s, String(n), { x, y, w: d, h: d, fontSize: d * 34, bold: true, color: C.background1, align: "center", valign: "middle" });
}

function shot(s, C, file, box, caption, capH = 0.45) {
  const f = fit(file, box);
  s.addImage({ path: file, ...f });
  if (caption) {
    text(s, caption, { x: box.x, y: box.y + box.h + 0.08, w: box.w, h: capH, fontSize: 10.5, color: C.accent4, italic: true });
  }
  return f;
}

// ═══════════════════════════════════════════════════════════════════════════
// Slides
// ═══════════════════════════════════════════════════════════════════════════
const SLIDES = [];

// 1 · Title ──────────────────────────────────────────────────────────────────
SLIDES.push(async (pres, C) => {
  const s = pres.addSlide({ masterName: "PS Title Dark", sectionTitle: "Introduction" });
  const photoX = 7.55;
  s.addImage({ path: path.join(FIG, "fig_photo_s05.jpg"), x: photoX, y: 0, w: W - photoX, h: H, sizing: { type: "cover", w: W - photoX, h: H } });
  text(s, "CS595 CAPSTONE · MIDTERM PROGRESS REVIEW", { x: 0.75, y: 1.55, w: 6.5, h: 0.3, fontSize: 12, bold: true, color: C.accent5, charSpacing: 2 });
  text(s, "ProofShape", { x: 0.75, y: 2.0, w: 6.6, h: 1.1, fontSize: 58, bold: true, color: C.background1, fontFace: "+mj-lt" });
  text(s, "A phone becomes a coarse 3D inspection tool for manufactured parts", { x: 0.75, y: 3.2, w: 6.3, h: 0.95, fontSize: 21, color: C.background2, fontFace: "+mj-lt" });
  text(s, "Tabeen  ·  Mamadou  ·  Yashi", { x: 0.75, y: 4.75, w: 6.3, h: 0.4, fontSize: 17, bold: true, color: C.background1 });
  text(s, "San Francisco Bay University · Fall 2026\ngithub.com/proofshape/proofshape", { x: 0.75, y: 5.2, w: 6.3, h: 0.7, fontSize: 13, color: C.accent5, lineSpacingMultiple: 1.2 });
  s.addShape("rect", { x: photoX, y: H - 0.55, w: W - photoX, h: 0.55, fill: { color: C.text2, transparency: 25 }, line: { color: C.text2, transparency: 25 } });
  text(s, "Golden capture s-05: a printed part on the 20 mm ChArUco reference board", { x: photoX + 0.2, y: H - 0.55, w: W - photoX - 0.4, h: 0.55, fontSize: 11, color: C.background1, valign: "middle" });
  s.addNotes(NOTES[0]);
});

// 2 · Team ───────────────────────────────────────────────────────────────────
SLIDES.push(async (pres, C) => {
  const s = contentSlide(pres, "Introduction", "Team information", "Presenter · Mamadou", "Three engineers, one shared backlog");
  const people = [
    { name: "Mamadou", handle: "@mbj1994", slides: "Presents slides 1–5", built: ["Shared GPU workspace", "Golden capture set", "Camera calibration", "MASt3R + COLMAP backbones", "Reconstruction API service", "Result screen, viewer, colouring"] },
    { name: "Yashi", handle: "@dalwalyk", slides: "Presents slides 6–9", built: ["Repo skeleton, CI, dev setup", "VGGT runner, metric alignment", "Segmentation, fusion, cleanup", "Provenance + uncertainty", "Live camera + capture loop", "Gate, orbit guide, app ↔ service"] },
    { name: "Tabeen Raoof", handle: "@TabeenRaoof", slides: "Presents slides 10–15", built: ["Branch protection, API contract", "Board detection + camera pose", "Container + cloud deployment", "Real-data checks and fixes", "Next-view scoring", "Upload page; the first phone attempt"] },
  ];
  const cw = 3.84;
  const gap = (W - 2 * M - 3 * cw) / 2;
  for (let i = 0; i < people.length; i++) {
    const p = people[i];
    const x = M + i * (cw + gap);
    const y = 1.62;
    s.addShape("roundRect", { x, y, w: cw, h: 4.85, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 } });
    s.addShape("ellipse", { x: x + 0.3, y: y + 0.3, w: 0.78, h: 0.78, fill: { color: C.accent1 }, line: { color: C.accent1 } });
    text(s, p.name[0], { x: x + 0.3, y: y + 0.3, w: 0.78, h: 0.78, fontSize: 28, bold: true, color: C.background1, align: "center", valign: "middle", fontFace: "+mj-lt" });
    text(s, p.name, { x: x + 1.25, y: y + 0.27, w: cw - 1.4, h: 0.42, fontSize: 19, bold: true, color: C.text1, fontFace: "+mj-lt", valign: "middle" });
    text(s, p.handle, { x: x + 1.25, y: y + 0.7, w: cw - 1.4, h: 0.3, fontSize: 12, color: C.accent1, valign: "middle" });
    text(s, "Student ID: ____________", { x: x + 0.3, y: y + 1.28, w: cw - 0.6, h: 0.3, fontSize: 12, color: C.accent4 });
    text(s, "BUILT", { x: x + 0.3, y: y + 1.78, w: 2, h: 0.25, fontSize: 10, bold: true, color: C.accent4, charSpacing: 1.5 });
    text(
      s,
      p.built.map((b, j) => ({ text: b, options: { bullet: { indent: 14 }, breakLine: j < p.built.length - 1 } })),
      { x: x + 0.3, y: y + 2.08, w: cw - 0.45, h: 2.05, fontSize: 13, color: C.text1, paraSpaceAfter: 4 },
    );
    chip(s, C, p.slides, { x: x + 0.3, y: y + 4.2, w: cw - 0.6, h: 0.42, fill: HEX.lt1, color: C.accent1, bold: true, fontSize: 12, line: HEX.accent6 });
  }
  text(s, "Lanes are places in the code, not people: anyone can take any story, and a different teammate reviews it.", { x: M, y: 6.6, w: W - 2 * M, h: 0.3, fontSize: 12, color: C.accent4, italic: true });
  s.addNotes(NOTES[1]);
});

// 3 · System overview + foundations ──────────────────────────────────────────
SLIDES.push(async (pres, C) => {
  const s = contentSlide(pres, "Completed work", "Completed work · system overview", "Presenter · Mamadou", "One pipeline: phone photos in, metric 3D model out");
  const top = 1.55;
  const ph = 2.5;
  s.addShape("roundRect", { x: M, y: top, w: 2.3, h: ph, rectRadius: 0.08, fill: { color: C.accent1 }, line: { color: C.accent1 } });
  s.addImage({ data: await icon("FaMobileScreenButton", HEX.lt1), x: M + 0.85, y: top + 0.28, w: 0.6, h: 0.6 });
  text(s, "Capture app", { x: M + 0.1, y: top + 1.03, w: 2.1, h: 0.35, fontSize: 15, bold: true, color: C.background1, align: "center" });
  text(s, "phone browser:\nphotos + gyro in,\ncoloured 3D model out", { x: M + 0.1, y: top + 1.42, w: 2.1, h: 0.9, fontSize: 11.5, color: C.background1, align: "center" });
  s.addShape("line", { x: 2.98, y: top + ph / 2 + 0.05, w: 0.62, h: 0, line: { color: C.accent4, width: 1.5, beginArrowType: "triangle", endArrowType: "triangle" } });
  text(s, "HTTPS\nAPI", { x: 2.9, y: top + ph / 2 - 0.55, w: 0.8, h: 0.5, fontSize: 9.5, color: C.accent4, align: "center" });
  const px = 3.72;
  const pw = W - M - px;
  s.addShape("roundRect", { x: px, y: top, w: pw, h: ph, rectRadius: 0.06, fill: { color: C.background2 }, line: { color: C.accent6 } });
  text(s, "RECONSTRUCTION SERVICE · CLOUD GPU", { x: px + 0.25, y: top + 0.12, w: 6, h: 0.28, fontSize: 10, bold: true, color: C.accent1, charSpacing: 1.5 });
  const stages = [
    ["1", "Board pose\nper photo"], ["2", "Learned 3D\nreconstruction"], ["3", "Metric\nalignment (mm)"], ["4", "Segment the\npart"],
    ["5", "TSDF fusion\nto a mesh"], ["6", "Mesh cleanup"], ["7", "Observed / unseen\n+ uncertainty σ"], ["8", "Labelled GLB\nmodel"],
  ];
  const cw = 1.95;
  const cgap = (pw - 0.5 - 4 * cw) / 3;
  for (let i = 0; i < stages.length; i++) {
    const row = Math.floor(i / 4);
    const col = i % 4;
    const x = px + 0.25 + col * (cw + cgap);
    const y = top + 0.5 + row * 0.98;
    s.addShape("roundRect", { x, y, w: cw, h: 0.78, rectRadius: 0.08, fill: { color: C.accent1 }, line: { color: C.accent1 } });
    text(s, stages[i][0], { x: x + 0.1, y: y + 0.07, w: 0.3, h: 0.3, fontSize: 11, bold: true, color: C.accent5 });
    text(s, stages[i][1], { x: x + 0.3, y, w: cw - 0.38, h: 0.78, fontSize: 12, bold: true, color: C.background1, valign: "middle", align: "center" });
    if (col < 3) {
      s.addShape("line", { x: x + cw + 0.04, y: y + 0.39, w: cgap - 0.08, h: 0, line: { color: C.accent1, width: 1.25, endArrowType: "triangle" } });
    }
  }
  // Foundations strip
  text(s, "BUILT FIRST, SO THREE PEOPLE COULD WORK IN PARALLEL", { x: M, y: 4.35, w: 9, h: 0.28, fontSize: 10, bold: true, color: C.accent4, charSpacing: 1.5 });
  const tiles = [
    { big: "5", label: "API operations, frozen", sub: "App and backend both test against one contract", ic: "FaFileContract" },
    { big: "142", label: "golden photos", sub: "5 sets of 4 objects; checksummed, never in git", ic: "FaCameraRetro" },
    { big: "T4", label: "shared cloud GPU", sub: "One command sets up a Mac or the GPU", ic: "FaMicrochip" },
    { big: "1+", label: "reviewer on every PR", sub: "Plus CI; protection applies to admins too", ic: "FaCodePullRequest" },
  ];
  const tw = (W - 2 * M - 3 * 0.22) / 4;
  for (let i = 0; i < tiles.length; i++) {
    const t = tiles[i];
    const x = M + i * (tw + 0.22);
    const y = 4.72;
    s.addShape("roundRect", { x, y, w: tw, h: 1.95, rectRadius: 0.07, fill: { color: C.background1 }, line: { color: C.accent6 } });
    s.addImage({ data: await icon(t.ic, HEX.accent1), x: x + tw - 0.55, y: y + 0.22, w: 0.32, h: 0.32 });
    text(s, t.big, { x: x + 0.22, y: y + 0.12, w: tw - 0.9, h: 0.6, fontSize: 28, bold: true, color: C.accent1, fontFace: "+mj-lt", valign: "middle" });
    text(s, t.label, { x: x + 0.22, y: y + 0.76, w: tw - 0.4, h: 0.3, fontSize: 13, bold: true, color: C.text1 });
    text(s, t.sub, { x: x + 0.22, y: y + 1.08, w: tw - 0.4, h: 0.75, fontSize: 11.5, color: C.accent4 });
  }
  s.addNotes(NOTES[2]);
});

// 4 · Backbones + latency ────────────────────────────────────────────────────
SLIDES.push(async (pres, C) => {
  const s = contentSlide(pres, "Completed work", "Completed work · reconstruction backbones", "Presenter · Mamadou", "Three reconstruction backbones behind one switch");
  text(s, "Measured wall-clock per stage · Tesla T4 · the same 30 photos (s-04) · seconds", { x: M, y: 1.55, w: 7.0, h: 0.3, fontSize: 12, bold: true, color: C.text1 });
  const vO = 80.21 - (36.12 + 19.72 + 16.61);
  const mO = 265.9 - (22.93 + 21.0 + 211.58);
  const labels = ["MASt3R · 266 s", "VGGT · 80 s"];
  s.addChart(
    pres.charts.BAR,
    [
      { name: "Model load", labels, values: [22.93, 36.12] },
      { name: "Preprocess", labels, values: [21.0, 19.72] },
      { name: "Inference / global alignment", labels, values: [211.58, 16.61] },
      { name: "Other (unproject, write, overhead)", labels, values: [+mO.toFixed(2), +vO.toFixed(2)] },
    ],
    {
      x: M, y: 1.9, w: 7.0, h: 2.75,
      barDir: "bar", barGrouping: "stacked", barGapWidthPct: 55,
      chartColors: [HEX.accent5, HEX.accent6, HEX.accent1, HEX.accent2],
      catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 13, catAxisLabelFontFace: "+mn-lt", catAxisLineShow: false,
      valAxisLabelColor: HEX.accent4, valAxisLabelFontSize: 10, valAxisLabelFontFace: "+mn-lt", valAxisMaxVal: 280, valAxisMajorUnit: 50,
      valGridLine: { color: "E3E9EB", size: 0.75 }, catGridLine: { style: "none" },
      showLegend: true, legendPos: "b", legendFontSize: 10.5, legendFontFace: "+mn-lt", legendColor: HEX.accent4,
      showValue: false,
    },
  );
  s.addShape("roundRect", { x: M, y: 4.85, w: 7.0, h: 0.72, rectRadius: 0.06, fill: { color: C.background1 }, line: { color: C.accent6 } });
  s.addImage({ data: await icon("FaHand", HEX.accent3), x: M + 0.22, y: 5.04, w: 0.34, h: 0.34 });
  text(s, [
    { text: "COLMAP registered 28 of 30 photos, so it refused ", options: { bold: true } },
    { text: "rather than invent the 2 missing views (81 s). A refusal is a valid result." },
  ], { x: M + 0.75, y: 4.85, w: 6.1, h: 0.72, fontSize: 12.5, color: C.text1, valign: "middle" });
  const rx = 8.05;
  const rw = W - M - rx;
  const bb = [
    ["VGGT-1B", "Primary. One forward pass over all photos returns camera poses and per-pixel depth."],
    ["MASt3R", "Fallback. Pairwise 3D matching plus an iterative global alignment, hence the long bar."],
    ["COLMAP", "Classic structure-from-motion, seeded with the board poses. Refuses if any photo fails."],
  ];
  for (let i = 0; i < bb.length; i++) {
    const y = 1.6 + i * 1.05;
    text(s, bb[i][0], { x: rx, y, w: rw, h: 0.32, fontSize: 15, bold: true, color: C.accent1, fontFace: "+mj-lt" });
    text(s, bb[i][1], { x: rx, y: y + 0.34, w: rw, h: 0.62, fontSize: 12.5, color: C.text1 });
  }
  s.addShape("roundRect", { x: rx, y: 4.85, w: rw, h: 1.55, rectRadius: 0.07, fill: { color: C.background2 }, line: { color: C.background2 } });
  text(s, [
    { text: "Same four output files from all three, chosen by one config value, ", options: { bold: true } },
    { text: "so nothing downstream changes when the backbone does.", options: { breakLine: true } },
    { text: "About 80 s per session means batch capture is the path we ship; live feedback would need about 2 s per update.", options: { paraSpaceBefore: 6 } },
  ], { x: rx + 0.22, y: 4.98, w: rw - 0.44, h: 1.3, fontSize: 12.5, color: C.text1, valign: "middle" });
  s.addNotes(NOTES[3]);
});

// 5 · Metric scale ───────────────────────────────────────────────────────────
SLIDES.push(async (pres, C) => {
  const s = contentSlide(pres, "Completed work", "Completed work · metric scale", "Presenter · Mamadou", "Millimetres come from the printed board");
  shot(s, C, path.join(FIG, "fig_cameras.png"), { x: M, y: 1.55, w: 5.15, h: 3.75 }, "Real poses for s-05: 29 cameras solved from the board, each line runs from the camera to where it was looking.", 0.5);
  const steps = [
    ["Detect the board, solve each camera's pose", "138 of 142 golden photos posed (ChArUco + PnP)"],
    ["Calibrate the lens from the board", "1.20 px reprojection RMS on s-04"],
    ["Fit one similarity transform (Umeyama)", "maps VGGT's arbitrary units into board millimetres"],
  ];
  const sx = 6.15;
  for (let i = 0; i < steps.length; i++) {
    const y = 1.6 + i * 0.78;
    numberDot(s, C, i + 1, sx, y + 0.04, 0.42);
    text(s, steps[i][0], { x: sx + 0.58, y, w: W - M - sx - 0.58, h: 0.3, fontSize: 14, bold: true, color: C.text1 });
    text(s, steps[i][1], { x: sx + 0.58, y: y + 0.32, w: W - M - sx - 0.58, h: 0.3, fontSize: 12, color: C.accent4 });
  }
  text(s, "Board as a ruler: the whole scene comes out 4–9% small", { x: sx, y: 4.0, w: W - M - sx, h: 0.32, fontSize: 13, bold: true, color: C.accent3 });
  const cats = ["s-01", "s-02", "s-03", "s-04", "s-05"];
  s.addChart(
    pres.charts.BAR,
    [
      { name: "Board measured in the reconstruction (true size = 1.00)", labels: cats, values: [0.928, 0.907, 0.938, 0.958, 0.943] },
      { name: "Focal-length ratio, board poses ÷ VGGT", labels: cats, values: [0.935, 0.918, 0.938, 0.956, 0.944] },
    ],
    {
      x: sx - 0.1, y: 4.35, w: W - M - sx + 0.1, h: 2.45,
      barDir: "col", barGrouping: "clustered", barGapWidthPct: 70,
      chartColors: [HEX.accent1, HEX.accent5],
      valAxisMinVal: 0.85, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00",
      catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt",
      valAxisLabelColor: HEX.accent4, valAxisLabelFontSize: 10, valAxisLabelFontFace: "+mn-lt",
      valGridLine: { color: "E3E9EB", size: 0.75 }, catGridLine: { style: "none" },
      showLegend: true, legendPos: "b", legendFontSize: 10, legendFontFace: "+mn-lt", legendColor: HEX.accent4,
    },
  );
  s.addNotes(NOTES[4]);
});

// 6 · Photos → mesh ──────────────────────────────────────────────────────────
SLIDES.push(async (pres, C) => {
  const s = contentSlide(pres, "Completed work", "Completed work · geometry", "Presenter · Yashi", "From photos to a fused mesh, on real data");
  shot(
    s, C, path.join(FIG, "fig_pipeline_strip.jpg"),
    { x: M, y: 1.75, w: W - 2 * M, h: 3.1 },
    "The geometry chain on golden sample s-05, the red prism: (a) one of 29 phone photos, (b) VGGT's depth for every pixel in board millimetres, (c) segmented — board plane fitted, low-confidence pixels dropped, (d) the TSDF-fused mesh before cleanup.",
    0.55,
  );
  s.addShape("roundRect", { x: M, y: 6.12, w: W - 2 * M, h: 0.6, rectRadius: 0.06, fill: { color: C.background2 }, line: { color: C.background2 } });
  text(s, [
    { text: "Every stage is its own tested module with a CLI and a JSON report, ", options: { bold: true } },
    { text: "so each one can be rerun and inspected alone. The fused mesh still has artifacts: slides 7 and 11." },
  ], { x: M + 0.25, y: 6.12, w: W - 2 * M - 0.5, h: 0.6, fontSize: 12.5, color: C.text1, valign: "middle" });
  s.addNotes(NOTES[5]);
});

// 7 · The hard rule ──────────────────────────────────────────────────────────
SLIDES.push(async (pres, C) => {
  const s = contentSlide(pres, "Completed work", "Completed work · provenance and uncertainty", "Presenter · Yashi", "Unseen geometry can never pass or fail a part");
  shot(s, C, path.join(FIG, "fig_provenance.jpg"), { x: M, y: 1.5, w: 4.5, h: 4.5 });
  const ly = 6.12;
  s.addShape("rect", { x: M, y: ly + 0.06, w: 0.22, h: 0.22, fill: { color: C.accent1 }, line: { color: C.accent1 } });
  text(s, "Observed", { x: M + 0.3, y: ly, w: 1.2, h: 0.34, fontSize: 12, color: C.text1, valign: "middle" });
  s.addShape("rect", { x: M + 1.45, y: ly + 0.06, w: 0.22, h: 0.22, fill: { color: "CFD3D6" }, line: { color: "B9BEC1" } });
  text(s, "Unobserved", { x: M + 1.75, y: ly, w: 1.4, h: 0.34, fontSize: 12, color: C.text1, valign: "middle" });
  text(s, "Real s-05 mesh, current pipeline: the part is Observed, the artifacts are not", { x: M, y: ly + 0.38, w: 4.8, h: 0.3, fontSize: 10.5, italic: true, color: C.accent4 });

  const rx = 5.55;
  const rw = W - M - rx;
  s.addShape("roundRect", { x: rx, y: 1.6, w: rw, h: 0.95, rectRadius: 0.07, fill: { color: C.text2 }, line: { color: C.text2 } });
  text(s, "If the camera did not really see it, it can only trigger a Rescan.", { x: rx + 0.3, y: 1.6, w: rw - 0.6, h: 0.95, fontSize: 19, color: C.background1, fontFace: "+mj-lt", valign: "middle" });
  const rules = [
    ["Observed", "seen by at least 2 photos, at least 2° apart, through pixels the model is confident about, and not hidden behind other surface"],
    ["σ per vertex", "spread of depth across the photos that see it; above 3 mm it is downgraded to Unobserved"],
    ["Enforced", "labels travel as vertex colours in the GLB; the API contract bars unseen regions from any verdict, and a test checks it"],
  ];
  for (let i = 0; i < rules.length; i++) {
    const y = 2.8 + i * 0.78;
    numberDot(s, C, i + 1, rx, y + 0.03, 0.4);
    text(s, [
      { text: rules[i][0] + ": ", options: { bold: true, color: C.text1 } },
      { text: rules[i][1], options: { color: C.text1 } },
    ], { x: rx + 0.55, y, w: rw - 0.55, h: 0.7, fontSize: 13 });
  }
  const stats = [
    ["13%", "of the s-05 mesh is Observed, mainly the part. The rest can only trigger a Rescan"],
    ["1.2–2.1 mm", "median σ per object across the five golden objects"],
  ];
  const sw = (rw - 0.25) / 2;
  for (let i = 0; i < stats.length; i++) {
    const x = rx + i * (sw + 0.25);
    s.addShape("roundRect", { x, y: 5.25, w: sw, h: 1.4, rectRadius: 0.07, fill: { color: C.background2 }, line: { color: C.background2 } });
    text(s, stats[i][0], { x: x + 0.22, y: 5.35, w: sw - 0.44, h: 0.55, fontSize: 26, bold: true, color: C.accent1, fontFace: "+mj-lt", valign: "middle" });
    text(s, stats[i][1], { x: x + 0.22, y: 5.92, w: sw - 0.44, h: 0.66, fontSize: 11.5, color: C.accent4 });
  }
  s.addNotes(NOTES[6]);
});

// 8 · Capture app ────────────────────────────────────────────────────────────
SLIDES.push(async (pres, C) => {
  const s = contentSlide(pres, "Completed work", "Completed work · capture app", "Presenter · Yashi", "The capture app: camera to coloured model");
  // Capture screen in a phone frame
  const capFile = path.join(FIG, "fig_desktop_capture.jpg");
  const cd = imageSize(capFile);
  const imgH = 4.4;
  const imgW = imgH * (cd.w / cd.h);
  const fx = M + 0.05;
  const fy = 1.85;
  text(s, "CAPTURE", { x: fx, y: 1.5, w: imgW + 0.24, h: 0.26, fontSize: 10, bold: true, color: C.accent4, charSpacing: 1.5, align: "center" });
  s.addShape("roundRect", { x: fx, y: fy, w: imgW + 0.24, h: imgH + 0.4, rectRadius: 0.2, fill: { color: C.text1 }, line: { color: C.text1 } });
  s.addImage({ path: capFile, x: fx + 0.12, y: fy + 0.2, w: imgW, h: imgH });
  // Result screen card
  const vFile = path.join(FIG, "fig_desktop_viewer.jpg");
  const vd = imageSize(vFile);
  const vx = fx + imgW + 0.24 + 0.3;
  const vw = 2.75;
  const vh = vw * (vd.h / vd.w);
  text(s, "RESULT", { x: vx, y: 1.5, w: vw, h: 0.26, fontSize: 10, bold: true, color: C.accent4, charSpacing: 1.5, align: "center" });
  s.addShape("roundRect", { x: vx - 0.06, y: fy - 0.06, w: vw + 0.12, h: vh + 0.12, rectRadius: 0.06, fill: { color: C.background1 }, line: { color: C.accent6 } });
  s.addImage({ path: vFile, x: vx, y: fy, w: vw, h: vh });
  text(s, "Desktop Chrome at phone size: a golden photo as the camera feed, our real s-05 mesh in the viewer.", { x: vx - 0.06, y: fy + vh + 0.15, w: vw + 0.12, h: 0.75, fontSize: 10, italic: true, color: C.accent4 });

  const rx = vx + vw + 0.45;
  const rw = W - M - rx;
  const feats = [
    ["FaLock", "Camera + gyro permissions", "iOS grants them only from a direct tap; verified on real iPhones"],
    ["FaCamera", "Live camera + capture loop", "Gate → JPEG → upload; only a server accept marks an orbit slot"],
    ["FaFilter", "On-device frame gate", "Rejects blur, clipped exposure and near-duplicates"],
    ["FaCompass", "Orbit guidance", "13 target views and a gyro-driven arrow; walked 13/13 on an iPhone"],
    ["FaCube", "Viewer with provenance colours", "three.js; blue Observed, slate Unobserved, with a legend"],
  ];
  for (let i = 0; i < feats.length; i++) {
    const [ic, head, sub] = feats[i];
    const y = 1.55 + i * 0.9;
    s.addShape("ellipse", { x: rx, y: y + 0.02, w: 0.56, h: 0.56, fill: { color: C.background2 }, line: { color: C.background2 } });
    s.addImage({ data: await icon(ic, HEX.accent1), x: rx + 0.14, y: y + 0.16, w: 0.28, h: 0.28 });
    text(s, head, { x: rx + 0.72, y, w: rw - 0.72, h: 0.3, fontSize: 14, bold: true, color: C.text1, valign: "middle" });
    text(s, sub, { x: rx + 0.72, y: y + 0.31, w: rw - 0.72, h: 0.5, fontSize: 11.5, color: C.accent4 });
  }
  s.addShape("roundRect", { x: rx, y: 6.12, w: rw, h: 0.6, rectRadius: 0.06, fill: { color: C.background2 }, line: { color: C.background2 } });
  text(s, [
    { text: "11 of 13 capture stories done. ", options: { bold: true } },
    { text: "Vite + TypeScript, no framework · 203 tests." },
  ], { x: rx + 0.2, y: 6.12, w: rw - 0.4, h: 0.6, fontSize: 12, color: C.text1, valign: "middle" });
  s.addNotes(NOTES[7]);
});

// 9 · Next-view scoring ──────────────────────────────────────────────────────
SLIDES.push(async (pres, C) => {
  const s = contentSlide(pres, "Completed work", "Completed work · next-view scoring", "Presenter · Yashi", "Where to point the camera next");
  shot(s, C, path.join(FIG, "fig_nextview.png"), { x: M, y: 1.5, w: 8.1, h: 4.05 });
  // Legend for the dots
  const ly = 5.72;
  const shades = ["E3E9EB", "7FB3BE", "16697A", "0E2F36"];
  text(s, "Each dot is a candidate camera (top-down). Darker = sees more of the unseen face.", { x: M, y: ly, w: 6.2, h: 0.3, fontSize: 11, color: C.accent4 });
  for (let i = 0; i < shades.length; i++) {
    s.addShape("ellipse", { x: 6.9 + i * 0.27, y: ly + 0.06, w: 0.18, h: 0.18, fill: { color: shades[i] }, line: { color: shades[i] } });
  }
  s.addShape("ellipse", { x: 8.15, y: ly + 0.02, w: 0.26, h: 0.26, fill: { type: "none" }, line: { color: HEX.accent3, width: 1.75 } });
  text(s, "best", { x: 8.47, y: ly, w: 0.6, h: 0.3, fontSize: 11, color: C.accent3, bold: true });

  const rx = 9.05;
  const rw = W - M - rx;
  const pts = [
    ["147 candidate views", "on a hemisphere round the part, 25–40 cm away, at most 60° from vertical"],
    ["Ray-cast to every unseen vertex", "the whole mesh blocks the rays, so Observed surface still hides what is behind it"],
    ["Score = unseen vertices in view", "ties go to the most head-on view"],
  ];
  for (let i = 0; i < pts.length; i++) {
    const y = 1.6 + i * 1.2;
    numberDot(s, C, i + 1, rx, y + 0.02, 0.4);
    text(s, pts[i][0], { x: rx + 0.55, y, w: rw - 0.55, h: 0.32, fontSize: 14, bold: true, color: C.text1 });
    text(s, pts[i][1], { x: rx + 0.55, y: y + 0.34, w: rw - 0.55, h: 0.8, fontSize: 12, color: C.accent4 });
  }
  s.addShape("roundRect", { x: M, y: 6.12, w: W - 2 * M, h: 0.6, rectRadius: 0.06, fill: { color: C.background1 }, line: { color: C.accent2, width: 1.25 } });
  text(s, [
    { text: "Pull request #144, in review. ", options: { bold: true, color: C.accent2 } },
    { text: "Tested on synthetic meshes. On today's real meshes the fusion artifacts are Unobserved too, so it would chase them; it pays off once the mesh fixes land." },
  ], { x: M + 0.25, y: 6.12, w: W - 2 * M - 0.5, h: 0.6, fontSize: 12.5, color: C.text1, valign: "middle" });
  s.addNotes(NOTES[8]);
});

// 10 · First real-phone attempt ──────────────────────────────────────────────
SLIDES.push(async (pres, C) => {
  const s = contentSlide(pres, "Completed work", "Completed work · connecting to the real service", "Presenter · Tabeen", "First real-phone attempt, and what it found");

  const colW = (W - 2 * M - 0.35) / 2;
  const colY = 1.55;
  const colH = 4.35;

  // Left: the deployment chicken-and-egg problem
  s.addShape("roundRect", { x: M, y: colY, w: colW, h: colH, rectRadius: 0.07, fill: { color: C.background2 }, line: { color: C.background2 } });
  text(s, "Attempt 1: the deployed service", { x: M + 0.3, y: colY + 0.22, w: colW - 0.6, h: 0.38, fontSize: 16, bold: true, color: C.text1, fontFace: "+mj-lt" });
  const warn1 = await icon("FaTriangleExclamation", HEX.accent2);
  const left = [
    ["Same chicken-and-egg R-15 found", "the deployed image only has code already on main — our own CORS fix couldn't be tested there before it merged"],
    ["Stopped that deployment", "rather than let it keep billing against a check it couldn't pass"],
  ];
  for (let i = 0; i < left.length; i++) {
    const y = colY + 0.85 + i * 1.0;
    s.addImage({ data: warn1, x: M + 0.3, y: y + 0.04, w: 0.26, h: 0.26 });
    text(s, [
      { text: left[i][0] + " — ", options: { bold: true } },
      { text: left[i][1] },
    ], { x: M + 0.72, y, w: colW - 1.0, h: 0.9, fontSize: 12.5, color: C.text1 });
  }
  const pivot = await icon("FaArrowRightArrowLeft", HEX.accent1);
  s.addShape("roundRect", { x: M + 0.3, y: colY + 2.85, w: colW - 0.6, h: 1.25, rectRadius: 0.06, fill: { color: C.background1 }, line: { color: C.accent6 } });
  s.addImage({ data: pivot, x: M + 0.52, y: colY + 3.08, w: 0.3, h: 0.3 });
  text(s, [
    { text: "Decision (D-049): ", options: { bold: true, color: C.accent1 } },
    { text: "move the end-to-end check to our M2 Max laptop, running the service locally, with Tabeen joining C-13 as co-owner." },
  ], { x: M + 0.95, y: colY + 2.98, w: colW - 1.25, h: 1.0, fontSize: 12.5, color: C.text1, valign: "middle" });

  // Right: the EXIF finding and the fix
  const rx = M + colW + 0.35;
  s.addShape("roundRect", { x: rx, y: colY, w: colW, h: colH, rectRadius: 0.07, fill: { color: C.background2 }, line: { color: C.background2 } });
  text(s, "Attempt 2: on the Mac — zero poses", { x: rx + 0.3, y: colY + 0.22, w: colW - 0.6, h: 0.38, fontSize: 16, bold: true, color: C.text1, fontFace: "+mj-lt" });
  const warn2 = await icon("FaTriangleExclamation", HEX.accent2);
  text(s, [
    { text: "19 and 17 frames, both failed", options: { bold: true } },
    { text: " — the board was detected in every frame, but every frame came back intrinsics_unavailable." },
  ], { x: rx + 0.72, y: colY + 0.85, w: colW - 1.0, h: 0.7, fontSize: 12.5, color: C.text1 });
  s.addImage({ data: warn2, x: rx + 0.3, y: colY + 0.9, w: 0.26, h: 0.26 });
  text(s, [
    { text: "Root cause: ", options: { bold: true } },
    { text: "frames from the live camera preview carry no EXIF focal length, unlike the golden capture's Camera-app photos." },
  ], { x: rx + 0.3, y: colY + 1.65, w: colW - 0.6, h: 0.65, fontSize: 12.5, color: C.text1 });
  const check2 = await icon("FaCircleCheck", HEX.accent1);
  s.addShape("roundRect", { x: rx + 0.3, y: colY + 2.4, w: colW - 0.6, h: 1.7, rectRadius: 0.06, fill: { color: C.background1 }, line: { color: C.accent6 } });
  s.addImage({ data: check2, x: rx + 0.52, y: colY + 2.6, w: 0.28, h: 0.28 });
  text(s, [
    { text: "Fix (D-050): ", options: { bold: true, color: C.accent1 } },
    { text: "calibrate the camera from the board itself when EXIF is missing." },
  ], { x: rx + 0.95, y: colY + 2.55, w: colW - 1.25, h: 0.55, fontSize: 12.5, color: C.text1 });
  text(s, [
    { text: "Validated offline, not yet live: ", options: { bold: true } },
    { text: "reprocessed the failed session's own 19 frames — 0.20 px calibration error, 18 of 19 posed, a real metric result. Confirming it on an actual phone is next (slide 14)." },
  ], { x: rx + 0.3, y: colY + 3.15, w: colW - 0.6, h: 0.9, fontSize: 11.5, color: C.accent4 });

  // Bottom: the cloud deployment, compact (unaffected by the finding above)
  const by = 6.15;
  s.addShape("roundRect", { x: M, y: by, w: W - 2 * M, h: 0.6, rectRadius: 0.06, fill: { color: C.background1 }, line: { color: C.accent6 } });
  s.addImage({ data: await icon("FaMicrochip", HEX.accent1), x: M + 0.22, y: by + 0.16, w: 0.28, h: 0.28 });
  text(s, [
    { text: "Cloud deployment (R-15): ", options: { bold: true } },
    { text: "image built and smoke-tested in CI → registry → one-command deploy to a Lightning T4 over token-gated HTTPS, verified 9 Oct. Browser → cloud GPU still open (O-005)." },
  ], { x: M + 0.65, y: by, w: W - 2 * M - 0.85, h: 0.6, fontSize: 12, color: C.text1, valign: "middle" });
  s.addNotes(NOTES[9]);
});

// 11 · Real data ─────────────────────────────────────────────────────────────
SLIDES.push(async (pres, C) => {
  const s = contentSlide(pres, "Completed work", "Completed work · testing on real data", "Presenter · Tabeen", "Real photos exposed what unit tests could not");
  shot(
    s, C, path.join(FIG, "fig_confidence.jpg"),
    { x: M, y: 1.7, w: 6.1, h: 3.2 },
    "s-05, segmented points: (a) every pixel kept, 17.7% stray; (b) VGGT's confidence floor applied (R-20), 2.5% stray. Stray = outside the part's main cluster.",
    0.55,
  );
  const rx = 7.05;
  const rw = W - M - rx;
  text(s, "A visual check of all five objects (R-17) passed none. Each cause became its own story:", { x: rx, y: 1.55, w: rw, h: 0.6, fontSize: 13, bold: true, color: C.text1 });
  const causes = [
    ["Stray low-confidence points", "Threshold set at VGGT's confidence floor: stray 11–29% → ≤2.5%. Provenance trusts only those pixels too", "R-20, R-24 · merged"],
    ["Base of every part cut off", "Cut sat 3.6–7.4 mm above the real board. Now fit the board plane and cut from its own noise", "R-21 · merged"],
    ["Everything comes out too small", "Whole scene 4–9% small, from camera-centre alignment (slide 5). R-23 is filed but not yet claimed. Walls shrink further: open", "R-22 · merged"],
  ];
  for (let i = 0; i < causes.length; i++) {
    const y = 2.3 + i * 1.3;
    s.addShape("roundRect", { x: rx, y, w: rw, h: 1.15, rectRadius: 0.07, fill: { color: C.background2 }, line: { color: C.background2 } });
    numberDot(s, C, i + 1, rx + 0.2, y + 0.18, 0.38, C.accent3);
    text(s, causes[i][0], { x: rx + 0.75, y: y + 0.12, w: rw - 2.55, h: 0.38, fontSize: 14, bold: true, color: C.text1, valign: "middle" });
    chip(s, C, causes[i][2], { x: rx + rw - 1.75, y: y + 0.15, w: 1.57, h: 0.32, fill: HEX.lt1, line: HEX.accent6, color: C.accent1, bold: true, fontSize: 10.5, r: 0.15 });
    text(s, causes[i][1], { x: rx + 0.75, y: y + 0.52, w: rw - 0.95, h: 0.58, fontSize: 12, color: C.accent4 });
  }
  s.addShape("roundRect", { x: M, y: 6.2, w: W - 2 * M, h: 0.55, rectRadius: 0.06, fill: { color: C.background1 }, line: { color: C.accent2, width: 1.25 } });
  text(s, [
    { text: "Honest status: ", options: { bold: true, color: C.accent2 } },
    { text: "meshes still don't close into a recognisable part. Next: the scale fix, then confirming the phone fix (slide 10) on a real device." },
  ], { x: M + 0.25, y: 6.2, w: W - 2 * M - 0.5, h: 0.55, fontSize: 12.5, color: C.text1, valign: "middle" });
  s.addNotes(NOTES[10]);
});

// 12 · Progress + process ────────────────────────────────────────────────────
SLIDES.push(async (pres, C) => {
  const s = contentSlide(pres, "Completed work", "Completed work · progress and process", "Presenter · Tabeen", "Progress so far, and how we keep it honest");
  const stats = [
    ["40 / 44", "stories done", "2 open PRs, 1 filed, 1 blocked"],
    ["130", "pull requests merged", "each approved by a teammate"],
    ["585", "automated tests passing", "404 Python · 181 TypeScript"],
    ["45", "decisions recorded", "each with its reasoning"],
  ];
  const sw = (W - 2 * M - 3 * 0.25) / 4;
  for (let i = 0; i < stats.length; i++) {
    const x = M + i * (sw + 0.25);
    s.addShape("roundRect", { x, y: 1.6, w: sw, h: 1.55, rectRadius: 0.07, fill: { color: C.background2 }, line: { color: C.background2 } });
    text(s, stats[i][0], { x: x + 0.25, y: 1.7, w: sw - 0.5, h: 0.7, fontSize: 34, bold: true, color: C.accent1, fontFace: "+mj-lt", valign: "middle" });
    text(s, stats[i][1], { x: x + 0.25, y: 2.42, w: sw - 0.5, h: 0.3, fontSize: 13.5, bold: true, color: C.text1 });
    text(s, stats[i][2], { x: x + 0.25, y: 2.74, w: sw - 0.5, h: 0.3, fontSize: 11.5, color: C.accent4 });
  }
  text(s, "Stories done, by area of the code", { x: M, y: 3.5, w: 5.6, h: 0.3, fontSize: 13, bold: true, color: C.text1 });
  const lanes = [["Foundations", 7, 7], ["Reconstruction engine", 22, 24], ["Capture app", 11, 13]];
  const bx = M + 2.35;
  const bw = 3.3;
  for (let i = 0; i < lanes.length; i++) {
    const y = 4.0 + i * 0.62;
    text(s, lanes[i][0], { x: M, y, w: 2.3, h: 0.36, fontSize: 12.5, color: C.text1, valign: "middle" });
    s.addShape("roundRect", { x: bx, y: y + 0.06, w: bw, h: 0.24, rectRadius: 0.12, fill: { color: C.background2 }, line: { color: C.background2 } });
    s.addShape("roundRect", { x: bx, y: y + 0.06, w: bw * (lanes[i][1] / lanes[i][2]), h: 0.24, rectRadius: 0.12, fill: { color: C.accent1 }, line: { color: C.accent1 } });
    text(s, `${lanes[i][1]}/${lanes[i][2]}`, { x: bx + bw + 0.12, y, w: 0.7, h: 0.36, fontSize: 12.5, bold: true, color: C.text1, valign: "middle" });
  }
  text(s, "Inspection, ground truth and generative-AI stories are written one sprint ahead, so they are not counted yet.", { x: M, y: 5.95, w: 5.9, h: 0.6, fontSize: 11, italic: true, color: C.accent4 });
  const px = 7.15;
  const pw = W - M - px;
  text(s, "Every story goes through the same gate", { x: px, y: 3.5, w: pw, h: 0.3, fontSize: 13, bold: true, color: C.text1 });
  const steps = ["Story file with testable criteria", "Claimed by one person, on main", "Tests written with the code", "Pull request + CI + teammate review", "Done; hours measured, never the estimate"];
  for (let i = 0; i < steps.length; i++) {
    const y = 3.95 + i * 0.52;
    numberDot(s, C, i + 1, px, y + 0.03, 0.34);
    text(s, steps[i], { x: px + 0.5, y, w: pw - 0.5, h: 0.4, fontSize: 13, color: C.text1, valign: "middle" });
  }
  s.addNotes(NOTES[11]);
});

// 13 · Timeline ──────────────────────────────────────────────────────────────
SLIDES.push(async (pres, C) => {
  const s = contentSlide(pres, "Plan", "Timeline", "Presenter · Tabeen", "Timeline: where we stand against the plan");
  const day0 = Date.UTC(2026, 8, 1);
  const dayN = Date.UTC(2026, 11, 21);
  const gx = 3.0;
  const gw = W - M - gx;
  const X = (m, d) => gx + gw * ((Date.UTC(2026, m - 1, d) - day0) / (dayN - day0));
  for (const [m, label] of [[9, "Sep"], [10, "Oct"], [11, "Nov"], [12, "Dec"]]) {
    const x = X(m, 1);
    s.addShape("line", { x, y: 1.95, w: 0, h: 4.42, line: { color: "E3E9EB", width: 0.75 } });
    text(s, label, { x: x + 0.06, y: 1.55, w: 0.8, h: 0.28, fontSize: 11, bold: true, color: C.accent4 });
  }
  for (const [m, d, label] of [[9, 25, "M1"], [10, 9, "M2"], [10, 16, "M3"], [10, 30, "M4"], [11, 13, "M5"], [12, 4, "M7"], [12, 18, "M8"]]) {
    const x = X(m, d);
    s.addShape("diamond", { x: x - 0.08, y: 1.98, w: 0.16, h: 0.16, fill: { color: C.text2 }, line: { color: C.text2 } });
    text(s, label, { x: x - 0.3, y: 2.15, w: 0.6, h: 0.22, fontSize: 9.5, bold: true, color: C.text2, align: "center" });
  }
  // Every "act" bar is capped at the 15 October cutoff; nothing drawn as done can extend past
  // the dashed "Today" line.
  const rows = [
    { name: "Foundations", plan: [[9, 3], [9, 25]], act: [[9, 3], [9, 24]], state: "done" },
    { name: "Reconstruction engine", plan: [[9, 9], [10, 9]], act: [[9, 25], [10, 9]], state: "done" },
    { name: "Capture app (11 of 13)", plan: [[9, 3], [11, 13]], act: [[9, 29], [10, 15]], state: "progress" },
    { name: "Real-data fixes", plan: null, act: [[10, 9], [10, 15]], state: "progress" },
    { name: "Phone → service link", plan: [[9, 18], [9, 25]], act: [[10, 15], [10, 15]], state: "late" },
    { name: "Ground truth (σ_floor)", plan: [[10, 12], [10, 16]], act: null, state: "late" },
    { name: "Inspection (Tier 2)", plan: [[10, 19], [11, 13]], act: null, state: "todo" },
    { name: "Generative AI", plan: [[9, 28], [11, 13]], act: null, state: "late" },
    { name: "Harden + record demo", plan: [[11, 16], [12, 4]], act: null, state: "todo" },
    { name: "Final defense", plan: [[12, 7], [12, 18]], act: null, state: "todo" },
  ];
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const y = 2.46 + i * 0.4;
    const late = r.state === "late";
    text(s, r.name, { x: M, y, w: gx - M - 0.1, h: 0.3, fontSize: 11.5, bold: late, color: late ? C.accent3 : C.text1, valign: "middle" });
    if (r.plan) {
      const x1 = X(...r.plan[0]);
      const x2 = X(...r.plan[1]);
      s.addShape("roundRect", { x: x1, y: y + 0.04, w: x2 - x1, h: 0.2, rectRadius: 0.05, fill: { color: C.background1 }, line: { color: late ? HEX.accent3 : HEX.accent6, width: 1, dashType: late ? "dash" : "solid" } });
    }
    if (r.act) {
      const x1 = X(...r.act[0]);
      const x2 = X(...r.act[1]);
      const col = r.state === "done" ? HEX.accent1 : r.state === "progress" ? HEX.accent2 : HEX.accent3;
      s.addShape("roundRect", { x: x1, y: y + 0.04, w: Math.max(x2 - x1, 0.08), h: 0.2, rectRadius: 0.05, fill: { color: col }, line: { color: col } });
    }
  }
  const tx = X(10, 15);
  s.addShape("line", { x: tx, y: 2.38, w: 0, h: 4.08, line: { color: C.accent3, width: 1.5, dashType: "dash" } });
  text(s, "Today · 15 Oct", { x: tx - 0.7, y: 6.5, w: 1.4, h: 0.26, fontSize: 10, bold: true, color: C.accent3, align: "center" });
  const ly = 6.84;
  const leg = [["Planned", HEX.lt1, HEX.accent6, false], ["Done", HEX.accent1, HEX.accent1, false], ["In progress", HEX.lt1, HEX.accent2, true], ["Behind plan", HEX.lt1, HEX.accent3, true]];
  let lx = gx;
  for (const [label, fill, line, dash] of leg) {
    s.addShape("roundRect", { x: lx, y: ly + 0.04, w: 0.36, h: 0.17, rectRadius: 0.04, fill: { color: fill }, line: { color: line, dashType: dash ? "dash" : "solid" } });
    text(s, label, { x: lx + 0.44, y: ly - 0.02, w: 1.6, h: 0.28, fontSize: 10.5, color: C.accent4, valign: "middle" });
    lx += lx === gx ? 1.45 : 1.95;
  }
  s.addNotes(NOTES[12]);
});

// 14 · Next steps ────────────────────────────────────────────────────────────
SLIDES.push(async (pres, C) => {
  const s = contentSlide(pres, "Plan", "Next steps", "Presenter · Tabeen", "Next: confirm the fix, then inspect");
  const cols = [
    { n: 1, head: "Geometry + phone reliability", when: "Now", items: ["Anchor scale on the board", "Confirm the EXIF fix (D-050) on a real device", "Capture at full camera resolution"], ids: "R-23 + new stories" },
    { n: 2, head: "Ground truth", when: "M3 · slipping", items: ["Print test parts", "Caliper-measure them", "Measure σ_floor → first real tolerance"], ids: "G-01 – G-05, R-16" },
    { n: 3, head: "Tier 2 inspection", when: "M4 · 30 Oct", items: ["CAD registration", "Deviation heatmap", "Pass / Fail / Rescan / Unverifiable"], ids: "I-01 – I-10" },
    { n: 4, head: "Generative AI", when: "Start now → 13 Nov", items: ["VLM checks the capture (first)", "Plain-English spec → structured spec", "Shape completion, capture agent, report"], ids: "A-01 – A-09" },
    { n: 5, head: "Cloud path + product", when: "M5 – M6", items: ["Browser → cloud GPU, order code checked", "Host the app properly", "Android, buyer and report pages"], ids: "O-005, C-14 – C-18" },
  ];
  const cw = (W - 2 * M - 4 * 0.15) / 5;
  for (let i = 0; i < cols.length; i++) {
    const c = cols[i];
    const x = M + i * (cw + 0.15);
    const y = 1.6;
    const genai = c.head === "Generative AI";
    s.addShape("roundRect", { x, y, w: cw, h: 4.3, rectRadius: 0.07, fill: { color: genai ? HEX.lt1 : HEX.lt2 }, line: { color: genai ? HEX.accent2 : HEX.lt2, width: genai ? 2.25 : 0.75 } });
    numberDot(s, C, c.n, x + 0.2, y + 0.22, 0.42, genai ? C.accent3 : C.accent1);
    text(s, c.head, { x: x + 0.2, y: y + 0.78, w: cw - 0.4, h: 0.66, fontSize: 15, bold: true, color: C.text1, fontFace: "+mj-lt" });
    chip(s, C, c.when, { x: x + 0.2, y: y + 1.5, w: cw - 0.4, h: 0.32, fill: genai ? HEX.accent2 : HEX.lt1, line: genai ? HEX.accent2 : HEX.accent6, color: genai ? C.text1 : (c.when.includes("slipping") ? C.accent3 : C.accent1), bold: true, fontSize: 10.5, r: 0.15 });
    text(s, c.items.map((t, j) => ({ text: t, options: { bullet: { indent: 12 }, breakLine: j < c.items.length - 1 } })), { x: x + 0.2, y: y + 2.0, w: cw - 0.3, h: 1.75, fontSize: 12.5, color: C.text1, paraSpaceAfter: 6 });
    text(s, c.ids, { x: x + 0.2, y: y + 3.85, w: cw - 0.4, h: 0.3, fontSize: 10.5, color: C.accent4 });
  }
  s.addShape("roundRect", { x: M, y: 6.12, w: W - 2 * M, h: 0.6, rectRadius: 0.06, fill: { color: C.background1 }, line: { color: C.accent2, width: 1.25 } });
  text(s, [
    { text: "Generative AI is about a third of the course requirement and has not started. ", options: { bold: true } },
    { text: "It is never cut; it starts now, alongside inspection." },
  ], { x: M + 0.25, y: 6.12, w: W - 2 * M - 0.5, h: 0.6, fontSize: 12.5, color: C.text1, valign: "middle" });
  s.addNotes(NOTES[13]);
});

// 15 · References ────────────────────────────────────────────────────────────
SLIDES.push(async (pres, C) => {
  const s = contentSlide(pres, "Plan", "References · methods and software we build on", "Presenter · Tabeen", "References");
  const left = [
    "J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, D. Novotny, “VGGT: Visual Geometry Grounded Transformer,” CVPR, 2025.",
    "V. Leroy, Y. Cabon, J. Revaud, “Grounding Image Matching in 3D with MASt3R,” ECCV, 2024.",
    "J. L. Schönberger, J.-M. Frahm, “Structure-from-Motion Revisited,” CVPR, 2016.",
    "J. L. Schönberger, E. Zheng, M. Pollefeys, J.-M. Frahm, “Pixelwise View Selection for Unstructured Multi-View Stereo,” ECCV, 2016.",
    "S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, M. J. Marín-Jiménez, “Automatic generation and detection of highly reliable fiducial markers under occlusion,” Pattern Recognition, 47(6), 2014.",
    "G. Terzakis, M. Lourakis, “A Consistently Fast and Globally Optimal Solution to the Perspective-n-Point Problem,” ECCV, 2020.",
    "S. Umeyama, “Least-squares estimation of transformation parameters between two point patterns,” IEEE TPAMI, 13(4), 1991.",
    "B. Curless, M. Levoy, “A Volumetric Method for Building Complex Models from Range Images,” SIGGRAPH, 1996.",
  ];
  const right = [
    "W. E. Lorensen, H. E. Cline, “Marching Cubes: A High Resolution 3D Surface Construction Algorithm,” SIGGRAPH, 1987.",
    "M. Kazhdan, M. Bolitho, H. Hoppe, “Poisson Surface Reconstruction,” Eurographics SGP, 2006.",
    "M. Garland, P. S. Heckbert, “Surface Simplification Using Quadric Error Metrics,” SIGGRAPH, 1997.",
    "G. Bradski, “The OpenCV Library,” Dr. Dobb’s Journal of Software Tools, 2000.",
    "Q.-Y. Zhou, J. Park, V. Koltun, “Open3D: A Modern Library for 3D Data Processing,” arXiv:1801.09847, 2018.",
    "S. van der Walt et al., “scikit-image: image processing in Python,” PeerJ, 2:e453, 2014.",
    "Khronos Group, “glTF 2.0 Specification”; three.js, FastAPI, Vite and Lightning AI documentation.",
    "ProofShape team, project repository, stories and decision log: github.com/proofshape/proofshape",
  ];
  const cw = (W - 2 * M - 0.4) / 2;
  const fmt = (arr, start) => arr.map((t, i) => ({ text: `[${start + i}]  ${t}`, options: { breakLine: i < arr.length - 1, paraSpaceAfter: 9 } }));
  text(s, fmt(left, 1), { x: M, y: 1.6, w: cw, h: 5.2, fontSize: 12, color: C.text1 });
  text(s, fmt(right, left.length + 1), { x: M + cw + 0.4, y: 1.6, w: cw, h: 5.2, fontSize: 12, color: C.text1 });
  s.addNotes(NOTES[14]);
});

// Professor's structure: intro (title, team) · completed work (10) · plan (timeline, next, refs).
function sectionFor(i) {
  if (i < 2) return "Introduction";
  if (i < 12) return "Completed work";
  return "Plan";
}

async function build(outPath, only) {
  const { pres, C } = makePres();
  const added = new Set();
  for (let i = 0; i < SLIDES.length; i++) {
    if (only !== undefined && i !== only) continue;
    const section = sectionFor(i);
    if (!added.has(section)) {
      pres.addSection({ title: section });
      added.add(section);
    }
    await SLIDES[i](pres, C);
  }
  await pres.writeFile({ fileName: outPath });
  await applyTheme(outPath, THEME);
  return outPath;
}

(async () => {
  const out = process.argv[2] || path.join(__dirname, "..", "submissions", "midterm-presentation-deck.pptx");
  if (process.argv.includes("--split")) {
    fs.mkdirSync(out, { recursive: true });
    for (let i = 0; i < SLIDES.length; i++) {
      await build(path.join(out, `slide-${String(i + 1).padStart(2, "0")}.pptx`), i);
    }
    console.log("split decks written to", out);
  } else {
    await build(out);
    console.log("wrote", out);
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});

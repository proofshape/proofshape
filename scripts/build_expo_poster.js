// ProofShape — expo poster (one 48 × 27 in slide, 16:9; scales to 56 × 31.5 or any 16:9 size).
//
//   node scripts/build_expo_poster.js [out.pptx]   # defaults to submissions/expo-poster.pptx
//
// Layout and colours follow the SFBU capstone poster template the team was given: beige header
// with the SFBU logo and mascot, a gold band, a navy body, four columns of alternating tan and
// grey panels with light-blue section headers. The 3D figures come from one real session
// (scripts/render_expo_figures.py, golden s-04, sess_f82401bb6852). Every number is a recorded
// result; the source is noted beside each one in RESULTS below.

const path = require("path");
const PptxGenJS = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa6");

const ROOT = path.join(__dirname, "..");
const FIG = path.join(ROOT, "submissions", "assets", "expo");

const W = 48;
const H = 27;

// Sampled from the template poster.
const C = {
  header: "DDC5A3",
  navy: "00406C",
  tan: "B59059",
  grey: "C4CCD7",
  secHead: "96B4D8",
  title: "17406D",
  ink: "111820",
  white: "FFFFFF",
  gold: "C9A14A",
  goldDeep: "BA7C15",
  tableHead: "204780",
  rowA: "F4F6FA",
  rowB: "E3E9F2",
  pass: "15803D",
  fail: "B91C1C",
  rescan: "D97706",
  unver: "64748B",
  blue: "2563EB",
};

const FONT = "Arial";

// ── icons (react-icons → PNG) ───────────────────────────────────────────────
const iconCache = new Map();
async function icon(name, hex, size = 256) {
  const key = name + hex + size;
  if (iconCache.has(key)) return iconCache.get(key);
  const Comp = fa[name];
  if (!Comp) throw new Error(`no icon ${name}`);
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size: String(size) }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  const data = "image/png;base64," + png.toString("base64");
  iconCache.set(key, data);
  return data;
}

async function goldBand() {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="4800" height="60">
    <defs><linearGradient id="g" x1="0" x2="1" y1="0" y2="0">
      <stop offset="0" stop-color="#D5B55E"/><stop offset="0.35" stop-color="#C9A14A"/>
      <stop offset="0.5" stop-color="#BA7C15"/><stop offset="0.7" stop-color="#C69433"/>
      <stop offset="1" stop-color="#E2C77A"/></linearGradient>
      <linearGradient id="s" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0" stop-color="#fff" stop-opacity="0.35"/><stop offset="0.5" stop-color="#fff" stop-opacity="0"/>
      <stop offset="1" stop-color="#000" stop-opacity="0.18"/></linearGradient></defs>
    <rect width="4800" height="60" fill="url(#g)"/><rect width="4800" height="60" fill="url(#s)"/></svg>`;
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

async function heroBackdrop() {
  // soft radial light behind the 3D scene, so it reads as a lit stage rather than a flat box
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="960">
    <defs><radialGradient id="r" cx="0.42" cy="0.55" r="0.75">
      <stop offset="0" stop-color="#FFFFFF"/><stop offset="0.55" stop-color="#EEF2F8"/>
      <stop offset="1" stop-color="#C9D5E6"/></radialGradient></defs>
    <rect width="2400" height="960" fill="url(#r)"/></svg>`;
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

// ── helpers ─────────────────────────────────────────────────────────────────
function text(s, t, o) {
  s.addText(t, { fontFace: FONT, color: C.ink, margin: 0, valign: "top", ...o });
}

function sectionHeader(s, label, x, y, w) {
  s.addShape("rect", { x, y, w, h: 0.95, fill: { color: C.secHead }, line: { color: C.secHead } });
  text(s, label, { x: x + 0.3, y, w: w - 0.6, h: 0.95, fontSize: 40, bold: true, color: "0D1B2A", valign: "middle" });
  return y + 0.95;
}

function para(s, t, x, y, w, h, o = {}) {
  text(s, t, { x, y, w, h, fontSize: 28, align: "justify", lineSpacingMultiple: 1.08, ...o });
}

// ── build ───────────────────────────────────────────────────────────────────
async function build(out) {
  const pres = new PptxGenJS();
  pres.defineLayout({ name: "POSTER", width: W, height: H });
  pres.layout = "POSTER";
  pres.title = "ProofShape — Expo poster";
  pres.subject = "CS595 Capstone, Fall 2026, San Francisco Bay University";
  const s = pres.addSlide();
  s.background = { color: C.navy };

  // header
  const HH = 4.35;
  s.addShape("rect", { x: 0, y: 0, w: W, h: HH, fill: { color: C.header }, line: { color: C.header } });
  s.addImage({ path: path.join(FIG, "sfbu-logo-crop.png"), x: 0.7, y: 0.35, w: 4.2, h: 3.66 });
  s.addImage({ path: path.join(FIG, "sfbu-mascot-crop.png"), x: W - 0.6 - 4.15, y: 0.35, w: 4.15, h: 3.6 });
  text(s, [
    { text: "ProofShape: Inspecting Manufactured Parts", options: { breakLine: true } },
    { text: "with a Smartphone, 3D Reconstruction & AI", options: { breakLine: true } },
    { text: "Project Advisor : Professor Ahmed Banafa", options: { fontSize: 46 } },
  ], { x: 5.4, y: 0.25, w: 25.6, h: 3.9, fontSize: 64, bold: true, color: C.title, align: "center", valign: "middle", lineSpacingMultiple: 1.12 });
  text(s, [
    { text: "CS595 Capstone · Fall 2026", options: { breakLine: true, fontSize: 40 } },
    { text: " ", options: { breakLine: true, fontSize: 14 } },
    { text: "Team members:", options: { breakLine: true, fontSize: 34 } },
    { text: "Mamadou Jallow · Yashi Dalwala", options: { breakLine: true, fontSize: 34 } },
    { text: "Tabeen Raoof", options: { fontSize: 34 } },
  ], { x: 31.0, y: 0.35, w: 11.9, h: 3.7, bold: true, color: C.title, align: "center", valign: "middle" });

  s.addImage({ data: await goldBand(), x: 0, y: HH, w: W, h: 0.5 });

  // columns
  const top = HH + 0.5 + 0.55;
  const bottom = H - 0.55;
  const M = 0.6;
  const G = 0.45;
  const CW = (W - 2 * M - 3 * G) / 4;
  const X = [0, 1, 2, 3].map((i) => M + i * (CW + G));
  const PAD = 0.3;

  // column 1 · tan ─────────────────────────────────────────────────────────
  s.addShape("rect", { x: X[0], y: top, w: CW, h: bottom - top, fill: { color: C.tan }, line: { color: "8C6A3A", width: 1.5 } });
  let y = sectionHeader(s, "Introduction", X[0] + PAD, top + PAD, CW - 2 * PAD);
  para(s, "A part shipped out of spec costs a second shipment and days of delay. A coordinate-measuring machine (CMM) is accurate but stays in the lab, and a photo emailed by the supplier measures nothing. ProofShape is a web app that turns a smartphone into a coarse 3D inspection tool: the supplier photographs the part on a printed board, and the buyer gets a measured 3D model and a verdict against the CAD.",
    X[0] + PAD + 0.05, y + 0.2, CW - 2 * PAD - 0.1, 3.75);
  y += 4.15;
  y = sectionHeader(s, "How It Works", X[0] + PAD, y, CW - 2 * PAD);
  const steps = [
    ["FaFileArrowUp", "Buyer", "uploads the CAD and writes the requirements in plain English."],
    ["FaMobileScreenButton", "Supplier", "opens a link on the phone, sets the part on the printed board and follows the on-screen guidance."],
    ["FaCubes", "Service", "rebuilds a metric 3D model and compares it with the CAD."],
    ["FaClipboardCheck", "Report", "shows the deviation heatmap and the verdict, ready to share."],
  ];
  y += 0.25;
  for (const [ic, who, what] of steps) {
    s.addShape("ellipse", { x: X[0] + PAD + 0.1, y: y + 0.05, w: 1.25, h: 1.25, fill: { color: C.navy }, line: { color: C.gold, width: 3 } });
    s.addImage({ data: await icon(ic, "FFFFFF"), x: X[0] + PAD + 0.42, y: y + 0.37, w: 0.62, h: 0.62 });
    text(s, [
      { text: who + " ", options: { bold: true, color: "0D1B2A" } },
      { text: what },
    ], { x: X[0] + PAD + 1.6, y, w: CW - 2 * PAD - 1.7, h: 1.4, fontSize: 27, valign: "middle" });
    y += 1.4;
  }
  y += 0.1;
  y = sectionHeader(s, "Four Honest Verdicts", X[0] + PAD, y, CW - 2 * PAD);
  const verdicts = [
    ["PASS", C.pass, "every measured feature is within tolerance."],
    ["FAIL", C.fail, "a measured feature is out of tolerance."],
    ["RESCAN", C.rescan, "too little of the part was seen, or its pose is ambiguous."],
    ["UNVERIFIABLE", C.unver, "the tolerance is tighter than the method can measure, refused when the order is made."],
  ];
  y += 0.25;
  for (const [label, col, why] of verdicts) {
    s.addShape("roundRect", { x: X[0] + PAD + 0.05, y: y + 0.1, w: 3.5, h: 0.8, fill: { color: col }, line: { color: C.white, width: 2 }, rectRadius: 0.2 });
    text(s, label, { x: X[0] + PAD + 0.05, y: y + 0.1, w: 3.5, h: 0.8, fontSize: 25, bold: true, color: C.white, align: "center", valign: "middle" });
    text(s, why, { x: X[0] + PAD + 3.8, y, w: CW - 2 * PAD - 3.85, h: 1.0, fontSize: 24, valign: "middle" });
    y += 1.05;
  }
  // the hard rule
  const ry = bottom - PAD - 2.6;
  s.addShape("rect", { x: X[0] + PAD, y: ry, w: CW - 2 * PAD, h: 2.6, fill: { color: C.navy }, line: { color: C.gold, width: 4 } });
  s.addImage({ data: await icon("FaShieldHalved", C.gold), x: X[0] + PAD + 0.35, y: ry + 0.3, w: 0.8, h: 0.8 });
  text(s, "THE HARD RULE", { x: X[0] + PAD + 1.35, y: ry + 0.3, w: 8, h: 0.8, fontSize: 30, bold: true, color: C.gold, valign: "middle", charSpacing: 3 });
  text(s, "Geometry the camera never saw can never pass or fail a part. It can only ask for a Rescan.",
    { x: X[0] + PAD + 0.35, y: ry + 1.1, w: CW - 2 * PAD - 0.7, h: 1.35, fontSize: 29, bold: true, color: C.white, valign: "middle" });

  // hero · spans columns 2–3 ───────────────────────────────────────────────
  const hx = X[1];
  const hw = 2 * CW + G;
  const hh = 9.0;
  s.addImage({ data: await heroBackdrop(), x: hx, y: top, w: hw, h: hh });
  s.addShape("rect", { x: hx, y: top, w: hw, h: hh, fill: { type: "none" }, line: { color: C.gold, width: 4 } });
  s.addShape("rect", { x: hx, y: top, w: hw, h: 1.05, fill: { color: C.navy }, line: { color: C.navy } });
  text(s, [
    { text: "30 phone photos  ", options: { color: C.white } },
    { text: "→", options: { color: C.gold } },
    { text: "  camera poses from the printed board  ", options: { color: C.white } },
    { text: "→", options: { color: C.gold } },
    { text: "  a measured 3D model", options: { color: C.white } },
  ], { x: hx + 0.3, y: top, w: hw - 0.6, h: 1.05, fontSize: 36, bold: true, align: "center", valign: "middle" });
  // scene: 1994 × 1443 px
  const sh = hh - 1.05 - 0.35;
  const sw = sh * (1994 / 1443);
  const sx = hx + 0.35;
  const sy = top + 1.05 + 0.2;
  s.addImage({ path: path.join(FIG, "hero-scene.png"), x: sx, y: sy, w: sw, h: sh });
  // magnifier: part position in the scene image ≈ (1090, 1180) px
  const px = sx + (1090 / 1994) * sw;
  const py = sy + (1185 / 1443) * sh;
  // close-up: 1999 × 1110 px
  const ch = 4.3;
  const cw = ch * (1999 / 1110);
  const cx = hx + hw - cw - 0.35;
  const cy = top + 1.05 + 0.4;
  s.addShape("line", { x: px + 0.2, y: cy + ch, w: cx - px - 0.2, h: py - (cy + ch), flipV: true, line: { color: C.navy, width: 2.5, dashType: "dash" } });
  s.addShape("line", { x: px + 0.2, y: cy + 0.05, w: cx - px - 0.2, h: py - cy, flipV: true, line: { color: C.navy, width: 2.5, dashType: "dash" } });
  s.addShape("ellipse", { x: px - 0.95, y: py - 0.75, w: 1.9, h: 1.3, fill: { type: "none" }, line: { color: C.gold, width: 4 } });
  s.addShape("roundRect", { x: cx - 0.15, y: cy - 0.15, w: cw + 0.3, h: ch + 0.3, fill: { color: C.white, transparency: 10 }, line: { color: C.navy, width: 3 }, rectRadius: 0.25 });
  s.addImage({ path: path.join(FIG, "heatmap-closeup.png"), x: cx, y: cy, w: cw, h: ch });
  // scale + legend
  let ly = cy + ch + 0.35;
  text(s, "Signed distance from the CAD (Observed surface)", { x: cx - 0.15, y: ly, w: cw + 0.3, h: 0.55, fontSize: 22, bold: true, color: C.title, align: "center" });
  ly += 0.6;
  const barH = 0.42;
  s.addImage({ path: path.join(FIG, "heatmap-bar.png"), x: cx + 0.6, y: ly, w: cw - 1.2, h: barH });
  [["−4", 0], ["−2", 0.25], ["0", 0.5], ["+2", 0.75], ["+4 mm", 1]].forEach(([t, f]) => {
    text(s, t, { x: cx + 0.6 + f * (cw - 1.2) - 0.8, y: ly + barH + 0.02, w: 1.6, h: 0.45, fontSize: 20, bold: true, color: C.title, align: "center" });
  });
  ly += barH + 0.55;
  const legend = [
    [C.blue, "missing material"],
    ["B91C1C", "extra material"],
    [C.unver, "never seen"],
  ];
  let lx = cx - 0.05;
  for (const [col, lab] of legend) {
    s.addShape("rect", { x: lx, y: ly + 0.1, w: 0.38, h: 0.38, fill: { color: col }, line: { color: "334155", width: 1 } });
    const lw = lab.length * 0.13 + 0.2;
    text(s, lab, { x: lx + 0.48, y: ly, w: lw, h: 0.6, fontSize: 20, valign: "middle" });
    lx += 0.48 + lw + 0.15;
  }
  ly += 0.7;
  text(s, "Real output: golden part s-04, 30 iPhone photos, 76% observed.",
    { x: cx - 0.15, y: ly, w: cw + 0.3, h: 0.5, fontSize: 20, italic: true, color: "334155", align: "center" });

  // column 2 · grey, below hero ───────────────────────────────────────────────
  const lowTop = top + hh + 0.45;
  s.addShape("rect", { x: X[1], y: lowTop, w: CW, h: bottom - lowTop, fill: { color: C.grey }, line: { color: "8E99A8", width: 1.5 } });
  y = sectionHeader(s, "Methodology", X[1] + PAD, lowTop + PAD, CW - 2 * PAD);
  const stages = [
    ["FaCamera", "Capture", "Phone web app. Blurry, dark and duplicate frames are rejected on the phone with a reason; a vision-language model checks what each frame shows."],
    ["FaCube", "Reconstruct", "VGGT transformer gives depth and camera poses; the ChArUco board fixes millimetre scale; fused mesh, every vertex labelled Observed or not."],
    ["FaRulerCombined", "Inspect", "Global registration + ICP align the CAD; signed distance on Observed vertices only, judged against the measured noise floor."],
    ["FaFileLines", "Report", "3D viewer with the heatmap, verdict and reasons, shareable with the buyer."],
  ];
  y += 0.3;
  const stH = 1.7;
  for (let i = 0; i < stages.length; i++) {
    const [ic, name, body] = stages[i];
    const bx = X[1] + PAD;
    const bw = CW - 2 * PAD;
    s.addShape("roundRect", { x: bx, y, w: bw, h: stH, fill: { color: C.white }, line: { color: C.navy, width: 2 }, rectRadius: 0.18, shadow: { type: "outer", color: "000000", opacity: 0.25, blur: 6, offset: 3, angle: 90 } });
    s.addShape("rect", { x: bx, y, w: 1.45, h: stH, fill: { color: C.navy }, line: { color: C.navy } });
    s.addImage({ data: await icon(ic, "FFFFFF"), x: bx + 0.38, y: y + 0.28, w: 0.7, h: 0.7 });
    text(s, String(i + 1), { x: bx, y: y + 1.05, w: 1.45, h: 0.6, fontSize: 24, bold: true, color: C.gold, align: "center" });
    text(s, [
      { text: name, options: { bold: true, color: C.title, fontSize: 27, breakLine: true } },
      { text: body, options: { fontSize: 21 } },
    ], { x: bx + 1.65, y: y + 0.1, w: bw - 1.8, h: stH - 0.2, valign: "middle" });
    y += stH;
    if (i < stages.length - 1) {
      s.addShape("downArrow", { x: bx + bw / 2 - 0.3, y: y + 0.04, w: 0.6, h: 0.34, fill: { color: C.goldDeep }, line: { color: C.goldDeep } });
      y += 0.42;
    }
  }
  y += 0.25;
  const stack = ["Python", "PyTorch", "FastAPI", "OpenCV", "Open3D", "TypeScript", "three.js", "Gemini"];
  let tx = X[1] + PAD;
  const chipW = (CW - 2 * PAD - 3 * 0.15) / 4;
  stack.forEach((t, i) => {
    const r = Math.floor(i / 4);
    const c = i % 4;
    const xx = tx + c * (chipW + 0.15);
    const yy = y + r * 0.68;
    s.addShape("roundRect", { x: xx, y: yy, w: chipW, h: 0.6, fill: { color: C.navy }, line: { color: C.gold, width: 1.5 }, rectRadius: 0.3 });
    text(s, t, { x: xx, y: yy, w: chipW, h: 0.6, fontSize: 20, bold: true, color: C.white, align: "center", valign: "middle" });
  });

  // column 3 · tan, below hero ────────────────────────────────────────────────
  s.addShape("rect", { x: X[2], y: lowTop, w: CW, h: bottom - lowTop, fill: { color: C.tan }, line: { color: "8C6A3A", width: 1.5 } });
  y = sectionHeader(s, "Analysis & Results", X[2] + PAD, lowTop + PAD, CW - 2 * PAD);
  text(s, "Measured on real 3D-printed parts and iPhone captures", { x: X[2] + PAD + 0.05, y: y + 0.2, w: CW - 2 * PAD, h: 0.6, fontSize: 27, bold: true, color: "0D1B2A" });
  y += 0.95;
  // RESULTS — sources: progress log 2026-11-16 (R-36), D-073 + R-16 note in decisions, D-078,
  // progress log G-01 entry, R-33 close-out, R-13 (PR #98).
  const RESULTS = [
    ["Phone to 3D model", "31–51 s after Finish (3 iPhone sessions, laptop service)"],
    ["Height accuracy", "−0.52 mm (s-04), −0.21 mm (s-05) against calipers"],
    ["Measured noise floor", "0.86 mm in height · 5.28 mm in-plane"],
    ["Ground-truth printer", "all axes within one 0.1 mm caliper step"],
    ["Wall artifacts fixed", "34–79% → 0.2–2.6% of faces, 5 parts"],
    ["Backbone chosen", "VGGT 80 s vs MASt3R 266 s; COLMAP refused"],
  ];
  const head = (t) => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.tableHead }, fontSize: 24, align: "center", valign: "middle" } });
  const rows = [[head("Measure"), head("Result")]];
  RESULTS.forEach(([k, v], i) => {
    const fill = { color: i % 2 ? C.rowB : C.rowA };
    rows.push([
      { text: k, options: { bold: true, fill, fontSize: 22, valign: "middle" } },
      { text: v, options: { fill, fontSize: 22, valign: "middle" } },
    ]);
  });
  const tw = CW - 2 * PAD;
  s.addTable(rows, { x: X[2] + PAD, y, w: tw, colW: [tw * 0.36, tw * 0.64], rowH: [0.62, ...RESULTS.map(() => 0.9)], fontFace: FONT, color: C.ink, border: { type: "solid", pt: 1, color: "9AA7B8" }, margin: 0.1 });
  y += 0.62 + RESULTS.length * 0.9 + 0.3;
  // honest limit
  const limH = bottom - PAD - y;
  s.addShape("rect", { x: X[2] + PAD, y, w: CW - 2 * PAD, h: limH, fill: { color: "F7EBD3" }, line: { color: C.goldDeep, width: 3 } });
  text(s, [
    { text: "Stated limit  ", options: { bold: true, color: C.fail } },
    { text: "This is coarse inspection, a few millimetres on hand-sized parts. It complements a CMM and does not replace it. A printed 4 mm hole shift sits below the 5.28 mm in-plane floor and reads like the control part." },
  ], { x: X[2] + PAD + 0.25, y: y + 0.1, w: CW - 2 * PAD - 0.5, h: limH - 0.2, fontSize: 25, valign: "middle" });

  // column 4 · grey ────────────────────────────────────────────────────────────
  s.addShape("rect", { x: X[3], y: top, w: CW, h: bottom - top, fill: { color: C.grey }, line: { color: "8E99A8", width: 1.5 } });
  y = sectionHeader(s, "Generative AI in the Loop", X[3] + PAD, top + PAD, CW - 2 * PAD);
  const GENAI = [
    ["FaEye", "VLM capture check", "Gemini Flash-Lite checks the frames and can block a Pass. Live, it flagged a photo of a screen."],
    ["FaRobot", "Capture agent", "An LLM tool-calling loop that coaches the supplier, with a deterministic fallback."],
    ["FaWandMagicSparkles", "Shape completion", "LaS-Comp proposes 8 completions of unseen geometry; where they disagree is where the next photo helps."],
    ["FaKeyboard", "Spec parser", "Turns the buyer's plain-English requirements into an editable tolerance table."],
    ["FaPenNib", "Report narrative", "Version 1 ships a fixed template; the LLM path is kept behind it."],
  ];
  y += 0.25;
  for (const [ic, name, body] of GENAI) {
    s.addShape("ellipse", { x: X[3] + PAD + 0.05, y: y + 0.2, w: 1.05, h: 1.05, fill: { color: C.navy }, line: { color: C.gold, width: 2.5 } });
    s.addImage({ data: await icon(ic, "FFFFFF"), x: X[3] + PAD + 0.32, y: y + 0.47, w: 0.51, h: 0.51 });
    text(s, [
      { text: name + "  ", options: { bold: true, color: C.title } },
      { text: body },
    ], { x: X[3] + PAD + 1.3, y, w: CW - 2 * PAD - 1.35, h: 1.5, fontSize: 24, valign: "middle" });
    y += 1.5;
  }
  text(s, "None of them can pass or fail a part.", { x: X[3] + PAD, y: y + 0.02, w: CW - 2 * PAD, h: 0.55, fontSize: 24, bold: true, italic: true, color: C.fail, align: "center" });
  y += 0.75;
  y = sectionHeader(s, "Summary / Conclusion", X[3] + PAD, y, CW - 2 * PAD);
  para(s, "ProofShape takes a part from phone photos to a metric 3D model in about a minute and checks it against the buyer's CAD. It reports only what it measured: unseen geometry triggers a Rescan, and a tolerance tighter than the measured noise floor is refused up front. The result is a quick, honest first check at the supplier's bench, before the part ships.",
    X[3] + PAD + 0.05, y + 0.2, CW - 2 * PAD - 0.1, 3.45, { fontSize: 25 });
  y += 3.7;
  y = sectionHeader(s, "Key References", X[3] + PAD, y, CW - 2 * PAD);
  const REFS = [
    "J. Wang et al., “VGGT: Visual Geometry Grounded Transformer,” CVPR 2025.",
    "S. Garrido-Jurado et al., “Automatic generation and detection of highly reliable fiducial markers under occlusion,” Pattern Recognition, 2014.",
    "R. B. Rusu et al., “Fast Point Feature Histograms (FPFH) for 3D Registration,” ICRA 2009.",
    "LaS-Comp, CVPR 2026, github.com/DavidYan2001/LaS-Comp.",
  ];
  text(s, REFS.map((r, i) => ({ text: r, options: { breakLine: i < REFS.length - 1, paraSpaceAfter: 6 } })),
    { x: X[3] + PAD + 0.05, y: y + 0.15, w: CW - 2 * PAD - 0.1, h: 3.1, fontSize: 21 });
  y += 3.3;
  y = sectionHeader(s, "Acknowledgement", X[3] + PAD, y, CW - 2 * PAD);
  para(s, "We thank our project advisor, Professor Ahmed Banafa, for the guidance and support throughout this project.",
    X[3] + PAD + 0.05, y + 0.15, CW - 2 * PAD - 0.1, bottom - PAD - y - 0.15, { fontSize: 24, valign: "middle" });

  await pres.writeFile({ fileName: out });
  console.log(`wrote ${out}`);
}

const out = process.argv[2] || path.join(ROOT, "submissions", "expo-poster.pptx");
build(out).catch((e) => {
  console.error(e);
  process.exit(1);
});

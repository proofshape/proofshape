// ProofShape — expo poster (one 48 × 27 in slide, 16:9; scales to 56 × 31.5 or any 16:9 size).
//
//   node scripts/build_expo_poster.js [out.pptx] [--ids <student-ids.json>]
//
// out.pptx defaults to submissions/expo-poster.pptx. --ids is for the print copy only: it adds
// each member's student ID beside their name. The file lives outside git
// (proofshape-data/expo/student-ids.json, {"<name as in TEAM>": "<id>", ...}) and the committed
// poster never carries IDs, because the repository is public (D-013).
//
// Layout and colours follow the SFBU capstone poster template the team was given: beige header
// with the SFBU logo and mascot, a gold band, a navy body, four columns of alternating tan and
// grey panels with light-blue section headers. The 3D figures and their label positions come
// from one real session (scripts/render_expo_figures.py, golden s-04, sess_f82401bb6852; see
// submissions/assets/expo/figure-facts.json). Every number is a recorded result, with its
// source beside it below.

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa6");

const ROOT = path.join(__dirname, "..");
const FIG = path.join(ROOT, "submissions", "assets", "expo");
const FACTS = JSON.parse(fs.readFileSync(path.join(FIG, "figure-facts.json"), "utf8"));

const W = 48;
const H = 27;

// Member order as the team confirmed it (2026-11-22).
const TEAM = ["Mamadou Jallow", "Yashi Dalwala", "Tabeen Raoof"];

// Sampled from the template poster, plus the status badges.
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
  cream: "F7EBD3",
  card: "FBFCFE",
  pass: "15803D",
  fail: "B91C1C",
  rescan: "D97706",
  unver: "64748B",
  blue: "2563EB",
  live: "0F766E",
  progress: "B45309",
  prototype: "6D28D9",
  template: "475569",
};

const FONT = "Arial";

// ── command line ─────────────────────────────────────────────────────────────
function parseArgs(argv) {
  let out = null;
  let ids = null;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--ids") {
      ids = argv[++i];
      if (!ids) throw new Error("--ids needs a file");
    } else if (argv[i].startsWith("--")) {
      throw new Error(`unknown option ${argv[i]}`);
    } else if (out === null) {
      out = argv[i];
    } else {
      throw new Error(`unexpected argument ${argv[i]}`);
    }
  }
  return { out: out || path.join(ROOT, "submissions", "expo-poster.pptx"), ids };
}

function loadIds(file) {
  const data = JSON.parse(fs.readFileSync(file, "utf8"));
  for (const name of TEAM) {
    if (typeof data[name] !== "string" || !/^\d{5,9}$/.test(data[name])) {
      throw new Error(`${file}: no student ID (digits only) for ${name}`);
    }
  }
  return data;
}

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
  text(s, t, { x, y, w, h, fontSize: 28, lineSpacingMultiple: 1.05, ...o });
}

// a label on the 3D scene, with a leader line to the point it names; `bounds` keeps the box
// inside the hero frame
function sceneLabel(s, label, box, anchor, bounds) {
  if (bounds) {
    box = { ...box, x: Math.min(Math.max(box.x, bounds.x), bounds.x + bounds.w - box.w) };
    box.y = Math.min(Math.max(box.y, bounds.y), bounds.y + bounds.h - box.h);
  }
  const [ax, ay] = anchor;
  const ex = Math.min(Math.max(ax, box.x), box.x + box.w);
  const ey = ay < box.y ? box.y : box.y + box.h;
  s.addShape("line", {
    x: Math.min(ax, ex), y: Math.min(ay, ey), w: Math.max(Math.abs(ex - ax), 0.01), h: Math.max(Math.abs(ey - ay), 0.01),
    flipH: ex < ax !== ey < ay, line: { color: C.navy, width: 2.5 },
  });
  s.addShape("ellipse", { x: ax - 0.09, y: ay - 0.09, w: 0.18, h: 0.18, fill: { color: C.navy }, line: { color: C.white, width: 1.5 } });
  s.addShape("roundRect", { ...box, fill: { color: C.white, transparency: 8 }, line: { color: C.navy, width: 2 }, rectRadius: 0.12 });
  text(s, label, { ...box, fontSize: 24, bold: true, color: C.title, align: "center", valign: "middle" });
}

// ── build ───────────────────────────────────────────────────────────────────
async function build(out, ids) {
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
  // SFBU's own site logo (sfbu.edu/themes/custom/sfbu/logo.svg wraps this 4096 × 751 PNG; fetched
  // 2026-11-22). It is the horizontal lockup, so it gets a wider slot than the template's stacked
  // one, on white: its gold lettering is too faint on the beige header.
  const logoW = 9.1;
  const logoH = logoW * (751 / 4096);
  s.addShape("roundRect", { x: 0.5, y: (4.35 - logoH) / 2 - 0.3, w: logoW + 0.5, h: logoH + 0.6, fill: { color: C.white }, line: { color: C.goldDeep, width: 1.5 }, rectRadius: 0.2 });
  s.addImage({ path: path.join(FIG, "sfbu-logo-web.png"), x: 0.75, y: (4.35 - logoH) / 2, w: logoW, h: logoH });
  s.addImage({ path: path.join(FIG, "sfbu-mascot-crop.png"), x: W - 0.6 - 4.15, y: 0.35, w: 4.15, h: 3.6 });
  text(s, [
    { text: "ProofShape: Smartphone 3D Inspection", options: { breakLine: true } },
    { text: "Before Parts Ship", options: { breakLine: true } },
    { text: "Guided capture → metric 3D model → CAD comparison → evidence-aware verdict", options: { fontSize: 30, bold: false, breakLine: true } },
    { text: "Project Advisor: Prof. Ahmed Banafa", options: { fontSize: 36 } },
  ], { x: 10.4, y: 0.2, w: 20.8, h: 4.0, fontSize: 58, bold: true, color: C.title, align: "center", valign: "middle", lineSpacingMultiple: 1.06 });
  const members = TEAM.map((name, i) => ({
    text: ids ? `${name} · ${ids[name]}` : name,
    options: { fontSize: 32, breakLine: i < TEAM.length - 1 },
  }));
  text(s, [
    { text: "CS595 Capstone · Fall 2026", options: { breakLine: true, fontSize: 38 } },
    { text: "Team ProofShape", options: { breakLine: true, fontSize: 34, color: C.goldDeep } },
    ...members,
  ], { x: 31.3, y: 0.3, w: 11.6, h: 3.8, bold: true, color: C.title, align: "center", valign: "middle", lineSpacingMultiple: 1.05 });

  s.addImage({ data: await goldBand(), x: 0, y: HH, w: W, h: 0.5 });

  // columns
  const top = HH + 0.5 + 0.55;
  const bottom = H - 0.55;
  const M = 0.6;
  const G = 0.45;
  const CW = (W - 2 * M - 3 * G) / 4;
  const X = [0, 1, 2, 3].map((i) => M + i * (CW + G));
  const PAD = 0.3;
  const IW = CW - 2 * PAD; // inner width

  // column 1 · tan: the problem, how it works, the four verdicts, the hard rule ─────────
  s.addShape("rect", { x: X[0], y: top, w: CW, h: bottom - top, fill: { color: C.tan }, line: { color: "8C6A3A", width: 1.5 } });
  let x = X[0] + PAD;
  let y = sectionHeader(s, "Introduction", x, top + PAD, IW);
  para(s, "A part shipped out of spec costs a second shipment and days. Photos from a supplier measure nothing, and a coordinate-measuring machine (CMM) means sending the part to a lab. ProofShape turns the supplier's phone and a printed board into a coarse check before the part ships.",
    x + 0.05, y + 0.15, IW - 0.1, 2.75);
  y += 3.0;
  s.addShape("rect", { x, y, w: IW, h: 1.75, fill: { color: C.cream }, line: { color: C.goldDeep, width: 2.5 } });
  text(s, [
    { text: "Our question  ", options: { bold: true, color: C.goldDeep } },
    { text: "Can a smartphone give a useful dimensional check before a part ships, without pretending unseen geometry was measured?", options: { italic: true, color: C.title } },
  ], { x: x + 0.25, y: y + 0.08, w: IW - 0.5, h: 1.6, fontSize: 28, bold: true, valign: "middle" });
  y += 1.75 + 0.3;
  y = sectionHeader(s, "How It Works", x, y, IW);
  const steps = [
    ["FaFileArrowUp", "Buyer", "uploads the CAD and states the tolerances in plain English."],
    ["FaMobileScreenButton", "Supplier", "sets the part on a printed board and follows the phone's guidance around it."],
    ["FaCubes", "Service", "builds a 3D model in millimetres and compares it with the CAD."],
    ["FaClipboardCheck", "Report", "shows where the part differs from the CAD, and the verdict."],
  ];
  y += 0.15;
  for (const [ic, who, what] of steps) {
    s.addShape("ellipse", { x: x + 0.05, y: y + 0.1, w: 1.05, h: 1.05, fill: { color: C.navy }, line: { color: C.gold, width: 3 } });
    s.addImage({ data: await icon(ic, "FFFFFF"), x: x + 0.32, y: y + 0.37, w: 0.51, h: 0.51 });
    text(s, [
      { text: who + " ", options: { bold: true, color: "0D1B2A" } },
      { text: what },
    ], { x: x + 1.35, y, w: IW - 1.4, h: 1.25, fontSize: 27, valign: "middle" });
    y += 1.25;
  }
  y += 0.2;
  y = sectionHeader(s, "Four Honest Verdicts", x, y, IW);
  // I-07's verdict table (inspect/inspection/verdict.py), I-09's intake check and A-02's veto
  const verdicts = [
    ["PASS", C.pass, "every required feature was seen clearly and measures within tolerance."],
    ["FAIL", C.fail, "a measured deviation is beyond tolerance, by more than its uncertainty."],
    ["RESCAN", C.rescan, "too little seen, an ambiguous pose, or the capture check found a problem."],
    ["UNVERIFIABLE", C.unver, "tighter than the method can check, so refused when the order is placed."],
  ];
  y += 0.15;
  for (const [label, col, why] of verdicts) {
    s.addShape("roundRect", { x: x + 0.05, y: y + 0.17, w: 3.25, h: 0.75, fill: { color: col }, line: { color: C.white, width: 2 }, rectRadius: 0.2 });
    text(s, label, { x: x + 0.05, y: y + 0.17, w: 3.25, h: 0.75, fontSize: 24, bold: true, color: C.white, align: "center", valign: "middle" });
    text(s, why, { x: x + 3.5, y, w: IW - 3.55, h: 1.1, fontSize: 25, valign: "middle" });
    y += 1.1;
  }
  // the hard rule
  const ry = bottom - PAD - 2.45;
  s.addShape("rect", { x, y: ry, w: IW, h: 2.45, fill: { color: C.navy }, line: { color: C.gold, width: 4 } });
  s.addImage({ data: await icon("FaShieldHalved", C.gold), x: x + 0.35, y: ry + 0.28, w: 0.75, h: 0.75 });
  text(s, "THE HARD RULE", { x: x + 1.3, y: ry + 0.28, w: 8, h: 0.75, fontSize: 30, bold: true, color: C.gold, valign: "middle", charSpacing: 3 });
  text(s, "Geometry the camera never saw can never pass or fail a part. It can only ask for a Rescan.",
    { x: x + 0.35, y: ry + 1.05, w: IW - 0.7, h: 1.3, fontSize: 29, bold: true, color: C.white, valign: "middle" });

  // hero · spans columns 2–3 ─────────────────────────────────────────────────
  const hx = X[1];
  const hw = 2 * CW + G;
  const hh = 9.6;
  const barH = 1.0;
  s.addImage({ data: await heroBackdrop(), x: hx, y: top, w: hw, h: hh });
  s.addShape("rect", { x: hx, y: top, w: hw, h: hh, fill: { type: "none" }, line: { color: C.gold, width: 4 } });
  s.addShape("rect", { x: hx, y: top, w: hw, h: barH, fill: { color: C.navy }, line: { color: C.navy } });
  text(s, [
    { text: "30 phone photos  ", options: { color: C.white } },
    { text: "→", options: { color: C.gold } },
    { text: "  camera positions from the printed board  ", options: { color: C.white } },
    { text: "→", options: { color: C.gold } },
    { text: "  a measured 3D model", options: { color: C.white } },
  ], { x: hx + 0.3, y: top, w: hw - 0.6, h: barH, fontSize: 34, bold: true, align: "center", valign: "middle" });

  // the scene, on the left
  const [scPxW, scPxH] = FACTS.scene.px;
  const sh = hh - barH - 0.4;
  const sw = sh * (scPxW / scPxH);
  const sx = hx + 0.25;
  const sy = top + barH + 0.2;
  s.addImage({ path: path.join(FIG, "hero-scene.png"), x: sx, y: sy, w: sw, h: sh });
  const at = (k) => [sx + FACTS.scene.anchors[k][0] * sw, sy + FACTS.scene.anchors[k][1] * sh];
  const sceneBounds = { x: hx + 0.15, y: top + barH + 0.1, w: sw + 0.2, h: hh - barH - 0.2 };
  const [cax, cay] = at("camera");
  sceneLabel(s, "Camera poses (29 photos)", { x: cax + 0.55, y: cay + 0.25, w: 4.6, h: 0.75 }, [cax, cay], sceneBounds);
  const [bax, bay] = at("board");
  sceneLabel(s, "Printed ChArUco board", { x: bax - 0.6, y: bay - 0.95, w: 4.3, h: 0.75 }, [bax, bay], sceneBounds);

  // the close-up, on the right, with its legend
  const [cuPxW, cuPxH] = FACTS.closeup.px;
  const rx = sx + sw + 0.55; // right region
  const rw = hx + hw - 0.3 - rx;
  const ch = 4.7;
  const cw = ch * (cuPxW / cuPxH);
  const cx = rx + (rw - cw) / 2;
  const cy = sy + 0.15;
  const [pax, pay] = at("part");
  s.addShape("line", { x: pax + 0.25, y: cy + ch, w: cx - pax - 0.25, h: pay - (cy + ch), flipV: true, line: { color: C.navy, width: 2.5, dashType: "dash" } });
  s.addShape("line", { x: pax + 0.25, y: cy + 0.05, w: cx - pax - 0.25, h: pay - cy, flipV: true, line: { color: C.navy, width: 2.5, dashType: "dash" } });
  s.addShape("ellipse", { x: pax - 1.1, y: pay - 0.55, w: 2.2, h: 1.45, fill: { type: "none" }, line: { color: C.gold, width: 4 } });
  sceneLabel(s, "Reconstructed part", { x: pax - 4.6, y: pay - 2.2, w: 3.7, h: 0.75 }, [pax - 0.75, pay - 0.35], sceneBounds);
  s.addShape("roundRect", { x: cx - 0.15, y: cy - 0.15, w: cw + 0.3, h: ch + 0.3, fill: { color: C.white, transparency: 10 }, line: { color: C.navy, width: 3 }, rectRadius: 0.25 });
  s.addImage({ path: path.join(FIG, "heatmap-closeup.png"), x: cx, y: cy, w: cw, h: ch });

  let ly = cy + ch + 0.3;
  text(s, "Distance from the CAD, on surface the camera saw", { x: rx, y: ly, w: rw, h: 0.5, fontSize: 24, bold: true, color: C.title, align: "center" });
  ly += 0.55;
  const bx = rx + 1.0;
  const bw = rw - 2.0;
  s.addImage({ path: path.join(FIG, "heatmap-bar.png"), x: bx, y: ly, w: bw, h: 0.4 });
  [["−4", 0], ["−2", 0.25], ["0", 0.5], ["+2", 0.75], ["+4 mm", 1]].forEach(([t, f]) => {
    text(s, t, { x: bx + f * bw - 0.8, y: ly + 0.42, w: 1.6, h: 0.42, fontSize: 22, bold: true, color: C.title, align: "center" });
  });
  ly += 0.95;
  const legend = [
    [C.blue, "material missing"],
    [C.fail, "extra material"],
    [C.unver, "Unobserved or not measured"],
  ];
  const legendW = legend.reduce((sum, [, lab]) => sum + 0.5 + lab.length * 0.155 + 0.35, 0);
  let lx = rx + (rw - legendW) / 2;
  for (const [col, lab] of legend) {
    s.addShape("rect", { x: lx, y: ly + 0.1, w: 0.38, h: 0.38, fill: { color: col }, line: { color: "334155", width: 1 } });
    const lw = lab.length * 0.155;
    text(s, lab, { x: lx + 0.5, y: ly, w: lw + 0.2, h: 0.58, fontSize: 22, valign: "middle" });
    lx += 0.5 + lw + 0.35;
  }
  ly += 0.62;
  text(s, "Colours do not mean pass or fail. ±4 mm is the colour range, not a tolerance.",
    { x: rx, y: ly, w: rw, h: 0.45, fontSize: 22, color: C.ink, align: "center" });
  ly += 0.5;
  const observedPct = Math.round(FACTS.observed_area_fraction * 100);
  text(s, `Real output: a 3D-printed test part, 30 iPhone photos, ${observedPct}% of its surface area observed.`,
    { x: rx, y: ly, w: rw, h: 0.45, fontSize: 22, italic: true, color: "334155", align: "center" });

  // column 2 · grey, below the hero: methodology ───────────────────────────────
  const lowTop = top + hh + 0.45;
  s.addShape("rect", { x: X[1], y: lowTop, w: CW, h: bottom - lowTop, fill: { color: C.grey }, line: { color: "8E99A8", width: 1.5 } });
  x = X[1] + PAD;
  y = sectionHeader(s, "Methodology", x, lowTop + PAD, IW);
  const stages = [
    ["FaCamera", "Capture and quality gate", "The phone guides the orbit and refuses blurry, dark or repeated frames. Gemini checks sampled frames."],
    ["FaCube", "Metric 3D reconstruction", "VGGT predicts depth and camera poses. The ChArUco board fixes millimetre scale. Depth is fused into one mesh."],
    ["FaEye", "Evidence and uncertainty", "Every vertex is labelled Observed or not, with its own uncertainty. Unseen surface can't count."],
    ["FaRulerCombined", "CAD inspection", "The CAD is registered to the model. Deviation on Observed surface goes through calibrated verdict rules."],
  ];
  y += 0.25;
  const stH = 1.95;
  const arrowH = 0.33;
  for (let i = 0; i < stages.length; i++) {
    const [ic, name, body] = stages[i];
    s.addShape("roundRect", { x, y, w: IW, h: stH, fill: { color: C.white }, line: { color: C.navy, width: 2 }, rectRadius: 0.18, shadow: { type: "outer", color: "000000", opacity: 0.25, blur: 6, offset: 3, angle: 90 } });
    s.addShape("rect", { x, y, w: 1.3, h: stH, fill: { color: C.navy }, line: { color: C.navy } });
    s.addImage({ data: await icon(ic, "FFFFFF"), x: x + 0.33, y: y + 0.3, w: 0.64, h: 0.64 });
    text(s, String(i + 1), { x, y: y + 1.12, w: 1.3, h: 0.6, fontSize: 26, bold: true, color: C.gold, align: "center" });
    text(s, [
      { text: name, options: { bold: true, color: C.title, fontSize: 28, breakLine: true } },
      { text: body, options: { fontSize: 24 } },
    ], { x: x + 1.5, y: y + 0.08, w: IW - 1.65, h: stH - 0.16, valign: "middle" });
    y += stH;
    if (i < stages.length - 1) {
      s.addShape("downArrow", { x: x + IW / 2 - 0.28, y: y + 0.02, w: 0.56, h: arrowH - 0.04, fill: { color: C.goldDeep }, line: { color: C.goldDeep } });
      y += arrowH;
    }
  }

  // column 3 · tan, below the hero: results ───────────────────────────────────
  s.addShape("rect", { x: X[2], y: lowTop, w: CW, h: bottom - lowTop, fill: { color: C.tan }, line: { color: "8C6A3A", width: 1.5 } });
  x = X[2] + PAD;
  y = sectionHeader(s, "Analysis & Results", x, lowTop + PAD, IW);
  // RESULTS — sources: progress log 2026-11-16 (R-36's three phone sessions); D-073 (n = 5) and
  // R-16 (s-04, s-05 heights); D-078 (n = 11); #407 and G-09 (the 4 mm hole shift, re-shot in #421).
  const RESULTS = [
    ["31–51 s", 54, "from Finish to a measured 3D model, on 3 iPhone sessions served from a laptop"],
    ["0.86 mm", 54, "p95 height error against calipers, from 5 comparisons; two test parts read −0.52 and −0.21 mm"],
    ["5.28 mm", 54, "p95 error in measured edge lengths, against calipers, from 11 comparisons"],
    ["4 mm shift", 40, "a printed hole moved 4 mm, and its control, both came back Rescan: not separable yet, and the system says so"],
  ];
  y += 0.25;
  const rH = 1.82;
  for (const [big, size, caption] of RESULTS) {
    s.addShape("roundRect", { x, y, w: IW, h: rH, fill: { color: C.card }, line: { color: C.navy, width: 2 }, rectRadius: 0.15 });
    text(s, big, { x: x + 0.15, y, w: 3.55, h: rH, fontSize: size, bold: true, color: C.navy, align: "center", valign: "middle" });
    s.addShape("line", { x: x + 3.8, y: y + 0.3, w: 0, h: rH - 0.6, line: { color: C.gold, width: 3 } });
    text(s, caption, { x: x + 4.0, y, w: IW - 4.2, h: rH, fontSize: 24, valign: "middle" });
    y += rH + 0.16;
  }
  // stated limit
  y += 0.05;
  const limH = bottom - PAD - y;
  s.addShape("rect", { x, y, w: IW, h: limH, fill: { color: C.cream }, line: { color: C.goldDeep, width: 3 } });
  text(s, [
    { text: "Stated limit  ", options: { bold: true, color: C.fail } },
    { text: "Coarse inspection: a few millimetres on hand-sized parts. It complements a CMM and does not replace one." },
  ], { x: x + 0.25, y: y + 0.05, w: IW - 0.5, h: limH - 0.1, fontSize: 25, valign: "middle" });

  // column 4 · grey: generative AI, summary, references ──────────────────────────
  s.addShape("rect", { x: X[3], y: top, w: CW, h: bottom - top, fill: { color: C.grey }, line: { color: "8E99A8", width: 1.5 } });
  x = X[3] + PAD;
  y = sectionHeader(s, "Generative AI in the Loop", x, top + PAD, IW);
  // Status on main at 2026-11-22, rechecked at print (2026-11-23): A-01/A-02/A-14 (D-084), A-07/A-13
  // with A-15 in draft #431, A-04/A-05/A-12 (unavailable without a CUDA runner), A-03 with C-16,
  // A-09 (D-082).
  const GENAI = [
    ["FaEye", "VLM capture check", "LIVE", C.live, "Gemini checks sampled frames for a wrong part, hands in the way, a photo of a screen or poor light. A failed or unchecked frame turns a Pass into Rescan."],
    ["FaRobot", "Capture agent", "IN PROGRESS", C.progress, "An LLM tool-calling loop that coaches the supplier, with a deterministic fallback. It is built and tested; serving it to a live session is in review."],
    ["FaWandMagicSparkles", "Shape completion", "PROTOTYPE · CUDA", C.prototype, "LaS-Comp samples 8 completions of unseen geometry; where they disagree, another photo helps. The demo laptop has no CUDA GPU, so it skips this."],
    ["FaKeyboard", "Spec parser", "LIVE · GEMINI KEY", C.live, "Turns the buyer's plain-English requirements into a tolerance table the buyer edits and confirms."],
    ["FaPenNib", "Report narrative", "TEMPLATE", C.template, "Version 1 writes the summary from a fixed template. The LLM path is built but not served."],
  ];
  y += 0.15;
  const gH = 1.9;
  for (const [ic, name, badge, col, body] of GENAI) {
    s.addShape("ellipse", { x: x + 0.05, y: y + 0.12, w: 1.0, h: 1.0, fill: { color: C.navy }, line: { color: C.gold, width: 2.5 } });
    s.addImage({ data: await icon(ic, "FFFFFF"), x: x + 0.3, y: y + 0.37, w: 0.5, h: 0.5 });
    text(s, name, { x: x + 1.25, y: y + 0.05, w: 5.2, h: 0.6, fontSize: 27, bold: true, color: C.title, valign: "middle" });
    const badgeW = badge.length * 0.19 + 0.55;
    s.addShape("roundRect", { x: x + IW - badgeW, y: y + 0.1, w: badgeW, h: 0.52, fill: { color: col }, line: { color: col }, rectRadius: 0.26 });
    text(s, badge, { x: x + IW - badgeW, y: y + 0.1, w: badgeW, h: 0.52, fontSize: 20, bold: true, color: C.white, align: "center", valign: "middle" });
    text(s, body, { x: x + 1.25, y: y + 0.66, w: IW - 1.3, h: gH - 0.72, fontSize: 24 });
    y += gH;
  }
  s.addShape("rect", { x, y: y + 0.05, w: IW, h: 1.2, fill: { color: C.navy }, line: { color: C.gold, width: 2.5 } });
  text(s, "AI may coach the capture or veto confidence. Measured geometry decides Pass or Fail.",
    { x: x + 0.3, y: y + 0.05, w: IW - 0.6, h: 1.2, fontSize: 26, bold: true, color: C.white, align: "center", valign: "middle" });
  y += 1.25 + 0.25;
  y = sectionHeader(s, "Summary", x, y, IW);
  para(s, "A smartphone can do a fast, coarse 3D inspection before a part ships, and ProofShape refuses to claim what it didn't measure.",
    x + 0.05, y + 0.12, IW - 0.1, 1.6, { fontSize: 29, bold: true, color: C.title });
  y += 1.85;
  y = sectionHeader(s, "Key References", x, y, IW);
  const REFS = [
    "J. Wang et al., “VGGT: Visual Geometry Grounded Transformer,” CVPR 2025.",
    "S. Garrido-Jurado et al., “Automatic generation and detection of highly reliable fiducial markers under occlusion,” Pattern Recognition, 2014.",
    "B. Curless and M. Levoy, “A volumetric method for building complex models from range images,” SIGGRAPH 1996.",
    "R. B. Rusu et al., “Fast Point Feature Histograms (FPFH) for 3D Registration,” ICRA 2009.",
    "LaS-Comp, CVPR 2026, github.com/DavidYan2001/LaS-Comp.",
  ];
  text(s, REFS.map((r, i) => ({ text: r, options: { breakLine: i < REFS.length - 1, paraSpaceAfter: 4 } })),
    { x: x + 0.05, y: y + 0.1, w: IW - 0.1, h: 2.45, fontSize: 19 });
  y += 2.6;
  y = sectionHeader(s, "Acknowledgement", x, y, IW);
  para(s, "We thank our project advisor, Prof. Ahmed Banafa, for the guidance and support throughout this project.",
    x + 0.05, y + 0.1, IW - 0.1, bottom - PAD - y - 0.1, { fontSize: 24, valign: "middle" });

  await pres.writeFile({ fileName: out });
  console.log(`wrote ${out}`);
}

const args = parseArgs(process.argv.slice(2));
build(args.out, args.ids ? loadIds(args.ids) : null).catch((e) => {
  console.error(e);
  process.exit(1);
});

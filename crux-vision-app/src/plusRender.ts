// plusRender.ts
// Canvas port of reconstruct_plus.py: a side-by-side analysis view.
//   LEFT panel  — a synthetic body schematic (dark background) with hand/foot
//                 weight circles colour-coded green→red by vertical load,
//                 limb segments colour-coded by axial force, and (when CoM
//                 motion data is available) velocity/acceleration gauges +
//                 a velocity direction arrow.
//   RIGHT panel — the real video frame with the skeleton, force-coloured
//                 limb segments, knee/elbow joint-angle dots, CoM, and small
//                 weight circles at the active hold positions.
//
// Reuses render.ts's drawPose/drawCom (via a canvas translate) for the base
// skeleton + CoM rather than duplicating that drawing code.
//
// Font glyphs and the velocity arrowhead are a close visual match to cv2's
// Hershey-font + arrowedLine rendering, not pixel-identical (no browser
// equivalent for either) — same caveat as render.ts.

import { LANDMARK_NAMES } from "./types";
import type { FrameEntry } from "./types";
import type { AnalysisResult, Hold, HoldKind, Side } from "./analysis";
import { drawCom, drawPose } from "./render";
import { EXTREMITY_NAMES, JOINT_DEFS, LIMB_SEGMENTS } from "./forces";
import type { ExtremityName, SegmentForce, WeightResult } from "./forces";
import type { MotionFrameEntry } from "./motion";

const VISIBILITY_THRESHOLD = 0.5;
const MAX_WEIGHT_PCT = 50.0;
const MAX_SEGMENT_PCT = 80.0;
const MAX_SPEED_DISPLAY = 300.0;
const MAX_ACCEL_DISPLAY = 1000.0;
const MAX_ARROW_PX = 110;
const SMOOTHING_ALPHA = 0.5;

const LM_IDX: Record<string, number> = {};
LANDMARK_NAMES.forEach((name, i) => (LM_IDX[name] = i));

// ── Colour helpers (OpenCV BGR HSV ramp -> canvas RGB) ─────────────────────

function hsvToRgb(hueDeg: number, s: number, v: number): [number, number, number] {
  const h = ((hueDeg % 360) + 360) % 360;
  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;
  let r = 0,
    g = 0,
    b = 0;
  if (h < 60) {
    r = c;
    g = x;
  } else if (h < 120) {
    r = x;
    g = c;
  } else if (h < 180) {
    g = c;
    b = x;
  } else if (h < 240) {
    g = x;
    b = c;
  } else if (h < 300) {
    r = x;
    b = c;
  } else {
    r = c;
    b = x;
  }
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

/** ratio 0->1 maps green->yellow->red, matching reconstruct_plus.py's _hsv_colour. */
function hsvColour(ratio: number): string {
  const clamped = Math.min(1, Math.max(0, ratio));
  const hueCv = 60 * (1 - clamped); // OpenCV-scale hue (0-180): 60=green, 0=red
  const [r, g, b] = hsvToRgb(hueCv * 2, 240 / 255, 230 / 255);
  return `rgb(${r}, ${g}, ${b})`;
}
function weightColour(pct: number): string {
  return hsvColour(pct / MAX_WEIGHT_PCT);
}
function segmentColour(pct: number): string {
  return hsvColour(pct / MAX_SEGMENT_PCT);
}
function jointAngleColour(angleDeg: number): string {
  const ratio = (180.0 - Math.min(180, Math.max(0, angleDeg))) / 90.0;
  return hsvColour(ratio);
}

// ── Smoothed weight-by-frame (EMA, ported from smooth_weight_data) ─────────

export interface SmoothedExtremity {
  active: true;
  axial_kg: number | null;
  axial_pct: number | null;
  vertical_kg: number;
  vertical_pct: number;
}
export interface SmoothedWeightFrame {
  extremities: Partial<Record<ExtremityName, SmoothedExtremity>>;
  limb_segments: Record<string, SegmentForce | null>;
  joint_angles: Record<string, number | null>;
}

function ema(prev: number | undefined, next: number, alpha: number): number {
  return prev !== undefined ? alpha * next + (1.0 - alpha) * prev : next;
}

/** Builds the per-frame lookup of EMA-smoothed display values, matching build_weight_by_frame + smooth_weight_data. */
export function buildSmoothedWeightByFrame(weight: WeightResult): Map<number, SmoothedWeightFrame> {
  const emaExt = new Map<ExtremityName, { vertical_pct: number; vertical_kg: number }>();
  const emaSeg = new Map<string, number>();
  const emaJoint = new Map<string, number>();
  const out = new Map<number, SmoothedWeightFrame>();

  const sorted = [...weight.per_frame].sort((a, b) => a.frame - b.frame);
  for (const rec of sorted) {
    const extremities: Partial<Record<ExtremityName, SmoothedExtremity>> = {};
    for (const name of EXTREMITY_NAMES) {
      const info = rec[name];
      if (!info.active) continue;
      const prev = emaExt.get(name);
      const sPct = ema(prev?.vertical_pct, info.vertical_pct as number, SMOOTHING_ALPHA);
      const sKg = ema(prev?.vertical_kg, info.vertical_kg as number, SMOOTHING_ALPHA);
      emaExt.set(name, { vertical_pct: sPct, vertical_kg: sKg });
      extremities[name] = { active: true, axial_kg: info.axial_kg, axial_pct: info.axial_pct, vertical_kg: sKg, vertical_pct: sPct };
    }

    const limbSegments: Record<string, SegmentForce | null> = {};
    for (const [segName, seg] of Object.entries(rec.limb_segments)) {
      if (seg === null) {
        limbSegments[segName] = null;
        continue;
      }
      const prevPct = emaSeg.get(segName);
      const sPct = ema(prevPct, seg.axial_pct, SMOOTHING_ALPHA);
      const bw = seg.axial_pct !== 0 ? seg.axial_kg / (seg.axial_pct / 100) : 70.0;
      emaSeg.set(segName, sPct);
      limbSegments[segName] = { axial_pct: sPct, axial_kg: (sPct / 100) * bw, angle_deg: seg.angle_deg };
    }

    const jointAngles: Record<string, number | null> = {};
    for (const [jName, angle] of Object.entries(rec.joint_angles)) {
      if (angle === null) {
        jointAngles[jName] = null;
        continue;
      }
      const sAngle = ema(emaJoint.get(jName), angle, SMOOTHING_ALPHA);
      emaJoint.set(jName, sAngle);
      jointAngles[jName] = sAngle;
    }

    out.set(rec.frame, { extremities, limb_segments: limbSegments, joint_angles: jointAngles });
  }

  return out;
}

// ── Right-panel drawing (real video coordinates) ────────────────────────────

function getCoords(entry: FrameEntry, width: number, height: number): Map<number, [number, number]> {
  const coords = new Map<number, [number, number]>();
  for (const lm of entry.landmarks) {
    if (lm.visibility >= VISIBILITY_THRESHOLD) {
      const idx = LM_IDX[lm.name];
      if (idx !== undefined) coords.set(idx, [lm.x * width, lm.y * height]);
    }
  }
  return coords;
}

function drawSegmentForces(
  ctx: CanvasRenderingContext2D,
  entry: FrameEntry,
  segData: Record<string, SegmentForce | null>,
  width: number,
  height: number
) {
  const coords = getCoords(entry, width, height);
  ctx.lineCap = "round";
  for (const seg of LIMB_SEGMENTS) {
    const d = coords.get(LM_IDX[seg.distal]);
    const p = coords.get(LM_IDX[seg.proximal]);
    if (!d || !p) continue;
    const info = segData[seg.name];
    const colour = info ? segmentColour(info.axial_pct) : "rgb(120, 120, 120)";
    ctx.lineWidth = 12;
    ctx.strokeStyle = colour;
    ctx.beginPath();
    ctx.moveTo(d[0], d[1]);
    ctx.lineTo(p[0], p[1]);
    ctx.stroke();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(d[0], d[1]);
    ctx.lineTo(p[0], p[1]);
    ctx.stroke();
  }
}

function drawJointAngles(
  ctx: CanvasRenderingContext2D,
  entry: FrameEntry,
  jointData: Record<string, number | null>,
  width: number,
  height: number
) {
  const coords = getCoords(entry, width, height);
  for (const j of JOINT_DEFS) {
    const pt = coords.get(LM_IDX[j.joint]);
    if (!pt) continue;
    const angle = jointData[j.name];
    const colour = angle != null ? jointAngleColour(angle) : "rgb(120, 120, 120)";
    ctx.beginPath();
    ctx.arc(pt[0], pt[1], 19, 0, 2 * Math.PI);
    ctx.fillStyle = colour;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#000000";
    ctx.stroke();
  }
}

function drawPastHoldRight(ctx: CanvasRenderingContext2D, px: number, py: number) {
  ctx.beginPath();
  ctx.arc(px, py, 14, 0, 2 * Math.PI);
  ctx.fillStyle = "#383838";
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#000000";
  ctx.stroke();
}

function drawActiveHoldSmall(ctx: CanvasRenderingContext2D, px: number, py: number, label: string, verticalPct: number) {
  const colour = weightColour(verticalPct);
  const R = 20;
  ctx.beginPath();
  ctx.arc(px, py, R, 0, 2 * Math.PI);
  ctx.fillStyle = colour;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#000000";
  ctx.stroke();

  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.font = "bold 18px sans-serif";
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#000000";
  ctx.strokeText(label, px, py - R - 8);
  ctx.fillStyle = colour;
  ctx.fillText(label, px, py - R - 8);
  ctx.textAlign = "left";
}

function drawPanelLabel(ctx: CanvasRenderingContext2D, text: string, x: number, y: number) {
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.font = "bold 34px sans-serif";
  ctx.lineWidth = 5;
  ctx.strokeStyle = "#000000";
  ctx.strokeText(text, x, y);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(text, x, y);
}

const HOLD_TO_EXT: Record<string, ExtremityName> = {
  "hand:left": "left_hand",
  "hand:right": "right_hand",
  "foot:left": "left_foot",
  "foot:right": "right_foot",
};
const HOLD_LABELS: Record<string, string> = {
  "hand:left": "LH",
  "hand:right": "RH",
  "foot:left": "LF",
  "foot:right": "RF",
};
function holdKey(kind: HoldKind, side: Side): string {
  return `${kind}:${side}`;
}

// ── Left-panel drawing (synthetic schematic, independent of real pose) ─────

const EXT_MAP: Record<ExtremityName, { joint: string; label: string }> = {
  left_hand: { joint: "l_wri", label: "LH" },
  right_hand: { joint: "r_wri", label: "RH" },
  left_foot: { joint: "l_ank", label: "LF" },
  right_foot: { joint: "r_ank", label: "RF" },
};
const SCHEMATIC_SEGS: [string, string, string][] = [
  ["left_upper_arm", "l_sho", "l_elb"],
  ["left_forearm", "l_elb", "l_wri"],
  ["right_upper_arm", "r_sho", "r_elb"],
  ["right_forearm", "r_elb", "r_wri"],
  ["left_upper_leg", "l_hip", "l_kne"],
  ["left_lower_leg", "l_kne", "l_ank"],
  ["right_upper_leg", "r_hip", "r_kne"],
  ["right_lower_leg", "r_kne", "r_ank"],
];

function drawArrow(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, colour: string, lineWidth: number) {
  ctx.strokeStyle = colour;
  ctx.fillStyle = colour;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  const angle = Math.atan2(y1 - y0, x1 - x0);
  const headLen = Math.hypot(x1 - x0, y1 - y0) * 0.38;
  const spread = Math.PI / 7;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - headLen * Math.cos(angle - spread), y1 - headLen * Math.sin(angle - spread));
  ctx.lineTo(x1 - headLen * Math.cos(angle + spread), y1 - headLen * Math.sin(angle + spread));
  ctx.closePath();
  ctx.fill();
}

function drawWeightSchematic(
  ctx: CanvasRenderingContext2D,
  extremities: Partial<Record<ExtremityName, SmoothedExtremity>>,
  limbSegs: Record<string, SegmentForce | null>,
  width: number,
  height: number,
  motion: MotionFrameEntry | null,
  xOffset: number
) {
  ctx.fillStyle = "#1e1919";
  ctx.fillRect(xOffset, 0, width, height);

  const cx = xOffset + width / 2;
  const px = (frac: number) => cx + frac * width;
  const py = (frac: number) => frac * height;

  const j: Record<string, [number, number]> = {
    head: [px(0), py(0.06)],
    l_sho: [px(-0.16), py(0.17)],
    r_sho: [px(0.16), py(0.17)],
    l_elb: [px(-0.26), py(0.29)],
    r_elb: [px(0.26), py(0.29)],
    l_wri: [px(-0.31), py(0.41)],
    r_wri: [px(0.31), py(0.41)],
    l_hip: [px(-0.09), py(0.49)],
    r_hip: [px(0.09), py(0.49)],
    l_kne: [px(-0.11), py(0.65)],
    r_kne: [px(0.11), py(0.65)],
    l_ank: [px(-0.12), py(0.83)],
    r_ank: [px(0.12), py(0.83)],
  };

  const midSho: [number, number] = [(j.l_sho[0] + j.r_sho[0]) / 2, (j.l_sho[1] + j.r_sho[1]) / 2];
  const midHip: [number, number] = [(j.l_hip[0] + j.r_hip[0]) / 2, (j.l_hip[1] + j.r_hip[1]) / 2];
  const GREY = "#555050";

  function line(a: [number, number], b: [number, number], colour: string, w: number) {
    ctx.lineWidth = w;
    ctx.strokeStyle = colour;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }

  line(j.head, midSho, GREY, 8);
  line(midSho, midHip, GREY, 14);
  line(j.l_sho, j.r_sho, GREY, 10);
  line(j.l_hip, j.r_hip, GREY, 10);
  ctx.beginPath();
  ctx.arc(j.head[0], j.head[1], 32, 0, 2 * Math.PI);
  ctx.fillStyle = GREY;
  ctx.fill();

  for (const [segKey, jFrom, jTo] of SCHEMATIC_SEGS) {
    const seg = limbSegs[segKey];
    const colour = seg ? segmentColour(seg.axial_pct) : "#37373c";
    line(j[jFrom], j[jTo], colour, 22);
    line(j[jFrom], j[jTo], "#ffffff", 2);
  }

  for (const jn of ["l_elb", "r_elb", "l_kne", "r_kne"]) {
    ctx.beginPath();
    ctx.arc(j[jn][0], j[jn][1], 12, 0, 2 * Math.PI);
    ctx.fillStyle = "#a5a0a0";
    ctx.fill();
  }

  const R = 75;
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  for (const extName of EXTREMITY_NAMES) {
    const { joint: jname, label } = EXT_MAP[extName];
    const pt = j[jname];
    const info = extremities[extName];
    if (info) {
      const colour = weightColour(info.vertical_pct);
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], R, 0, 2 * Math.PI);
      ctx.fillStyle = colour;
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#000000";
      ctx.stroke();

      const kgText = `${info.vertical_kg.toFixed(1)}kg`;
      const pctText = `${Math.round(info.vertical_pct)}%`;
      ctx.font = "bold 26px sans-serif";
      ctx.lineWidth = 5;
      ctx.strokeStyle = "#000000";
      ctx.strokeText(kgText, pt[0], pt[1] - 14);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(kgText, pt[0], pt[1] - 14);

      ctx.font = "bold 22px sans-serif";
      ctx.lineWidth = 5;
      ctx.strokeStyle = "#000000";
      ctx.strokeText(pctText, pt[0], pt[1] + 24);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(pctText, pt[0], pt[1] + 24);
    } else {
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], R, 0, 2 * Math.PI);
      ctx.fillStyle = "#322d2d";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#5f5a5a";
      ctx.stroke();
      ctx.font = "bold 28px sans-serif";
      ctx.fillStyle = "#5f5a5a";
      ctx.fillText(label, pt[0], pt[1] + 10);
    }
  }

  if (motion) {
    const vx = motion.vx_px_s ?? 0;
    const vyUp = motion.vy_px_s ?? 0;
    const speed = motion.speed_px_s ?? 0;
    const accel = motion.accel_px_s2 ?? 0;

    const BAR_W = 38;
    const MARGIN = 14;
    const BAR_TOP = height * 0.04;
    const BAR_BOTTOM = height * 0.96;
    const barH = BAR_BOTTOM - BAR_TOP;
    const centerY = (BAR_TOP + BAR_BOTTOM) / 2;
    const DARK = "#2a2626",
      BORDER = "#645f5f",
      TEXT_COL = "#a09b9b";

    const bx0 = xOffset + MARGIN,
      bx1 = xOffset + MARGIN + BAR_W;
    ctx.fillStyle = DARK;
    ctx.fillRect(bx0, BAR_TOP, bx1 - bx0, barH);

    const ratioV = Math.min(1, Math.max(0, Math.abs(vyUp) / MAX_SPEED_DISPLAY));
    const fillV = (ratioV * barH) / 2;
    if (vyUp >= 0) {
      ctx.fillStyle = "#3cc83c";
      ctx.fillRect(bx0, centerY - fillV, bx1 - bx0, fillV);
    } else {
      ctx.fillStyle = "#dc3c3c";
      ctx.fillRect(bx0, centerY, bx1 - bx0, fillV);
    }
    ctx.strokeStyle = "#c8c8c8";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(bx0, centerY);
    ctx.lineTo(bx1, centerY);
    ctx.stroke();
    ctx.strokeStyle = BORDER;
    ctx.lineWidth = 1;
    ctx.strokeRect(bx0, BAR_TOP, bx1 - bx0, barH);
    for (const frac of [0.25, 0.5, 0.75]) {
      const ty = BAR_TOP + frac * barH;
      ctx.beginPath();
      ctx.moveTo(bx1, ty);
      ctx.lineTo(bx1 + 5, ty);
      ctx.stroke();
    }

    ctx.textAlign = "left";
    ctx.font = "16px sans-serif";
    ctx.fillStyle = TEXT_COL;
    ctx.fillText("VEL", bx0, BAR_TOP - 10);
    ctx.font = "12px sans-serif";
    ctx.fillStyle = "#505050";
    ctx.fillText("UP", bx0, BAR_TOP + 22);
    ctx.fillText("DN", bx0, BAR_BOTTOM - 10);
    const valV = `${vyUp >= 0 ? "+" : ""}${Math.round(vyUp)}`;
    ctx.textAlign = "center";
    ctx.font = "15px sans-serif";
    ctx.fillStyle = TEXT_COL;
    ctx.fillText(valV, bx0 + BAR_W / 2, BAR_BOTTOM + 24);
    ctx.textAlign = "left";
    ctx.font = "11px sans-serif";
    ctx.fillStyle = "#646464";
    ctx.fillText("px/s", bx0, BAR_BOTTOM + 44);

    const bx0r = xOffset + width - MARGIN - BAR_W,
      bx1r = xOffset + width - MARGIN;
    ctx.fillStyle = DARK;
    ctx.fillRect(bx0r, BAR_TOP, bx1r - bx0r, barH);
    const ratioA = Math.min(1, Math.max(0, accel / MAX_ACCEL_DISPLAY));
    const fillA = ratioA * barH;
    ctx.fillStyle = hsvColour(ratioA);
    ctx.fillRect(bx0r, BAR_BOTTOM - fillA, bx1r - bx0r, fillA);
    ctx.strokeStyle = BORDER;
    ctx.lineWidth = 1;
    ctx.strokeRect(bx0r, BAR_TOP, bx1r - bx0r, barH);
    for (const frac of [0.25, 0.5, 0.75]) {
      const ty = BAR_TOP + frac * barH;
      ctx.beginPath();
      ctx.moveTo(bx0r - 5, ty);
      ctx.lineTo(bx0r, ty);
      ctx.stroke();
    }
    ctx.textAlign = "left";
    ctx.font = "16px sans-serif";
    ctx.fillStyle = TEXT_COL;
    ctx.fillText("ACC", bx0r, BAR_TOP - 10);
    const valA = `${Math.round(accel)}`;
    ctx.textAlign = "center";
    ctx.font = "15px sans-serif";
    ctx.fillText(valA, bx0r + BAR_W / 2, BAR_BOTTOM + 24);
    ctx.textAlign = "left";
    ctx.font = "11px sans-serif";
    ctx.fillStyle = "#646464";
    ctx.fillText("px/s2", bx0r - 4, BAR_BOTTOM + 44);

    if (speed > 2.0) {
      const torsoX = xOffset + width * 0.5;
      const torsoY = height * 0.33;
      const mag = Math.max(speed, 1e-6);
      const dxN = vx / mag;
      const dyN = -vyUp / mag;
      const arrowLen = Math.min(1, Math.max(0, speed / MAX_SPEED_DISPLAY)) * MAX_ARROW_PX;
      if (arrowLen > 6) {
        const ex = torsoX + dxN * arrowLen;
        const ey = torsoY + dyN * arrowLen;
        const aCol = hsvColour(speed / MAX_SPEED_DISPLAY);
        drawArrow(ctx, torsoX, torsoY, ex, ey, "#000000", 9);
        drawArrow(ctx, torsoX, torsoY, ex, ey, aCol, 4);
      }
    }
  }
}

// ── Combined entry point ────────────────────────────────────────────────────

/**
 * Draws the full climbing_plus.mp4-equivalent dual-panel overlay for one
 * frame. Assumes the caller has already drawn the video frame itself into
 * the RIGHT half of the canvas (x: videoWidth..2*videoWidth) — the left
 * half is entirely synthetic (the weight/motion schematic) and this
 * function fills it from scratch.
 */
export function renderPlusOverlay(
  ctx: CanvasRenderingContext2D,
  entry: FrameEntry | null,
  analysis: AnalysisResult | null,
  weightByFrame: Map<number, SmoothedWeightFrame> | null,
  motionByFrame: Map<number, MotionFrameEntry> | null,
  videoWidth: number,
  videoHeight: number
) {
  if (!entry) return;
  const fi = entry.frame;
  const wRec = weightByFrame?.get(fi);
  const extremities = wRec?.extremities ?? {};
  const limbSegs = wRec?.limb_segments ?? {};
  const jointAngles = wRec?.joint_angles ?? {};
  const motion = motionByFrame?.get(fi) ?? null;

  // LEFT panel: synthetic schematic.
  drawWeightSchematic(ctx, extremities, limbSegs, videoWidth, videoHeight, motion, 0);

  // RIGHT panel: video (already drawn by caller) + overlay.
  const xOff = videoWidth;

  if (analysis) {
    for (const hold of [...analysis.handholds, ...analysis.footholds]) {
      if (fi > hold.end_frame) drawPastHoldRight(ctx, hold.px + xOff, hold.py);
    }
  }

  if (entry.detected && entry.landmarks && entry.landmarks.length) {
    ctx.save();
    ctx.translate(xOff, 0);
    drawPose(ctx, { width: videoWidth, height: videoHeight } as HTMLCanvasElement, entry);
    drawSegmentForces(ctx, entry, limbSegs, videoWidth, videoHeight);
    drawJointAngles(ctx, entry, jointAngles, videoWidth, videoHeight);
    ctx.restore();
  }

  if (analysis) {
    const com = analysis.com[fi];
    if (com && com.px !== null && com.py !== null) {
      ctx.save();
      ctx.translate(xOff, 0);
      drawCom(ctx, com.px, com.py);
      ctx.restore();
    }
  }

  if (analysis) {
    for (const hold of [...analysis.handholds, ...analysis.footholds] as Hold[]) {
      if (fi < hold.start_frame || fi > hold.end_frame) continue;
      const ext = HOLD_TO_EXT[holdKey(hold.kind, hold.side)];
      const label = HOLD_LABELS[holdKey(hold.kind, hold.side)];
      const pt: [number, number] = [hold.px + xOff, hold.py];
      const wInfo = extremities[ext];
      if (wInfo) {
        drawActiveHoldSmall(ctx, pt[0], pt[1], label, wInfo.vertical_pct);
      } else {
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], 40, 0, 2 * Math.PI);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#b4b4b4";
        ctx.stroke();
      }
    }
  }

  drawPanelLabel(ctx, `${entry.timestamp_s.toFixed(2)}s`, xOff + 20, 55);
}

// render.ts
// Canvas port of reconstruct_video.py's draw_pose / draw_holds / draw_com —
// reproduces the Python-generated pose_overlay.mp4's visual style (per-region
// skeleton colouring, black-outlined dots, hold markers with past/active
// states, the CoM diamond, and the frame/timestamp stamp) in the browser, so
// the live preview and the exported overlay video match the original tool's
// output rather than the earlier flat single-colour overlay.
//
// Note: exact font glyphs won't match cv2's Hershey Simplex font pixel-for-
// pixel (no browser equivalent), but colours, positions, sizing, and the
// black-outline-then-fill text technique are matched.

import { CONNECTIONS } from "./types";
import type { FrameEntry } from "./types";
import type { AnalysisResult, Hold, HoldKind, Side } from "./analysis";

const VISIBILITY_THRESHOLD = 0.5;

const HAND_INDICES = new Set([15, 16, 17, 18, 19, 20, 21, 22]);
const FOOT_INDICES = new Set([27, 28, 29, 30, 31, 32]);
const FACE_INDICES = new Set(Array.from({ length: 11 }, (_, i) => i));

// Colours converted from reconstruct_video.py's OpenCV BGR tuples to RGB hex.
const COLOUR_BODY = "#64dc00"; // green — torso & limbs
const COLOUR_FACE = "#c8c8c8"; // grey — face landmarks
const COLOUR_HANDS = "#ffa000"; // orange — wrists/hands
const COLOUR_FEET = "#5050ff"; // blue — ankles/feet
const COLOUR_SKELETON = "#50c800"; // connection line colour

function landmarkColour(idx: number): string {
  if (FACE_INDICES.has(idx)) return COLOUR_FACE;
  if (HAND_INDICES.has(idx)) return COLOUR_HANDS;
  if (FOOT_INDICES.has(idx)) return COLOUR_FEET;
  return COLOUR_BODY;
}

const HOLD_COLOURS: Record<string, string> = {
  "hand:left": "#ffa500",
  "hand:right": "#dc0000",
  "foot:left": "#00dcff",
  "foot:right": "#0050ff",
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

function fadeColour(hex: string, factor: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgb(${Math.round(r * factor)}, ${Math.round(g * factor)}, ${Math.round(b * factor)})`;
}

/** Draws the skeleton (connection lines + per-region coloured, black-outlined dots). */
function drawPose(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, entry: FrameEntry) {
  const visible = entry.landmarks.map((lm) => lm.visibility >= VISIBILITY_THRESHOLD);

  ctx.lineWidth = 3;
  ctx.strokeStyle = COLOUR_SKELETON;
  ctx.lineCap = "round";
  for (const [a, b] of CONNECTIONS) {
    if (!visible[a] || !visible[b]) continue;
    const la = entry.landmarks[a],
      lb = entry.landmarks[b];
    ctx.beginPath();
    ctx.moveTo(la.x * canvas.width, la.y * canvas.height);
    ctx.lineTo(lb.x * canvas.width, lb.y * canvas.height);
    ctx.stroke();
  }

  entry.landmarks.forEach((lm, i) => {
    if (!visible[i]) return;
    const px = lm.x * canvas.width;
    const py = lm.y * canvas.height;
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, 2 * Math.PI);
    ctx.fillStyle = landmarkColour(i);
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#000000";
    ctx.stroke();
  });
}

/** Draws past (faded) and active (bright + labelled) hold markers for the current frame. */
function drawHolds(ctx: CanvasRenderingContext2D, frameIdx: number, analysis: AnalysisResult) {
  const allHolds: Hold[] = [...analysis.handholds, ...analysis.footholds];

  // Past holds first — drawn underneath any active hold at the same spot.
  for (const hold of allHolds) {
    if (frameIdx <= hold.end_frame) continue;
    const colour = HOLD_COLOURS[holdKey(hold.kind, hold.side)];
    ctx.beginPath();
    ctx.arc(hold.px, hold.py, 14, 0, 2 * Math.PI);
    ctx.fillStyle = fadeColour(colour, 0.35);
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#000000";
    ctx.stroke();
  }

  // Active holds — prominent ring + filled circle + label.
  for (const hold of allHolds) {
    if (frameIdx < hold.start_frame || frameIdx > hold.end_frame) continue;
    const colour = HOLD_COLOURS[holdKey(hold.kind, hold.side)];
    const label = HOLD_LABELS[holdKey(hold.kind, hold.side)];

    ctx.beginPath();
    ctx.arc(hold.px, hold.py, 30, 0, 2 * Math.PI);
    ctx.lineWidth = 3;
    ctx.strokeStyle = colour;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(hold.px, hold.py, 20, 0, 2 * Math.PI);
    ctx.fillStyle = colour;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#000000";
    ctx.stroke();

    const textX = hold.px - 20;
    const textY = hold.py - 35;
    ctx.font = "bold 24px sans-serif";
    ctx.textBaseline = "alphabetic";
    ctx.lineWidth = 5;
    ctx.strokeStyle = "#000000";
    ctx.strokeText(label, textX, textY);
    ctx.fillStyle = colour;
    ctx.fillText(label, textX, textY);
  }
}

/** Draws the yellow CoM diamond + label at the current frame's position. */
function drawCom(ctx: CanvasRenderingContext2D, px: number, py: number) {
  const size = 18;
  ctx.beginPath();
  ctx.moveTo(px, py - size);
  ctx.lineTo(px + size, py);
  ctx.lineTo(px, py + size);
  ctx.lineTo(px - size, py);
  ctx.closePath();
  ctx.fillStyle = "#ffe600";
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#000000";
  ctx.stroke();

  ctx.font = "18px sans-serif";
  ctx.textBaseline = "alphabetic";
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#000000";
  ctx.strokeText("CoM", px + 22, py + 6);
  ctx.fillStyle = "#ffe600";
  ctx.fillText("CoM", px + 22, py + 6);
}

/** Draws the "frame N  T.TTs" stamp in the top-left corner, shown even on undetected frames. */
function drawFrameStamp(ctx: CanvasRenderingContext2D, entry: FrameEntry) {
  const label = `frame ${entry.frame}  ${entry.timestamp_s.toFixed(2)}s`;
  ctx.font = "bold 28px sans-serif";
  ctx.textBaseline = "alphabetic";
  ctx.lineWidth = 4;
  ctx.strokeStyle = "#000000";
  ctx.strokeText(label, 20, 50);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, 20, 50);
}

/**
 * Draws the full pose_overlay.mp4-equivalent overlay for one frame: holds
 * (underneath), skeleton, CoM (on top of skeleton), then the frame stamp —
 * same layering order as reconstruct_video.py's reconstruct().
 */
export function renderOverlay(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  entry: FrameEntry | null,
  analysis: AnalysisResult | null
) {
  if (!entry) return;

  if (analysis && (analysis.handholds.length || analysis.footholds.length)) {
    drawHolds(ctx, entry.frame, analysis);
  }

  if (entry.detected && entry.landmarks && entry.landmarks.length) {
    drawPose(ctx, canvas, entry);
  }

  if (analysis) {
    const com = analysis.com[entry.frame];
    if (com && com.px !== null && com.py !== null) {
      drawCom(ctx, com.px, com.py);
    }
  }

  drawFrameStamp(ctx, entry);
}

// analysis.ts
// TypeScript port of generate_step_1.py: hold detection (handholds/footholds
// via stationary-wrist/ankle segments) and centre-of-mass estimation (Winter's
// 2009 body-segment method), computed entirely client-side from the pose
// landmark data already produced in-browser. Mirrors the Python script's
// algorithm, constants, and rounding exactly, so the browser's overlay can
// show the same hold markers and CoM trace as pose_overlay.mp4.

import { LANDMARK_NAMES } from "./types";
import type { FrameEntry, Landmark, PoseData } from "./types";

const IDX: Record<string, number> = {};
LANDMARK_NAMES.forEach((name, i) => (IDX[name] = i));

// Same tuning constants as generate_step_1.py.
const VELOCITY_THRESHOLD = 0.005;
const MIN_HOLD_FRAMES = 85;
const SMOOTH_WINDOW = 4;
const MAX_GAP_FRAMES = 5;

export type Side = "left" | "right";
export type HoldKind = "hand" | "foot";

export interface Hold {
  id: number;
  kind: HoldKind;
  side: Side;
  start_frame: number;
  end_frame: number;
  start_time_s: number;
  end_time_s: number;
  duration_s: number;
  x: number;
  y: number;
  px: number;
  py: number;
}

export interface ComFrameEntry {
  frame: number;
  timestamp_s: number;
  x: number | null;
  y: number | null;
  px: number | null;
  py: number | null;
}

export interface AnalysisResult {
  handholds: Hold[];
  footholds: Hold[];
  com: ComFrameEntry[];
}

function round3(v: number): number {
  return Math.round(v * 1000) / 1000;
}
function round5(v: number): number {
  return Math.round(v * 100000) / 100000;
}

/**
 * Rolling mean matching numpy's `np.convolve(arr, ones(window)/window, mode="same")`
 * exactly, including its true (zero-padded, not edge-replicated) boundary
 * behaviour — needed for byte-for-byte-equivalent hold/CoM detection.
 */
function rollingMean(arr: Float64Array, window: number): Float64Array {
  const N = arr.length;
  const out = new Float64Array(N);
  if (window <= 0) return arr.slice();

  const prefix = new Float64Array(N + 1);
  for (let i = 0; i < N; i++) prefix[i + 1] = prefix[i] + arr[i];

  const offset = Math.floor((window - 1) / 2);
  for (let i = 0; i < N; i++) {
    const fullIdx = i + offset;
    const jStart = Math.max(0, fullIdx - window + 1);
    const jEnd = Math.min(N - 1, fullIdx);
    if (jEnd < jStart) {
      out[i] = 0;
      continue;
    }
    out[i] = (prefix[jEnd + 1] - prefix[jStart]) / window;
  }
  return out;
}

function extractPositions(
  frames: FrameEntry[],
  landmarkIdx: number
): { x: Float64Array; y: Float64Array } {
  const N = frames.length;
  const x = new Float64Array(N).fill(NaN);
  const y = new Float64Array(N).fill(NaN);
  for (const frame of frames) {
    const fi = frame.frame;
    if (fi >= N) continue;
    if (frame.detected && frame.landmarks && frame.landmarks.length) {
      const lm = frame.landmarks[landmarkIdx];
      x[fi] = lm.x;
      y[fi] = lm.y;
    }
  }
  return { x, y };
}

function computeVelocity(x: Float64Array, y: Float64Array, smoothWindow: number): Float64Array {
  const N = x.length;
  const nanMask = new Uint8Array(N);
  const xClean = new Float64Array(N);
  const yClean = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    if (Number.isNaN(x[i])) {
      nanMask[i] = 1;
    } else {
      xClean[i] = x[i];
      yClean[i] = y[i];
    }
  }
  const xSmooth = rollingMean(xClean, smoothWindow);
  const ySmooth = rollingMean(yClean, smoothWindow);

  const speed = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const dx = i === 0 ? 0 : xSmooth[i] - xSmooth[i - 1];
    const dy = i === 0 ? 0 : ySmooth[i] - ySmooth[i - 1];
    speed[i] = nanMask[i] ? NaN : Math.sqrt(dx * dx + dy * dy);
  }

  const speedClean = new Float64Array(N);
  for (let i = 0; i < N; i++) speedClean[i] = Number.isNaN(speed[i]) ? 0 : speed[i];
  const speedSmooth = rollingMean(speedClean, smoothWindow);
  for (let i = 0; i < N; i++) if (nanMask[i]) speedSmooth[i] = NaN;

  return speedSmooth;
}

function findStationarySegments(
  speed: Float64Array,
  velocityThreshold: number,
  minHoldFrames: number,
  maxGapFrames: number
): [number, number][] {
  const N = speed.length;
  const stationary = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    if (!Number.isNaN(speed[i])) stationary[i] = speed[i] < velocityThreshold ? 1 : 0;
  }

  if (maxGapFrames > 0) {
    let i = 0;
    while (i < N) {
      if (!stationary[i]) {
        const gapStart = i;
        while (i < N && !stationary[i]) i++;
        const gapEnd = i;
        const gapLen = gapEnd - gapStart;
        if (gapLen <= maxGapFrames && gapStart > 0 && gapEnd < N) {
          for (let k = gapStart; k < gapEnd; k++) stationary[k] = 1;
        }
      } else {
        i++;
      }
    }
  }

  const segments: [number, number][] = [];
  let inSeg = false;
  let segStart = 0;
  for (let i = 0; i < N; i++) {
    if (stationary[i] && !inSeg) {
      segStart = i;
      inSeg = true;
    } else if (!stationary[i] && inSeg) {
      const segEnd = i - 1;
      if (segEnd - segStart + 1 >= minHoldFrames) segments.push([segStart, segEnd]);
      inSeg = false;
    }
  }
  if (inSeg && N - 1 - segStart + 1 >= minHoldFrames) segments.push([segStart, N - 1]);

  return segments;
}

function segmentsToHolds(
  segments: [number, number][],
  x: Float64Array,
  y: Float64Array,
  side: Side,
  kind: HoldKind,
  fps: number,
  width: number,
  height: number
): Hold[] {
  const holds: Hold[] = [];
  for (const [start, end] of segments) {
    let sumX = 0,
      sumY = 0,
      count = 0;
    for (let i = start; i <= end; i++) {
      if (!Number.isNaN(x[i])) {
        sumX += x[i];
        sumY += y[i];
        count++;
      }
    }
    if (count === 0) continue;
    const avgX = sumX / count;
    const avgY = sumY / count;
    holds.push({
      id: 0,
      kind,
      side,
      start_frame: start,
      end_frame: end,
      start_time_s: round3(start / fps),
      end_time_s: round3(end / fps),
      duration_s: round3((end - start) / fps),
      x: round5(avgX),
      y: round5(avgY),
      px: Math.round(avgX * width),
      py: Math.round(avgY * height),
    });
  }
  return holds;
}

// ── Centre of Mass (Winter's 2009 body-segment parameters) ────────────────
interface SegmentDef {
  mass: number;
  prox: string;
  dist: string;
  frac: number;
}

const SEGMENTS: SegmentDef[] = [
  { mass: 0.497, prox: "MID_SHOULDER", dist: "MID_HIP", frac: 0.43 },
  { mass: 0.028, prox: "LEFT_SHOULDER", dist: "LEFT_ELBOW", frac: 0.436 },
  { mass: 0.016, prox: "LEFT_ELBOW", dist: "LEFT_WRIST", frac: 0.43 },
  { mass: 0.006, prox: "LEFT_WRIST", dist: "LEFT_INDEX", frac: 0.506 },
  { mass: 0.028, prox: "RIGHT_SHOULDER", dist: "RIGHT_ELBOW", frac: 0.436 },
  { mass: 0.016, prox: "RIGHT_ELBOW", dist: "RIGHT_WRIST", frac: 0.43 },
  { mass: 0.006, prox: "RIGHT_WRIST", dist: "RIGHT_INDEX", frac: 0.506 },
  { mass: 0.1, prox: "LEFT_HIP", dist: "LEFT_KNEE", frac: 0.433 },
  { mass: 0.047, prox: "LEFT_KNEE", dist: "LEFT_ANKLE", frac: 0.433 },
  { mass: 0.015, prox: "LEFT_ANKLE", dist: "LEFT_FOOT_INDEX", frac: 0.5 },
  { mass: 0.1, prox: "RIGHT_HIP", dist: "RIGHT_KNEE", frac: 0.433 },
  { mass: 0.047, prox: "RIGHT_KNEE", dist: "RIGHT_ANKLE", frac: 0.433 },
  { mass: 0.015, prox: "RIGHT_ANKLE", dist: "RIGHT_FOOT_INDEX", frac: 0.5 },
];
const HEAD_MASS = 0.081;

function comFromFrame(landmarks: Landmark[]): { x: number; y: number } | null {
  const lmMap: Record<string, [number, number]> = {};
  for (const lm of landmarks) {
    if (lm.visibility >= 0.3) lmMap[lm.name] = [lm.x, lm.y];
  }
  if (lmMap["LEFT_SHOULDER"] && lmMap["RIGHT_SHOULDER"]) {
    const [lx, ly] = lmMap["LEFT_SHOULDER"];
    const [rx, ry] = lmMap["RIGHT_SHOULDER"];
    lmMap["MID_SHOULDER"] = [(lx + rx) / 2, (ly + ry) / 2];
  }
  if (lmMap["LEFT_HIP"] && lmMap["RIGHT_HIP"]) {
    const [lx, ly] = lmMap["LEFT_HIP"];
    const [rx, ry] = lmMap["RIGHT_HIP"];
    lmMap["MID_HIP"] = [(lx + rx) / 2, (ly + ry) / 2];
  }

  let totalWeight = 0,
    wx = 0,
    wy = 0;

  if (lmMap["NOSE"]) {
    const [nx, ny] = lmMap["NOSE"];
    wx += HEAD_MASS * nx;
    wy += HEAD_MASS * ny;
    totalWeight += HEAD_MASS;
  }

  for (const seg of SEGMENTS) {
    const p = lmMap[seg.prox];
    const d = lmMap[seg.dist];
    if (p && d) {
      const cx = p[0] + seg.frac * (d[0] - p[0]);
      const cy = p[1] + seg.frac * (d[1] - p[1]);
      wx += seg.mass * cx;
      wy += seg.mass * cy;
      totalWeight += seg.mass;
    }
  }

  if (totalWeight < 0.3) return null;
  return { x: wx / totalWeight, y: wy / totalWeight };
}

function computeComSeries(
  frames: FrameEntry[],
  fps: number,
  width: number,
  height: number
): ComFrameEntry[] {
  const results: ComFrameEntry[] = [];
  for (const frame of frames) {
    const fi = frame.frame;
    const ts = Math.round((fi / fps) * 10000) / 10000;
    let com: { x: number; y: number } | null = null;
    if (frame.detected && frame.landmarks && frame.landmarks.length) {
      com = comFromFrame(frame.landmarks);
    }
    if (com) {
      results.push({
        frame: fi,
        timestamp_s: ts,
        x: round5(com.x),
        y: round5(com.y),
        px: Math.round(com.x * width),
        py: Math.round(com.y * height),
      });
    } else {
      results.push({ frame: fi, timestamp_s: ts, x: null, y: null, px: null, py: null });
    }
  }
  return results;
}

/** Runs hold detection + CoM estimation over an entire extracted pose sequence. */
export function computeAnalysis(poseData: PoseData): AnalysisResult {
  const { fps, width, height } = poseData.video;
  const frames = poseData.landmarks;

  const tasks: { lmIdx: number; side: Side; kind: HoldKind }[] = [
    { lmIdx: IDX["LEFT_WRIST"], side: "left", kind: "hand" },
    { lmIdx: IDX["RIGHT_WRIST"], side: "right", kind: "hand" },
    { lmIdx: IDX["LEFT_ANKLE"], side: "left", kind: "foot" },
    { lmIdx: IDX["RIGHT_ANKLE"], side: "right", kind: "foot" },
  ];

  const handholds: Hold[] = [];
  const footholds: Hold[] = [];

  for (const { lmIdx, side, kind } of tasks) {
    const { x, y } = extractPositions(frames, lmIdx);
    const speed = computeVelocity(x, y, SMOOTH_WINDOW);
    const segments = findStationarySegments(speed, VELOCITY_THRESHOLD, MIN_HOLD_FRAMES, MAX_GAP_FRAMES);
    const holds = segmentsToHolds(segments, x, y, side, kind, fps, width, height);
    if (kind === "hand") handholds.push(...holds);
    else footholds.push(...holds);
  }

  handholds.sort((a, b) => a.start_frame - b.start_frame);
  footholds.sort((a, b) => a.start_frame - b.start_frame);
  handholds.forEach((h, i) => (h.id = i));
  footholds.forEach((h, i) => (h.id = i));

  const com = computeComSeries(frames, fps, width, height);

  return { handholds, footholds, com };
}

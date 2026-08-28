// motion.ts
// TypeScript port of stats2.py: Savitzky-Golay-smoothed centre-of-mass
// trajectory, differentiated (via a non-uniform-spacing finite-difference
// scheme matching numpy.gradient) into per-frame velocity and acceleration.
//
// Note on "smoothing" here: this is standard numerical-differentiation
// practice (smoothing a position signal before taking its 2nd derivative,
// which otherwise amplifies detection noise enormously), applied to the
// already-computed CoM trajectory — a different thing from, and unrelated
// to, temporally smoothing raw pose landmarks (which was explicitly ruled
// out earlier as a way to paper over MediaPipe's per-frame detection error).
// This is ported as-is because it's what the reference climbing_plus.mp4
// output actually shows.

import type { ComFrameEntry } from "./analysis";

const SMOOTH_WINDOW = 51;
const SMOOTH_POLY = 3;
const ACCEL_SMOOTH_WINDOW = 51;
const ACCEL_SMOOTH_POLY = 2;
const STATIC_THRESHOLD = 15.0;
const DYNAMIC_THRESHOLD_PERCENTILE = 90;

export interface MotionFrameEntry {
  frame: number;
  timestamp_s: number;
  px: number | null;
  py: number | null;
  x: number | null;
  y: number | null;
  vx_px_s: number | null;
  vy_px_s: number | null;
  speed_px_s: number | null;
  ax_px_s2: number | null;
  ay_px_s2: number | null;
  accel_px_s2: number | null;
  is_dynamic: boolean | null;
  is_static: boolean | null;
}

function round2(v: number): number {
  return Math.round(v * 100) / 100;
}
function round6(v: number): number {
  return Math.round(v * 1000000) / 1000000;
}

// ── Linear algebra helpers for the per-window polynomial fit ──────────────

function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    [M[col], M[pivot]] = [M[pivot], M[col]];
    const pv = M[col][col];
    if (Math.abs(pv) < 1e-14) continue;
    for (let c = col; c <= n; c++) M[col][c] /= pv;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      if (factor === 0) continue;
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }
  return M.map((row) => row[n]);
}

function polyfit(xs: number[], ys: number[], degree: number): number[] {
  const m = degree + 1;
  const XtX: number[][] = Array.from({ length: m }, () => new Array(m).fill(0));
  const Xty: number[] = new Array(m).fill(0);
  for (let i = 0; i < xs.length; i++) {
    const powers = new Array(m);
    let p = 1;
    for (let k = 0; k < m; k++) {
      powers[k] = p;
      p *= xs[i];
    }
    for (let r = 0; r < m; r++) {
      Xty[r] += powers[r] * ys[i];
      for (let c = 0; c < m; c++) XtX[r][c] += powers[r] * powers[c];
    }
  }
  return solveLinearSystem(XtX, Xty);
}

function evalPoly(coeffs: number[], x: number): number {
  let result = 0,
    p = 1;
  for (const c of coeffs) {
    result += c * p;
    p *= x;
  }
  return result;
}

/**
 * Savitzky-Golay filter matching scipy.signal.savgol_filter's default
 * mode="interp": interior points use a centred local polynomial fit;
 * the first/last `window//2` points are filled by fitting ONE polynomial to
 * the boundary window and evaluating it at each edge offset (not truncating
 * or zero-padding the window at the edges).
 */
function savgol(arr: number[], window: number, poly: number): number[] {
  const N = arr.length;
  const half = Math.floor(window / 2);
  const out = new Array(N);

  if (N <= window) {
    const xs = arr.map((_, i) => i);
    const coeffs = polyfit(xs, arr, Math.min(poly, N - 1));
    for (let i = 0; i < N; i++) out[i] = evalPoly(coeffs, i);
    return out;
  }

  for (let i = half; i < N - half; i++) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let k = -half; k <= half; k++) {
      xs.push(k);
      ys.push(arr[i + k]);
    }
    const coeffs = polyfit(xs, ys, poly);
    out[i] = evalPoly(coeffs, 0);
  }

  {
    const xs = Array.from({ length: window }, (_, k) => k);
    const ys = arr.slice(0, window);
    const coeffs = polyfit(xs, ys, poly);
    for (let i = 0; i < half; i++) out[i] = evalPoly(coeffs, i);
  }

  {
    const start = N - window;
    const xs = Array.from({ length: window }, (_, k) => k);
    const ys = arr.slice(start, N);
    const coeffs = polyfit(xs, ys, poly);
    for (let i = N - half; i < N; i++) out[i] = evalPoly(coeffs, i - start);
  }

  return out;
}

/**
 * numpy.gradient equivalent for possibly-non-uniformly-spaced x (edge_order=1
 * default): simple one-sided differences at the edges, and the exact
 * non-uniform central-difference formula for interior points.
 */
function gradient(f: number[], x: number[]): number[] {
  const N = f.length;
  const out = new Array(N).fill(0);
  if (N < 2) return out;
  out[0] = (f[1] - f[0]) / (x[1] - x[0]);
  out[N - 1] = (f[N - 1] - f[N - 2]) / (x[N - 1] - x[N - 2]);
  for (let i = 1; i < N - 1; i++) {
    const hs = x[i] - x[i - 1];
    const hd = x[i + 1] - x[i];
    out[i] = (hs * hs * f[i + 1] + (hd * hd - hs * hs) * f[i] - hd * hd * f[i - 1]) / (hs * hd * (hd + hs));
  }
  return out;
}

function percentile(arr: number[], p: number): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const frac = idx - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

/**
 * Runs the full stats2.py CoM-motion pipeline. Returns null when there are
 * fewer detected CoM frames than SMOOTH_WINDOW, matching the Python script's
 * abort condition (no com_motion.json is produced in that case).
 */
export function computeMotion(comSeries: ComFrameEntry[], fps: number): MotionFrameEntry[] | null {
  const detected = comSeries.filter((e) => e.px !== null && e.py !== null && e.x !== null && e.y !== null);
  if (detected.length < SMOOTH_WINDOW) return null;

  const frames = detected.map((e) => e.frame);
  const times = detected.map((e) => e.timestamp_s);
  const pxRaw = detected.map((e) => e.px as number);
  const pyRaw = detected.map((e) => e.py as number);

  let w = Math.min(SMOOTH_WINDOW, pxRaw.length);
  if (w % 2 === 0) w -= 1;
  w = Math.max(w, 3);

  const pxSm = savgol(pxRaw, w, SMOOTH_POLY);
  const pySm = savgol(pyRaw, w, SMOOTH_POLY);

  const vx = gradient(pxSm, times);
  const vyUp = gradient(pySm, times).map((v) => -v);
  const speed = vx.map((v, i) => Math.sqrt(v * v + vyUp[i] * vyUp[i]));

  let ax = gradient(vx, times);
  let ayUp = gradient(vyUp, times);
  let accel = ax.map((v, i) => Math.sqrt(v * v + ayUp[i] * ayUp[i]));

  if (ACCEL_SMOOTH_WINDOW > 1) {
    let aw = Math.min(ACCEL_SMOOTH_WINDOW, accel.length);
    if (aw % 2 === 0) aw -= 1;
    aw = Math.max(aw, 3);
    ax = savgol(ax, aw, ACCEL_SMOOTH_POLY);
    ayUp = savgol(ayUp, aw, ACCEL_SMOOTH_POLY);
    accel = savgol(accel, aw, ACCEL_SMOOTH_POLY);
  }

  const dynamicThreshold = percentile(accel, DYNAMIC_THRESHOLD_PERCENTILE);
  const isDynamic = accel.map((a) => a >= dynamicThreshold);
  const isStatic = speed.map((s) => s < STATIC_THRESHOLD);

  const lookup = new Map<number, MotionFrameEntry>();
  for (let i = 0; i < detected.length; i++) {
    lookup.set(frames[i], {
      frame: frames[i],
      timestamp_s: times[i],
      px: round2(pxSm[i]),
      py: round2(pySm[i]),
      x: round6(detected[i].x as number),
      y: round6(detected[i].y as number),
      vx_px_s: round2(vx[i]),
      vy_px_s: round2(vyUp[i]),
      speed_px_s: round2(speed[i]),
      ax_px_s2: round2(ax[i]),
      ay_px_s2: round2(ayUp[i]),
      accel_px_s2: round2(accel[i]),
      is_dynamic: isDynamic[i],
      is_static: isStatic[i],
    });
  }

  const out: MotionFrameEntry[] = [];
  for (const e of comSeries) {
    const m = lookup.get(e.frame);
    if (m) {
      out.push(m);
    } else {
      out.push({
        frame: e.frame,
        timestamp_s: e.timestamp_s,
        px: null,
        py: null,
        x: null,
        y: null,
        vx_px_s: null,
        vy_px_s: null,
        speed_px_s: null,
        ax_px_s2: null,
        ay_px_s2: null,
        accel_px_s2: null,
        is_dynamic: null,
        is_static: null,
      });
    }
  }
  return out;
}

// shape.ts
// Turns the path someone drags across the tunnel into something the solver can
// use: a coverage mask in the body's own frame, plus the numbers the rotation
// needs — where the area centroid is, how much area there is, how hard it is to
// spin, and which way its long axis runs.
//
// Everything here is measured from the RASTERISED mask rather than from the
// polygon. Shoelace formulas are exact for a simple polygon and quietly wrong
// for a self-intersecting one, and a freehand scribble crosses itself all the
// time. Counting pixels is immune to that, and it has the better property of
// measuring exactly the body the fluid will go on to see.

/** "Tunnel units": y runs 0..1 bottom to top, x runs 0..aspect left to right,
 *  so distances mean the same thing on both axes and a circle is round. */
export interface Vec2 {
  x: number;
  y: number;
}

export interface Shape {
  /** Area centroid, in tunnel units. */
  centroid: Vec2;
  /** Covered area, in tunnel units squared. */
  area: number;
  /** Second moment of area about the centroid. Tunnel units to the fourth. */
  polarMoment: number;
  /** Distance from the centroid to the furthest covered pixel. */
  radius: number;
  /** Unit vector along the long axis, in the body's own (undrawn) frame. The
   *  balance point slides along this. */
  axis: Vec2;
  /** Square coverage bitmap in the body's frame, centred on the centroid and
   *  spanning 2 * `extent` tunnel units. 0 = open air, 255 = solid. */
  mask: Uint8Array;
  maskSize: number;
  /** Half-width of the mask, in tunnel units. */
  extent: number;
}

/** Working resolution for measuring. Fine enough that the second moment of a
 *  thin plate is not dominated by its own staircase. */
const MEASURE_SIZE = 512;
/** Resolution of the mask handed to the GPU. Downsampling into it is what gives
 *  the edge a one-texel ramp, which the immersed boundary wants: a hard 0/1
 *  edge makes the measured torque rattle as the body turns and whole cells flip
 *  state at once. */
const MASK_SIZE = 256;
/** How much of the mask the shape fills, leaving room for that soft edge. */
const MASK_FILL = 0.86;
/** Below this the drawing was a stray click, or a line enclosing nothing. */
const MIN_AREA = 0.00035;

function fillPath(ctx: CanvasRenderingContext2D, path: Vec2[], toPixel: (p: Vec2) => Vec2) {
  ctx.beginPath();
  const first = toPixel(path[0]);
  ctx.moveTo(first.x, first.y);
  for (let i = 1; i < path.length; i++) {
    const p = toPixel(path[i]);
    ctx.lineTo(p.x, p.y);
  }
  ctx.closePath();
  // A freehand loop crosses itself constantly. "evenodd" would punch a hole at
  // every crossing; nonzero fills the whole enclosed blob, which is plainly
  // what the person drawing meant.
  ctx.fill("nonzero");
}

/**
 * Builds a Shape from a drawn path, or returns null when the drawing encloses
 * too little to be a body. `path` is in tunnel units; `aspect` is the tunnel's
 * width in those units.
 */
export function buildShape(path: Vec2[], aspect: number): Shape | null {
  if (path.length < 3) return null;

  // ── Pass 1: draw it where it was drawn, and measure ──────────────────────
  const height = MEASURE_SIZE;
  const width = Math.max(8, Math.round(MEASURE_SIZE * aspect));
  const measure = document.createElement("canvas");
  measure.width = width;
  measure.height = height;
  const mctx = measure.getContext("2d", { willReadFrequently: true });
  if (!mctx) return null;

  const pxPerUnit = height;
  const unitPerPx = 1 / pxPerUnit;
  mctx.fillStyle = "#fff";
  // Canvas y grows downward and tunnel y grows upward, so it is flipped here
  // and nowhere else; everything downstream is in tunnel orientation.
  fillPath(mctx, path, (p) => ({ x: p.x * pxPerUnit, y: (1 - p.y) * pxPerUnit }));

  const data = mctx.getImageData(0, 0, width, height).data;

  let covered = 0;
  let sx = 0;
  let sy = 0;
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      // Antialiased coverage, not a yes/no test: a thin plate is nearly all
      // edge, and rounding its edge away would lose most of the shape.
      const c = data[(py * width + px) * 4 + 3] / 255;
      if (c <= 0) continue;
      covered += c;
      sx += c * (px + 0.5) * unitPerPx;
      sy += c * (1 - (py + 0.5) * unitPerPx);
    }
  }

  const cellArea = unitPerPx * unitPerPx;
  const area = covered * cellArea;
  if (area < MIN_AREA) return null;

  const centroid: Vec2 = { x: sx / covered, y: sy / covered };

  // Second moments about the centroid, in tunnel units.
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  let maxR2 = 0;
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      const c = data[(py * width + px) * 4 + 3] / 255;
      if (c <= 0) continue;
      const dx = (px + 0.5) * unitPerPx - centroid.x;
      const dy = 1 - (py + 0.5) * unitPerPx - centroid.y;
      sxx += c * dx * dx;
      syy += c * dy * dy;
      sxy += c * dx * dy;
      const r2 = dx * dx + dy * dy;
      if (r2 > maxR2) maxR2 = r2;
    }
  }
  const polarMoment = (sxx + syy) * cellArea;
  const radius = Math.sqrt(maxR2);
  if (!(radius > 0)) return null;

  // The long axis is the principal direction of greatest spread — the larger
  // eigenvector of the covariance of the covered pixels.
  const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const axis: Vec2 = { x: Math.cos(theta), y: Math.sin(theta) };

  // ── Pass 2: redraw centred on the centroid, at mask resolution ───────────
  const extent = radius / MASK_FILL;
  const big = document.createElement("canvas");
  big.width = MASK_SIZE * 2;
  big.height = MASK_SIZE * 2;
  const bctx = big.getContext("2d", { willReadFrequently: true });
  if (!bctx) return null;
  bctx.fillStyle = "#fff";
  const half = big.width / 2;
  const maskPxPerUnit = half / extent;
  fillPath(bctx, path, (p) => ({
    x: half + (p.x - centroid.x) * maskPxPerUnit,
    y: half - (p.y - centroid.y) * maskPxPerUnit,
  }));

  // Halving it with the browser's own filter is the cheapest way to turn the
  // boundary into a one-texel ramp instead of a cliff.
  const small = document.createElement("canvas");
  small.width = MASK_SIZE;
  small.height = MASK_SIZE;
  const sctx = small.getContext("2d", { willReadFrequently: true });
  if (!sctx) return null;
  sctx.imageSmoothingEnabled = true;
  sctx.imageSmoothingQuality = "high";
  sctx.drawImage(big, 0, 0, MASK_SIZE, MASK_SIZE);

  const maskData = sctx.getImageData(0, 0, MASK_SIZE, MASK_SIZE).data;
  const mask = new Uint8Array(MASK_SIZE * MASK_SIZE);
  for (let i = 0; i < mask.length; i++) mask[i] = maskData[i * 4 + 3];

  return { centroid, area, polarMoment, radius, axis, mask, maskSize: MASK_SIZE, extent };
}

/** Closed outlines to open with, in tunnel units, so the tunnel is running
 *  before anyone has drawn anything. */
export function presetPath(name: "teardrop" | "plate" | "disc", aspect: number): Vec2[] {
  const cx = aspect / 2;
  const cy = 0.5;
  const pts: Vec2[] = [];
  const n = 96;

  if (name === "disc") {
    for (let i = 0; i < n; i++) {
      const t = (i / n) * Math.PI * 2;
      pts.push({ x: cx + 0.11 * Math.cos(t), y: cy + 0.11 * Math.sin(t) });
    }
    return pts;
  }

  // Both of the others are drawn at a slant. Square to the flow is an
  // equilibrium — for some shapes an unstable one — and a body started exactly
  // on it has no reason to leave.
  const tilt = name === "plate" ? 0.5 : 1.05;
  const c = Math.cos(tilt);
  const s = Math.sin(tilt);
  const place = (x: number, y: number) => pts.push({ x: cx + x * c - y * s, y: cy + x * s + y * c });

  if (name === "plate") {
    const hw = 0.2;
    const ht = 0.016;
    place(-hw, -ht);
    place(hw, -ht);
    place(hw, ht);
    place(-hw, ht);
    return pts;
  }

  // Teardrop: blunt at one end, drawn out to a point at the other.
  for (let i = 0; i < n; i++) {
    const t = (i / n) * Math.PI * 2;
    const rr = 0.135 * (1 - 0.55 * Math.cos(t));
    place(rr * Math.cos(t) * 0.8, rr * Math.sin(t));
  }
  return pts;
}

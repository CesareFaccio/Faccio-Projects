// trace.ts
// Torque against angle, drawn 1-bit on a small canvas.
//
// This is the static-stability curve an aerodynamicist would plot, and it reads
// directly: where the cloud crosses zero going DOWNWARDS, the body has found an
// orientation it can hold, because a nudge either way produces a torque pushing
// it back. A crossing going upwards is an equilibrium it will fall off.
//
// It plots torque rather than drag because torque is the quantity that could be
// measured honestly here — see BODY_FORCE_SHADER in tunnel.ts for the three
// attempts at drag and why each one failed its check against a disc.
//
// One series, so there is no legend — the window's title names it — and no
// categorical palette to validate: the whole site is black on white by design,
// and the density of the dot cloud carries weight in place of colour. Axes are
// hairlines, ticks are three pixels, and the only things labelled directly are
// the zero rule and where the body is right now.
//
// There is deliberately no hover layer. The trace is a live instrument that
// appends points several times a second, so a tooltip would be chasing a moving
// target; the current state is printed under the plot instead, which answers
// the same question without asking anyone to aim at a 1px dot.

export interface TracePoint {
  /** Degrees, 0..360. */
  angle: number;
  torque: number;
}

const PAD_L = 22;
const PAD_R = 6;
const PAD_T = 8;
const PAD_B = 16;
/** Past a few thousand the cloud is saturated and older points only cost
 *  memory. */
const MAX_POINTS = 2600;

export class Trace {
  private points: TracePoint[] = [];
  private ctx: CanvasRenderingContext2D | null;
  /** The y axis is symmetric about zero, because the sign of the torque is the
   *  whole point: it says which way the body is being pushed. */
  private span = 0;
  private current: TracePoint | null = null;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d");
  }

  clear() {
    this.points = [];
    this.span = 0;
    this.current = null;
    this.draw();
  }

  push(angleDeg: number, torque: number) {
    let a = angleDeg % 360;
    if (a < 0) a += 360;
    const p = { angle: a, torque };
    this.current = p;
    // A body that has stopped moving would otherwise pile thousands of
    // identical samples onto one pixel.
    const last = this.points[this.points.length - 1];
    if (last && Math.abs(last.angle - a) < 0.35 && Math.abs(last.torque - torque) < this.span * 0.01) {
      return;
    }
    this.points.push(p);
    if (this.points.length > MAX_POINTS) this.points.shift();
    if (Math.abs(torque) > this.span) this.span = Math.abs(torque);
  }

  /** Resizes the backing store to match the element. */
  resize() {
    const w = Math.max(80, Math.round(this.canvas.clientWidth));
    const h = Math.max(60, Math.round(this.canvas.clientHeight));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  private plotX(angle: number, w: number) {
    return PAD_L + (angle / 360) * (w - PAD_L - PAD_R);
  }

  private plotY(torque: number, h: number) {
    const top = this.span > 0 ? this.span : 1;
    const frac = Math.max(-1, Math.min(1, torque / top));
    const mid = (PAD_T + (h - PAD_B)) / 2;
    return mid - frac * ((h - PAD_T - PAD_B) / 2);
  }

  draw() {
    const ctx = this.ctx;
    if (!ctx) return;
    this.resize();
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, w, h);

    ctx.font = '8px "Pixelify Sans", Geneva, Verdana, sans-serif';
    ctx.textBaseline = "top";

    // Recessive hairlines, not a box and not a grid. The zero rule is dashed
    // because it is a reference rather than data — and it is the line the whole
    // plot is read against.
    const zeroY = Math.round(this.plotY(0, h));
    ctx.fillStyle = "#000";
    ctx.fillRect(PAD_L, PAD_T, 1, h - PAD_T - PAD_B);
    ctx.fillRect(PAD_L, h - PAD_B, w - PAD_L - PAD_R, 1);
    ctx.fillStyle = "#a8a8a8";
    for (let x = PAD_L + 2; x < w - PAD_R; x += 4) ctx.fillRect(x, zeroY, 2, 1);

    for (const d of [0, 90, 180, 270, 360]) {
      const x = Math.round(this.plotX(d, w));
      ctx.fillStyle = "#000";
      ctx.fillRect(x, h - PAD_B + 1, 1, 3);
      ctx.fillStyle = "#6b6b6b";
      const label = String(d);
      const tw = ctx.measureText(label).width;
      ctx.fillText(label, Math.min(Math.max(x - tw / 2, 0), w - tw), h - PAD_B + 6);
    }

    ctx.textBaseline = "middle";
    ctx.fillStyle = "#6b6b6b";
    ctx.fillText("ccw", 1, PAD_T + 5);
    ctx.fillText("0", 1, zeroY);
    ctx.fillText("cw", 1, h - PAD_B - 5);

    if (this.points.length === 0) {
      ctx.fillText("waiting for flow", PAD_L + 14, zeroY - 12);
      return;
    }

    // The cloud. Single pixels: where the body lingers they pile up and read
    // darker, which is density doing the work colour would do elsewhere.
    ctx.fillStyle = "#000";
    for (const p of this.points) {
      ctx.fillRect(Math.round(this.plotX(p.angle, w)), Math.round(this.plotY(p.torque, h)), 1, 1);
    }

    if (this.current) {
      const x = Math.round(this.plotX(this.current.angle, w));
      const y = Math.round(this.plotY(this.current.torque, h));
      // A white ring under the marker so it stays visible inside a dense cloud.
      ctx.fillStyle = "#fff";
      ctx.fillRect(x - 4, y - 4, 9, 9);
      ctx.fillStyle = "#000";
      ctx.fillRect(x - 4, y - 4, 9, 1);
      ctx.fillRect(x - 4, y + 4, 9, 1);
      ctx.fillRect(x - 4, y - 4, 1, 9);
      ctx.fillRect(x + 4, y - 4, 1, 9);
      ctx.fillRect(x - 1, y - 1, 3, 3);
    }
  }

  /** One line for the window's status bar, since the plot carries no tooltip. */
  status(settled: boolean): string {
    if (!this.current) return "no samples yet";
    const deg = Math.round(this.current.angle);
    if (settled) return `${deg}° · holding — the torque crosses zero here`;
    return `${deg}° · ${this.current.torque > 0 ? "pushing ccw" : "pushing cw"}`;
  }
}

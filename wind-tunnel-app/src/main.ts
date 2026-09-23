// main.ts
// Wiring: the drawing surface, the sliders, the trace, and the tunnel itself.

import { startTunnel } from "./tunnel";
import type { TunnelHandle, TunnelParams } from "./tunnel";
import { buildShape, presetPath } from "./shape";
import type { Vec2 } from "./shape";
import { Trace } from "./trace";
import { makeWindowsDraggable } from "./windows";

const tunnelCanvas = document.getElementById("tunnel-canvas") as HTMLCanvasElement;
const inkCanvas = document.getElementById("ink-canvas") as HTMLCanvasElement;
const traceCanvas = document.getElementById("trace-canvas") as HTMLCanvasElement;
const fallback = document.querySelector<HTMLElement>(".tunnel__fallback");
const tunnelStatus = document.getElementById("tunnel-status") as HTMLElement;
const traceStatus = document.getElementById("trace-status") as HTMLElement;
const angleReadout = document.getElementById("angle-readout") as HTMLElement;
const pauseBtn = document.getElementById("pause") as HTMLButtonElement;

makeWindowsDraggable(document);

const trace = new Trace(traceCanvas);
let tunnel: TunnelHandle | null = null;

// Booting a frame late, not immediately: the solver sizes its grids from the
// canvas exactly once, and a module script can run before the stylesheet has
// given the canvas a height. Starting on a zero-sized canvas would fix the
// whole simulation to the wrong aspect ratio with nothing to show for it.
requestAnimationFrame(() => {
  tunnel = startTunnel(tunnelCanvas);
  if (!tunnel) {
    tunnelCanvas.style.display = "none";
    if (fallback) fallback.hidden = false;
    tunnelStatus.textContent = "This browser cannot run the simulation.";
    return;
  }
  // Exposed for stepping the solver and reading its state from the console or
  // from a test, the same way the other apps expose their internals.
  (window as unknown as Record<string, unknown>).__windTunnel = tunnel;
  start(tunnel);
});

function aspect(): number {
  const r = tunnelCanvas.getBoundingClientRect();
  return r.height > 0 ? r.width / r.height : 1;
}

/** Pointer position in tunnel units: y up, x scaled so a circle stays round. */
function toTunnel(e: PointerEvent): Vec2 {
  const r = tunnelCanvas.getBoundingClientRect();
  return {
    x: ((e.clientX - r.left) / r.height),
    y: (r.bottom - e.clientY) / r.height,
  };
}

function applyShape(path: Vec2[]) {
  if (!tunnel) return;
  const shape = buildShape(path, aspect());
  if (!shape) {
    tunnelStatus.textContent = "That did not enclose anything — draw a closed blob.";
    return;
  }
  tunnel.setShape(shape);
  trace.clear();
  tunnelStatus.textContent = "Air runs bottom to top. The shape is pinned at the cross and free to turn.";
}

function start(t: TunnelHandle) {
  // ── Drawing ────────────────────────────────────────────────────────────
  // The ink canvas sits over the tunnel and shows the stroke as it is made;
  // once the shape is handed to the solver it is the solver that draws it.
  const ink = inkCanvas.getContext("2d");
  let drawing = false;
  let path: Vec2[] = [];

  function sizeInk() {
    const r = tunnelCanvas.getBoundingClientRect();
    const w = Math.round(r.width);
    const h = Math.round(r.height);
    if (inkCanvas.width !== w || inkCanvas.height !== h) {
      inkCanvas.width = w;
      inkCanvas.height = h;
    }
  }

  function drawInk() {
    if (!ink) return;
    sizeInk();
    ink.clearRect(0, 0, inkCanvas.width, inkCanvas.height);
    if (path.length < 2) return;
    const h = inkCanvas.height;
    ink.strokeStyle = "#000";
    ink.lineWidth = 2;
    ink.lineJoin = "round";
    ink.lineCap = "round";
    ink.beginPath();
    ink.moveTo(path[0].x * h, (1 - path[0].y) * h);
    for (let i = 1; i < path.length; i++) ink.lineTo(path[i].x * h, (1 - path[i].y) * h);
    ink.stroke();
    // A faint chord back to the start, so it is obvious the loop will close.
    ink.setLineDash([3, 3]);
    ink.strokeStyle = "#8b8b8b";
    ink.lineWidth = 1;
    ink.beginPath();
    ink.moveTo(path[path.length - 1].x * h, (1 - path[path.length - 1].y) * h);
    ink.lineTo(path[0].x * h, (1 - path[0].y) * h);
    ink.stroke();
    ink.setLineDash([]);
  }

  tunnelCanvas.parentElement?.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    drawing = true;
    path = [toTunnel(e)];
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    tunnelStatus.textContent = "Drawing — release to close the shape.";
    e.preventDefault();
  });

  tunnelCanvas.parentElement?.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    const p = toTunnel(e);
    const last = path[path.length - 1];
    // Decimate: a freehand drag fires far more events than the outline needs,
    // and every extra vertex is another edge for the rasteriser to walk.
    if (Math.hypot(p.x - last.x, p.y - last.y) > 0.006) {
      path.push(p);
      drawInk();
    }
  });

  function endStroke(e: PointerEvent) {
    if (!drawing) return;
    drawing = false;
    const el = e.currentTarget as HTMLElement;
    if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
    if (ink) ink.clearRect(0, 0, inkCanvas.width, inkCanvas.height);
    applyShape(path);
    path = [];
  }
  tunnelCanvas.parentElement?.addEventListener("pointerup", endStroke);
  tunnelCanvas.parentElement?.addEventListener("pointercancel", endStroke);

  // ── Sliders ────────────────────────────────────────────────────────────
  const sliders: Array<[string, string, keyof TunnelParams, (v: number) => string]> = [
    ["s-wind", "o-wind", "wind", (v) => `${Math.round(v * 100)}%`],
    ["s-turb", "o-turb", "turbulence", (v) => `${Math.round(v * 100)}%`],
    ["s-inertia", "o-inertia", "inertia", (v) => `${Math.round(v * 100)}%`],
    ["s-damp", "o-damp", "damping", (v) => `${Math.round(v * 100)}%`],
    ["s-balance", "o-balance", "balance", (v) =>
      Math.abs(v) < 0.04 ? "centre" : `${v > 0 ? "fwd" : "aft"} ${Math.abs(Math.round(v * 100))}%`],
  ];

  for (const [inputId, outId, key, fmt] of sliders) {
    const input = document.getElementById(inputId) as HTMLInputElement;
    const out = document.getElementById(outId) as HTMLOutputElement;
    const sync = () => {
      const v = Number(input.value);
      out.textContent = fmt(v);
      const patch: Partial<TunnelParams> = {};
      patch[key] = v;
      t.setParams(patch);
      // Where the body hangs changes what the equilibrium even is, so the old
      // samples are no longer a curve for the same system.
      if (key === "balance") trace.clear();
    };
    input.addEventListener("input", sync);
    sync();
  }

  // ── Buttons ────────────────────────────────────────────────────────────
  document.getElementById("preset-teardrop")?.addEventListener("click", () => {
    applyShape(presetPath("teardrop", aspect()));
  });
  document.getElementById("preset-plate")?.addEventListener("click", () => {
    applyShape(presetPath("plate", aspect()));
  });
  document.getElementById("reset-flow")?.addEventListener("click", () => {
    t.resetFlow();
    trace.clear();
  });

  let paused = false;
  pauseBtn.addEventListener("click", () => {
    paused = !paused;
    t.setPaused(paused);
    pauseBtn.textContent = paused ? "Run" : "Pause";
    pauseBtn.setAttribute("aria-pressed", String(paused));
  });

  // ── Readouts ───────────────────────────────────────────────────────────
  // The trace samples at a fraction of the frame rate: the plot is about where
  // the body spends its time, and 60 points a second only piles them up.
  let tick = 0;
  function poll() {
    requestAnimationFrame(poll);
    const r = t.read();
    tick++;
    if (!paused && tick % 4 === 0) {
      trace.push((r.angle * 180) / Math.PI, r.torque);
      trace.draw();
      traceStatus.textContent = trace.status(r.settled);
    }
    if (tick % 8 === 0) {
      const deg = Math.round((r.angle * 180) / Math.PI);
      angleReadout.textContent = r.settled
        ? `settled at ${deg}°`
        : `${deg}°  ·  turning ${r.omega >= 0 ? "ccw" : "cw"}`;
    }
  }

  // Open with a shape already in the flow, so the tunnel is running before
  // anyone has drawn anything.
  applyShape(presetPath("teardrop", aspect()));
  requestAnimationFrame(poll);

  window.addEventListener("resize", () => {
    sizeInk();
    trace.draw();
  });
}

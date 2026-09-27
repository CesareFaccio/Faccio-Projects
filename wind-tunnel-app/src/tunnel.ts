// tunnel.ts
// A vertical wind tunnel with a body immersed in it, free to rotate about its
// own centre and nothing else.
//
// The flow is the same family of solver as the landing page's hero — semi-
// Lagrangian advection, vorticity confinement, a Jacobi pressure projection —
// turned on its side so the air runs bottom to top, with the fixed cylinders
// replaced by one body whose shape someone drew.
//
// ── How the body and the fluid talk to each other ───────────────────────────
//
// The body is an immersed boundary. Rather than carving it out of the grid, we
// let the fluid advect everywhere and then, each step, push the velocity inside
// the body toward what a solid spinning at omega would have there:
//
//     u := mix(u, omega x r, coverage)
//
// Driving it toward `omega x r` rather than toward zero matters. It is the
// correct no-slip condition for a rotating solid, and it is also what makes the
// fluid resist the spin on its own: a body that is turning fast has to drag the
// air around with it, and pays for that in torque. Without it the only thing
// opposing rotation is the damping slider, and the body would happily spin up
// forever at low damping.
//
// The momentum that step removes from the fluid is, by Newton's third law, the
// force the fluid puts on the body:
//
//     f_fluid = coverage * (omega x r - u) / dt        (per unit area, rho = 1)
//     f_body  = -f_fluid
//
// so the torque about the pivot is sum(r x f_body) over the grid. That sum is
// done on the GPU — a force texture, then four box-sum passes down to a single
// texel — and only that one texel is read back per frame. Reading the whole
// field back every frame would cost more than the simulation.
//
// ── Units ───────────────────────────────────────────────────────────────────
//
// "Tunnel units": y runs 0..1 over the tunnel's height, x runs 0..aspect, so
// distances mean the same thing on both axes and a circle is round. The grid's
// cells are square in these units.
//
// Velocity is in the solver's own units, where advection moves a parcel by
// dt * u * texelSize per step; physical speed in tunnel units is u / gridHeight.
// Torque and inertia inherit that scale. None of it is SI, and there is no
// pretence that it is — the sliders are calibrated so that the behaviour is
// right, not so that the numbers are.

import type { Shape } from "./shape";

export interface TunnelParams {
  /** 0..1 — how hard the air blows. */
  wind: number;
  /** 0..1 — resistance to rotation, over and above what the fluid provides. */
  damping: number;
  /** 0..1 — multiplier on the shape's own moment of area. */
  inertia: number;
  /**
   * 0..1 — inverse viscosity, and so the Reynolds number.
   *
   * At 0 the air is thick: the boundary layer stays attached, the wake closes
   * quietly behind the body and it settles quickly. Toward 1 the air thins, the
   * layer separates, and the wake becomes a broad unsteady one shedding
   * vortices off alternate sides, which buffets the body and makes it hunt
   * around its equilibrium rather than sitting on it.
   *
   * There is a ceiling the slider cannot pass: a 180-cell grid has a numerical
   * diffusion of its own, so the top of the range is "as sharp as this grid
   * gets", not "inviscid".
   */
  turbulence: number;
  /**
   * -1..1 — where the body is pinned, along its own long axis, as a fraction of
   * its radius. 0 pins it at the area centroid.
   *
   * This slider is the whole reason the thing behaves the way people expect.
   * A body pinned at its centroid does NOT seek the low-drag orientation: at a
   * small angle of attack the centre of pressure sits ahead of mid-chord, so
   * the moment pushes the angle wider, and the stable resting place is
   * broadside — maximum drag. It is exactly why a dropped card settles flat
   * rather than edge-on. Weathervanes, darts and shuttlecocks self-orient
   * because they are held AHEAD of their centre of pressure, and moving the
   * balance point forward here is the same trick.
   */
  balance: number;
}

export interface TunnelReadout {
  /** Radians, counter-clockwise, relative to the shape as it was drawn. */
  angle: number;
  /** Radians per second. */
  omega: number;
  /** Torque about the pivot, smoothed. Arbitrary but consistent units; the
   *  sign and the zero crossings are what mean anything. There is deliberately
   *  no drag here — see BODY_FORCE_SHADER for why it could not be measured
   *  honestly. */
  torque: number;
  /** True once the body has been turning slowly for a while. */
  settled: boolean;
  /** How many grid cells the fastest part of the flow crosses per step. Above
   *  about 2 the advection stops resolving the flow and starts smearing it. */
  cflCells: number;
  /** Reynolds number on the body's width. Indicative: it counts the grid's own
   *  numerical diffusion, which is an estimate. */
  reynolds: number;
  /** The grid height this device was given, and the sub-steps it runs. */
  grid: number;
  subSteps: number;
  /** What the startup probe measured, in texels shaded per millisecond. */
  throughput: number;
}

export interface TunnelHandle {
  setShape(shape: Shape | null): void;
  setParams(params: Partial<TunnelParams>): void;
  setPaused(paused: boolean): void;
  /** Clears the flow field but keeps the body and its angle. */
  resetFlow(): void;
  /** Runs the solver for a fixed number of steps without waiting for frames.
   *  Used to settle the flow after a change, and to drive the simulation from
   *  a test that cannot rely on the frame loop. */
  advance(steps: number): void;
  /** Places the body at a given angle and stops it dead. */
  setAngle(radians: number): void;
  read(): TunnelReadout;
  destroy(): void;
}

// ── Grid and solver constants ───────────────────────────────────────────────
/** The grid the solver's constants were measured on. Quantities calibrated
 *  against it are normalised back to it rather than re-measured per tier. */
const REFERENCE_GRID = 180;

/**
 * Candidate grid heights, coarsest first.
 *
 * Resolution is the one thing that buys visible turbulence, because the
 * advection scheme's own diffusion falls with cell size — so a finer grid
 * genuinely raises the Reynolds number the tunnel can reach, where asking the
 * slider for less viscosity only saturates against it.
 *
 * It is not cheap. Cells go as the square, and a finer grid also needs a
 * shorter timestep: velocity in cells per second is proportional to the grid
 * height, so the same physical wind crosses more cells per step, and past about
 * two the scheme is smearing again. Sub-stepping restores that and costs
 * another multiple on top. The top tier here is roughly five times the work of
 * the bottom one, which is why the device gets asked first.
 */
const RESOLUTION_TIERS = [180, 240, 320];
/** Smoke is carried on its own finer grid; this is the ratio to the solver's. */
const DYE_SCALE = 420 / 180;
/** Cells of travel per step to stay under. */
const CFL_TARGET = 2.2;

/**
 * Sub-steps per frame for a given grid height. Velocity in cells per second
 * scales with the grid, so a finer grid covers more cells per step for the same
 * physical wind; this splits the frame until the fastest the tunnel can be
 * driven stays inside what the advection resolves.
 */
function subStepsFor(gridHeight: number): number {
  const cellsPerStep = (WIND_MAX * gridHeight) / 60;
  return Math.max(1, Math.ceil(cellsPerStep / CFL_TARGET));
}
/** The force field is summed by repeated 4x4 box passes, so it is a power of
 *  four on a side and the reduction divides exactly at every level. */
const FORCE_RESOLUTION = 256;

// The smoke has to survive the whole height of the tunnel, because the
// interesting part of the picture is the wake ABOVE the body, not the rakes
// below it. At 0.5 a streakline faded under the display's black point before
// it got there and the wake was invisible; at 0.12 there was no sink strong
// enough to balance the injection and the whole frame filled in solid.
const DENSITY_DISSIPATION = 0.28;
const VELOCITY_DISSIPATION = 0.08;
// The pressure field is integrated to get the force on the body, not just used
// to project the velocity, so it has to be a good deal more trustworthy here
// than it does in a purely decorative solver. Warm-starting it at 0.8 let a
// large-scale background build up over many frames and swamp the body-scale
// signal; more iterations and a shorter memory keep each solve close to its own
// answer.
const PRESSURE_DISSIPATION = 0.3;
const PRESSURE_ITERATIONS = 40;

const CURL_CAP = 900;
// A safety net, and — since the turbulence slider became a viscosity rather
// than a vorticity-confinement strength — no longer a working part. Measured
// with this lifted to 420 so the physics had to hold unaided, the peak stays at
// 1.9 to 2.0 cells of travel per advection step across the whole slider. It
// used to reach 6.5, and the clamp was the only thing standing in the way.
// 190 bounds each component, so a diagonal velocity could still reach about 4.5
// cells if anything ever pushed that hard. Nothing currently does.
const SPEED_CAP = 190 / REFERENCE_GRID;


const INLET_WIDTH = 0.05;
const INLET_DYE_RATE = 0.95;
const RAKE_COUNT = 22;

const DISPLAY_GAIN = 1.5;
const DPR = 1;

const WARMUP_STEPS = 80;
const DT = 1 / 60;

// ── Units ───────────────────────────────────────────────────────────────────
// Everything below is in TUNNEL units — tunnel heights per second, tunnel
// heights squared per second — and is converted to the solver's cell units at
// the point of use, by multiplying by the grid height (velocity) or its square
// (viscosity). That indirection is what lets the grid change with the device
// without changing the physics: a wind of 0.7 means the same flow whether it is
// resolved on 180 cells or 380.
//
// The ceiling on wind is set by the advection step, not by taste. Past roughly
// two cells of travel per step the scheme stops resolving the flow and starts
// smearing it, and the pressure field goes with it — measured drag on a disc
// came out NEGATIVE at a higher ceiling, and steady and positive at this one.
const WIND_MIN = 45 / REFERENCE_GRID;
const WIND_MAX = 128 / REFERENCE_GRID;
// Vorticity confinement, off.
//
// It went from being the turbulence slider, to a small fixed corrective, to
// nothing — and the reason it is nothing is that MacCormack advection does the
// job it was standing in for, honestly. Confinement existed to claw back detail
// the first-order scheme smeared away, and it did that by adding energy in
// proportion to the vorticity already present: a positive feedback with no
// equilibrium, which is what ran away every time it was turned up. Measuring
// the advection error and subtracting it removes the smearing at the source, so
// there is nothing left to compensate for.
//
// The vorticity pass is still run, because it is also where the speed ceiling
// is applied; with a strength of zero it costs two passes and adds no energy.
const CURL_STRENGTH = 0;

// Kinematic viscosity, in cells squared per second, at the laminar end of the
// slider. At the other end the slider asks for zero and the flow is left to
// whatever the grid itself imposes.
const VISCOSITY_MAX = 90 / (REFERENCE_GRID * REFERENCE_GRID);
/** Jacobi sweeps for the diffusion solve. Diffusion is a smooth operator and
 *  does not need the convergence the pressure solve does. */
const VISCOSITY_ITERATIONS = 16;
/**
 * No floor. It existed only to give the vorticity confinement something to push
 * against; with the confinement gone there is no energy source to balance, and
 * the top of the slider can ask for genuinely inviscid air. Verified with the
 * safety clamp lifted: the peak holds at about 2 cells of travel per step right
 * across the slider, where it used to reach 6.5.
 */
const VISCOSITY_FLOOR = 0;
/** Below this the explicit solve costs more than it changes. */
const VISCOSITY_MIN = 0.4 / (REFERENCE_GRID * REFERENCE_GRID);
/**
 * The advection scheme's own diffusion, in the same units, as an order-of-
 * magnitude estimate for this grid. It exists whether or not any viscosity is
 * asked for, which is why the Reynolds number shown to the viewer is computed
 * against the SUM of the two and is indicative rather than calibrated: the
 * simulation cannot be made sharper than its own grid, only blunter.
 */
/**
 * The advection scheme's own diffusion, as a coefficient rather than a number:
 * it is proportional to the flow speed and to the cell size, so on a finer grid
 * it genuinely falls. That is the whole reason resolution buys turbulence.
 * Calibrated at the reference grid, and indicative rather than measured — the
 * Reynolds number shown to the viewer inherits that.
 */
const NUMERICAL_VISCOSITY_COEFF = 0.162;

// Both of these come from a measured torque curve rather than a guess, which
// is the only reason the body moves at all: swept through angle with its
// rotation frozen, a plate at mid wind feels torques of roughly 0.1 to 1.6, and
// its second moment about the pivot is about 2.4e-4. The first guess at
// INERTIA_SCALE was 5.2e4 — some sixty times too stiff — so the body sat
// motionless while the torque quietly did nothing.
//
// INERTIA_SCALE is set so the angular acceleration at a typical torque swings a
// plate through a right angle in about a second and a half. DAMPING_MAX is set
// from the stiffness of the same curve near its stable zero crossing
// (dtau/dtheta of about 1.4), so mid-slider lands a little past critical and
// the body settles instead of hunting. The fluid adds damping of its own
// through the omega x r boundary condition, so the true value is a shade
// higher than this alone.
const INERTIA_SCALE = 900;
const INERTIA_MIN = 0.35;
const INERTIA_MAX = 4.0;
const DAMPING_MAX = 5;
/** Ceiling on spin. `omega x r` is written into the velocity field, so an
 *  unbounded omega would breach the advection limit from the inside. Kept low
 *  enough that the rim of the largest sensible body stays under SPEED_CAP. */
const OMEGA_CAP = 5;
/** How far the balance point can slide along the long axis, as a fraction of
 *  the shape's radius. At the extreme the pivot sits right at the shape's tip,
 *  which is what a dart or a weathervane actually does; 0.75 did not move it
 *  far enough ahead of the centre of pressure to change which equilibrium a
 *  plate chose. */
const BALANCE_RANGE = 1.0;
/** The measured torque is a per-frame momentum difference and is inherently
 *  noisy; this is the time constant of the low-pass applied before it is
 *  integrated. Real measurements get smoothed too. */
// About an eight-sample average. The per-sample scatter on the torque is
// roughly 0.15 against signals of 0.6 to 1.6, so this pulls the noise well
// under the signal without making the body sluggish to respond.
const TORQUE_SMOOTHING = 0.12;
const SETTLED_OMEGA = 0.08;
const SETTLED_FRAMES = 70;

const BASE_VERTEX_SHADER = `#version 300 es
precision highp float;
in vec2 aPosition;
out vec2 vUv;
out vec2 vL;
out vec2 vR;
out vec2 vT;
out vec2 vB;
uniform vec2 texelSize;
void main () {
  vUv = aPosition * 0.5 + 0.5;
  vL = vUv - vec2(texelSize.x, 0.0);
  vR = vUv + vec2(texelSize.x, 0.0);
  vT = vUv + vec2(0.0, texelSize.y);
  vB = vUv - vec2(0.0, texelSize.y);
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

/** Shared by every pass that needs to know where the body is, so the fluid,
 *  the force measurement and the picture on screen can never disagree about
 *  it. */
const BODY_GLSL = `
uniform sampler2D uBody;
uniform vec2 uPivot;
uniform vec2 uPivotInBody;
uniform float uAngle;
uniform float uExtent;
uniform float uAspect;
uniform float uHasBody;

// Offset from the PIVOT, in tunnel units. Torque is taken about this point and
// a spinning body's velocity field is built from it.
vec2 bodyOffset (vec2 uv) {
  return vec2((uv.x - uPivot.x) * uAspect, uv.y - uPivot.y);
}

float bodyCoverage (vec2 uv) {
  if (uHasBody < 0.5) return 0.0;
  vec2 q = bodyOffset(uv);
  float c = cos(uAngle), s = sin(uAngle);
  // Rotating by -angle takes a tunnel point back into the body's own frame,
  // which is the frame the mask was drawn in.
  vec2 r = vec2(c * q.x + s * q.y, -s * q.x + c * q.y);
  // The mask is centred on the area centroid, but the body hangs from the
  // pivot, so shift back by where the pivot sits inside the body.
  vec2 p = (r + uPivotInBody) / uExtent;
  if (abs(p.x) >= 1.0 || abs(p.y) >= 1.0) return 0.0;
  return texture(uBody, p * 0.5 + 0.5).r;
}`;

const CLEAR_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv; uniform sampler2D uTexture; uniform float value;
out vec4 fragColor;
void main () { fragColor = value * texture(uTexture, vUv); }`;

/**
 * Semi-Lagrangian advection: look back along the velocity to find where this
 * parcel came from, and take its value.
 *
 * `texelSize` here is the VELOCITY grid's, never the target's, and that
 * distinction was a real bug. Velocity is carried in cells per second of the
 * simulation grid, so the distance a parcel travels is dt * u / simGridHeight
 * whatever field is being carried. Passing the dye grid's texel size instead —
 * 420 cells against the simulation's 180 — advected the smoke at 43% of the
 * speed of the fluid it was supposed to be tracing.
 */
const ADVECTION_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 texelSize;
uniform float dt;
uniform float dissipation;
out vec4 fragColor;
void main () {
  vec2 coord = vUv - dt * texture(uVelocity, vUv).xy * texelSize;
  vec4 result = texture(uSource, coord);
  fragColor = result / (1.0 + dissipation * dt);
}`;

/**
 * The MacCormack correction, which is the whole reason this solver can show a
 * wake at all.
 *
 * Plain semi-Lagrangian advection is first-order: every step it interpolates
 * between grid values, and that interpolation is a low-pass filter. Applied
 * sixty times a second it behaves exactly like a viscosity — one nobody asked
 * for, which on a 180-cell grid was large enough to swamp any viscosity the
 * slider set and hold the whole tunnel in the laminar regime.
 *
 * The trick is to measure that error rather than model it. Advect forward, then
 * advect the result BACKWARD: with a perfect scheme you would land exactly
 * where you started, so whatever you are off by is the error, and half of it
 * subtracts out:
 *
 *     phi_new = phi_forward + (phi_original - phi_back_and_forth) / 2
 *
 * That is second-order accurate and costs two extra passes rather than the four
 * times the cells a doubled grid would.
 *
 * The clamp at the end is not optional. The correction can overshoot into
 * values that exist nowhere in the source field, which on a velocity field
 * means invented energy; limiting it to the range of the four cells the forward
 * trace interpolated between is what keeps it stable.
 */
const MACCORMACK_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform sampler2D uForward;
uniform sampler2D uBackward;
uniform vec2 texelSize;
uniform vec2 uSourceTexel;
uniform float dt;
uniform float dissipation;
out vec4 fragColor;
void main () {
  vec4 original = texture(uSource, vUv);
  vec4 forward = texture(uForward, vUv);
  vec4 back = texture(uBackward, vUv);
  vec4 corrected = forward + 0.5 * (original - back);

  // The four source cells the forward trace interpolated between.
  vec2 coord = vUv - dt * texture(uVelocity, vUv).xy * texelSize;
  vec2 base = (floor(coord / uSourceTexel - 0.5) + 0.5) * uSourceTexel;
  vec4 s00 = texture(uSource, base);
  vec4 s10 = texture(uSource, base + vec2(uSourceTexel.x, 0.0));
  vec4 s01 = texture(uSource, base + vec2(0.0, uSourceTexel.y));
  vec4 s11 = texture(uSource, base + uSourceTexel);
  vec4 lo = min(min(s00, s10), min(s01, s11));
  vec4 hi = max(max(s00, s10), max(s01, s11));

  fragColor = clamp(corrected, lo, hi) / (1.0 + dissipation * dt);
}`;

// Bottom inlet, top outlet. Mode 0 sets velocity (a boundary condition, so it
// relaxes toward a target); mode 1 lays down the smoke rakes.
const INLET_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv;
uniform sampler2D uTarget;
uniform float uMode;
uniform float uSpeed;
uniform float uAmount;
uniform float uRakes;
uniform float uWidth;
uniform float uTime;
out vec4 fragColor;
void main () {
  vec4 base = texture(uTarget, vUv);
  float band = smoothstep(uWidth, 0.0, vUv.y);
  if (uMode < 0.5) {
    // A slow wander across the inlet. Without it the wake sheds in perfect
    // lockstep and a symmetric body sits balanced on a knife edge forever.
    float shear = 0.05 * sin(vUv.x * 7.0 + uTime * 0.5)
                + 0.03 * sin(vUv.x * 13.0 - uTime * 0.31);
    fragColor = vec4(mix(base.xy, vec2(uSpeed * shear, uSpeed), band), 0.0, 1.0);
  } else {
    float rakes = pow(0.5 + 0.5 * cos(vUv.x * uRakes * 6.2831853), 6.0);
    fragColor = vec4(base.x + band * rakes * uAmount, 0.0, 0.0, 1.0);
  }
}`;

/**
 * One Jacobi sweep of implicit viscous diffusion, solving
 *
 *     (I - nu * dt * laplacian) u_new = u_old
 *
 * This is the term the "turbulence" slider actually moves, and it is the only
 * honest place to put it. What was there before — vorticity confinement — is
 * not a physical effect at all: it is a numerical corrector that pushes
 * vorticity back toward local maxima to undo the smearing of a coarse advection
 * scheme. Turning it up does not make a flow more turbulent, it injects energy
 * at the grid scale, which is exactly why it ran away.
 *
 * Viscosity is the real knob. It sets the Reynolds number, and the Reynolds
 * number is what decides whether the boundary layer stays attached and the wake
 * closes quietly behind the body, or separates into a broad unsteady wake that
 * sheds vortices alternately off each side.
 *
 * The solve is implicit because explicit diffusion needs nu*dt/dx^2 <= 1/4,
 * which at this timestep caps nu at about 15 — well below the laminar end of
 * the range. Velocities here are already in cells per second and the grid
 * spacing is one cell, so nu is in cells squared per second and no unit
 * conversion is needed.
 */
const VISCOSITY_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv; in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform float alpha;
uniform float rBeta;
out vec4 fragColor;
void main () {
  vec2 L = texture(uVelocity, vL).xy;
  vec2 R = texture(uVelocity, vR).xy;
  vec2 T = texture(uVelocity, vT).xy;
  vec2 B = texture(uVelocity, vB).xy;
  vec2 b = texture(uSource, vUv).xy;
  fragColor = vec4((L + R + T + B + alpha * b) * rBeta, 0.0, 1.0);
}`;

const DIVERGENCE_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv; in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
uniform sampler2D uVelocity;
out vec4 fragColor;
void main () {
  float L = texture(uVelocity, vL).x;
  float R = texture(uVelocity, vR).x;
  float T = texture(uVelocity, vT).y;
  float B = texture(uVelocity, vB).y;
  vec2 C = texture(uVelocity, vUv).xy;
  // Free-slip walls left and right: mirror the normal component.
  if (vL.x < 0.0) { L = -C.x; }
  if (vR.x > 1.0) { R = -C.x; }
  // The bottom is a prescribed inflow, so the clamped sample is already the
  // value we want. The top is an open outflow — zero gradient, so air leaves
  // instead of being reflected back down into the tunnel.
  if (vT.y > 1.0) { T = C.y; }
  fragColor = vec4(0.5 * (R - L + T - B), 0.0, 0.0, 1.0);
}`;

const CURL_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
uniform sampler2D uVelocity;
out vec4 fragColor;
void main () {
  float L = texture(uVelocity, vL).y;
  float R = texture(uVelocity, vR).y;
  float T = texture(uVelocity, vT).x;
  float B = texture(uVelocity, vB).x;
  fragColor = vec4(0.5 * (R - L - T + B), 0.0, 0.0, 1.0);
}`;

const VORTICITY_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv; in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform float curl;
uniform float curlCap;
uniform float speedCap;
uniform float dt;
out vec4 fragColor;
void main () {
  float L = texture(uCurl, vL).x;
  float R = texture(uCurl, vR).x;
  float T = texture(uCurl, vT).x;
  float B = texture(uCurl, vB).x;
  float C = texture(uCurl, vUv).x;
  vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
  force /= length(force) + 0.0001;
  // Confinement adds energy in proportion to the vorticity already present, so
  // it is a positive feedback loop and needs a ceiling.
  force *= clamp(curl * C, -curlCap, curlCap);
  force.y *= -1.0;
  vec2 velocity = texture(uVelocity, vUv).xy + force * dt;
  fragColor = vec4(clamp(velocity, -speedCap, speedCap), 0.0, 1.0);
}`;

const PRESSURE_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv; in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
out vec4 fragColor;
void main () {
  float L = texture(uPressure, vL).x;
  float R = texture(uPressure, vR).x;
  float T = texture(uPressure, vT).x;
  float B = texture(uPressure, vB).x;
  float divergence = texture(uDivergence, vUv).x;
  float pressure = (L + R + B + T - divergence) * 0.25;
  // Pinning pressure to zero at the outflow is what actually lets mass leave;
  // the zero-gradient velocity condition alone is not enough.
  if (vT.y > 1.0) { pressure = 0.0; }
  fragColor = vec4(pressure, 0.0, 0.0, 1.0);
}`;

const GRADIENT_SUBTRACT_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv; in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
uniform float speedCap;
out vec4 fragColor;
void main () {
  float L = texture(uPressure, vL).x;
  float R = texture(uPressure, vR).x;
  float T = texture(uPressure, vT).x;
  float B = texture(uPressure, vB).x;
  vec2 velocity = texture(uVelocity, vUv).xy - vec2(R - L, T - B);
  // The same ceiling as the vorticity pass, applied again here. Without it the
  // projection could hand the advection a field faster than the clamp allows —
  // measured at 4.3 cells per step against a ceiling meant to be 3.2 — and it
  // is the field coming OUT of the projection that gets advected.
  fragColor = vec4(clamp(velocity, -speedCap, speedCap), 0.0, 1.0);
}`;

/** Drives the velocity inside the body toward that of a solid spinning at
 *  omega. `uOmegaSim` is omega already converted into solver velocity units. */
const BODY_APPLY_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv;
uniform sampler2D uVelocity;
uniform float uOmegaSim;
out vec4 fragColor;
${BODY_GLSL}
void main () {
  vec2 u = texture(uVelocity, vUv).xy;
  float cov = bodyCoverage(vUv);
  vec2 r = bodyOffset(vUv);
  vec2 target = uOmegaSim * vec2(-r.y, r.x);
  fragColor = vec4(mix(u, target, cov), 0.0, 1.0);
}`;

/**
 * The torque the fluid puts on the body, per cell.
 *
 * The obvious measurement — the momentum the forcing step takes out of the
 * fluid — is nearly useless at steady state, and it is worth saying why. Once
 * the flow has settled, the velocity inside the body has ALREADY been driven to
 * zero on previous steps, so each new step removes only the little that
 * advected in since: the number tends to zero while the real force does not.
 * The momentum never left through that step. It left through the pressure
 * projection, which is what pushes the oncoming air around the body.
 *
 * So the force is taken from the pressure field. For a body immersed in a
 * smoothed indicator function the surface integral becomes a volume one:
 *
 *     F = -closed_integral(p * n_out) dS  =  integral(p * grad(coverage)) dA
 *
 * because coverage rises inward, so grad(coverage) is the inward normal times
 * the surface delta. Only the boundary ramp contributes, which is where a
 * pressure force lives. An additive constant in p contributes nothing, which is
 * just as well, since a Poisson solve only pins pressure up to one.
 *
 * Only the MOMENT of that force is used, never the force itself, and the
 * distinction is not fussiness. A large-scale pressure gradient across the
 * tunnel adds -grad(p) * V to the force — measured on a disc it swamped the
 * real signal and flipped its sign — but it adds no moment about the centroid,
 * because the resultant of a uniform body force acts at the centroid itself.
 * Torque is immune to it; drag was not, and three attempts at correcting drag
 * (the raw integral, a background-gradient subtraction, and a downstream wake
 * rake) all failed validation on a disc, whose drag must be positive, steady
 * and independent of angle. Torque passed the matching test — a disc's must be
 * zero at every angle — with a standard deviation of 0.003 against plate
 * torques of 0.1 to 1.6, so torque is what this reports.
 */
const BODY_FORCE_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
uniform vec2 uStep;
uniform float uGrid;
uniform float uDt;
out vec4 fragColor;
${BODY_GLSL}
void main () {
  vec2 r = bodyOffset(vUv);

  // The grid's cells are square in tunnel units, so one factor of uGrid
  // converts both derivatives at once.
  float cR = bodyCoverage(vUv + vec2(uStep.x, 0.0));
  float cL = bodyCoverage(vUv - vec2(uStep.x, 0.0));
  float cT = bodyCoverage(vUv + vec2(0.0, uStep.y));
  float cB = bodyCoverage(vUv - vec2(0.0, uStep.y));
  vec2 gradCov = 0.5 * uGrid * vec2(cR - cL, cT - cB);
  // The projection's pressure is a potential with units of velocity x length;
  // dividing by dt puts it back into force per unit area.
  float p = texture(uPressure, vUv).x / uDt;
  vec2 fPressure = p * gradCov;

  // Green carries the local speed, which the reduction maxes rather than sums.
  // The advection step moves a parcel dt*u cells, so this number divided by 60
  // is how many cells the fastest part of the flow jumps per step — the single
  // most useful thing to know about whether the solver is still resolving
  // anything or just smearing.
  float speed = length(texture(uVelocity, vUv).xy);

  fragColor = vec4(r.x * fPressure.y - r.y * fPressure.x, speed, 0.0, 1.0);
}`;

/** Clears the smoke out of the body's interior. Without it the dye advects
 *  straight through the solid and re-emerges above it, which reads as the body
 *  leaking even though the velocity field is doing the right thing. */
const DYE_MASK_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv;
uniform sampler2D uTexture;
out vec4 fragColor;
${BODY_GLSL}
void main () {
  float d = texture(uTexture, vUv).x;
  fragColor = vec4(d * (1.0 - bodyCoverage(vUv)), 0.0, 0.0, 1.0);
}`;

/** One level of a 4x4 box sum. Run enough times and the whole field collapses
 *  into a single texel holding its total. */
const REDUCE_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv;
uniform sampler2D uTexture;
uniform vec2 uSourceTexel;
out vec4 fragColor;
void main () {
  // vUv is the centre of a destination texel, which covers exactly 4x4 source
  // texels; step back to the first of them.
  vec2 base = vUv - 1.5 * uSourceTexel;
  vec4 acc = vec4(0.0);
  for (int y = 0; y < 4; y++) {
    for (int x = 0; x < 4; x++) {
      vec4 s = texture(uTexture, base + vec2(float(x), float(y)) * uSourceTexel);
      // Red accumulates (it is an integral); green takes the maximum (it is a
      // peak). Both survive the whole chain down to one texel.
      acc.r += s.r;
      acc.g = max(acc.g, s.g);
    }
  }
  fragColor = acc;
}`;

const DISPLAY_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv;
uniform sampler2D uTexture;
uniform float uGain;
out vec4 fragColor;
${BODY_GLSL}

// The 4x4 ordered (Bayer) matrix a 1-bit Mac used to fake grey.
float bayer4 (vec2 pixel) {
  int x = int(mod(pixel.x, 4.0));
  int y = int(mod(pixel.y, 4.0));
  int i = y * 4 + x;
  float m[16] = float[16](
     0.0,  8.0,  2.0, 10.0,
    12.0,  4.0, 14.0,  6.0,
     3.0, 11.0,  1.0,  9.0,
    15.0,  7.0, 13.0,  5.0
  );
  return (m[i] + 0.5) / 16.0;
}

void main () {
  float d = texture(uTexture, vUv).x;
  float v = 1.0 - exp(-uGain * max(d, 0.0));
  float ink = clamp((v - 0.14) / 0.78, 0.0, 1.0);
  float lit = step(bayer4(gl_FragCoord.xy), ink);
  float shade = 1.0 - lit;

  float cov = bodyCoverage(vUv);
  // The body is solid black with a white rule around it, so it never merges
  // into a patch of dark smoke sitting against it.
  if (cov > 0.5) shade = 0.0;
  if (abs(cov - 0.5) < 0.16) shade = 1.0;

  // A small cross marks the pivot, which is the one point that never moves.
  vec2 q = bodyOffset(vUv);
  if (uHasBody > 0.5) {
    float armX = step(abs(q.x), 0.022) * step(abs(q.y), 0.0035);
    float armY = step(abs(q.y), 0.022) * step(abs(q.x), 0.0035);
    if (armX + armY > 0.0) shade = 1.0;
  }

  fragColor = vec4(vec3(shade), 1.0);
}`;

interface Measured {
  torque: number;
  /** Largest speed anywhere in the field, in solver units. */
  peakSpeed: number;
}

interface FBO {
  texture: WebGLTexture;
  fbo: WebGLFramebuffer;
  width: number;
  height: number;
  texelSizeX: number;
  texelSizeY: number;
  attach(id: number): number;
}

interface DoubleFBO {
  read: FBO;
  write: FBO;
  swap(): void;
  width: number;
  height: number;
  texelSizeX: number;
  texelSizeY: number;
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`tunnel: shader compile failed — ${gl.getShaderInfoLog(shader)}`);
  }
  return shader;
}

class Program {
  readonly program: WebGLProgram;
  readonly uniforms: Record<string, WebGLUniformLocation | null> = {};
  constructor(private gl: WebGL2RenderingContext, vertexShader: WebGLShader, fragmentSource: string) {
    const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
    this.program = gl.createProgram()!;
    gl.attachShader(this.program, vertexShader);
    gl.attachShader(this.program, fragmentShader);
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error(`tunnel: program link failed — ${gl.getProgramInfoLog(this.program)}`);
    }
    const count = gl.getProgramParameter(this.program, gl.ACTIVE_UNIFORMS) as number;
    for (let i = 0; i < count; i++) {
      const name = gl.getActiveUniform(this.program, i)!.name;
      this.uniforms[name] = gl.getUniformLocation(this.program, name);
    }
  }
  bind() {
    this.gl.useProgram(this.program);
  }
}

export function startTunnel(canvas: HTMLCanvasElement): TunnelHandle | null {
  const gl = canvas.getContext("webgl2", {
    alpha: false,
    depth: false,
    stencil: false,
    antialias: false,
    preserveDrawingBuffer: false,
  });
  if (!gl) return null;
  if (!gl.getExtension("EXT_color_buffer_float")) return null;
  const linear = gl.getExtension("OES_texture_float_linear") ? gl.LINEAR : gl.NEAREST;

  const vertexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
  const indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(0);

  function blit(target: FBO | null) {
    if (target === null) {
      gl!.viewport(0, 0, gl!.drawingBufferWidth, gl!.drawingBufferHeight);
      gl!.bindFramebuffer(gl!.FRAMEBUFFER, null);
    } else {
      gl!.viewport(0, 0, target.width, target.height);
      gl!.bindFramebuffer(gl!.FRAMEBUFFER, target.fbo);
    }
    gl!.drawElements(gl!.TRIANGLES, 6, gl!.UNSIGNED_SHORT, 0);
  }

  const textures: WebGLTexture[] = [];
  const framebuffers: WebGLFramebuffer[] = [];

  function createFBO(w: number, h: number, internalFormat: number, format: number, type: number, param: number): FBO {
    const texture = gl!.createTexture()!;
    textures.push(texture);
    gl!.activeTexture(gl!.TEXTURE0);
    gl!.bindTexture(gl!.TEXTURE_2D, texture);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_MIN_FILTER, param);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_MAG_FILTER, param);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_WRAP_S, gl!.CLAMP_TO_EDGE);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_WRAP_T, gl!.CLAMP_TO_EDGE);
    gl!.texImage2D(gl!.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

    const fbo = gl!.createFramebuffer()!;
    framebuffers.push(fbo);
    gl!.bindFramebuffer(gl!.FRAMEBUFFER, fbo);
    gl!.framebufferTexture2D(gl!.FRAMEBUFFER, gl!.COLOR_ATTACHMENT0, gl!.TEXTURE_2D, texture, 0);
    gl!.viewport(0, 0, w, h);
    gl!.clear(gl!.COLOR_BUFFER_BIT);

    return {
      texture,
      fbo,
      width: w,
      height: h,
      texelSizeX: 1 / w,
      texelSizeY: 1 / h,
      attach(id: number) {
        gl!.activeTexture(gl!.TEXTURE0 + id);
        gl!.bindTexture(gl!.TEXTURE_2D, texture);
        return id;
      },
    };
  }

  function createDoubleFBO(w: number, h: number, internalFormat: number, format: number, type: number, param: number): DoubleFBO {
    let fbo1 = createFBO(w, h, internalFormat, format, type, param);
    let fbo2 = createFBO(w, h, internalFormat, format, type, param);
    return {
      width: w,
      height: h,
      texelSizeX: 1 / w,
      texelSizeY: 1 / h,
      get read() {
        return fbo1;
      },
      get write() {
        return fbo2;
      },
      swap() {
        const t = fbo1;
        fbo1 = fbo2;
        fbo2 = t;
      },
    };
  }

  // Size the drawing buffer before the grids are derived from it, or they get
  // computed from the canvas's default 300x150 and the flow comes out stretched.
  function sizeCanvas(): boolean {
    const w = Math.round(canvas.clientWidth * DPR);
    const h = Math.round(canvas.clientHeight * DPR);
    if (w > 0 && h > 0 && (canvas.width !== w || canvas.height !== h)) {
      canvas.width = w;
      canvas.height = h;
      return true;
    }
    return false;
  }
  sizeCanvas();

  function canvasAspect() {
    return gl!.drawingBufferWidth / gl!.drawingBufferHeight || 1;
  }

  /** Square cells in tunnel units: the grid is stretched to match the canvas. */
  function gridFor(resolution: number) {
    const aspect = canvasAspect();
    return aspect >= 1
      ? { width: Math.round(resolution * aspect), height: Math.round(resolution) }
      : { width: Math.round(resolution), height: Math.round(resolution / aspect) };
  }

  /**
   * Rough GPU throughput, in texels shaded per millisecond.
   *
   * Timed around a batch of full-screen passes with a one-pixel readback at the
   * end, because readPixels is what forces the queue to actually finish —
   * without it this would be timing command submission, not work. The first
   * batch is discarded: it pays for shader compilation and pipeline warm-up.
   *
   * This is an estimate, and deliberately a conservative one. It distinguishes
   * an integrated laptop GPU from a discrete or Apple Silicon one, which is all
   * the tiering needs; it is not a benchmark.
   */
  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, BASE_VERTEX_SHADER);

  function probeThroughput(): number {
    const size = 512;
    // Its own program, because this runs before the solver's are built, and its
    // own try/catch, because a device that cannot complete the probe should
    // quietly get the smallest grid rather than no simulation at all.
    try {
      // The pressure solve is 40 of the ~70 passes in a step, so it is what
      // the estimate should be built on. A one-tap shader measures bandwidth
      // this workload never sees and flatters a slow device badly — a software
      // rasteriser came out fast enough for the top tier.
      const probeProgram = new Program(gl!, vertexShader, PRESSURE_SHADER);
      // Two targets, alternating. Rendering into the same texture being sampled
      // is undefined behaviour, and drivers are entitled to short-circuit it —
      // this probe reported an impossible three billion texels a second on a
      // software rasteriser until it ping-ponged properly.
      const probeA = createFBO(size, size, gl!.RGBA32F, gl!.RGBA, gl!.FLOAT, gl!.NEAREST);
      const probeB = createFBO(size, size, gl!.RGBA32F, gl!.RGBA, gl!.FLOAT, gl!.NEAREST);
      const scratch = new Float32Array(4);
      const run = (passes: number) => {
        probeProgram.bind();
        gl!.uniform2f(probeProgram.uniforms.texelSize!, 1 / size, 1 / size);
        let src = probeA;
        let dst = probeB;
        for (let i = 0; i < passes; i++) {
          gl!.uniform1i(probeProgram.uniforms.uPressure!, src.attach(0));
          gl!.uniform1i(probeProgram.uniforms.uDivergence!, src.attach(1));
          blit(dst);
          const t = src;
          src = dst;
          dst = t;
        }
        // readPixels is what forces the queue to finish. Without it this would
        // be timing command submission, not work.
        gl!.bindFramebuffer(gl!.FRAMEBUFFER, src.fbo);
        gl!.readPixels(0, 0, 1, 1, gl!.RGBA, gl!.FLOAT, scratch);
      };

      run(24); // discarded: pays for shader compilation and pipeline warm-up
      const start = performance.now();
      const passes = 180;
      run(passes);
      const elapsed = Math.max(performance.now() - start, 0.01);
      return (passes * size * size) / elapsed;
    } catch {
      return 0;
    }
  }

  /**
   * Picks the finest grid whose estimated frame cost fits the budget.
   *
   * A simulation step is around 70 full-grid passes (most of them the pressure
   * solve), plus three on the dye grid, which has DYE_SCALE^2 times the cells.
   */
  function chooseResolution(throughput: number): number {
    if (!isFinite(throughput) || throughput <= 0) return RESOLUTION_TIERS[0];

    const aspect = canvasAspect();
    const budgetMs = 7;
    let chosen = RESOLUTION_TIERS[0];
    for (const tier of RESOLUTION_TIERS) {
      const cells = tier * tier * Math.max(aspect, 1 / aspect);
      const passes = 70 + 3 * DYE_SCALE * DYE_SCALE;
      const steps = subStepsFor(tier);
      const costMs = (cells * passes * steps) / throughput;
      if (costMs <= budgetMs) chosen = tier;
    }
    return chosen;
  }

  const measuredThroughput = probeThroughput();
  const simResolution = chooseResolution(measuredThroughput);
  const subSteps = subStepsFor(simResolution);
  const simGrid = gridFor(simResolution);
  const dyeGrid = gridFor(Math.round(simResolution * DYE_SCALE));

  const velocity = createDoubleFBO(simGrid.width, simGrid.height, gl.RG16F, gl.RG, gl.HALF_FLOAT, linear);
  const dye = createDoubleFBO(dyeGrid.width, dyeGrid.height, gl.R16F, gl.RED, gl.HALF_FLOAT, linear);
  const divergence = createFBO(simGrid.width, simGrid.height, gl.R16F, gl.RED, gl.HALF_FLOAT, gl.NEAREST);
  const curlField = createFBO(simGrid.width, simGrid.height, gl.R16F, gl.RED, gl.HALF_FLOAT, gl.NEAREST);
  const viscositySource = createFBO(simGrid.width, simGrid.height, gl.RG16F, gl.RG, gl.HALF_FLOAT, gl.NEAREST);
  // The forward and back-and-forth passes MacCormack needs, one pair per field.
  const velForward = createFBO(simGrid.width, simGrid.height, gl.RG16F, gl.RG, gl.HALF_FLOAT, linear);
  const velBack = createFBO(simGrid.width, simGrid.height, gl.RG16F, gl.RG, gl.HALF_FLOAT, linear);
  const dyeForward = createFBO(dyeGrid.width, dyeGrid.height, gl.R16F, gl.RED, gl.HALF_FLOAT, linear);
  const dyeBack = createFBO(dyeGrid.width, dyeGrid.height, gl.R16F, gl.RED, gl.HALF_FLOAT, linear);
  // Full float, and linear because the force pass samples it on its own grid
  // rather than this one. Half precision costs about three decimal digits, and
  // the quantity being integrated is a small difference across the body's
  // boundary ramp sitting on top of a much larger field — exactly the case
  // where those digits are the signal.
  const pressure = createDoubleFBO(simGrid.width, simGrid.height, gl.R32F, gl.RED, gl.FLOAT, linear);

  // The force field and its reduction chain: 256 -> 64 -> 16 -> 4 -> 1.
  const forceField = createFBO(FORCE_RESOLUTION, FORCE_RESOLUTION, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST);
  const reduceChain: FBO[] = [];
  for (let size = FORCE_RESOLUTION / 4; size >= 1; size /= 4) {
    reduceChain.push(createFBO(size, size, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST));
  }
  // Every measurement comes back through this one texel, so if the driver
  // cannot render to it there is no point starting: better to report no
  // simulation than to integrate whatever readPixels happens to return.
  const finalLevel = reduceChain[reduceChain.length - 1];
  gl.bindFramebuffer(gl.FRAMEBUFFER, finalLevel.fbo);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) return null;

  // The body mask lives in its own frame and is only re-uploaded when the shape
  // changes; rotating it is the sampler's job, not the CPU's.
  const bodyTexture = gl.createTexture()!;
  textures.push(bodyTexture);
  gl.bindTexture(gl.TEXTURE_2D, bodyTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([0]));

  const clearProgram = new Program(gl, vertexShader, CLEAR_SHADER);
  const advectionProgram = new Program(gl, vertexShader, ADVECTION_SHADER);
  const macCormackProgram = new Program(gl, vertexShader, MACCORMACK_SHADER);
  const inletProgram = new Program(gl, vertexShader, INLET_SHADER);
  const divergenceProgram = new Program(gl, vertexShader, DIVERGENCE_SHADER);
  const curlProgram = new Program(gl, vertexShader, CURL_SHADER);
  const vorticityProgram = new Program(gl, vertexShader, VORTICITY_SHADER);
  const viscosityProgram = new Program(gl, vertexShader, VISCOSITY_SHADER);
  const pressureProgram = new Program(gl, vertexShader, PRESSURE_SHADER);
  const gradientProgram = new Program(gl, vertexShader, GRADIENT_SUBTRACT_SHADER);
  const bodyApplyProgram = new Program(gl, vertexShader, BODY_APPLY_SHADER);
  const bodyForceProgram = new Program(gl, vertexShader, BODY_FORCE_SHADER);
  const dyeMaskProgram = new Program(gl, vertexShader, DYE_MASK_SHADER);
  const reduceProgram = new Program(gl, vertexShader, REDUCE_SHADER);
  const displayProgram = new Program(gl, vertexShader, DISPLAY_SHADER);

  // ── Body state ───────────────────────────────────────────────────────────
  let shape: Shape | null = null;
  let angle = 0;
  let omega = 0;
  let torqueSmoothed = 0;
  let peakSpeed = 0;
  let settledFor = 0;
  let time = 0;
  let paused = false;

  // Defaults set from a measured sweep rather than taste. Released at the same
  // angle each time, a plate settles broadside with the pivot at its centroid
  // (|sin| to the flow of 0.15) and edge-on with the pivot at its tip (0.95);
  // halfway out is already firmly in the second regime (0.87), and it keeps the
  // pivot visibly inside the shape rather than hanging it off the nose. Damping
  // at 0.6 settles instead of pendulumming — lower it and the body flutters,
  // which is equally real.
  const params: TunnelParams = {
    wind: 0.55,
    damping: 0.6,
    inertia: 0.45,
    turbulence: 0.6,
    balance: 0.5,
  };
  const readBuffer = new Float32Array(4);

  /** Where the pivot sits inside the body, in the body's own frame: along the
   *  long axis, a fraction of the radius out from the area centroid. */
  function pivotInBody(): { x: number; y: number } {
    if (!shape) return { x: 0, y: 0 };
    const d = params.balance * BALANCE_RANGE * shape.radius;
    return { x: shape.axis.x * d, y: shape.axis.y * d };
  }

  function setBodyUniforms(p: Program) {
    const aspect = canvasAspect();
    const d = pivotInBody();
    gl!.uniform1i(p.uniforms.uBody!, 7);
    gl!.activeTexture(gl!.TEXTURE7);
    gl!.bindTexture(gl!.TEXTURE_2D, bodyTexture);
    // The pivot is pinned to the middle of the tunnel: the body turns, and
    // nothing else about where it sits ever changes.
    gl!.uniform2f(p.uniforms.uPivot!, 0.5, 0.5);
    gl!.uniform2f(p.uniforms.uPivotInBody!, d.x, d.y);
    gl!.uniform1f(p.uniforms.uAngle!, angle);
    gl!.uniform1f(p.uniforms.uExtent!, shape ? shape.extent : 1);
    gl!.uniform1f(p.uniforms.uAspect!, aspect);
    gl!.uniform1f(p.uniforms.uHasBody!, shape ? 1 : 0);
  }

  /** Tunnel heights per second. */
  function windPhysical() {
    return WIND_MIN + (WIND_MAX - WIND_MIN) * params.wind;
  }

  /** The same speed in the solver's units: cells per second. */
  function windSpeed() {
    return windPhysical() * simGrid.height;
  }

  function runInlet(target: DoubleFBO, mode: 0 | 1) {
    inletProgram.bind();
    gl!.uniform1i(inletProgram.uniforms.uTarget!, target.read.attach(0));
    gl!.uniform1f(inletProgram.uniforms.uMode!, mode);
    gl!.uniform1f(inletProgram.uniforms.uSpeed!, windSpeed());
    gl!.uniform1f(inletProgram.uniforms.uAmount!, INLET_DYE_RATE * DT);
    gl!.uniform1f(inletProgram.uniforms.uRakes!, RAKE_COUNT);
    gl!.uniform1f(inletProgram.uniforms.uWidth!, INLET_WIDTH);
    gl!.uniform1f(inletProgram.uniforms.uTime!, time);
    blit(target.write);
    target.swap();
  }

  /** Kinematic viscosity the slider is currently asking for. Squared so the
   *  interesting, nearly-inviscid end of the range gets most of the travel. */
  /** Tunnel heights squared per second. */
  function viscosityPhysical() {
    const t = 1 - params.turbulence;
    return VISCOSITY_FLOOR + (VISCOSITY_MAX - VISCOSITY_FLOOR) * t * t;
  }

  /** The same, in cells squared per second. */
  function viscosity() {
    return viscosityPhysical() * simGrid.height * simGrid.height;
  }

  /** What the grid imposes whatever the slider asks for: proportional to speed
   *  and to cell size, so it halves when the grid doubles. */
  function numericalViscosity() {
    return (NUMERICAL_VISCOSITY_COEFF * windPhysical()) / simGrid.height;
  }

  /** Reynolds number, on the body's own width, against the total viscosity —
   *  what the slider adds plus what the grid imposes regardless. */
  function reynolds() {
    if (!shape) return 0;
    // All three in tunnel units, so the grid cancels out of the ratio as it
    // should — except through the numerical viscosity, where it belongs.
    return (
      (windPhysical() * 2 * shape.radius) /
      (viscosityPhysical() + numericalViscosity())
    );
  }

  function diffuse(dt: number) {
    const nu = viscosity();
    if (viscosityPhysical() < VISCOSITY_MIN) return;

    // Keep the pre-diffusion field: every Jacobi sweep needs it as the
    // right-hand side, not just the previous iterate.
    clearProgram.bind();
    gl!.uniform1i(clearProgram.uniforms.uTexture!, velocity.read.attach(0));
    gl!.uniform1f(clearProgram.uniforms.value!, 1);
    blit(viscositySource);

    const alpha = 1 / (nu * dt);
    viscosityProgram.bind();
    gl!.uniform2f(viscosityProgram.uniforms.texelSize!, velocity.texelSizeX, velocity.texelSizeY);
    gl!.uniform1f(viscosityProgram.uniforms.alpha!, alpha);
    gl!.uniform1f(viscosityProgram.uniforms.rBeta!, 1 / (4 + alpha));
    gl!.uniform1i(viscosityProgram.uniforms.uSource!, viscositySource.attach(1));
    for (let i = 0; i < VISCOSITY_ITERATIONS; i++) {
      gl!.uniform1i(viscosityProgram.uniforms.uVelocity!, velocity.read.attach(0));
      blit(velocity.write);
      velocity.swap();
    }
  }

  /**
   * Advects `field` through the velocity field, second-order. `forward` and
   * `back` are scratch targets matching the field's own grid.
   */
  function advect(
    field: DoubleFBO,
    forward: FBO,
    back: FBO,
    dissipation: number,
    dt: number,
  ) {
    advectionProgram.bind();
    // Always the velocity grid's texel size: that is what sets how far a parcel
    // moves, whichever field is being carried.
    gl!.uniform2f(advectionProgram.uniforms.texelSize!, velocity.texelSizeX, velocity.texelSizeY);
    gl!.uniform1i(advectionProgram.uniforms.uVelocity!, velocity.read.attach(0));
    gl!.uniform1f(advectionProgram.uniforms.dissipation!, 0);

    gl!.uniform1i(advectionProgram.uniforms.uSource!, field.read.attach(1));
    gl!.uniform1f(advectionProgram.uniforms.dt!, dt);
    blit(forward);

    // Back along the same velocity field. Where this lands short of where we
    // started is the scheme's own error.
    gl!.uniform1i(advectionProgram.uniforms.uSource!, forward.attach(1));
    gl!.uniform1f(advectionProgram.uniforms.dt!, -dt);
    blit(back);

    macCormackProgram.bind();
    gl!.uniform2f(macCormackProgram.uniforms.texelSize!, velocity.texelSizeX, velocity.texelSizeY);
    gl!.uniform2f(macCormackProgram.uniforms.uSourceTexel!, field.texelSizeX, field.texelSizeY);
    gl!.uniform1i(macCormackProgram.uniforms.uVelocity!, velocity.read.attach(0));
    gl!.uniform1i(macCormackProgram.uniforms.uSource!, field.read.attach(1));
    gl!.uniform1i(macCormackProgram.uniforms.uForward!, forward.attach(2));
    gl!.uniform1i(macCormackProgram.uniforms.uBackward!, back.attach(3));
    gl!.uniform1f(macCormackProgram.uniforms.dt!, dt);
    gl!.uniform1f(macCormackProgram.uniforms.dissipation!, dissipation);
    blit(field.write);
    field.swap();
  }

  function applyBody() {
    bodyApplyProgram.bind();
    gl!.uniform1i(bodyApplyProgram.uniforms.uVelocity!, velocity.read.attach(0));
    gl!.uniform1f(bodyApplyProgram.uniforms.uOmegaSim!, omega * simGrid.height);
    setBodyUniforms(bodyApplyProgram);
    blit(velocity.write);
    velocity.swap();
  }

  /**
   * Renders the per-cell force the fluid puts on the body, sums it on the GPU,
   * and reads back the single texel that results.
   *
   * Returns torque and drag already scaled out of the solver's grid units:
   * the sum is over FORCE_RESOLUTION^2 cells each standing for the same slice
   * of the tunnel, and solver velocity is gridHeight times tunnel speed.
   */
  function measureForce(): Measured {
    if (!shape) return { torque: 0, peakSpeed: 0 };

    bodyForceProgram.bind();
    gl!.uniform1i(bodyForceProgram.uniforms.uPressure!, pressure.read.attach(1));
    gl!.uniform1i(bodyForceProgram.uniforms.uVelocity!, velocity.read.attach(2));
    gl!.uniform2f(bodyForceProgram.uniforms.uStep!, velocity.texelSizeX, velocity.texelSizeY);
    gl!.uniform1f(bodyForceProgram.uniforms.uGrid!, simGrid.height);
    gl!.uniform1f(bodyForceProgram.uniforms.uDt!, DT);
    setBodyUniforms(bodyForceProgram);
    blit(forceField);

    reduceProgram.bind();
    let source = forceField;
    for (const level of reduceChain) {
      gl!.uniform1i(reduceProgram.uniforms.uTexture!, source.attach(0));
      gl!.uniform2f(reduceProgram.uniforms.uSourceTexel!, source.texelSizeX, source.texelSizeY);
      blit(level);
      source = level;
    }

    gl!.bindFramebuffer(gl!.FRAMEBUFFER, source.fbo);
    gl!.readPixels(0, 0, 1, 1, gl!.RGBA, gl!.FLOAT, readBuffer);

    const aspect = canvasAspect();
    // Each force texel stands for this much tunnel area.
    const cellArea = (aspect / FORCE_RESOLUTION) * (1 / FORCE_RESOLUTION);
    // Solver velocity is gridHeight x tunnel speed, and force is linear in it.
    // The measured torque scales with the grid: the projection's pressure is in
    // solver velocity units, which are proportional to the grid height, and the
    // coverage gradient carries another factor of it. Left alone, the same
    // shape would turn harder on a machine that earned a finer grid. Normalised
    // back to the grid the rotation constants were calibrated on.
    const scale = (cellArea / simGrid.height) * (REFERENCE_GRID / simGrid.height);
    return { torque: readBuffer[0] * scale, peakSpeed: readBuffer[1] };
  }

  function step(dt: number) {
    gl!.disable(gl!.BLEND);

    // ── Velocity ──────────────────────────────────────────────────────────
    gl!.viewport(0, 0, simGrid.width, simGrid.height);
    curlProgram.bind();
    gl!.uniform2f(curlProgram.uniforms.texelSize!, velocity.texelSizeX, velocity.texelSizeY);
    gl!.uniform1i(curlProgram.uniforms.uVelocity!, velocity.read.attach(0));
    blit(curlField);

    vorticityProgram.bind();
    gl!.uniform2f(vorticityProgram.uniforms.texelSize!, velocity.texelSizeX, velocity.texelSizeY);
    gl!.uniform1i(vorticityProgram.uniforms.uVelocity!, velocity.read.attach(0));
    gl!.uniform1i(vorticityProgram.uniforms.uCurl!, curlField.attach(1));
    gl!.uniform1f(vorticityProgram.uniforms.curl!, CURL_STRENGTH);
    gl!.uniform1f(vorticityProgram.uniforms.curlCap!, CURL_CAP);
    gl!.uniform1f(vorticityProgram.uniforms.speedCap!, SPEED_CAP * simGrid.height);
    gl!.uniform1f(vorticityProgram.uniforms.dt!, dt);
    blit(velocity.write);
    velocity.swap();

    // A small fixed bleed. This is a linear drag, not viscosity — it damps every
    // scale equally — so it is kept low and left alone; the diffusion step below
    // is what the slider moves.
    advect(velocity, velForward, velBack, VELOCITY_DISSIPATION, dt);

    diffuse(dt);
    runInlet(velocity, 0);
    applyBody();

    // ── Projection ────────────────────────────────────────────────────────
    divergenceProgram.bind();
    gl!.uniform2f(divergenceProgram.uniforms.texelSize!, velocity.texelSizeX, velocity.texelSizeY);
    gl!.uniform1i(divergenceProgram.uniforms.uVelocity!, velocity.read.attach(0));
    blit(divergence);

    clearProgram.bind();
    gl!.uniform1i(clearProgram.uniforms.uTexture!, pressure.read.attach(0));
    gl!.uniform1f(clearProgram.uniforms.value!, PRESSURE_DISSIPATION);
    blit(pressure.write);
    pressure.swap();

    pressureProgram.bind();
    gl!.uniform2f(pressureProgram.uniforms.texelSize!, velocity.texelSizeX, velocity.texelSizeY);
    gl!.uniform1i(pressureProgram.uniforms.uDivergence!, divergence.attach(0));
    for (let i = 0; i < PRESSURE_ITERATIONS; i++) {
      gl!.uniform1i(pressureProgram.uniforms.uPressure!, pressure.read.attach(1));
      blit(pressure.write);
      pressure.swap();
    }

    gradientProgram.bind();
    gl!.uniform2f(gradientProgram.uniforms.texelSize!, velocity.texelSizeX, velocity.texelSizeY);
    gl!.uniform1i(gradientProgram.uniforms.uPressure!, pressure.read.attach(0));
    gl!.uniform1i(gradientProgram.uniforms.uVelocity!, velocity.read.attach(1));
    gl!.uniform1f(gradientProgram.uniforms.speedCap!, SPEED_CAP * simGrid.height);
    blit(velocity.write);
    velocity.swap();

    // The force is read here, after the projection, because the pressure field
    // it integrates is what that projection just produced.
    const measured = measureForce();

    // Projection leaks a little flow back into the solid, so the condition is
    // re-imposed afterwards.
    gl!.viewport(0, 0, simGrid.width, simGrid.height);
    applyBody();

    // ── Smoke ─────────────────────────────────────────────────────────────
    gl!.viewport(0, 0, dyeGrid.width, dyeGrid.height);
    advect(dye, dyeForward, dyeBack, DENSITY_DISSIPATION, dt);
    runInlet(dye, 1);

    dyeMaskProgram.bind();
    gl!.uniform1i(dyeMaskProgram.uniforms.uTexture!, dye.read.attach(0));
    setBodyUniforms(dyeMaskProgram);
    blit(dye.write);
    dye.swap();

    // ── Rotation ──────────────────────────────────────────────────────────
    if (shape) {
      torqueSmoothed += (measured.torque - torqueSmoothed) * TORQUE_SMOOTHING;
      peakSpeed = measured.peakSpeed;

      const inertiaMul = INERTIA_MIN + (INERTIA_MAX - INERTIA_MIN) * params.inertia;
      // A bigger shape is genuinely harder to turn: the slider multiplies the
      // shape's own second moment of area rather than replacing it. Moving the
      // balance point off the centroid adds the parallel-axis term, so a body
      // pinned near its nose is correctly harder to swing than one pinned in
      // the middle.
      const d = pivotInBody();
      const aboutPivot = shape.polarMoment + shape.area * (d.x * d.x + d.y * d.y);
      const I = Math.max(aboutPivot * INERTIA_SCALE * inertiaMul, 1e-6);
      const c = DAMPING_MAX * params.damping;
      omega += ((torqueSmoothed - c * omega) / I) * dt;
      omega = Math.max(-OMEGA_CAP, Math.min(OMEGA_CAP, omega));
      angle += omega * dt;
      if (angle > Math.PI) angle -= 2 * Math.PI;
      if (angle < -Math.PI) angle += 2 * Math.PI;

      settledFor = Math.abs(omega) < SETTLED_OMEGA ? settledFor + 1 : 0;
    }

    time += dt;
  }

  function render() {
    displayProgram.bind();
    gl!.uniform1i(displayProgram.uniforms.uTexture!, dye.read.attach(0));
    gl!.uniform1f(displayProgram.uniforms.uGain!, DISPLAY_GAIN);
    setBodyUniforms(displayProgram);
    blit(null);
  }

  function clearField() {
    for (const target of [velocity, dye, pressure]) {
      clearProgram.bind();
      gl!.viewport(0, 0, target.width, target.height);
      gl!.uniform1i(clearProgram.uniforms.uTexture!, target.read.attach(0));
      gl!.uniform1f(clearProgram.uniforms.value!, 0);
      blit(target.write);
      target.swap();
    }
    torqueSmoothed = 0;
    settledFor = 0;
  }

  // ── Frame loop ───────────────────────────────────────────────────────────
  let rafId = 0;

  function frame() {
    rafId = requestAnimationFrame(frame);
    if (paused) {
      render();
      return;
    }
    // The grids are sized once from the canvas, so resizing is only safe
    // because the tunnel's aspect ratio is pinned in CSS: the drawing buffer
    // changes resolution, never shape, and the flow carries on undisturbed.
    sizeCanvas();
    // One frame of simulated time, however many sub-steps this grid needs.
    for (let i = 0; i < subSteps; i++) step(DT / subSteps);
    render();
  }

  for (let i = 0; i < WARMUP_STEPS * subSteps; i++) step(DT / subSteps);
  render();
  rafId = requestAnimationFrame(frame);

  return {
    setShape(next) {
      shape = next;
      angle = 0;
      omega = 0;
      torqueSmoothed = 0;
      settledFor = 0;
      if (next) {
        gl!.bindTexture(gl!.TEXTURE_2D, bodyTexture);
        gl!.pixelStorei(gl!.UNPACK_ALIGNMENT, 1);
        // The mask was drawn on a 2D canvas, whose first row is its top; the
        // tunnel's y runs the other way, so it is flipped on upload.
        gl!.pixelStorei(gl!.UNPACK_FLIP_Y_WEBGL, true);
        gl!.texImage2D(
          gl!.TEXTURE_2D, 0, gl!.R8, next.maskSize, next.maskSize, 0,
          gl!.RED, gl!.UNSIGNED_BYTE, next.mask,
        );
        gl!.pixelStorei(gl!.UNPACK_FLIP_Y_WEBGL, false);
      }
    },
    setParams(next) {
      Object.assign(params, next);
    },
    setPaused(next) {
      paused = next;
    },
    resetFlow() {
      clearField();
      omega = 0;
      for (let i = 0; i < WARMUP_STEPS * subSteps; i++) step(DT / subSteps);
    },
    advance(steps) {
      for (let i = 0; i < steps * subSteps; i++) step(DT / subSteps);
      render();
    },
    setAngle(radians) {
      angle = radians;
      omega = 0;
      torqueSmoothed = 0;
      settledFor = 0;
    },
    read() {
      return {
        angle,
        omega,
        torque: torqueSmoothed,
        cflCells: (peakSpeed * DT) / subSteps,
        reynolds: reynolds(),
        grid: simGrid.height,
        subSteps,
        throughput: measuredThroughput,
        settled: settledFor > SETTLED_FRAMES,
      };
    },
    destroy() {
      cancelAnimationFrame(rafId);
      for (const t of textures) gl!.deleteTexture(t);
      for (const f of framebuffers) gl!.deleteFramebuffer(f);
      gl!.deleteBuffer(vertexBuffer);
      gl!.deleteBuffer(indexBuffer);
    },
  };
}

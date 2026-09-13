// fluid.ts
// A compact GPU fluid simulation (incompressible Navier-Stokes, solved on the
// GPU with WebGL2) used as the hero background. Each frame it advects a
// velocity field through itself, applies vorticity confinement to keep the
// small eddies from being smeared away by numerical diffusion, projects the
// velocity back to divergence-free with a Jacobi pressure solve, and advects a
// dye field through the result. Pointer movement injects velocity and dye.
//
// The field is driven by a fixed inflow along the bottom edge — a row of small
// jets, like air rising through a perforated grate — rather than by random
// impulses. Pointer movement disturbs the plumes on top of that.
//
// Everything is deliberately dependency-free and runs at a lower internal
// resolution than the canvas — the dye field is what you see, and it is
// upsampled by the GPU for free when drawn.

export interface FluidHandle {
  destroy(): void;
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

// Tuning. SIM_RESOLUTION drives cost almost entirely; DYE_RESOLUTION only
// affects how crisp the smoke looks.
const SIM_RESOLUTION = 128;
const DYE_RESOLUTION = 512;
// The primary density control now that the sponge only covers the top strip.
// Applied uniformly, so a plume thins as it climbs — which reads as a column
// dispersing, rather than the hard horizontal line a wide sponge draws.
const DENSITY_DISSIPATION = 1.90; // how fast the dye fades
// Absorbing layer near the outflow. Still needed even though the top boundary
// is genuinely open: the open boundary lets momentum leave, but advection never
// carries dye out — the top row's backtrace just resamples from below — so
// without this, dye piles against the ceiling. Keep the ramp wide enough that
// it doesn't draw a visible horizontal line where it starts to bite.
const DYE_HEIGHT_FADE = 2.6;
// Hard ceiling on dye density, so the field can never wash out to flat white.
const DYE_CEILING = 1.15;
const VELOCITY_DISSIPATION = 0.11; // how fast the motion dies down
// Upward acceleration proportional to local dye density.
//
// This is what gathers the flow into vertical columns rather than a diffuse
// haze — without it the hero reads as fine grain. It is safe to raise only
// because BUOYANCY_CAP bounds the feedback; see the vorticity shader.
const BUOYANCY = 260;
// Dye density above which no further lift is added.
const BUOYANCY_CAP = 0.30;
const PRESSURE_DISSIPATION = 0.8;
const PRESSURE_ITERATIONS = 20;
// Vorticity confinement re-injects the small eddies that numerical diffusion
// smears away. High values give churning, curly smoke; that reads as grain
// rather than flow, so a directed river wants this low.
const CURL = 7;
// Gaussian falloff denominator, in normalised-UV units squared. Small changes
// here matter a lot: this is what separates thin wisps from billowing smoke.
const SPLAT_RADIUS = 0.30;
const SPLAT_FORCE = 5200;

// Tuning notes. These interact, the failure modes are easy to hit, and all of
// them look fine for the first ten seconds — always check against a long run.
//
// WHAT EACH KNOB ACTUALLY DOES, learned the hard way:
//
//   Overall density  — DENSITY_DISSIPATION, then INLET_DYE_RATE. Dissipation
//     is the better handle: it is applied uniformly, so a plume thins as it
//     climbs. Response is smooth but sub-linear; expect to move it by a lot.
//
//   Character        — CURL. Vorticity confinement re-injects small eddies.
//     High (20+) gives churning, curly smoke that reads as grain; low (<10)
//     gives smoother, more directed flow.
//
//   Vertical reach   — the open top boundary (divergence + pressure shaders)
//     does most of this. A closed lid forces recirculation, which caps how far
//     any column can run, no matter how hard it is driven.
//
// TRAPS:
//
//   Speed. Advection backtraces dt*velocity*texelSize per step, so on a
//   128-wide grid at 60fps a velocity near 900 moves a fifth of the screen in
//   one step. Past the CFL limit the scheme stops advecting and starts
//   scrambling, and the result looks WEAKER, not stronger.
//
//   Buoyancy feedback. Lift proportional to dye density is a loop, and
//   uncapped it makes the system bistable — the field either dies or fills the
//   screen, with no usable range between. BUOYANCY_CAP bounds it. If you ever
//   see a knob flip between "nothing" and "everything" with no middle, suspect
//   this rather than reaching for finer steps.
//
//   Dye never leaves through the open top. The boundary lets momentum out, but
//   the advection backtrace at the top row just resamples from below, so the
//   dye budget still has to balance via dissipation and DYE_HEIGHT_FADE.
//
// The values below were checked against a 900-step run (~22s simulated). To
// judge a change, screenshot that long-run state and measure its mean
// luminance rather than eyeballing it — the eye is a poor judge here, and the
// first ten seconds tell you nothing. Change one thing at a time.

// ── Grate inflow ────────────────────────────────────────────────────────────
// Rates are per second and multiplied by the frame's dt, so the look does not
// change with frame rate. INLET_BAND is the grate's thickness in UV units;
// jets are spaced roughly every INLET_JET_SPACING_PX across the canvas.
const INLET_BAND = 0.032;
const INLET_JET_SPACING_PX = 95;
const INLET_VELOCITY_RATE = 560;
const INLET_DYE_RATE = 1.10;
// Sim steps run before the first paint, so the hero opens with plumes already
// risen rather than an empty black frame that fills in over a few seconds.
const WARMUP_STEPS = 320;
const WARMUP_DT = 1 / 40;
// Ceiling on warm-up cost, so a slow device gets a shorter warm-up rather than
// a stalled first paint.
const WARMUP_BUDGET_MS = 140;

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

const CLEAR_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv; uniform sampler2D uTexture; uniform float value;
out vec4 fragColor;
void main () { fragColor = value * texture(uTexture, vUv); }`;

const SPLAT_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv;
uniform sampler2D uTarget;
uniform float aspectRatio;
uniform vec3 color;
uniform vec2 point;
uniform float radius;
out vec4 fragColor;
void main () {
  vec2 p = vUv - point.xy;
  p.x *= aspectRatio;
  vec3 splat = exp(-dot(p, p) / radius) * color;
  vec3 base = texture(uTarget, vUv).xyz;
  fragColor = vec4(base + splat, 1.0);
}`;

const ADVECTION_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 texelSize;
uniform float dt;
uniform float dissipation;
uniform float maxValue;
uniform float heightFade;
out vec4 fragColor;
void main () {
  vec2 coord = vUv - dt * texture(uVelocity, vUv).xy * texelSize;
  vec4 result = texture(uSource, coord);
  // The simulation box is closed, so without an outflow the dye injected at
  // the grate simply accumulates until the field washes out. Fading it towards
  // the top stands in for smoke leaving the frame, and keeps the plumes
  // bottom-weighted, which is what rising smoke actually looks like.
  float decay = 1.0 + (dissipation + heightFade * smoothstep(0.70, 1.0, vUv.y)) * dt;
  // Ceiling: semi-Lagrangian advection isn't mass-conserving, and a backtrace
  // that clamps at a boundary re-samples the source row, so a steady inflow
  // can compound without bound. Effectively disabled for velocity.
  fragColor = min(result / decay, vec4(maxValue));
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
  // Free-slip walls on the sides and floor: mirror the normal component.
  if (vL.x < 0.0) { L = -C.x; }
  if (vR.x > 1.0) { R = -C.x; }
  if (vB.y < 0.0) { B = -C.y; }
  // The top is an open outflow — zero gradient, so fluid can leave. Mirroring
  // here (as a wall does) is what forced every rising plume to turn back down:
  // in a sealed box the pressure solve requires anything going up to come back
  // down somewhere, which caps how far a column can run.
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
uniform sampler2D uDye;
uniform float curl;
uniform float buoyancy;
uniform float buoyancyCap;
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
  force *= curl * C;
  force.y *= -1.0;
  // Denser dye is "warmer" and rises. Sampled from the dye field, which is a
  // finer grid than the velocity field — normalised UVs make that a non-issue.
  //
  // The lift is capped. Uncapped, this is a feedback loop — dye lifts faster,
  // which draws up more dye — and it makes the whole system bistable: the
  // field either dies out or runs away and fills the screen, with no usable
  // range between. Capping keeps the organising effect that gathers the flow
  // into columns while bounding the loop.
  force.y += buoyancy * min(texture(uDye, vUv).x, buoyancyCap);
  vec2 velocity = texture(uVelocity, vUv).xy;
  velocity += force * dt;
  velocity = min(max(velocity, -1000.0), 1000.0);
  fragColor = vec4(velocity, 0.0, 1.0);
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
  // Open outflow: pinning pressure to zero along the top row is what actually
  // lets mass leave the domain. The zero-gradient velocity condition in the
  // divergence pass is not enough on its own.
  if (vT.y > 1.0) { pressure = 0.0; }
  fragColor = vec4(pressure, 0.0, 0.0, 1.0);
}`;

const GRADIENT_SUBTRACT_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv; in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
out vec4 fragColor;
void main () {
  float L = texture(uPressure, vL).x;
  float R = texture(uPressure, vR).x;
  float T = texture(uPressure, vT).x;
  float B = texture(uPressure, vB).x;
  vec2 velocity = texture(uVelocity, vUv).xy;
  velocity -= vec2(R - L, T - B);
  fragColor = vec4(velocity, 0.0, 1.0);
}`;

// The dye field is monochrome smoke; this maps its density onto the page's
// palette and adds a vignette so the quote in the middle stays readable.
// A row of jets along the bottom edge. Used twice per frame — once to add
// upward velocity, once to add dye — selected by uMode.
const INLET_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv;
uniform sampler2D uTarget;
uniform float uJets;
uniform float uBand;
uniform float uTime;
uniform float uMode;
uniform float uAmount;
out vec4 fragColor;

float hash(float n) { return fract(sin(n * 127.1) * 43758.5453123); }

void main () {
  // A thin band just above the bottom edge — not flush with it, so the
  // advection backtrace above the grate doesn't clamp into the source row.
  float dy = vUv.y - 0.015;
  float band = exp(-(dy * dy) / (uBand * uBand));
  // Sharp, evenly spaced holes rather than a continuous slot.
  float holes = pow(0.5 + 0.5 * cos(vUv.x * uJets * 6.28318530718), 10.0);
  // Each hole breathes at its own rate, so the plumes stay unsteady instead of
  // settling into a static, obviously synthetic pattern.
  float h = hash(floor(vUv.x * uJets));
  float wobble = 0.62 + 0.5 * sin(uTime * (0.5 + h * 1.4) + h * 6.2832);

  float flow = band * holes * wobble;
  vec3 base = texture(uTarget, vUv).xyz;
  vec3 add;
  if (uMode < 0.5) {
    // Mostly upward, with a little sway so neighbouring plumes interact
    // instead of rising as independent parallel columns.
    float sway = sin(uTime * (0.35 + h) + h * 3.14159) * 0.10;
    add = vec3(sway * uAmount * flow, uAmount * flow, 0.0);
  } else {
    add = vec3(uAmount * flow);
  }
  fragColor = vec4(base + add, 1.0);
}`;

const DISPLAY_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv;
uniform sampler2D uTexture;
out vec4 fragColor;
void main () {
  vec3 c = texture(uTexture, vUv).rgb;
  float d = clamp(max(max(c.r, c.g), c.b), 0.0, 1.0);
  d = pow(d, 0.85);
  vec3 deep = vec3(0.031, 0.035, 0.043);
  vec3 mid  = vec3(0.325, 0.396, 0.427);
  vec3 hot  = vec3(0.706, 0.784, 0.776);
  vec3 col = mix(deep, mid, smoothstep(0.0, 0.34, d));
  col = mix(col, hot, smoothstep(0.34, 0.95, d));
  vec2 q = vUv - 0.5;
  col *= 1.0 - 0.75 * dot(q, q);
  fragColor = vec4(col, 1.0);
}`;

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`fluid: shader compile failed — ${gl.getShaderInfoLog(shader)}`);
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
      throw new Error(`fluid: program link failed — ${gl.getProgramInfoLog(this.program)}`);
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

/**
 * Boots the simulation onto `canvas`. Returns a handle whose destroy() releases
 * every GL resource and listener, or null when WebGL2 / float render targets
 * are unavailable — the caller's cue to leave its CSS fallback in place.
 */
export function startFluid(canvas: HTMLCanvasElement): FluidHandle | null {
  const gl = canvas.getContext("webgl2", {
    alpha: false,
    depth: false,
    stencil: false,
    antialias: false,
    preserveDrawingBuffer: false,
  });
  if (!gl) return null;

  // Rendering *into* float textures is an extension even under WebGL2, and the
  // velocity/pressure fields need it. Without it there is no simulation to run.
  if (!gl.getExtension("EXT_color_buffer_float")) return null;
  const linearFilteringSupported = !!gl.getExtension("OES_texture_float_linear");
  const filtering = linearFilteringSupported ? gl.LINEAR : gl.NEAREST;

  // ── Fullscreen quad shared by every pass ────────────────────────────────
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

  const createdTextures: WebGLTexture[] = [];
  const createdFramebuffers: WebGLFramebuffer[] = [];

  function createFBO(w: number, h: number, internalFormat: number, format: number, type: number, param: number): FBO {
    const texture = gl!.createTexture()!;
    createdTextures.push(texture);
    gl!.activeTexture(gl!.TEXTURE0);
    gl!.bindTexture(gl!.TEXTURE_2D, texture);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_MIN_FILTER, param);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_MAG_FILTER, param);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_WRAP_S, gl!.CLAMP_TO_EDGE);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_WRAP_T, gl!.CLAMP_TO_EDGE);
    gl!.texImage2D(gl!.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

    const fbo = gl!.createFramebuffer()!;
    createdFramebuffers.push(fbo);
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
        const temp = fbo1;
        fbo1 = fbo2;
        fbo2 = temp;
      },
    };
  }

  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, BASE_VERTEX_SHADER);
  const clearProgram = new Program(gl, vertexShader, CLEAR_SHADER);
  const splatProgram = new Program(gl, vertexShader, SPLAT_SHADER);
  const advectionProgram = new Program(gl, vertexShader, ADVECTION_SHADER);
  const divergenceProgram = new Program(gl, vertexShader, DIVERGENCE_SHADER);
  const curlProgram = new Program(gl, vertexShader, CURL_SHADER);
  const vorticityProgram = new Program(gl, vertexShader, VORTICITY_SHADER);
  const pressureProgram = new Program(gl, vertexShader, PRESSURE_SHADER);
  const gradientSubtractProgram = new Program(gl, vertexShader, GRADIENT_SUBTRACT_SHADER);
  const inletProgram = new Program(gl, vertexShader, INLET_SHADER);
  const displayProgram = new Program(gl, vertexShader, DISPLAY_SHADER);

  // Aspect-correct simulation grids, so eddies stay round on a wide viewport.
  function getResolution(resolution: number) {
    const aspect = gl!.drawingBufferWidth / gl!.drawingBufferHeight || 1;
    const ratio = aspect < 1 ? 1 / aspect : aspect;
    const min = Math.round(resolution);
    const max = Math.round(resolution * ratio);
    return aspect > 1 ? { width: max, height: min } : { width: min, height: max };
  }

  // Phones get a coarser grid and fewer solver iterations. The dye field is
  // upsampled to the canvas either way, so at this size the difference is
  // hard to see — but the saving in fill rate and battery is not.
  const compact = Math.min(window.innerWidth, window.innerHeight) < 760;
  const pressureIterations = compact ? 14 : PRESSURE_ITERATIONS;
  const simRes = getResolution(compact ? 96 : SIM_RESOLUTION);
  const dyeRes = getResolution(compact ? 320 : DYE_RESOLUTION);

  const dye = createDoubleFBO(dyeRes.width, dyeRes.height, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, filtering);
  const velocity = createDoubleFBO(simRes.width, simRes.height, gl.RG16F, gl.RG, gl.HALF_FLOAT, filtering);
  const divergence = createFBO(simRes.width, simRes.height, gl.R16F, gl.RED, gl.HALF_FLOAT, gl.NEAREST);
  const curlFBO = createFBO(simRes.width, simRes.height, gl.R16F, gl.RED, gl.HALF_FLOAT, gl.NEAREST);
  const pressure = createDoubleFBO(simRes.width, simRes.height, gl.R16F, gl.RED, gl.HALF_FLOAT, gl.NEAREST);

  function splat(x: number, y: number, dx: number, dy: number, intensity: number) {
    splatProgram.bind();
    gl!.uniform1i(splatProgram.uniforms.uTarget!, velocity.read.attach(0));
    gl!.uniform1f(splatProgram.uniforms.aspectRatio!, canvas.width / canvas.height);
    gl!.uniform2f(splatProgram.uniforms.point!, x, y);
    gl!.uniform3f(splatProgram.uniforms.color!, dx, dy, 0);
    // Widen the splat on wide viewports so it stays circular on screen rather
    // than being squashed by the aspect correction applied in the shader.
    const aspect = canvas.width / canvas.height;
    const radius = (SPLAT_RADIUS / 100) * (aspect > 1 ? aspect : 1);
    gl!.uniform1f(splatProgram.uniforms.radius!, radius);
    blit(velocity.write);
    velocity.swap();

    gl!.uniform1i(splatProgram.uniforms.uTarget!, dye.read.attach(0));
    gl!.uniform3f(splatProgram.uniforms.color!, intensity, intensity, intensity);
    blit(dye.write);
    dye.swap();
  }

  function step(dt: number) {
    gl!.disable(gl!.BLEND);

    curlProgram.bind();
    gl!.uniform2f(curlProgram.uniforms.texelSize!, velocity.texelSizeX, velocity.texelSizeY);
    gl!.uniform1i(curlProgram.uniforms.uVelocity!, velocity.read.attach(0));
    blit(curlFBO);

    vorticityProgram.bind();
    gl!.uniform2f(vorticityProgram.uniforms.texelSize!, velocity.texelSizeX, velocity.texelSizeY);
    gl!.uniform1i(vorticityProgram.uniforms.uVelocity!, velocity.read.attach(0));
    gl!.uniform1i(vorticityProgram.uniforms.uCurl!, curlFBO.attach(1));
    gl!.uniform1i(vorticityProgram.uniforms.uDye!, dye.read.attach(2));
    gl!.uniform1f(vorticityProgram.uniforms.curl!, CURL);
    gl!.uniform1f(vorticityProgram.uniforms.buoyancy!, BUOYANCY);
    gl!.uniform1f(vorticityProgram.uniforms.buoyancyCap!, BUOYANCY_CAP);
    gl!.uniform1f(vorticityProgram.uniforms.dt!, dt);
    blit(velocity.write);
    velocity.swap();

    divergenceProgram.bind();
    gl!.uniform2f(divergenceProgram.uniforms.texelSize!, velocity.texelSizeX, velocity.texelSizeY);
    gl!.uniform1i(divergenceProgram.uniforms.uVelocity!, velocity.read.attach(0));
    blit(divergence);

    // Decay the previous frame's pressure rather than starting from zero —
    // a warm start means far fewer Jacobi iterations are needed to converge.
    clearProgram.bind();
    gl!.uniform1i(clearProgram.uniforms.uTexture!, pressure.read.attach(0));
    gl!.uniform1f(clearProgram.uniforms.value!, PRESSURE_DISSIPATION);
    blit(pressure.write);
    pressure.swap();

    pressureProgram.bind();
    gl!.uniform2f(pressureProgram.uniforms.texelSize!, velocity.texelSizeX, velocity.texelSizeY);
    gl!.uniform1i(pressureProgram.uniforms.uDivergence!, divergence.attach(0));
    for (let i = 0; i < pressureIterations; i++) {
      gl!.uniform1i(pressureProgram.uniforms.uPressure!, pressure.read.attach(1));
      blit(pressure.write);
      pressure.swap();
    }

    gradientSubtractProgram.bind();
    gl!.uniform2f(gradientSubtractProgram.uniforms.texelSize!, velocity.texelSizeX, velocity.texelSizeY);
    gl!.uniform1i(gradientSubtractProgram.uniforms.uPressure!, pressure.read.attach(0));
    gl!.uniform1i(gradientSubtractProgram.uniforms.uVelocity!, velocity.read.attach(1));
    blit(velocity.write);
    velocity.swap();

    advectionProgram.bind();
    gl!.uniform2f(advectionProgram.uniforms.texelSize!, velocity.texelSizeX, velocity.texelSizeY);
    gl!.uniform1i(advectionProgram.uniforms.uVelocity!, velocity.read.attach(0));
    gl!.uniform1i(advectionProgram.uniforms.uSource!, velocity.read.attach(0));
    gl!.uniform1f(advectionProgram.uniforms.dt!, dt);
    gl!.uniform1f(advectionProgram.uniforms.dissipation!, VELOCITY_DISSIPATION);
    gl!.uniform1f(advectionProgram.uniforms.maxValue!, 1e6);
    gl!.uniform1f(advectionProgram.uniforms.heightFade!, 0);
    blit(velocity.write);
    velocity.swap();

    gl!.uniform1i(advectionProgram.uniforms.uVelocity!, velocity.read.attach(0));
    gl!.uniform1i(advectionProgram.uniforms.uSource!, dye.read.attach(1));
    gl!.uniform1f(advectionProgram.uniforms.dissipation!, DENSITY_DISSIPATION);
    gl!.uniform1f(advectionProgram.uniforms.maxValue!, DYE_CEILING);
    gl!.uniform1f(advectionProgram.uniforms.heightFade!, DYE_HEIGHT_FADE);
    blit(dye.write);
    dye.swap();
  }

  function render() {
    displayProgram.bind();
    gl!.uniform1i(displayProgram.uniforms.uTexture!, dye.read.attach(0));
    blit(null);
  }

  // ── Pointer input ────────────────────────────────────────────────────────
  let pointerActive = false;
  let lastX = 0;
  let lastY = 0;

  function toSimCoords(clientX: number, clientY: number) {
    const rect = canvas.getBoundingClientRect();
    return { x: (clientX - rect.left) / rect.width, y: 1 - (clientY - rect.top) / rect.height };
  }

  function onPointerMove(e: PointerEvent) {
    const { x, y } = toSimCoords(e.clientX, e.clientY);
    if (!pointerActive) {
      pointerActive = true;
      lastX = x;
      lastY = y;
      return;
    }
    const dx = (x - lastX) * SPLAT_FORCE;
    const dy = (y - lastY) * SPLAT_FORCE;
    lastX = x;
    lastY = y;
    if (Math.abs(dx) < 0.01 && Math.abs(dy) < 0.01) return;
    splat(x, y, dx, dy, 0.4);
  }

  function onPointerLeave() {
    pointerActive = false;
  }

  window.addEventListener("pointermove", onPointerMove, { passive: true });
  window.addEventListener("pointerleave", onPointerLeave, { passive: true });

  // ── Grate inflow ─────────────────────────────────────────────────────────
  // Injects upward velocity and dye through a row of holes along the bottom
  // edge, every frame. This is what keeps the hero moving on its own.
  let jetCount = 20;
  function inlet(dt: number, timeSeconds: number) {
    inletProgram.bind();
    gl!.uniform1f(inletProgram.uniforms.uJets!, jetCount);
    gl!.uniform1f(inletProgram.uniforms.uBand!, INLET_BAND);
    gl!.uniform1f(inletProgram.uniforms.uTime!, timeSeconds);

    gl!.uniform1f(inletProgram.uniforms.uMode!, 0);
    gl!.uniform1f(inletProgram.uniforms.uAmount!, INLET_VELOCITY_RATE * dt);
    gl!.uniform1i(inletProgram.uniforms.uTarget!, velocity.read.attach(0));
    blit(velocity.write);
    velocity.swap();

    gl!.uniform1f(inletProgram.uniforms.uMode!, 1);
    gl!.uniform1f(inletProgram.uniforms.uAmount!, INLET_DYE_RATE * dt);
    gl!.uniform1i(inletProgram.uniforms.uTarget!, dye.read.attach(0));
    blit(dye.write);
    dye.swap();
  }

  // ── Sizing ───────────────────────────────────────────────────────────────
  // Cap the device pixel ratio: the dye field is upsampled anyway, so a 3x
  // backing store costs fill rate for no visible gain.
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
    const w = Math.round(canvas.clientWidth * dpr);
    const h = Math.round(canvas.clientHeight * dpr);
    if (w > 0 && h > 0 && (canvas.width !== w || canvas.height !== h)) {
      canvas.width = w;
      canvas.height = h;
    }
    // Keep the holes a roughly constant size on screen at any width.
    jetCount = Math.max(6, Math.min(48, Math.round(canvas.clientWidth / INLET_JET_SPACING_PX)));
  }
  resize();
  window.addEventListener("resize", resize);

  // ── Frame loop ───────────────────────────────────────────────────────────
  // Only runs while the hero is actually on screen and the tab is visible;
  // a fluid sim churning behind a scrolled-past page is pure battery drain.
  let rafId = 0;
  let running = false;
  let onScreen = true;
  let lastTime = performance.now();

  function frame(now: number) {
    const dt = Math.min((now - lastTime) / 1000, 0.0166);
    lastTime = now;
    resize();
    inlet(dt, now / 1000);
    step(dt);
    render();
    rafId = requestAnimationFrame(frame);
  }

  function start() {
    if (running) return;
    running = true;
    lastTime = performance.now();
    rafId = requestAnimationFrame(frame);
  }
  function stop() {
    if (!running) return;
    running = false;
    cancelAnimationFrame(rafId);
  }
  function sync() {
    if (onScreen && document.visibilityState === "visible") start();
    else stop();
  }

  const observer = new IntersectionObserver(
    (entries) => {
      onScreen = entries.some((e) => e.isIntersecting);
      sync();
    },
    { threshold: 0 }
  );
  observer.observe(canvas);
  document.addEventListener("visibilitychange", sync);

  // Run the inflow forward before the first paint so the hero opens mid-plume
  // rather than as an empty frame that slowly fills.
  {
    const warmDt = WARMUP_DT;
    const warmSteps = compact ? 150 : WARMUP_STEPS;
    const deadline = performance.now() + (compact ? WARMUP_BUDGET_MS * 0.6 : WARMUP_BUDGET_MS);
    for (let i = 0; i < warmSteps; i++) {
      inlet(warmDt, i * warmDt);
      step(warmDt);
      // Checked every 16 steps: often enough to bound the cost, rarely enough
      // that the timing calls themselves don't show up in it.
      if ((i & 15) === 15 && performance.now() > deadline) break;
    }
  }
  sync();

  return {
    destroy() {
      stop();
      observer.disconnect();
      document.removeEventListener("visibilitychange", sync);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerleave", onPointerLeave);
      window.removeEventListener("resize", resize);
      for (const t of createdTextures) gl!.deleteTexture(t);
      for (const f of createdFramebuffers) gl!.deleteFramebuffer(f);
      gl!.deleteBuffer(vertexBuffer);
      gl!.deleteBuffer(indexBuffer);
    },
  };
}

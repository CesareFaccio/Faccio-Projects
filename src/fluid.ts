// fluid.ts
// A compact GPU fluid simulation (incompressible Navier-Stokes, solved on the
// GPU with WebGL2) used as the hero background. Each frame it advects a
// velocity field through itself, applies vorticity confinement to keep the
// small eddies from being smeared away by numerical diffusion, projects the
// velocity back to divergence-free with a Jacobi pressure solve, and advects a
// dye field through the result. Pointer movement injects velocity and dye.
//
// The set-up is a channel: a steady inflow across the left edge, an open
// outflow on the right, free-slip top and bottom, and a staggered row of small
// cylinders just downstream of the inlet. The cylinders shed vortices which
// break down across the rest of the frame — grid turbulence, the standard way
// a wind tunnel is made turbulent. Horizontal smoke rakes at the inlet make it
// visible as streaklines, the way a real flow-visualisation rig does.
//
// Everything enters at the inlet and leaves at the outlet, continuously.
// Nothing is ever added in the middle of the frame, so nothing can read as a
// puff.
//
// Tuning: INLET_DYE_RATE sets how thick the smoke is; CURL and
// VELOCITY_DISSIPATION together set how turbulent it looks (more curl, less
// dissipation = higher effective Reynolds number); INLET_SPEED sets how fast
// it crosses. Keep INLET_SPEED well under the CFL limit — advection backtraces
// dt*velocity*texelSize per step, so a few hundred on a ~200-wide grid is
// already a large fraction of a cell.
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
const SIM_RESOLUTION = 160;
const DYE_RESOLUTION = 384;
const DENSITY_DISSIPATION = 0.45; // how fast the dye fades
const VELOCITY_DISSIPATION = 0.06; // how fast the motion dies down
const PRESSURE_DISSIPATION = 0.8;
const PRESSURE_ITERATIONS = 20;
const CURL = 30; // vorticity confinement strength
// Ceilings that keep the simulation bounded no matter what is thrown at it.
// CURL_CAP bounds the confinement feedback; SPEED_CAP bounds the velocity
// field itself, at a few times the inflow speed.
const CURL_CAP = 900;
const SPEED_CAP = 520;
// Gaussian falloff denominator, in normalised-UV units squared. Small changes
// here matter a lot: this is what separates thin wisps from billowing smoke.
const SPLAT_RADIUS = 0.30;
// Pointer forcing. SPLAT_FORCE converts cursor travel (in screen fractions)
// into velocity; MAX_POINTER_IMPULSE is the hard cap on what one frame can
// inject, which is what actually keeps a fast flick from destabilising the
// channel. POINTER_DYE is deliberately small — the cursor is there to stir
// the smoke that is already flowing, not to paint new smoke into the frame.
const SPLAT_FORCE = 1400;
const MAX_POINTER_IMPULSE = 260;
const POINTER_DYE = 0.09;

// ── Channel ─────────────────────────────────────────────────────────────────
const INLET_SPEED = 130; // sim units; well inside the CFL limit
const INLET_DYE_RATE = 0.40; // dye per second at the rakes
const INLET_WIDTH = 0.05; // how far in from the left the inflow is imposed
const RAKE_COUNT = 24; // horizontal smoke lines seeded at the inlet
const OBSTACLE_COUNT = 7; // cylinders spanning the channel
const OBSTACLE_X = 0.17; // their distance from the left edge
const OBSTACLE_R = 0.028; // radius, in units of canvas height
// Brightness response: higher makes thin smoke show up more. It only changes
// how the field is drawn, never the simulation, so it is the safer of the two
// brightness knobs to reach for.
const DISPLAY_GAIN = 1.5;

// Sim steps run before the first paint, so the hero opens mid-turbulence.
// One device pixel per CSS pixel. The display shader dithers against a 4x4
// matrix in gl_FragCoord space, so this is what fixes that pattern at a
// visible 4px grid instead of shrinking it to invisibility on a retina screen —
// and the chunky output is the point, not a compromise. The dye field is
// upsampled to the canvas either way, so nothing is lost but fill rate.
const DPR = 1;

const WARMUP_STEPS = 260;
const WARMUP_DT = 1 / 60;

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
out vec4 fragColor;
void main () {
  vec2 coord = vUv - dt * texture(uVelocity, vUv).xy * texelSize;
  vec4 result = texture(uSource, coord);
  float decay = 1.0 + dissipation * dt;
  fragColor = result / decay;
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
  // Free-slip top and bottom: mirror the normal component.
  if (vT.y > 1.0) { T = -C.y; }
  if (vB.y < 0.0) { B = -C.y; }
  // Left is a prescribed inflow, so the clamped sample is already the value we
  // want. Right is an open outflow — zero gradient, so fluid can leave rather
  // than being reflected back and forcing the channel to recirculate.
  if (vR.x > 1.0) { R = C.x; }
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
  // Confinement adds energy in proportion to the vorticity already there, so
  // it is a positive feedback loop: a strong eddy makes itself stronger. Left
  // uncapped, one hard flick of the cursor is enough to make it run away.
  force *= clamp(curl * C, -curlCap, curlCap);
  force.y *= -1.0;
  vec2 velocity = texture(uVelocity, vUv).xy;
  velocity += force * dt;
  // Hard ceiling on speed. The previous 1000 was ~17 cells of travel per step
  // at 60fps — far more headroom than any part of this flow needs, and enough
  // for a disturbance to persist long after it should have washed downstream.
  velocity = clamp(velocity, -speedCap, speedCap);
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
  // Pinning pressure to zero at the outflow is what actually lets mass leave;
  // the zero-gradient velocity condition alone is not enough.
  if (vR.x > 1.0) { pressure = 0.0; }
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
// Shared by the obstacle mask and the display, so the cylinders drawn on
// screen are exactly the ones the fluid sees.
const OBSTACLE_GLSL = `
float solidAt(vec2 uv, float count, float cx0, float r, float aspect) {
  float slot = floor(uv.y * count);
  float cy = (slot + 0.5) / count;
  // Alternate cylinders sit slightly fore and aft, so the wakes interleave
  // instead of shedding in lockstep across the whole span.
  float cx = cx0 + (mod(slot, 2.0) - 0.5) * r * 2.2;
  vec2 d = vec2((uv.x - cx) * aspect, uv.y - cy);
  return 1.0 - smoothstep(r * 0.82, r, length(d));
}`;

// Imposes the inflow on the left edge. Used twice per frame: once to set
// velocity (a boundary condition, so it relaxes toward a target rather than
// accumulating) and once to lay down the smoke rakes.
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
  vec3 base = texture(uTarget, vUv).xyz;
  float band = smoothstep(uWidth, 0.0, vUv.x);
  if (uMode < 0.5) {
    // A slow shear across the inlet, so the cylinder wakes are not all
    // identical and the turbulence downstream stays irregular.
    float shear = 0.05 * sin(vUv.y * 7.0 + uTime * 0.5) + 0.03 * sin(vUv.y * 13.0 - uTime * 0.31);
    fragColor = vec4(mix(base.xy, vec2(uSpeed, uSpeed * shear), band), 0.0, 1.0);
  } else {
    float rakes = pow(0.5 + 0.5 * cos(vUv.y * uRakes * 6.2831853), 6.0);
    fragColor = vec4(base + band * rakes * uAmount, 1.0);
  }
}`;

// Zeroes whatever it is given inside the cylinders — applied to velocity (so
// the flow has to go around them) and to dye (so they read as solid).
const OBSTACLE_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv;
uniform sampler2D uTarget;
uniform float uCount;
uniform float uX;
uniform float uR;
uniform float uAspect;
out vec4 fragColor;
${OBSTACLE_GLSL}
void main () {
  fragColor = texture(uTarget, vUv) * (1.0 - solidAt(vUv, uCount, uX, uR, uAspect));
}`;

const DISPLAY_SHADER = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 vUv;
uniform sampler2D uTexture;
uniform float uCount;
uniform float uX;
uniform float uR;
uniform float uAspect;
uniform float uGain;
out vec4 fragColor;
${OBSTACLE_GLSL}
// The 4x4 ordered (Bayer) matrix a 1-bit Mac used to fake grey. Thresholding
// the continuous density against it turns a smooth field into the crosshatched
// patterns of a black-and-white QuickDraw screen — the simulation itself is
// untouched, this is purely how it is drawn.
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
  // +0.5 centres the thresholds, so a flat 50% field dithers to a true
  // checkerboard rather than tipping wholly one way.
  return (m[i] + 0.5) / 16.0;
}

void main () {
  vec3 c = texture(uTexture, vUv).rgb;
  float d = max(max(c.r, c.g), c.b);
  // A smooth saturating response, not a smoothstep knee. The knee behaved as a
  // cliff: the density field is fairly uniform, so once it crossed, the whole
  // frame flipped from black to washed-out at once and no dye rate in between
  // gave a usable picture. This is monotonic everywhere, so brightness tracks
  // density gradually and mid densities show their structure.
  float v = 1.0 - exp(-uGain * max(d, 0.0));
  // Dye is drawn dark on white paper, so density raises ink coverage. The
  // density field sits in a narrow band around the middle, and a linear map of
  // it dithers to near-uniform 50% noise with the flow structure buried in it —
  // so the band is stretched across the full range first. This is a contrast
  // expansion, not a threshold: it stays monotonic, so no value of the dye rate
  // makes the picture flip all at once the way the old smoothstep knee did.
  float ink = clamp((v - 0.16) / 0.66, 0.0, 1.0);
  // A slight vignette keeps the far corners of the window clean.
  vec2 q = vUv - 0.5;
  ink *= 1.0 - 0.55 * dot(q, q);

  float solid = solidAt(vUv, uCount, uX, uR, uAspect);
  ink = mix(ink, 1.0, solid);

  // Dither in device pixels, so the pattern stays a crisp 4px grid however the
  // window is sized — scaling it with the simulation grid would make it crawl.
  float lit = step(bayer4(gl_FragCoord.xy), ink);
  vec3 col = vec3(1.0 - lit);
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
  const displayProgram = new Program(gl, vertexShader, DISPLAY_SHADER);
  const inletProgram = new Program(gl, vertexShader, INLET_SHADER);
  const obstacleProgram = new Program(gl, vertexShader, OBSTACLE_SHADER);

  // Aspect-correct simulation grids, so eddies stay round on a wide viewport.
  function getResolution(resolution: number) {
    const aspect = gl!.drawingBufferWidth / gl!.drawingBufferHeight || 1;
    const ratio = aspect < 1 ? 1 / aspect : aspect;
    const min = Math.round(resolution);
    const max = Math.round(resolution * ratio);
    return aspect > 1 ? { width: max, height: min } : { width: min, height: max };
  }

  // Size the drawing buffer before the grids are derived from it. Without
  // this they are computed from the canvas's default 300x150, so the
  // simulation runs at the wrong aspect ratio and the flow comes out stretched.
  {
    const w0 = Math.round(canvas.clientWidth * DPR);
    const h0 = Math.round(canvas.clientHeight * DPR);
    if (w0 > 0 && h0 > 0) {
      canvas.width = w0;
      canvas.height = h0;
    }
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

  function canvasAspect() {
    return canvas.width / canvas.height || 1;
  }

  /** Zeroes the given field inside the cylinders. */
  function maskObstacles(target: DoubleFBO) {
    obstacleProgram.bind();
    gl!.uniform1f(obstacleProgram.uniforms.uCount!, OBSTACLE_COUNT);
    gl!.uniform1f(obstacleProgram.uniforms.uX!, OBSTACLE_X);
    gl!.uniform1f(obstacleProgram.uniforms.uR!, OBSTACLE_R);
    gl!.uniform1f(obstacleProgram.uniforms.uAspect!, canvasAspect());
    gl!.uniform1i(obstacleProgram.uniforms.uTarget!, target.read.attach(0));
    blit(target.write);
    target.swap();
  }

  /** Imposes the inflow and lays down the smoke rakes at the left edge. */
  function inflow(dt: number, timeSeconds: number) {
    inletProgram.bind();
    gl!.uniform1f(inletProgram.uniforms.uWidth!, INLET_WIDTH);
    gl!.uniform1f(inletProgram.uniforms.uRakes!, RAKE_COUNT);
    gl!.uniform1f(inletProgram.uniforms.uTime!, timeSeconds);
    gl!.uniform1f(inletProgram.uniforms.uSpeed!, INLET_SPEED);

    gl!.uniform1f(inletProgram.uniforms.uMode!, 0);
    gl!.uniform1i(inletProgram.uniforms.uTarget!, velocity.read.attach(0));
    blit(velocity.write);
    velocity.swap();

    gl!.uniform1f(inletProgram.uniforms.uMode!, 1);
    gl!.uniform1f(inletProgram.uniforms.uAmount!, INLET_DYE_RATE * dt);
    gl!.uniform1i(inletProgram.uniforms.uTarget!, dye.read.attach(0));
    blit(dye.write);
    dye.swap();
  }

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
    gl!.uniform1f(vorticityProgram.uniforms.curl!, CURL);
    gl!.uniform1f(vorticityProgram.uniforms.curlCap!, CURL_CAP);
    gl!.uniform1f(vorticityProgram.uniforms.speedCap!, SPEED_CAP);
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

    // The cylinders are enforced by zeroing velocity inside them after the
    // projection. Not a true no-slip boundary, but it sheds convincingly and
    // costs one pass.
    maskObstacles(velocity);

    advectionProgram.bind();
    gl!.uniform2f(advectionProgram.uniforms.texelSize!, velocity.texelSizeX, velocity.texelSizeY);
    gl!.uniform1i(advectionProgram.uniforms.uVelocity!, velocity.read.attach(0));
    gl!.uniform1i(advectionProgram.uniforms.uSource!, velocity.read.attach(0));
    gl!.uniform1f(advectionProgram.uniforms.dt!, dt);
    gl!.uniform1f(advectionProgram.uniforms.dissipation!, VELOCITY_DISSIPATION);
    blit(velocity.write);
    velocity.swap();

    gl!.uniform1i(advectionProgram.uniforms.uVelocity!, velocity.read.attach(0));
    gl!.uniform1i(advectionProgram.uniforms.uSource!, dye.read.attach(1));
    gl!.uniform1f(advectionProgram.uniforms.dissipation!, DENSITY_DISSIPATION);
    blit(dye.write);
    dye.swap();

    maskObstacles(dye);
  }

  function render() {
    displayProgram.bind();
    gl!.uniform1f(displayProgram.uniforms.uCount!, OBSTACLE_COUNT);
    gl!.uniform1f(displayProgram.uniforms.uX!, OBSTACLE_X);
    gl!.uniform1f(displayProgram.uniforms.uR!, OBSTACLE_R);
    gl!.uniform1f(displayProgram.uniforms.uAspect!, canvasAspect());
    gl!.uniform1f(displayProgram.uniforms.uGain!, DISPLAY_GAIN);
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

  // Pointer movement is accumulated here and applied once per frame, rather
  // than splatted per event. A fast drag fires many pointermove events between
  // frames, and splatting each one let a single flick inject an unbounded
  // amount of momentum — the main cause of the runaway.
  let pendingDX = 0;
  let pendingDY = 0;
  let pendingX = 0;
  let pendingY = 0;
  let pointerPending = false;

  function onPointerMove(e: PointerEvent) {
    const { x, y } = toSimCoords(e.clientX, e.clientY);
    if (!pointerActive) {
      pointerActive = true;
      lastX = x;
      lastY = y;
      return;
    }
    pendingDX += x - lastX;
    pendingDY += y - lastY;
    lastX = x;
    lastY = y;
    pendingX = x;
    pendingY = y;
    pointerPending = true;
  }

  /** Applies one frame's worth of accumulated cursor movement, magnitude-capped. */
  function applyPointer() {
    if (!pointerPending) return;
    pointerPending = false;
    let vx = pendingDX * SPLAT_FORCE;
    let vy = pendingDY * SPLAT_FORCE;
    pendingDX = 0;
    pendingDY = 0;
    const mag = Math.hypot(vx, vy);
    if (mag < 0.5) return;
    if (mag > MAX_POINTER_IMPULSE) {
      const k = MAX_POINTER_IMPULSE / mag;
      vx *= k;
      vy *= k;
    }
    splat(pendingX, pendingY, vx, vy, POINTER_DYE);
  }

  function onPointerLeave() {
    pointerActive = false;
  }

  window.addEventListener("pointermove", onPointerMove, { passive: true });
  window.addEventListener("pointerleave", onPointerLeave, { passive: true });

  // ── Emitters ─────────────────────────────────────────────────────────────
  // Each emitter rides a slow Lissajous loop, injecting continuously along its
  // own direction of travel. Because they never stop and never jump, the field
  // is stirred smoothly rather than punched — no moment where a discrete puff
  // appears — and the four of them together keep the whole frame in motion.
  // ── Sizing ───────────────────────────────────────────────────────────────
  function resize() {
    const dpr = DPR;
    const w = Math.round(canvas.clientWidth * dpr);
    const h = Math.round(canvas.clientHeight * dpr);
    if (w > 0 && h > 0 && (canvas.width !== w || canvas.height !== h)) {
      canvas.width = w;
      canvas.height = h;
    }
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
    inflow(dt, now / 1000);
    applyPointer();
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

  // Run the emitters forward before the first paint so the hero opens with
  // developed turbulence rather than an empty frame that fills in.
  {
    const warmSteps = compact ? 80 : WARMUP_STEPS;
    for (let i = 0; i < warmSteps; i++) {
      inflow(WARMUP_DT, i * WARMUP_DT);
      step(WARMUP_DT);
    }
    // Paint the warmed state immediately rather than waiting for the first
    // animation frame. Without this the canvas stays blank until rAF runs,
    // which is a visible flash of empty hero on a slow first frame.
    render();
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

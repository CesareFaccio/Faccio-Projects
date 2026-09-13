// fluid.ts
// A compact GPU fluid simulation (incompressible Navier-Stokes, solved on the
// GPU with WebGL2) used as the hero background. Each frame it advects a
// velocity field through itself, applies vorticity confinement to keep the
// small eddies from being smeared away by numerical diffusion, projects the
// velocity back to divergence-free with a Jacobi pressure solve, and advects a
// dye field through the result. Pointer movement injects velocity and dye.
//
// Between pointer input, the field is kept alive by six emitters drifting on
// slow looping paths, each kicking the fluid several times a second in a
// direction that sweeps as it goes. Nothing here reads the dye field back, so
// there is no feedback and nothing that can run away — density is simply
// injection over dissipation, and the response to every knob is monotonic.
//
// Tuning: EMITTER_DYE sets how thick the smoke is (roughly linear), CURL sets
// how curly versus smooth it looks, and EMITTER_IMPULSE sets how vigorously it
// churns. Emitter centres are spread across the frame on purpose — emitters
// that all orbit the middle leave the edges bare and the page never fills.
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
const DENSITY_DISSIPATION = 0.55; // how fast the dye fades
const VELOCITY_DISSIPATION = 0.14; // how fast the motion dies down
const PRESSURE_DISSIPATION = 0.8;
const PRESSURE_ITERATIONS = 20;
const CURL = 26; // vorticity confinement strength
// Gaussian falloff denominator, in normalised-UV units squared. Small changes
// here matter a lot: this is what separates thin wisps from billowing smoke.
const SPLAT_RADIUS = 0.30;
const SPLAT_FORCE = 5200;

// ── Emitters ────────────────────────────────────────────────────────────────
// Rates are per second and scaled by the frame's dt, so the look is
// frame-rate independent. Density is linear in EMITTER_DYE_RATE and inversely
// linear in DENSITY_DISSIPATION — no thresholds, no bistability.
// Forcing is impulsive, not spread evenly over every frame. That matters: the
// same momentum dribbled in continuously becomes a quasi-steady push, which
// the pressure solve largely cancels and which produces almost no vorticity.
// Discrete kicks roll up into vortices — that is what turbulence is made of.
// The cadence is fast enough (several per second, per emitter) that nothing
// reads as an individual puff.
const EMITTER_PERIOD = 0.22; // seconds between kicks, per emitter
const EMITTER_IMPULSE = 1200; // velocity magnitude per kick
const EMITTER_DYE = 0.22; // dye added per kick
// Each emitter orbits its own centre (cx, cy) on its own pair of path
// frequencies (Hz), and its kick direction sweeps at its own angular rate `fr`
// (rad/s). The centres are spread across the frame deliberately: emitters that
// all orbit the middle leave the edges empty and the page never fills.
// No two frequencies share a common factor, so nothing visibly repeats.
const EMITTERS = [
  { cx: 0.24, cy: 0.30, ax: 0.18, ay: 0.16, fx: 0.091, fy: 0.067, px: 0.0, py: 1.7, fr: 0.73 },
  { cx: 0.76, cy: 0.33, ax: 0.17, ay: 0.18, fx: 0.058, fy: 0.107, px: 2.4, py: 0.5, fr: -0.51 },
  { cx: 0.30, cy: 0.72, ax: 0.19, ay: 0.15, fx: 0.127, fy: 0.047, px: 4.1, py: 3.3, fr: 0.94 },
  { cx: 0.70, cy: 0.70, ax: 0.16, ay: 0.19, fx: 0.041, fy: 0.087, px: 5.6, py: 2.2, fr: -1.13 },
  { cx: 0.50, cy: 0.50, ax: 0.26, ay: 0.22, fx: 0.073, fy: 0.113, px: 1.2, py: 4.4, fr: 0.62 },
  { cx: 0.50, cy: 0.17, ax: 0.30, ay: 0.12, fx: 0.103, fy: 0.059, px: 3.7, py: 5.1, fr: -0.83 },
];
// Sim steps run before the first paint, so the hero opens mid-turbulence.
const WARMUP_STEPS = 150;
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
  // Free-slip walls: mirror the normal component at the boundary.
  if (vL.x < 0.0) { L = -C.x; }
  if (vR.x > 1.0) { R = -C.x; }
  if (vT.y > 1.0) { T = -C.y; }
  if (vB.y < 0.0) { B = -C.y; }
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
  fragColor = vec4((L + R + B + T - divergence) * 0.25, 0.0, 0.0, 1.0);
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
    gl!.uniform1f(vorticityProgram.uniforms.curl!, CURL);
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
    blit(velocity.write);
    velocity.swap();

    gl!.uniform1i(advectionProgram.uniforms.uVelocity!, velocity.read.attach(0));
    gl!.uniform1i(advectionProgram.uniforms.uSource!, dye.read.attach(1));
    gl!.uniform1f(advectionProgram.uniforms.dissipation!, DENSITY_DISSIPATION);
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

  // ── Emitters ─────────────────────────────────────────────────────────────
  // Each emitter rides a slow Lissajous loop, injecting continuously along its
  // own direction of travel. Because they never stop and never jump, the field
  // is stirred smoothly rather than punched — no moment where a discrete puff
  // appears — and the four of them together keep the whole frame in motion.
  const emitterClocks = EMITTERS.map((_, i) => (i * EMITTER_PERIOD) / EMITTERS.length);
  function stir(dt: number, t: number) {
    for (let i = 0; i < EMITTERS.length; i++) {
      emitterClocks[i] += dt;
      if (emitterClocks[i] < EMITTER_PERIOD) continue;
      emitterClocks[i] -= EMITTER_PERIOD;
      const e = EMITTERS[i];
      const x = e.cx + e.ax * Math.sin(t * e.fx * 6.2832 + e.px);
      const y = e.cy + e.ay * Math.cos(t * e.fy * 6.2832 + e.py);
      // The kick direction sweeps independently of the path, so successive
      // kicks from one emitter fan around rather than pushing the same way —
      // which is what deposits vorticity instead of a steady current.
      const angle = t * e.fr + e.px;
      splat(x, y, Math.cos(angle) * EMITTER_IMPULSE, Math.sin(angle) * EMITTER_IMPULSE, EMITTER_DYE);
    }
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
    stir(dt, now / 1000);
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
      stir(WARMUP_DT, i * WARMUP_DT);
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

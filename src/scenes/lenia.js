// «Ления» — continuous cellular automaton (Lenia, Bert Chan 2019), simulated entirely on the
// GPU with float ping-pong textures. Every cell in a toroidal grid follows one rule:
//   A(t+dt) = clip(A(t) + dt * G(K * A(t)), 0, 1)
// where K is a smooth ring-shaped kernel (a weighted average of the neighbourhood) and G is a
// bell curve that rewards neighbourhoods near a target density and punishes everything else.
// From that single local rule, self-sustaining, shape-holding, swimming "creatures" (the
// species Orbium unicaudatus) emerge — nothing about a creature's shape or motion is coded
// directly, it is entirely a consequence of the kernel + growth math finding a stable orbit.
(function () {
  'use strict';
  const SCENES = (window.SCENES = window.SCENES || {});

  // ---- Lenia parameters (Orbium unicaudatus, as published by Bert Chan) -------------------
  const R = 13;          // kernel radius, in cells
  const MU = 0.15;        // growth curve centre
  const SIGMA = 0.015;     // growth curve width
  const DT = 0.1;         // time step (T = 1/dt = 10)
  const SHORT_SIDE = 200;  // world cells along the shorter side of the viewport
  const TRAIL_RATE = 0.015; // EMA rate for the phosphorescent trail buffer
  const MAX_STIR_POINTS = 10;

  // Orbium unicaudatus — the 20x20 seed pattern from Bert Chan's Lenia tutorial notebook.
  // Verified offline (scratchpad/test-lenia-math.cjs): under these exact kernel/growth
  // parameters it survives 600+ steps with mass stable to within ~1% (well inside the +-30%
  // tolerance) after step 100, and its centroid travels ~370 cells of path length — it is a
  // genuine, robust glider, not a knife-edge pattern that happens to survive a few frames.
  const ORBIUM = [
    [0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0],
    [0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0],
    [0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0],
    [0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0],
    [0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0],
    [0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0],
    [0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0],
    [0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0],
    [0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0],
    [0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07],
    [0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11],
    [0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1],
    [0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05],
    [0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01],
    [0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0],
    [0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0],
    [0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0],
    [0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0],
    [0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0],
    [0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]
  ];

  // ---- GLSL sources -------------------------------------------------------------------------
  // A single fullscreen triangle, shared by every pass — no vertex buffers needed.
  const VERT = `#version 300 es
const vec2 POS[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
out vec2 vUv;
void main() {
  vec2 p = POS[gl_VertexID];
  vUv = p * 0.5 + 0.5;
  gl_Position = vec4(p, 0.0, 1.0);
}`;

  // The simulation step: convolve with the ring kernel, apply the growth bell curve, integrate.
  // Runs as a multiple-render-target pass so the growth field (used later for a faint glow)
  // is produced for free, without resampling the neighbourhood a second time.
  function makeUpdateSrc(kernelN) {
    return `#version 300 es
precision highp float;
#define KERNEL_N ${kernelN}
uniform sampler2D uState;
uniform sampler2D uKernel;   // Nx1 texture: (dx, dy, weight) per tap, weights sum to 1
uniform vec2 uGridSize;
uniform float uDt;
uniform float uMu;
uniform float uSigma;
in vec2 vUv;
layout(location = 0) out float outState;
layout(location = 1) out float outGrowth;

void main() {
  // K * A : a weighted average of the neighbourhood through the precomputed ring kernel.
  // uState wraps with REPEAT, so this samples across the torus edge for free.
  float u = 0.0;
  for (int i = 0; i < KERNEL_N; i++) {
    vec3 k = texelFetch(uKernel, ivec2(i, 0), 0).xyz;
    u += k.z * texture(uState, vUv + k.xy / uGridSize).r;
  }
  // G(u): a bell curve — neighbourhoods near mu grow, everything else decays back to 0.
  // This one curve is the entire "biology": no birth/survival rule tables, just calculus.
  float z = (u - uMu) / uSigma;
  float g = 2.0 * exp(-0.5 * z * z) - 1.0;
  float a = texture(uState, vUv).r;
  outState = clamp(a + uDt * g, 0.0, 1.0);
  outGrowth = g;
}`;
  }

  const FS_TRAIL = `#version 300 es
precision highp float;
uniform sampler2D uState;
uniform sampler2D uTrailPrev;
uniform float uRate;
in vec2 vUv;
layout(location = 0) out float outTrail;
void main() {
  // Exponential moving average of the state — a slow-decaying phosphor memory of where
  // living mass has recently been, so a creature's swim path glows faintly behind it.
  float a = texture(uState, vUv).r;
  float prev = texture(uTrailPrev, vUv).r;
  outTrail = prev + (a - prev) * uRate;
}`;

  const FS_STAMP = `#version 300 es
precision highp float;
uniform sampler2D uState;
uniform sampler2D uPattern;
uniform vec2 uGridSize;
uniform vec2 uCenter;       // stamp centre, in grid cells
uniform float uAngle;
uniform float uMirror;      // 0 or 1
uniform float uPatternHalf; // half-extent of the (padded) pattern texture, in cells
in vec2 vUv;
layout(location = 0) out float outState;

// Shortest signed offset on a torus of the given size (so a stamp near the wrap edge doesn't
// get sampled from the wrong, far side of the world).
vec2 wrapDelta(vec2 d, vec2 size) { return d - size * floor(d / size + 0.5); }

void main() {
  vec2 cell = vUv * uGridSize;
  vec2 delta = wrapDelta(cell - uCenter, uGridSize);
  float c = cos(-uAngle), s = sin(-uAngle);
  vec2 local = mat2(c, -s, s, c) * delta;  // rotate the query point into the pattern's frame
  if (uMirror > 0.5) local.x = -local.x;
  vec2 puv = local / (uPatternHalf * 2.0) + 0.5;
  float pv = 0.0;
  if (puv.x >= 0.0 && puv.x <= 1.0 && puv.y >= 0.0 && puv.y <= 1.0) {
    pv = texture(uPattern, puv).r;
  }
  float a = texture(uState, vUv).r;
  outState = max(a, pv); // overlay the new creature without erasing whatever is already there
}`;

  const FS_STIR = `#version 300 es
precision highp float;
#define MAX_POINTS ${MAX_STIR_POINTS}
uniform sampler2D uState;
uniform vec2 uGridSize;
uniform vec2 uPoints[MAX_POINTS];
uniform int uPointCount;
uniform float uRadius;
uniform float uAmp;
uniform float uSeed;
in vec2 vUv;
layout(location = 0) out float outState;

vec2 wrapDelta(vec2 d, vec2 size) { return d - size * floor(d / size + 0.5); }
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }

void main() {
  vec2 cell = vUv * uGridSize;
  float a = texture(uState, vUv).r;
  float add = 0.0;
  for (int i = 0; i < MAX_POINTS; i++) {
    if (i >= uPointCount) break;
    vec2 d = wrapDelta(cell - uPoints[i], uGridSize);
    float r = length(d) / uRadius;
    float fall = smoothstep(1.0, 0.0, r);
    float n = hash(cell * 0.15 + uSeed + float(i) * 17.0);
    add += fall * uAmp * (0.35 + 0.65 * n); // soft, slightly grainy — feeding the dish, not painting it
  }
  outState = clamp(a + add, 0.0, 1.0);
}`;

  const FS_DISPLAY = `#version 300 es
precision highp float;
uniform sampler2D uState;
uniform sampler2D uTrail;
uniform sampler2D uGrowth;
uniform vec2 uGridSize;
uniform vec2 uResolution;
uniform float uTime;
in vec2 vUv;
out vec4 fragColor;

int wrapi(int v, int n) { int m = v % n; return m < 0 ? m + n : m; }
float texelAt(sampler2D tex, ivec2 c, ivec2 size) {
  return texelFetch(tex, ivec2(wrapi(c.x, size.x), wrapi(c.y, size.y)), 0).r;
}
// Manual smooth (bilinear + smootherstep) upsampling of the low-res sim grid, done with
// texelFetch so it works on plain NEAREST float textures without needing linear-filterable
// float support from the GPU/driver.
float sampleSmooth(sampler2D tex, vec2 uv, ivec2 size) {
  vec2 texel = uv * vec2(size) - 0.5;
  vec2 i0f = floor(texel);
  vec2 f = smoothstep(0.0, 1.0, texel - i0f);
  ivec2 i0 = ivec2(i0f);
  float v00 = texelAt(tex, i0, size);
  float v10 = texelAt(tex, i0 + ivec2(1, 0), size);
  float v01 = texelAt(tex, i0 + ivec2(0, 1), size);
  float v11 = texelAt(tex, i0 + ivec2(1, 1), size);
  return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
}
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }

void main() {
  ivec2 size = ivec2(uGridSize);
  float a = clamp(sampleSmooth(uState, vUv, size), 0.0, 1.0);
  float t = clamp(sampleSmooth(uTrail, vUv, size), 0.0, 1.0);
  float g = sampleSmooth(uGrowth, vUv, size);

  // Bioluminescent microscopy palette: dark ground -> dim tissue trail -> living body ->
  // bright tissue -> near-white cores, layered by increasing state intensity.
  vec3 ground  = vec3(0.0118, 0.0314, 0.0392);
  vec3 trailC  = vec3(0.0510, 0.2275, 0.2510);
  vec3 bodyC   = vec3(0.1843, 0.7255, 0.6471);
  vec3 brightC = vec3(0.8471, 0.9490, 0.7686);
  vec3 coreC   = vec3(0.9647, 1.0, 0.9412);

  vec3 col = mix(ground, trailC, smoothstep(0.03, 0.5, t));
  col = mix(col, bodyC, smoothstep(0.06, 0.32, a));
  col = mix(col, brightC, smoothstep(0.34, 0.66, a));
  col = mix(col, coreC, smoothstep(0.68, 0.97, a));
  col += coreC * clamp(g, 0.0, 1.0) * a * 0.18; // faint glow on actively-growing membrane

  // Dark-field microscope vignette.
  vec2 p = (vUv * 2.0 - 1.0);
  p.x *= uResolution.x / uResolution.y;
  float vig = smoothstep(1.15, 0.35, length(p));
  col *= mix(0.5, 1.0, vig);

  // Very faint animated grain — keeps the dish from reading as a flat digital gradient.
  float n = hash(vUv * uResolution.xy * 0.5 + uTime * 97.0);
  col += (n - 0.5) * 0.018;

  fragColor = vec4(col, 1.0);
}`;

  // ---- small GL helpers ----------------------------------------------------------------------
  function compile(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.error('lenia: shader compile error:', gl.getShaderInfoLog(sh));
      gl.deleteShader(sh);
      return null;
    }
    return sh;
  }
  function link(gl, vsSrc, fsSrc) {
    const vs = compile(gl, gl.VERTEX_SHADER, vsSrc);
    const fs = compile(gl, gl.FRAGMENT_SHADER, fsSrc);
    if (!vs || !fs) return null;
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error('lenia: program link error:', gl.getProgramInfoLog(prog));
      gl.deleteProgram(prog);
      return null;
    }
    return prog;
  }
  function uniforms(gl, prog, names) {
    const u = {};
    for (const n of names) u[n] = gl.getUniformLocation(prog, n);
    return u;
  }

  SCENES.lenia = {
    id: 'lenia',
    name: 'Ления',
    tagline: 'Непрерывный клеточный автомат, в котором сами собой живут существа',
    about: 'Каждая клетка следует одному правилу: взвешенная сумма соседей через кольцевое ' +
      'ядро проходит через колоколообразную функцию роста и решает, прибавить клетке ' +
      'вещества или отнять. Из этой единственной формулы, без единой строчки про «тело» или ' +
      '«поведение», рождаются существа, которые держат форму, плавают и сталкиваются — вид ' +
      'называется Orbium unicaudatus.',
    spec: [
      ['Ядро', 'кольцо, R = 13 клеток'],
      ['Рост', 'μ = 0,15, σ = 0,015'],
      ['Шаг времени', 'Δt = 0,1 (T = 10)'],
      ['Мир', 'тор, короткая сторона ≈ 200 клеток, длинная — по пропорциям экрана'],
      ['Вид', 'Orbium unicaudatus'],
      ['Автор модели', 'Берт Чан, 2019']
    ],
    hint: 'Нажмите — новое существо · проведите — подкормить',
    accent: '#7FE0C4',

    mount(host, ui) {
      const canvas = document.createElement('canvas');
      canvas.style.display = 'block';
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      canvas.style.touchAction = 'none';
      host.appendChild(canvas);

      const gl = canvas.getContext('webgl2', {
        antialias: false, alpha: false, depth: false, stencil: false,
        preserveDrawingBuffer: false, powerPreference: 'high-performance'
      });
      const floatExt = gl && gl.getExtension('EXT_color_buffer_float');
      if (!gl || !floatExt) {
        const div = document.createElement('div');
        div.className = 'scene-fallback';
        div.textContent = 'Для этой сцены нужен WebGL2 — откройте страницу в свежем Chrome, Firefox или Safari.';
        host.innerHTML = '';
        host.appendChild(div);
        return { unmount() { host.innerHTML = ''; ui.innerHTML = ''; } };
      }

      // ---- build the kernel (CPU) and upload as a texture of (dx, dy, weight) taps ----------
      const kernelTaps = [];
      { // Local scope: bump-shell profile exp(4 - 1/(r(1-r))) on normalised radius r in (0,1).
        const R2 = R * R;
        for (let dy = -R; dy <= R; dy++) {
          for (let dx = -R; dx <= R; dx++) {
            const d2 = dx * dx + dy * dy;
            if (d2 === 0 || d2 >= R2) continue;
            const r = Math.sqrt(d2) / R;
            const w = Math.exp(4 - 1 / (r * (1 - r)));
            if (w > 1e-6) kernelTaps.push(dx, dy, w);
          }
        }
      }
      const kernelN = kernelTaps.length / 3;
      {
        let sum = 0;
        for (let i = 0; i < kernelN; i++) sum += kernelTaps[i * 3 + 2];
        for (let i = 0; i < kernelN; i++) kernelTaps[i * 3 + 2] /= sum;
      }
      const kernelData = new Float32Array(kernelN * 4);
      for (let i = 0; i < kernelN; i++) {
        kernelData[i * 4 + 0] = kernelTaps[i * 3 + 0];
        kernelData[i * 4 + 1] = kernelTaps[i * 3 + 1];
        kernelData[i * 4 + 2] = kernelTaps[i * 3 + 2];
        kernelData[i * 4 + 3] = 0;
      }
      const kernelTex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, kernelTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, kernelN, 1, 0, gl.RGBA, gl.FLOAT, kernelData);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

      // ---- Orbium pattern texture, padded so rotated sampling never bleeds off the edge -----
      const PAD = 2, PSIZE = 20 + PAD * 2;
      const patternData = new Uint8Array(PSIZE * PSIZE);
      for (let y = 0; y < 20; y++) {
        for (let x = 0; x < 20; x++) {
          patternData[(y + PAD) * PSIZE + (x + PAD)] = Math.round(Math.min(1, ORBIUM[y][x]) * 255);
        }
      }
      const patternTex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, patternTex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, PSIZE, PSIZE, 0, gl.RED, gl.UNSIGNED_BYTE, patternData);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const patternHalf = PSIZE / 2;

      // ---- programs ---------------------------------------------------------------------------
      const progUpdate = link(gl, VERT, makeUpdateSrc(kernelN));
      const progTrail = link(gl, VERT, FS_TRAIL);
      const progStamp = link(gl, VERT, FS_STAMP);
      const progStir = link(gl, VERT, FS_STIR);
      const progDisplay = link(gl, VERT, FS_DISPLAY);
      if (!progUpdate || !progTrail || !progStamp || !progStir || !progDisplay) {
        const div = document.createElement('div');
        div.className = 'scene-fallback';
        div.textContent = 'Для этой сцены нужен WebGL2 — откройте страницу в свежем Chrome, Firefox или Safari.';
        host.innerHTML = '';
        host.appendChild(div);
        return { unmount() { host.innerHTML = ''; ui.innerHTML = ''; } };
      }
      const uUpdate = uniforms(gl, progUpdate, ['uState', 'uKernel', 'uGridSize', 'uDt', 'uMu', 'uSigma']);
      const uTrail = uniforms(gl, progTrail, ['uState', 'uTrailPrev', 'uRate']);
      const uStamp = uniforms(gl, progStamp, ['uState', 'uPattern', 'uGridSize', 'uCenter', 'uAngle', 'uMirror', 'uPatternHalf']);
      const uStir = uniforms(gl, progStir, ['uState', 'uGridSize', 'uPoints', 'uPointCount', 'uRadius', 'uAmp', 'uSeed']);
      const uDisplay = uniforms(gl, progDisplay, ['uState', 'uTrail', 'uGrowth', 'uGridSize', 'uResolution', 'uTime']);

      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);

      // ---- sim-grid resources (rebuilt on significant resize) -------------------------------
      let gridW = 0, gridH = 0;
      let stateTex = [null, null], growthTex = null, trailTex = [null, null];
      let simFbo = [null, null], trailFbo = [null, null];
      let stateIdx = 0, trailIdx = 0;

      function makeFloatTex(w, h, internalFormat, format, type) {
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
        return tex;
      }
      function attachCheck(label) {
        const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
        if (status !== gl.FRAMEBUFFER_COMPLETE) console.error('lenia: framebuffer incomplete (' + label + '):', status);
      }

      function destroyGrid() {
        for (const t of stateTex) if (t) gl.deleteTexture(t);
        for (const t of trailTex) if (t) gl.deleteTexture(t);
        if (growthTex) gl.deleteTexture(growthTex);
        for (const f of simFbo) if (f) gl.deleteFramebuffer(f);
        for (const f of trailFbo) if (f) gl.deleteFramebuffer(f);
      }

      function buildGrid(w, h) {
        destroyGrid();
        gridW = w; gridH = h;
        stateTex = [makeFloatTex(w, h, gl.R32F, gl.RED, gl.FLOAT), makeFloatTex(w, h, gl.R32F, gl.RED, gl.FLOAT)];
        growthTex = makeFloatTex(w, h, gl.R16F, gl.RED, gl.FLOAT);
        trailTex = [makeFloatTex(w, h, gl.R16F, gl.RED, gl.FLOAT), makeFloatTex(w, h, gl.R16F, gl.RED, gl.FLOAT)];
        simFbo = [gl.createFramebuffer(), gl.createFramebuffer()];
        for (let i = 0; i < 2; i++) {
          gl.bindFramebuffer(gl.FRAMEBUFFER, simFbo[i]);
          gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, stateTex[i], 0);
          gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, growthTex, 0);
          attachCheck('sim' + i);
        }
        trailFbo = [gl.createFramebuffer(), gl.createFramebuffer()];
        for (let i = 0; i < 2; i++) {
          gl.bindFramebuffer(gl.FRAMEBUFFER, trailFbo[i]);
          gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, trailTex[i], 0);
          attachCheck('trail' + i);
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        stateIdx = 0; trailIdx = 0;
      }

      function clearAll() {
        gl.clearColor(0, 0, 0, 0);
        gl.viewport(0, 0, gridW, gridH);
        for (const f of simFbo) {
          gl.bindFramebuffer(gl.FRAMEBUFFER, f);
          gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
          gl.clear(gl.COLOR_BUFFER_BIT);
        }
        for (const f of trailFbo) {
          gl.bindFramebuffer(gl.FRAMEBUFFER, f);
          gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
          gl.clear(gl.COLOR_BUFFER_BIT);
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        stateIdx = 0; trailIdx = 0;
      }

      // ---- drawing one creature / one soft noise blob into the live state texture -----------
      function drawStamp(cx, cy, angle, mirror) {
        const dst = 1 - stateIdx;
        gl.viewport(0, 0, gridW, gridH);
        gl.bindFramebuffer(gl.FRAMEBUFFER, simFbo[dst]);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
        gl.useProgram(progStamp);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, stateTex[stateIdx]);
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, patternTex);
        gl.uniform1i(uStamp.uState, 0);
        gl.uniform1i(uStamp.uPattern, 1);
        gl.uniform2f(uStamp.uGridSize, gridW, gridH);
        gl.uniform2f(uStamp.uCenter, cx, cy);
        gl.uniform1f(uStamp.uAngle, angle);
        gl.uniform1f(uStamp.uMirror, mirror ? 1 : 0);
        gl.uniform1f(uStamp.uPatternHalf, patternHalf);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        stateIdx = dst;
      }

      function stampNoiseBlob(cx, cy, radius, amp, seed) {
        const dst = 1 - stateIdx;
        gl.viewport(0, 0, gridW, gridH);
        gl.bindFramebuffer(gl.FRAMEBUFFER, simFbo[dst]);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
        gl.useProgram(progStir);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, stateTex[stateIdx]);
        gl.uniform1i(uStir.uState, 0);
        gl.uniform2f(uStir.uGridSize, gridW, gridH);
        gl.uniform2fv(uStir.uPoints, new Float32Array([cx, cy]));
        gl.uniform1i(uStir.uPointCount, 1);
        gl.uniform1f(uStir.uRadius, radius);
        gl.uniform1f(uStir.uAmp, amp);
        gl.uniform1f(uStir.uSeed, seed);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        stateIdx = dst;
      }

      function seedDish() {
        clearAll();
        const short = Math.min(gridW, gridH);
        const nOrbium = 8 + Math.floor(Math.random() * 7); // 8..14
        const placed = [];
        // Keep freshly-seeded creatures outside each other's kernel reach (radius R) at spawn
        // time, so the dish opens with everyone swimming independently rather than already
        // merging. Two Orbium bodies (radius ~patternHalf) only start influencing each other's
        // growth once their centres are closer than roughly R + patternHalf; this gives a
        // comfortable margin beyond that so early collisions are a matter of them swimming
        // into each other, not bad luck at spawn.
        const minDist = 2 * R + patternHalf;
        for (let i = 0; i < nOrbium; i++) {
          let cx = 0, cy = 0, ok = false;
          for (let attempt = 0; attempt < 30 && !ok; attempt++) {
            cx = Math.random() * gridW;
            cy = Math.random() * gridH;
            ok = true;
            for (const p of placed) {
              let dx = Math.abs(cx - p[0]); dx = Math.min(dx, gridW - dx);
              let dy = Math.abs(cy - p[1]); dy = Math.min(dy, gridH - dy);
              if (Math.hypot(dx, dy) < minDist) { ok = false; break; }
            }
          }
          placed.push([cx, cy]);
          drawStamp(cx, cy, Math.random() * Math.PI * 2, Math.random() < 0.5);
        }
        const nBlobs = 2 + Math.floor(Math.random() * 2); // 2..3
        for (let i = 0; i < nBlobs; i++) {
          stampNoiseBlob(Math.random() * gridW, Math.random() * gridH,
            short * 0.09 + Math.random() * short * 0.05, 0.5 + Math.random() * 0.25, Math.random() * 100);
        }
      }

      // ---- sizing: world aspect follows the viewport; sim resolution is independent of the
      // adaptive display "quality" (which only affects the display canvas' pixel resolution) --
      function computeGrid() {
        const cw = Math.max(1, host.clientWidth), ch = Math.max(1, host.clientHeight);
        let w, h;
        if (cw <= ch) { w = SHORT_SIDE; h = Math.round(SHORT_SIDE * ch / cw); }
        else { h = SHORT_SIDE; w = Math.round(SHORT_SIDE * cw / ch); }
        const maxLong = Math.round(SHORT_SIDE * 2.4);
        w = Math.min(Math.max(w, 60), maxLong);
        h = Math.min(Math.max(h, 60), maxLong);
        return { w, h };
      }

      { const g0 = computeGrid(); buildGrid(g0.w, g0.h); seedDish(); }

      // ---- adaptive quality (display resolution) + adaptive sim step count ------------------
      const fixedQuality = host.dataset.fixedQuality !== undefined;
      let quality = fixedQuality ? parseFloat(host.dataset.fixedQuality) : 1.0;
      let stepsPerFrame = 2;
      const frameTimes = [];

      function applyCanvasSize() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.max(1, Math.round(host.clientWidth * dpr * quality));
        const h = Math.max(1, Math.round(host.clientHeight * dpr * quality));
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      }

      // ---- one Euler step of the whole grid ---------------------------------------------------
      function simStep() {
        window.__leniaSteps = (window.__leniaSteps || 0) + 1;
        const dst = 1 - stateIdx;
        gl.viewport(0, 0, gridW, gridH);
        gl.bindFramebuffer(gl.FRAMEBUFFER, simFbo[dst]);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
        gl.useProgram(progUpdate);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, stateTex[stateIdx]);
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, kernelTex);
        gl.uniform1i(uUpdate.uState, 0);
        gl.uniform1i(uUpdate.uKernel, 1);
        gl.uniform2f(uUpdate.uGridSize, gridW, gridH);
        gl.uniform1f(uUpdate.uDt, DT);
        gl.uniform1f(uUpdate.uMu, MU);
        gl.uniform1f(uUpdate.uSigma, SIGMA);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        stateIdx = dst;
      }

      function trailStep() {
        const dst = 1 - trailIdx;
        gl.viewport(0, 0, gridW, gridH);
        gl.bindFramebuffer(gl.FRAMEBUFFER, trailFbo[dst]);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
        gl.useProgram(progTrail);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, stateTex[stateIdx]);
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, trailTex[trailIdx]);
        gl.uniform1i(uTrail.uState, 0);
        gl.uniform1i(uTrail.uTrailPrev, 1);
        gl.uniform1f(uTrail.uRate, TRAIL_RATE);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        trailIdx = dst;
      }

      function renderDisplay(t) {
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.useProgram(progDisplay);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, stateTex[stateIdx]);
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, trailTex[trailIdx]);
        gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, growthTex);
        gl.uniform1i(uDisplay.uState, 0);
        gl.uniform1i(uDisplay.uTrail, 1);
        gl.uniform1i(uDisplay.uGrowth, 2);
        gl.uniform2f(uDisplay.uGridSize, gridW, gridH);
        gl.uniform2f(uDisplay.uResolution, canvas.width, canvas.height);
        gl.uniform1f(uDisplay.uTime, t);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }

      // ---- interaction: tap = new creature, drag = soft feeding brush -----------------------
      // The brush is re-applied every single automaton sub-step (not once per rendered frame):
      // scratchpad/tune-stir.cjs showed that with 2-3 sim steps per frame (the adaptive default),
      // a "paint once per frame" brush gets fully erased by that frame's own decay before it is
      // ever seen, because 2-3 consecutive Euler steps can subtract more than a modest brush
      // adds. Painting at the *same* cadence as decay keeps a steady, controllable glow instead.
      let paused = false;
      const pointers = new Map();
      const activeStir = new Map(); // pointerId -> [cx, cy], live position while held & moved
      const CLICK_SLOP = 6; // px

      function eventToCell(e) {
        const rect = canvas.getBoundingClientRect();
        const fx = (e.clientX - rect.left) / Math.max(1, rect.width);
        const fy = (e.clientY - rect.top) / Math.max(1, rect.height);
        return [fx * gridW, fy * gridH];
      }

      function onPointerDown(e) {
        canvas.setPointerCapture(e.pointerId);
        const [cx, cy] = eventToCell(e);
        pointers.set(e.pointerId, { sx: e.clientX, sy: e.clientY, moved: false });
      }
      function onPointerMove(e) {
        const p = pointers.get(e.pointerId);
        if (!p) return;
        const dx = e.clientX - p.sx, dy = e.clientY - p.sy;
        if (!p.moved && Math.hypot(dx, dy) < CLICK_SLOP) return;
        p.moved = true;
        const [cx, cy] = eventToCell(e);
        activeStir.set(e.pointerId, [cx, cy]);
      }
      function onPointerUp(e) {
        const p = pointers.get(e.pointerId);
        pointers.delete(e.pointerId);
        activeStir.delete(e.pointerId);
        try { canvas.releasePointerCapture(e.pointerId); } catch (err) { /* already released */ }
        if (p && !p.moved) {
          const [cx, cy] = eventToCell(e);
          drawStamp(cx, cy, Math.random() * Math.PI * 2, Math.random() < 0.5);
        }
      }
      canvas.addEventListener('pointerdown', onPointerDown);
      canvas.addEventListener('pointermove', onPointerMove);
      canvas.addEventListener('pointerup', onPointerUp);
      canvas.addEventListener('pointercancel', onPointerUp);

      // Applies the feeding brush at every currently-held pointer's live position. Called once
      // per automaton sub-step so painting and decay stay in lockstep regardless of stepsPerFrame.
      function applyActiveStir(seed) {
        if (activeStir.size === 0) return;
        const pts = new Float32Array(MAX_STIR_POINTS * 2);
        let count = 0;
        for (const [cx, cy] of activeStir.values()) {
          if (count >= MAX_STIR_POINTS) break;
          pts[count * 2] = cx; pts[count * 2 + 1] = cy; count++;
        }
        const dst = 1 - stateIdx;
        gl.viewport(0, 0, gridW, gridH);
        gl.bindFramebuffer(gl.FRAMEBUFFER, simFbo[dst]);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
        gl.useProgram(progStir);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, stateTex[stateIdx]);
        gl.uniform1i(uStir.uState, 0);
        gl.uniform2f(uStir.uGridSize, gridW, gridH);
        gl.uniform2fv(uStir.uPoints, pts);
        gl.uniform1i(uStir.uPointCount, count);
        // Tuned in scratchpad/tune-stir.cjs against the real growth/kernel constants, assuming
        // one brush application per automaton step: below ~0.28 an isolated stroke always
        // dissolves within roughly a second of release (no runaway "spontaneous life"); 0.22
        // sits comfortably under that critical mass while building a clearly visible glow
        // within a few steps of continuous dragging.
        gl.uniform1f(uStir.uRadius, Math.min(gridW, gridH) * 0.04);
        gl.uniform1f(uStir.uAmp, 0.22);
        gl.uniform1f(uStir.uSeed, seed);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        stateIdx = dst;
      }

      // ---- ui buttons -------------------------------------------------------------------------
      const btnSeed = document.createElement('button');
      btnSeed.type = 'button'; btnSeed.className = 'ctl'; btnSeed.textContent = 'Засеять';
      btnSeed.addEventListener('click', () => seedDish());

      const btnClear = document.createElement('button');
      btnClear.type = 'button'; btnClear.className = 'ctl'; btnClear.textContent = 'Очистить';
      btnClear.addEventListener('click', () => clearAll());

      const btnPause = document.createElement('button');
      btnPause.type = 'button'; btnPause.className = 'ctl'; btnPause.textContent = 'Пауза';
      btnPause.setAttribute('aria-pressed', 'false');
      btnPause.addEventListener('click', () => setPaused(!paused));

      ui.appendChild(btnSeed);
      ui.appendChild(btnClear);
      ui.appendChild(btnPause);

      function setPaused(v) {
        paused = v;
        btnPause.setAttribute('aria-pressed', v ? 'true' : 'false');
      }

      function onKey(e) {
        const t = e.target;
        if (t && (t.tagName === 'INPUT' || t.tagName === 'BUTTON')) return;
        if (e.code === 'Space') { e.preventDefault(); setPaused(!paused); }
        else if (e.code === 'KeyR') { seedDish(); }
      }
      window.addEventListener('keydown', onKey);

      // ---- resize: world aspect follows the viewport; only rebuild on a real size change ----
      let resizeTimer = null;
      const ro = new ResizeObserver(() => {
        if (resizeTimer) clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
          const g = computeGrid();
          if (Math.abs(g.w - gridW) >= 8 || Math.abs(g.h - gridH) >= 8) {
            buildGrid(g.w, g.h);
            seedDish();
          }
        }, 200);
      });
      ro.observe(host);

      // ---- main loop ----------------------------------------------------------------------------
      const reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      let raf = 0, lastNow = null, simTime = 0;
      function frame(now) {
        raf = requestAnimationFrame(frame);
        const dt = lastNow === null ? 16.7 : Math.min(now - lastNow, 100);
        lastNow = now;
        frameTimes.push(dt);
        if (frameTimes.length >= 30) {
          let sum = 0; for (const v of frameTimes) sum += v;
          const avg = sum / frameTimes.length;
          frameTimes.length = 0;
          if (!fixedQuality) {
            if (avg > 22) quality = Math.max(0.3, quality * 0.85);
            else if (avg < 12) quality = Math.min(1, quality * 1.1);
          }
          if (avg > 26) stepsPerFrame = Math.max(1, stepsPerFrame - 1);
          else if (avg < 14) stepsPerFrame = Math.min(3, stepsPerFrame + 1);
        }
        applyCanvasSize();

        if (!paused) {
          simTime += dt * 0.001;
          // Paint immediately before each decay step, at the same cadence, so a held/dragged
          // brush reaches a stable visible glow instead of being outrun by multi-step decay.
          for (let i = 0; i < stepsPerFrame; i++) {
            applyActiveStir(now * 0.001 + i * 0.017);
            simStep();
          }
          trailStep();
        }

        renderDisplay(reducedMotion ? 0 : simTime);
      }
      raf = requestAnimationFrame(frame);

      return {
        unmount() {
          cancelAnimationFrame(raf);
          ro.disconnect();
          if (resizeTimer) clearTimeout(resizeTimer);
          window.removeEventListener('keydown', onKey);
          canvas.removeEventListener('pointerdown', onPointerDown);
          canvas.removeEventListener('pointermove', onPointerMove);
          canvas.removeEventListener('pointerup', onPointerUp);
          canvas.removeEventListener('pointercancel', onPointerUp);
          destroyGrid();
          gl.deleteTexture(kernelTex);
          gl.deleteTexture(patternTex);
          gl.deleteProgram(progUpdate);
          gl.deleteProgram(progTrail);
          gl.deleteProgram(progStamp);
          gl.deleteProgram(progStir);
          gl.deleteProgram(progDisplay);
          gl.deleteVertexArray(vao);
          gl.getExtension('WEBGL_lose_context') && gl.getExtension('WEBGL_lose_context').loseContext();
          canvas.remove();
          ui.innerHTML = '';
        }
      };
    }
  };
})();

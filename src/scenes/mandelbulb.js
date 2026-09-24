(function () {
  'use strict';
  const SCENES = (window.SCENES = window.SCENES || {});

  // ---------------------------------------------------------------------
  // GLSL: full-screen triangle vertex shader
  // ---------------------------------------------------------------------
  const VERT_SRC = `#version 300 es
layout(location = 0) in vec2 aPos;
void main() {
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

  // ---------------------------------------------------------------------
  // GLSL: ray-marched Mandelbulb fragment shader
  // ---------------------------------------------------------------------
  const FRAG_SRC = `#version 300 es
precision highp float;

uniform vec2  uRes;
uniform float uTime;
uniform float uPower;
uniform vec3  uCamPos;
uniform vec3  uCamTarget;
uniform float uFov;
out vec4 fragColor;

// cheap 2D hash, used to dither the dark background so its smooth
// gradient doesn't band.
float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

// Distance estimator for the Mandelbulb: iterate z -> z^n + c in spherical
// (triplex) coordinates and use Hubbard-Douady's DE = 0.5 * ln(r) * r / dr,
// where dr tracks the running derivative magnitude of the orbit. "trap"
// collects the minimum |x|, |y|, |z| and radius seen along the orbit -
// classic "orbit trap" data used later to colour the surface by its
// internal structure instead of by iteration count.
const int ITER = 9;
const float BAILOUT = 4.0;

float mandelbulbDE(vec3 p, float power, out vec4 trap) {
  vec3 z = p;
  float dr = 1.0;
  float r = 0.0;
  trap = vec4(1e5);
  for (int i = 0; i < ITER; i++) {
    r = max(length(z), 1e-8);
    if (r > BAILOUT) break;
    trap = min(trap, vec4(abs(z.x), abs(z.y), abs(z.z), r));

    float theta = acos(clamp(z.z / r, -1.0, 1.0));
    float phi = atan(z.y, z.x);
    dr = pow(r, power - 1.0) * power * dr + 1.0;

    float zr = pow(r, power);
    theta *= power;
    phi *= power;

    z = zr * vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
    z += p;
  }
  return 0.5 * log(r) * r / dr;
}

vec3 calcNormal(vec3 p, float power, float e) {
  vec4 tr;
  vec2 k = vec2(1.0, -1.0);
  return normalize(
    k.xyy * mandelbulbDE(p + k.xyy * e, power, tr) +
    k.yyx * mandelbulbDE(p + k.yyx * e, power, tr) +
    k.yxy * mandelbulbDE(p + k.yxy * e, power, tr) +
    k.xxx * mandelbulbDE(p + k.xxx * e, power, tr)
  );
}

// Short march toward the light, taking the minimum ratio of distance to
// travelled length as a cheap soft-shadow penumbra term.
float softShadow(vec3 ro, vec3 rd, float power) {
  float res = 1.0;
  float t = 0.01;
  for (int i = 0; i < 20; i++) {
    vec4 tr;
    float d = mandelbulbDE(ro + rd * t, power, tr);
    if (d < 0.0006) return 0.0;
    res = min(res, 9.0 * d / t);
    t += clamp(d, 0.01, 0.18);
    if (t > 2.2) break;
  }
  return clamp(res, 0.0, 1.0);
}

// Ambient occlusion by sampling the DE a few steps out along the normal -
// a surface tucked into a crevice reports a smaller distance than its
// nominal offset, which this turns into a darkening factor.
float calcAO(vec3 p, vec3 n, float power) {
  float occ = 0.0;
  float sca = 1.0;
  for (int i = 0; i < 5; i++) {
    float h = 0.02 + 0.09 * float(i) / 4.0;
    vec4 tr;
    float d = mandelbulbDE(p + n * h, power, tr);
    occ += (h - d) * sca;
    sca *= 0.7;
  }
  return clamp(1.0 - 2.1 * occ, 0.0, 1.0);
}

// Filmic (ACES-ish) tone curve - keeps the ivory highlights from clipping
// to flat white and gives the shadows a gentle roll-off.
vec3 tonemap(vec3 x) {
  const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Orbit-trap palette: a mineral specimen - pale ivory/bone on the smooth
// outer lobes, oxidised verdigris and warm copper welling up in the folds
// and creases where the orbit stayed close to the axes.
vec3 shadeAlbedo(vec4 trap, float ao, vec3 p) {
  vec3 verdigris = vec3(0.110, 0.440, 0.355);
  vec3 copper    = vec3(0.680, 0.335, 0.130);
  vec3 bone      = vec3(0.580, 0.535, 0.440);
  vec3 ivory     = vec3(0.950, 0.910, 0.800);

  // smooth outer lobes: bone through to bright ivory, varied a little by
  // how deep the orbit dove during iteration (trap.w).
  float core = clamp(trap.w * 2.3, 0.0, 1.0);
  vec3 base = mix(bone, ivory, smoothstep(0.15, 0.85, core));

  // crevices - detected from the orbit trap, which dips reliably wherever
  // the surface folds toward the symmetry axes - pick up an oxidised
  // patina. Which of copper or verdigris dominates a given fold is a slow
  // spatial pattern over the surface point itself, deliberately decoupled
  // from the fold detector so both colours appear as separate patches
  // rather than one hue always winning where the patina actually shows.
  float fold = 1.0 - smoothstep(0.0, 0.32, min(trap.x, trap.y));
  float marble = sin(p.x * 5.0) * sin(p.y * 4.3 + 1.7) * sin(p.z * 6.1 + 0.4);
  float hueMix = smoothstep(-0.2, 0.2, marble);
  vec3 patina = mix(verdigris, copper, hueMix);
  return mix(base, patina, fold * 0.72);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uRes) / uRes.y;

  vec3 fwd = normalize(uCamTarget - uCamPos);
  vec3 right = normalize(cross(fwd, vec3(0.0, 1.0, 0.0)));
  vec3 up = cross(right, fwd);
  // uv spans roughly [-0.5, 0.5] vertically, so the focal length that maps
  // that span to the requested vertical FOV is 0.5 / tan(fov/2), not 1/tan(fov/2).
  // On a portrait/narrow canvas that alone would only fit the *vertical*
  // extent and crop the sides, so the FOV is widened by the aspect ratio
  // whenever width < height - the bulb then always fits the smaller side.
  float fitScale = min(uRes.x / uRes.y, 1.0);
  float focal = (0.5 / tan(uFov * 0.5)) * fitScale;
  vec3 rd = normalize(uv.x * right + uv.y * up + focal * fwd);
  vec3 ro = uCamPos;
  float pixelAngle = 1.0 / (uRes.y * focal);

  // background: a soft warm-to-cool pool of light - like a spotlight
  // falling on black velvet - offset toward the key light, fading fast
  // to near-black at the edges so the shell's dark UI reads cleanly on top.
  // Normalized by the SHORTER side (not uRes.y) so the pool stays a
  // compact, roughly circular glow on a tall/narrow phone canvas too -
  // uv itself is stretched vertically there and would otherwise leave
  // the "warm" zone covering almost the entire portrait screen.
  vec2 uvBg = (gl_FragCoord.xy - 0.5 * uRes) / min(uRes.x, uRes.y);
  vec2 poolCenter = vec2(-0.12, 0.10);
  float rUV = length(uvBg - poolCenter);
  vec3 bgWarm = vec3(0.075, 0.061, 0.046);
  vec3 bgCool = vec3(0.013, 0.015, 0.023);
  vec3 bgEdge = vec3(0.0020, 0.0023, 0.0035);
  vec3 bg = mix(bgWarm, bgCool, smoothstep(0.0, 0.55, rUV));
  bg = mix(bg, bgEdge, smoothstep(0.45, 1.15, rUV));

  bool hit = false;
  vec4 trap = vec4(1e5);
  float glow = 0.0;
  float t = 0.0;

  // bounding-sphere early out: the bulb never strays past this radius, so
  // rays that miss it entirely skip straight to the background.
  const float BOUND = 1.3;
  float b = dot(ro, rd);
  float cq = dot(ro, ro) - BOUND * BOUND;
  float h2 = b * b - cq;
  if (h2 > 0.0) {
    float sh = sqrt(h2);
    t = max(-b - sh, 0.0);
    float tFar = -b + sh;
    for (int i = 0; i < 140; i++) {
      if (t > tFar) break;
      vec3 p = ro + rd * t;
      vec4 tr;
      float d = mandelbulbDE(p, uPower, tr);
      glow += exp(-d * 34.0) * 0.045;
      // epsilon scales with the ray's footprint (distance travelled times
      // the angle one pixel subtends) so marching stops a fraction of a
      // pixel from the true surface instead of chasing sub-pixel detail
      // that would otherwise alias into stippled noise at low quality.
      float hitEps = max(pixelAngle * t * 1.3, 0.0004);
      if (d < hitEps) { hit = true; trap = tr; break; }
      t += d * 0.92; // relaxed step: the DE overshoots slightly at high n
    }
  }

  vec3 col = bg;
  if (hit) {
    vec3 p = ro + rd * t;
    // the normal-estimation offset scales with the same ray footprint as
    // hitEps, for the same reason: a fixed tiny offset resolves detail
    // finer than a pixel can represent and reads back as speckle.
    float nEps = clamp(pixelAngle * t * 1.0, 0.00015, 0.01);
    vec3 n = calcNormal(p, uPower, nEps);
    float ao = calcAO(p, n, uPower);
    vec3 albedo = shadeAlbedo(trap, ao, p);

    // key: one strong warm ~3200K spotlight from upper-left. This alone
    // does most of the modelling - everything else is a supporting light.
    vec3 lightDir = normalize(vec3(-0.55, 0.78, 0.30));
    float diff = max(dot(n, lightDir), 0.0);
    float shadow = softShadow(p + n * 0.003, lightDir, uPower);
    vec3 keyColor = vec3(1.0, 0.78, 0.53); // ~3200K tungsten
    vec3 key = keyColor * pow(diff, 1.4) * shadow * 1.3;

    // fill: much weaker and cool - stands in for bounced light so the
    // shadow side doesn't go pure flat; AO does the real darkening.
    vec3 fillColor = vec3(0.026, 0.046, 0.082);
    vec3 fill = fillColor * (0.5 + 0.5 * n.y);

    // rim/back light: a cool edge light that brightens the silhouette as
    // seen from the camera, independent of the key direction - contrasts
    // with the warm key like a spotlight's rim falling into a dark room.
    float rim = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
    vec3 rimCol = vec3(0.42, 0.56, 0.60) * rim * 0.55;

    // specular: Blinn-Phong (a tight core plus a softer, wider lobe) times
    // a Schlick fresnel term - a cheap GGX-lite that gives the ivory a
    // polished sheen that blooms a little at grazing angles, rather than a
    // single hard plastic dot. Masked to the paler albedo so patina stays matte.
    vec3 halfV = normalize(lightDir - rd);
    float NoH = max(dot(n, halfV), 0.0);
    float NoV = max(dot(n, -rd), 0.0);
    float fres = 0.04 + 0.96 * pow(1.0 - NoV, 5.0);
    float spec = (pow(NoH, 70.0) * 0.7 + pow(NoH, 16.0) * 0.3) * fres * shadow;
    float albedoLum = dot(albedo, vec3(0.299, 0.587, 0.114));
    float shineMask = smoothstep(0.22, 0.6, albedoLum);

    vec3 lit = albedo * (key + fill * ao) + rimCol * ao
             + vec3(1.0, 0.95, 0.88) * spec * shineMask * 0.9;

    // ambient occlusion pushes deep folds toward near-black rather than
    // flat grey - this is most of what makes the surface read as carved
    // stone rather than a uniformly-lit blob, and gives the wide range
    // from bright bone highlights down to near-black recesses.
    vec3 shadowTint = vec3(0.009, 0.015, 0.030);
    lit *= mix(shadowTint, vec3(1.0), pow(ao, 1.6));

    col = mix(lit, bg, smoothstep(BOUND * 2.0, 9.0, t));
  }

  // faint atmospheric haze from near-miss rays - kept subtle so it reads as
  // depth/fog rather than a glow outlining the whole silhouette.
  vec3 glowColor = vec3(0.16, 0.32, 0.28);
  col += glowColor * min(glow, 0.7) * 0.10;

  col = tonemap(col * 1.0);
  col = pow(max(col, 0.0), vec3(1.0 / 2.2));
  col *= 1.0 - 0.30 * dot(uv, uv); // vignette
  col += (hash21(gl_FragCoord.xy * 1.7 + 11.0) - 0.5) * 0.006; // de-band

  fragColor = vec4(col, 1.0);
}
`;

  function compileShader(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.error('mandelbulb shader compile error:', gl.getShaderInfoLog(sh));
      gl.deleteShader(sh);
      return null;
    }
    return sh;
  }

  function buildProgram(gl) {
    const vs = compileShader(gl, gl.VERTEX_SHADER, VERT_SRC);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
    if (!vs || !fs) return null;
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error('mandelbulb program link error:', gl.getProgramInfoLog(prog));
      gl.deleteProgram(prog);
      return null;
    }
    return prog;
  }

  SCENES.mandelbulb = {
    id: 'mandelbulb',
    name: 'Мандельбульб',
    tagline: 'Трёхмерный фрактал: бесконечная деталь из одной строчки формулы',
    about:
      'Возьмите точку, возведите её в n-ю степень в сферических координатах и прибавьте исходную ' +
      'точку — и повторяйте. Точки, которые не улетают в бесконечность, образуют эту минеральную ' +
      'поверхность. Приблизьтесь — и деталей меньше не станет: показатель степени медленно ' +
      'меняется, и в породе прорастают новые грани.',
    spec: [
      ['Формула', 'z ↦ zⁿ + c'],
      ['Степень', 'n = 6…9, плавно'],
      ['Итераций', '9'],
      ['Оценка расстояния', '½·ln r · r / dr'],
      ['Шагов луча', 'до 140'],
      ['Освещение', 'мягкие тени, AO, орбитальные ловушки']
    ],
    hint: 'Тяните — вращение · колесо или щипок — зум',
    accent: '#C68A56',

    mount(host, ui) {
      const canvas = document.createElement('canvas');
      canvas.style.display = 'block';
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      canvas.style.touchAction = 'none';
      host.appendChild(canvas);

      const gl = canvas.getContext('webgl2');
      if (!gl) {
        const div = document.createElement('div');
        div.className = 'scene-fallback';
        div.textContent = 'Для этой сцены нужен WebGL2 — откройте страницу в свежем Chrome, Firefox или Safari.';
        host.appendChild(div);
        return {
          unmount() {
            host.removeChild(div);
            host.removeChild(canvas);
          }
        };
      }

      const program = buildProgram(gl);
      let raf = 0;
      let disposed = false;

      // -- geometry: one full-screen triangle -----------------------------
      const vbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);

      const uniforms = {};
      if (program) {
        ['uRes', 'uTime', 'uPower', 'uCamPos', 'uCamTarget', 'uFov'].forEach((name) => {
          uniforms[name] = gl.getUniformLocation(program, name);
        });
      }

      // -- camera state -----------------------------------------------------
      const DEFAULTS = { azimuth: 0.6, elevation: 0.22, zoom: 1.0 };
      let azimuth = DEFAULTS.azimuth;
      let elevation = DEFAULTS.elevation;
      let zoomTarget = DEFAULTS.zoom;
      let zoom = DEFAULTS.zoom; // damped toward zoomTarget each frame
      let lastInteract = -999;

      // -- toggles ------------------------------------------------------
      let morphing = true;
      let paused = false;
      let power = 8.0;

      // -- adaptive quality (contract: heavy ray-marcher starts at 0.5) ---
      const fixedQ = host.dataset.fixedQuality ? parseFloat(host.dataset.fixedQuality) : null;
      let quality = fixedQ != null && !Number.isNaN(fixedQ) ? fixedQ : 0.5;
      let frameTimeSum = 0;
      let frameTimeCount = 0;

      function resize() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.max(1, Math.round(host.clientWidth * dpr * quality));
        const h = Math.max(1, Math.round(host.clientHeight * dpr * quality));
        if (canvas.width !== w || canvas.height !== h) {
          canvas.width = w;
          canvas.height = h;
        }
      }
      resize();
      const ro = new ResizeObserver(() => resize());
      ro.observe(host);

      // -- pointer input: drag to rotate, wheel / pinch to zoom -----------
      const pointers = new Map();
      let pinchStartDist = 0;
      let pinchStartZoom = 1;
      // release inertia: a single-finger drag leaves behind an angular
      // velocity (rad/s) that keeps spinning the view and decays away.
      let velAzimuth = 0;
      let velElevation = 0;
      let lastMoveTime = 0;

      function pointerDist() {
        const pts = Array.from(pointers.values());
        const dx = pts[0].x - pts[1].x;
        const dy = pts[0].y - pts[1].y;
        return Math.hypot(dx, dy);
      }

      function onPointerDown(e) {
        canvas.setPointerCapture(e.pointerId);
        pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
        lastInteract = performance.now() / 1000;
        lastMoveTime = lastInteract;
        // grabbing the view again cancels any fling still playing out
        velAzimuth = 0;
        velElevation = 0;
        if (pointers.size === 2) {
          pinchStartDist = pointerDist();
          pinchStartZoom = zoomTarget;
        }
      }
      function onPointerMove(e) {
        const pt = pointers.get(e.pointerId);
        if (!pt) return;
        const dx = e.clientX - pt.x;
        const dy = e.clientY - pt.y;
        const now = performance.now() / 1000;
        lastInteract = now;
        if (pointers.size === 1) {
          const dAz = -dx * 0.0055;
          const dEl = -dy * 0.0055;
          azimuth += dAz;
          elevation = Math.max(-1.3, Math.min(1.3, elevation + dEl));
          // instantaneous drag speed, smoothed a little, becomes the fling
          // velocity carried into the release-inertia phase below. Clamped
          // so a very short dtMove (e.g. two synthetic events fired back
          // to back) can't produce an absurd speed that takes far longer
          // than intended to decay away.
          const dtMove = Math.max(1 / 240, now - lastMoveTime);
          velAzimuth = clamp(velAzimuth + (dAz / dtMove - velAzimuth) * 0.6, -9, 9);
          velElevation = clamp(velElevation + (dEl / dtMove - velElevation) * 0.6, -9, 9);
          lastMoveTime = now;
          pt.x = e.clientX;
          pt.y = e.clientY;
        } else if (pointers.size === 2) {
          pt.x = e.clientX;
          pt.y = e.clientY;
          const d = pointerDist();
          if (pinchStartDist > 1) {
            zoomTarget = clamp(pinchStartZoom * (pinchStartDist / Math.max(d, 1)), 0.45, 2.4);
          }
        }
      }
      function onPointerUp(e) {
        pointers.delete(e.pointerId);
        if (pointers.size < 2) { pinchStartDist = 0; }
        lastInteract = performance.now() / 1000;
      }
      function onWheel(e) {
        e.preventDefault();
        zoomTarget = clamp(zoomTarget * Math.exp(e.deltaY * 0.0011), 0.45, 2.4);
        lastInteract = performance.now() / 1000;
      }
      function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

      canvas.addEventListener('pointerdown', onPointerDown);
      canvas.addEventListener('pointermove', onPointerMove);
      canvas.addEventListener('pointerup', onPointerUp);
      canvas.addEventListener('pointercancel', onPointerUp);
      canvas.addEventListener('wheel', onWheel, { passive: false });

      // -- controls ---------------------------------------------------------
      const btnMorph = document.createElement('button');
      btnMorph.type = 'button';
      btnMorph.className = 'ctl';
      btnMorph.textContent = 'Морфинг';
      btnMorph.setAttribute('aria-pressed', 'true');
      btnMorph.addEventListener('click', () => {
        morphing = !morphing;
        btnMorph.setAttribute('aria-pressed', String(morphing));
      });

      const btnPause = document.createElement('button');
      btnPause.type = 'button';
      btnPause.className = 'ctl';
      btnPause.textContent = 'Пауза';
      btnPause.setAttribute('aria-pressed', 'false');
      btnPause.addEventListener('click', () => togglePause());

      const btnReset = document.createElement('button');
      btnReset.type = 'button';
      btnReset.className = 'ctl';
      btnReset.textContent = 'Сброс';
      btnReset.addEventListener('click', () => reset());

      ui.appendChild(btnMorph);
      ui.appendChild(btnPause);
      ui.appendChild(btnReset);

      function togglePause() {
        paused = !paused;
        btnPause.setAttribute('aria-pressed', String(paused));
      }
      function reset() {
        azimuth = DEFAULTS.azimuth;
        elevation = DEFAULTS.elevation;
        zoomTarget = DEFAULTS.zoom;
        zoom = DEFAULTS.zoom;
        simTime = 0;
        paused = false;
        btnPause.setAttribute('aria-pressed', 'false');
      }

      function onKeyDown(e) {
        const target = e.target;
        if (target && (target.tagName === 'INPUT' || target.tagName === 'BUTTON')) return;
        if (e.code === 'Space') { e.preventDefault(); togglePause(); }
        else if (e.code === 'KeyR') { reset(); }
      }
      window.addEventListener('keydown', onKeyDown);

      // -- animation loop -----------------------------------------------
      const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)');
      let simTime = 0;
      let last = performance.now() / 1000;

      function frame(now) {
        raf = requestAnimationFrame(frame);
        const t0 = performance.now();
        const nowS = t0 / 1000;
        const dt = Math.min(0.05, nowS - last);
        last = nowS;
        if (!paused) simTime += dt;

        // fractal breathing: power drifts between ~6 and ~9, or eases to
        // exactly 8 when morphing is frozen off.
        if (morphing) {
          power = 7.5 + 1.5 * Math.sin(simTime * 0.08);
        } else {
          power += (8.0 - power) * Math.min(1, dt * 1.5);
        }

        // release inertia: while no finger is on the canvas, spend down the
        // velocity left over from the last drag - an exponential decay
        // with a ~0.2s time constant, i.e. essentially stopped by ~0.6s.
        if (pointers.size === 0 && !paused && (Math.abs(velAzimuth) > 1e-4 || Math.abs(velElevation) > 1e-4)) {
          azimuth += velAzimuth * dt;
          elevation = Math.max(-1.3, Math.min(1.3, elevation + velElevation * dt));
          const decay = Math.exp(-dt * 5.0);
          velAzimuth *= decay;
          velElevation *= decay;
        }

        const motionScale = reduceMotion && reduceMotion.matches ? 0.35 : 1.0;
        const idle = nowS - lastInteract > 4.0;
        if (idle && !paused) {
          azimuth += 0.028 * motionScale * dt;
          const elevTarget = 0.20 * Math.sin(simTime * 0.037);
          elevation += (elevTarget - elevation) * Math.min(1, dt * 0.5);
        }
        // damp the user's zoom toward its target so wheel/pinch input eases
        // in rather than snapping the camera distance in one frame.
        zoom += (zoomTarget - zoom) * Math.min(1, dt * 8);

        // orbit distance breathes slowly between a full-specimen view (bulb
        // filling ~60% of the shorter side) and a close pass where the
        // surface fills most of the frame and fine detail shows; zoom
        // scales on top. The lower safety clamp keeps the camera from ever
        // punching through the bounding sphere even at maximum pinch-in.
        const distBreathe = 2.6 + 1.6 * (0.5 + 0.5 * Math.sin(simTime * 0.05 * motionScale + 1.3));
        const distance = clamp(distBreathe * zoom, 1.7, 9.0);

        const camTarget = [0, 0, 0];
        const camPos = [
          distance * Math.cos(elevation) * Math.sin(azimuth),
          distance * Math.sin(elevation),
          distance * Math.cos(elevation) * Math.cos(azimuth)
        ];

        if (program) {
          gl.viewport(0, 0, canvas.width, canvas.height);
          gl.useProgram(program);
          gl.bindVertexArray(vao);
          gl.uniform2f(uniforms.uRes, canvas.width, canvas.height);
          gl.uniform1f(uniforms.uTime, simTime);
          gl.uniform1f(uniforms.uPower, power);
          gl.uniform3f(uniforms.uCamPos, camPos[0], camPos[1], camPos[2]);
          gl.uniform3f(uniforms.uCamTarget, camTarget[0], camTarget[1], camTarget[2]);
          gl.uniform1f(uniforms.uFov, 38 * Math.PI / 180);
          gl.drawArrays(gl.TRIANGLES, 0, 3);
          gl.bindVertexArray(null);
        }

        if (fixedQ == null) {
          const ft = performance.now() - t0;
          frameTimeSum += ft;
          frameTimeCount++;
          if (frameTimeCount >= 30) {
            const avg = frameTimeSum / frameTimeCount;
            if (avg > 22) quality = Math.max(0.3, quality * 0.85);
            else if (avg < 12) quality = Math.min(1, quality * 1.1);
            frameTimeSum = 0;
            frameTimeCount = 0;
            resize();
          }
        }
      }
      raf = requestAnimationFrame(frame);

      return {
        unmount() {
          if (disposed) return;
          disposed = true;
          cancelAnimationFrame(raf);
          ro.disconnect();
          canvas.removeEventListener('pointerdown', onPointerDown);
          canvas.removeEventListener('pointermove', onPointerMove);
          canvas.removeEventListener('pointerup', onPointerUp);
          canvas.removeEventListener('pointercancel', onPointerUp);
          canvas.removeEventListener('wheel', onWheel);
          window.removeEventListener('keydown', onKeyDown);
          if (program) gl.deleteProgram(program);
          gl.deleteBuffer(vbo);
          gl.deleteVertexArray(vao);
          gl.getExtension('WEBGL_lose_context')?.loseContext();
          host.removeChild(canvas);
          ui.innerHTML = '';
        }
      };
    }
  };
})();

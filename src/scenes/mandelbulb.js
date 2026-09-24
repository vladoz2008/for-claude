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

// cheap 2D hash, used for a sub-pixel jitter (poor-man's AA) and to
// dither the dark background so smooth gradients don't band.
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

vec3 calcNormal(vec3 p, float power) {
  const float e = 0.0006;
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
  return clamp(1.0 - 1.5 * occ, 0.0, 1.0);
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

  // crevices - wherever the surface is occluded (low ao) - pick up an
  // oxidised patina. Which of copper or verdigris dominates a given fold
  // is a slow spatial pattern over the *surface point itself*, not a trap
  // channel - the trap metrics dip in lockstep with ao at every crease, so
  // keying hue off them would always pick the same colour where it matters.
  float marble = sin(p.x * 5.0) * sin(p.y * 4.3 + 1.7) * sin(p.z * 6.1 + 0.4);
  float hueMix = smoothstep(-0.2, 0.2, marble);
  vec3 patina = mix(verdigris, copper, hueMix);
  float crevice = 1.0 - smoothstep(0.28, 0.82, ao);
  return mix(base, patina, crevice * 0.85);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uRes) / uRes.y;
  // tiny per-pixel, per-frame jitter: softens staircasing on the silhouette
  // once the canvas is bilinearly upscaled to CSS size.
  float j = hash21(gl_FragCoord.xy + uTime * 61.0);
  uv += (j - 0.5) * (1.0 / uRes.y);

  vec3 fwd = normalize(uCamTarget - uCamPos);
  vec3 right = normalize(cross(fwd, vec3(0.0, 1.0, 0.0)));
  vec3 up = cross(right, fwd);
  // uv spans roughly [-0.5, 0.5] vertically, so the focal length that maps
  // that span to the requested vertical FOV is 0.5 / tan(fov/2), not 1/tan(fov/2).
  float focal = 0.5 / tan(uFov * 0.5);
  vec3 rd = normalize(uv.x * right + uv.y * up + focal * fwd);
  vec3 ro = uCamPos;
  float pixelAngle = 1.0 / (uRes.y * focal);

  // subtle radial gradient behind everything, plus room for glow to build on
  vec3 bg = mix(vec3(0.035, 0.039, 0.052), vec3(0.006, 0.007, 0.012),
                 smoothstep(0.0, 1.15, length(uv)));

  bool hit = false;
  vec4 trap = vec4(1e5);
  float glow = 0.0;
  float t = 0.0;

  // bounding-sphere early out: the bulb never strays past this radius, so
  // rays that miss it entirely skip straight to the background.
  const float BOUND = 1.25;
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
      float hitEps = max(pixelAngle * t * 0.6, 0.0006);
      if (d < hitEps) { hit = true; trap = tr; break; }
      t += d * 0.92; // relaxed step: the DE overshoots slightly at high n
    }
  }

  vec3 col = bg;
  if (hit) {
    vec3 p = ro + rd * t;
    vec3 n = calcNormal(p, uPower);
    float ao = calcAO(p, n, uPower);
    // crevice colour is driven by ambient occlusion, not the orbit trap
    // alone, so the verdigris/copper patina lands exactly where the
    // surface folds into shadow - like oxidation pooling in real crevices.
    vec3 albedo = shadeAlbedo(trap, ao, p);

    // a raking, slightly side-on key light carves a clear light/dark
    // terminator across the lobes instead of a flat, evenly-lit blob.
    vec3 lightDir = normalize(vec3(0.30, 0.70, 0.55));
    float diff = max(dot(n, lightDir), 0.0);
    float shadow = softShadow(p + n * 0.0025, lightDir, uPower);

    vec3 halfV = normalize(lightDir - rd);
    float spec = pow(max(dot(n, halfV), 0.0), 34.0) * shadow;

    vec3 key = vec3(1.0, 0.96, 0.88) * pow(diff, 1.3) * shadow;
    vec3 fill = vec3(0.045, 0.07, 0.10) * (0.5 + 0.5 * n.y) * ao;
    float rim = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
    vec3 rimCol = vec3(0.28, 0.55, 0.47) * rim * 0.4 * ao;

    vec3 lit = albedo * (key * 1.05 + fill) + rimCol + vec3(1.0, 0.97, 0.9) * spec * 0.4;

    // deep, occluded folds go toward blue-black rather than flat grey -
    // this is what reads as "shadow" on a mineral specimen rather than haze.
    vec3 shadowTint = vec3(0.02, 0.035, 0.07);
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
      let zoom = DEFAULTS.zoom;
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
        if (pointers.size === 2) {
          pinchStartDist = pointerDist();
          pinchStartZoom = zoom;
        }
      }
      function onPointerMove(e) {
        const pt = pointers.get(e.pointerId);
        if (!pt) return;
        const dx = e.clientX - pt.x;
        const dy = e.clientY - pt.y;
        lastInteract = performance.now() / 1000;
        if (pointers.size === 1) {
          azimuth -= dx * 0.0055;
          elevation = Math.max(-1.3, Math.min(1.3, elevation - dy * 0.0055));
          pt.x = e.clientX;
          pt.y = e.clientY;
        } else if (pointers.size === 2) {
          pt.x = e.clientX;
          pt.y = e.clientY;
          const d = pointerDist();
          if (pinchStartDist > 1) {
            zoom = clamp(pinchStartZoom * (pinchStartDist / Math.max(d, 1)), 0.45, 2.4);
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
        zoom = clamp(zoom * Math.exp(e.deltaY * 0.0011), 0.45, 2.4);
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

        const motionScale = reduceMotion && reduceMotion.matches ? 0.35 : 1.0;
        const idle = nowS - lastInteract > 4.0;
        if (idle) {
          azimuth += 0.028 * motionScale * dt;
          const elevTarget = 0.20 * Math.sin(simTime * 0.037);
          elevation += (elevTarget - elevation) * Math.min(1, dt * 0.5);
        }
        // orbit distance breathes slowly between a full-specimen view and a
        // closer pass that reveals fine surface detail; zoom scales on top.
        const distBreathe = 3.6 + 2.2 * (0.5 + 0.5 * Math.sin(simTime * 0.05 * motionScale + 1.3));
        const distance = clamp(distBreathe * zoom, 1.3, 9.0);

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

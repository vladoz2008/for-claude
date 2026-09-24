// DEV STUB — stands in for src/scenes/lenia.js while the shell is being
// built and screenshotted. Not part of the shipped product. Same mount/unmount
// contract as a real scene, minus the real cellular automaton.
(function () {
  'use strict';
  const SCENES = (window.SCENES = window.SCENES || {});

  const VERT = `#version 300 es
const vec2 POS[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
void main() {
  gl_Position = vec4(POS[gl_VertexID], 0.0, 1.0);
}`;

  const FRAG = `#version 300 es
precision highp float;
uniform vec2 uResolution;
uniform float uTime;
uniform vec3 uAccent;
uniform vec2 uPhase;
uniform float uZoom;
uniform float uKernel;
out vec4 outColor;
void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y;
  uv = uv * uZoom + uPhase;
  float n = sin(uv.x * 8.0 + uTime) * sin(uv.y * 8.0 - uTime * 0.7)
          + sin((uv.x + uv.y) * 5.0 + uTime * 1.3) * 0.6;
  float field = smoothstep(-0.2, 1.1, n);
  vec3 col = mix(vec3(0.02, 0.03, 0.035), uAccent, field);
  // "Ядро" toggle: overlays a faint ring at the convolution kernel's radius.
  float ring = 1.0 - smoothstep(0.0, 0.02, abs(length(uv) - 0.5));
  col += uKernel * ring * uAccent * 0.6;
  outColor = vec4(col, 1.0);
}`;

  SCENES.lenia = {
    id: 'lenia',
    name: 'Ления',
    tagline: 'Заглушка для проверки вёрстки оболочки — не боевая сцена искусственной жизни.',
    about:
      'Черновая заглушка: слоистый sin-узор на полноэкранном WebGL2-шейдере имитирует органические ' +
      'пятна колонии. Нужна только затем, чтобы увидеть, как реальный канвас выглядит под ' +
      'интерфейсом оболочки, прежде чем настоящая сцена будет готова.',
    spec: [
      ['Рендер', 'WebGL2, 1 draw call'],
      ['Разрешение', 'адаптивное 0.3–1.0'],
      ['Узор', 'слоистые sin-волны'],
      ['Ядро', 'имитация ring-kernel'],
      ['DPR', 'min(devicePixelRatio, 2)']
    ],
    hint: 'Тяните — панорама · колесо или щипок — зум',
    accent: '#8fd8c5',
    mount(host, ui) {
      return createStub(host, ui, FRAG, hexToRgb01('#8fd8c5'));
    }
  };

  function hexToRgb01(hex) {
    const n = parseInt(hex.replace('#', ''), 16);
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  }

  function compile(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(sh));
      gl.deleteShader(sh);
      return null;
    }
    return sh;
  }

  function clamp(v, a, b) {
    return Math.min(b, Math.max(a, v));
  }

  function createStub(host, ui, fragSrc, accentRgb) {
    const canvas = document.createElement('canvas');
    canvas.style.display = 'block';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.touchAction = 'none';
    host.appendChild(canvas);

    const gl = canvas.getContext('webgl2');
    if (!gl) {
      const fb = document.createElement('div');
      fb.className = 'scene-fallback';
      fb.textContent = 'Для этой сцены нужен WebGL2 — откройте страницу в свежем Chrome, Firefox или Safari.';
      host.appendChild(fb);
      return {
        unmount() {
          host.innerHTML = '';
          ui.innerHTML = '';
        }
      };
    }

    const vs = compile(gl, gl.VERTEX_SHADER, VERT);
    const fs = compile(gl, gl.FRAGMENT_SHADER, fragSrc);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error(gl.getProgramInfoLog(prog));
    }
    gl.useProgram(prog);

    const uResolution = gl.getUniformLocation(prog, 'uResolution');
    const uTime = gl.getUniformLocation(prog, 'uTime');
    const uAccent = gl.getUniformLocation(prog, 'uAccent');
    const uPhase = gl.getUniformLocation(prog, 'uPhase');
    const uZoom = gl.getUniformLocation(prog, 'uZoom');
    const uKernel = gl.getUniformLocation(prog, 'uKernel');

    const fixedQuality = !!host.dataset.fixedQuality;
    let quality = fixedQuality ? parseFloat(host.dataset.fixedQuality) : 0.6;
    let paused = false;
    let kernelView = false;
    let phase = [0, 0];
    let zoom = 1;
    let last = performance.now();
    let elapsed = 0;
    let frameTimes = [];

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(host.clientWidth * dpr * quality));
      const h = Math.max(1, Math.round(host.clientHeight * dpr * quality));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      gl.viewport(0, 0, canvas.width, canvas.height);
    }

    const ro = new ResizeObserver(resize);
    ro.observe(host);
    resize();

    const pointers = new Map();
    let dragLast = null;
    let pinchDist = null;

    function onPointerDown(e) {
      canvas.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 1) dragLast = { x: e.clientX, y: e.clientY };
      if (pointers.size === 2) pinchDist = null;
    }
    function onPointerMove(e) {
      if (!pointers.has(e.pointerId)) return;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 1 && dragLast) {
        const dx = (e.clientX - dragLast.x) / host.clientHeight;
        const dy = (e.clientY - dragLast.y) / host.clientHeight;
        phase[0] -= dx * zoom;
        phase[1] += dy * zoom;
        dragLast = { x: e.clientX, y: e.clientY };
      } else if (pointers.size === 2) {
        const pts = Array.from(pointers.values());
        const d = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        if (pinchDist) zoom = clamp(zoom * (pinchDist / d), 0.3, 4);
        pinchDist = d;
      }
    }
    function onPointerUp(e) {
      pointers.delete(e.pointerId);
      pinchDist = null;
      dragLast = pointers.size === 1 ? Array.from(pointers.values())[0] : null;
    }
    function onWheel(e) {
      e.preventDefault();
      zoom = clamp(zoom * Math.pow(1.0015, e.deltaY), 0.3, 4);
    }

    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointercancel', onPointerUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });

    function onKey(e) {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON')) return;
      if (e.code === 'Space') {
        paused = !paused;
        e.preventDefault();
      }
      if (e.code === 'KeyR') {
        phase = [0, 0];
        zoom = 1;
        elapsed = 0;
      }
    }
    window.addEventListener('keydown', onKey);

    const btnPause = document.createElement('button');
    btnPause.type = 'button';
    btnPause.className = 'ctl';
    btnPause.textContent = 'Пауза';
    btnPause.setAttribute('aria-pressed', 'false');
    btnPause.addEventListener('click', () => {
      paused = !paused;
      btnPause.setAttribute('aria-pressed', String(paused));
    });
    const btnReset = document.createElement('button');
    btnReset.type = 'button';
    btnReset.className = 'ctl';
    btnReset.textContent = 'Сброс вида';
    btnReset.addEventListener('click', () => {
      phase = [0, 0];
      zoom = 1;
      elapsed = 0;
    });
    const btnKernel = document.createElement('button');
    btnKernel.type = 'button';
    btnKernel.className = 'ctl';
    btnKernel.textContent = 'Ядро';
    btnKernel.setAttribute('aria-pressed', 'false');
    btnKernel.addEventListener('click', () => {
      kernelView = !kernelView;
      btnKernel.setAttribute('aria-pressed', String(kernelView));
    });
    ui.appendChild(btnPause);
    ui.appendChild(btnReset);
    ui.appendChild(btnKernel);

    let raf = 0;
    function frame(now) {
      raf = requestAnimationFrame(frame);
      const dt = (now - last) / 1000;
      last = now;
      if (!paused) elapsed += dt;

      frameTimes.push(dt * 1000);
      if (frameTimes.length >= 30) {
        const avg = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
        frameTimes = [];
        if (!fixedQuality) {
          if (avg > 22) quality = clamp(quality * 0.85, 0.3, 1);
          else if (avg < 12) quality = clamp(quality * 1.1, 0.3, 1);
          resize();
        }
      }

      gl.useProgram(prog);
      gl.uniform2f(uResolution, canvas.width, canvas.height);
      gl.uniform1f(uTime, elapsed);
      gl.uniform3f(uAccent, accentRgb[0], accentRgb[1], accentRgb[2]);
      gl.uniform2f(uPhase, phase[0], phase[1]);
      gl.uniform1f(uZoom, zoom);
      gl.uniform1f(uKernel, kernelView ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    raf = requestAnimationFrame(frame);

    return {
      unmount() {
        cancelAnimationFrame(raf);
        ro.disconnect();
        window.removeEventListener('keydown', onKey);
        canvas.removeEventListener('pointerdown', onPointerDown);
        canvas.removeEventListener('pointermove', onPointerMove);
        canvas.removeEventListener('pointerup', onPointerUp);
        canvas.removeEventListener('pointercancel', onPointerUp);
        canvas.removeEventListener('wheel', onWheel);
        gl.deleteProgram(prog);
        gl.deleteShader(vs);
        gl.deleteShader(fs);
        const lose = gl.getExtension('WEBGL_lose_context');
        if (lose) lose.loseContext();
        host.innerHTML = '';
        ui.innerHTML = '';
      }
    };
  }
})();

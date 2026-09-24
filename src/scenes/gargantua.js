(function () {
  'use strict';
  const SCENES = (window.SCENES = window.SCENES || {});

  // ===========================================================================================
  // Гаргантюа — real-time Schwarzschild black hole ray tracer.
  //
  // Physics summary (see comments inline for the exact formulas used):
  //  - Units: Schwarzschild radius r_s = 1, black hole at the origin.
  //  - Photons are traced BACKWARDS from the camera through flat 3D space, but bent by the
  //    "fake curvature" acceleration a = -1.5 h^2 x / |x|^5, h = |x x v| (angular momentum,
  //    conserved). This is the well known cheap-but-exact trick that reproduces the Schwarzschild
  //    null-geodesic orbit equation d^2u/dphi^2 + u = 1.5 r_s u^2 (u = 1/r) without touching a
  //    metric tensor: it gives the correct photon sphere at r = 1.5 r_s and shadow at ~2.6 r_s.
  //  - The thin accretion disk (y = 0 plane) is sampled every time the ray crosses the plane;
  //    because bent rays cross the plane more than once, the far side of the disk reappears
  //    arched over and under the shadow, plus a thin photon ring - all for free from the geometry.
  // ===========================================================================================

  // ---------------------------------------------------------------------------------------------
  // Shader sources
  // ---------------------------------------------------------------------------------------------

  // Full-screen triangle, no vertex buffers needed (uses gl_VertexID).
  const VERT_SRC = `#version 300 es
    const vec2 POS[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    out vec2 vUv;
    void main() {
      vec2 p = POS[gl_VertexID];
      vUv = p * 0.5 + 0.5;
      gl_Position = vec4(p, 0.0, 1.0);
    }
  `;

  const RAYTRACE_FRAG = `#version 300 es
    precision highp float;
    in vec2 vUv;
    out vec4 outColor;

    uniform vec2 u_resolution;
    uniform vec3 u_camPos;
    uniform vec3 u_camRight;
    uniform vec3 u_camUp;
    uniform vec3 u_camForward;
    uniform float u_tanHalfFov;
    uniform float u_aspect;
    uniform float u_diskTime;   // wall-clock time for disk turbulence, frozen while paused
    uniform float u_doppler;    // 1.0 = relativistic shading on, 0.0 = off (comparison toggle)
    uniform int u_maxSteps;     // adaptive step budget (quality-scaled)
    uniform float u_pixelAngle; // angular footprint of one output pixel (radians) - used to size
                                 // stars as true pinpoints regardless of render resolution

    // ---- physical constants (r_s = 1) --------------------------------------------------------
    const float DISK_INNER = 3.0;    // ISCO for Schwarzschild: r = 3 r_s
    const float DISK_OUTER = 13.0;
    const float T0 = 9200.0;         // reference temperature scale for the blackbody map (K):
                                      // tuned so the disk spans white-hot / pale-yellow inner gas
                                      // to deep orange/amber at the outer edge, like the real
                                      // T(r) profile (peak near the inner edge, r^-0.75 falloff).
    const float BASE_BRIGHT = 1.75;
    const float OMEGA0 = 1.15;       // Keplerian angular-speed constant: omega(r) = OMEGA0 * r^-1.5
    const int MAX_STEPS_CONST = 225; // hard cap the compiler can prove termination on

    // ---- small hash / noise toolbox ----------------------------------------------------------
    float hash21(vec2 p) {
      p = fract(p * vec2(123.34, 456.21));
      p += dot(p, p + 45.32);
      return fract(p.x * p.y);
    }
    float hash31(vec3 p) {
      p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
      p *= 17.0;
      return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
    }
    float noise2(vec2 p) {
      vec2 i = floor(p), f = fract(p);
      float a = hash21(i), b = hash21(i + vec2(1.0, 0.0));
      float c = hash21(i + vec2(0.0, 1.0)), d = hash21(i + vec2(1.0, 1.0));
      vec2 u = f * f * (3.0 - 2.0 * f);
      return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
    }
    float fbm2(vec2 p, int octaves) {
      float v = 0.0, amp = 0.55;
      for (int i = 0; i < 6; i++) {
        if (i >= octaves) break;
        v += amp * noise2(p);
        p = p * 2.03 + 11.0;
        amp *= 0.55;
      }
      return v;
    }
    // 3D value noise (trilinear) - used for the disk turbulence so it can be sampled in plain
    // Cartesian coordinates (rotated into a co-rotating frame) instead of raw polar angle, which
    // has a hard seam at phi = +-pi that shows up as a sharp line through the disk.
    float noise3(vec3 p) {
      vec3 i = floor(p), f = fract(p);
      f = f * f * (3.0 - 2.0 * f);
      float n000 = hash31(i + vec3(0.0, 0.0, 0.0)), n100 = hash31(i + vec3(1.0, 0.0, 0.0));
      float n010 = hash31(i + vec3(0.0, 1.0, 0.0)), n110 = hash31(i + vec3(1.0, 1.0, 0.0));
      float n001 = hash31(i + vec3(0.0, 0.0, 1.0)), n101 = hash31(i + vec3(1.0, 0.0, 1.0));
      float n011 = hash31(i + vec3(0.0, 1.0, 1.0)), n111 = hash31(i + vec3(1.0, 1.0, 1.0));
      float nx00 = mix(n000, n100, f.x), nx10 = mix(n010, n110, f.x);
      float nx01 = mix(n001, n101, f.x), nx11 = mix(n011, n111, f.x);
      return mix(mix(nx00, nx10, f.y), mix(nx01, nx11, f.y), f.z);
    }
    float fbm3(vec3 p, int octaves) {
      float v = 0.0, amp = 0.55;
      for (int i = 0; i < 6; i++) {
        if (i >= octaves) break;
        v += amp * noise3(p);
        p = p * 2.05 + 7.0;
        amp *= 0.55;
      }
      return v;
    }

    // ---- Tanner Helland blackbody approximation (input Kelvin -> linear-ish RGB in [0,1]) ----
    vec3 blackbody(float tempK) {
      float t = clamp(tempK, 1000.0, 40000.0) / 100.0;
      float r, g, b;
      r = t <= 66.0 ? 255.0 : 329.698727446 * pow(t - 60.0, -0.1332047592);
      g = t <= 66.0
        ? 99.4708025861 * log(t) - 161.1195681661
        : 288.1221695283 * pow(t - 60.0, -0.0755148492);
      b = t >= 66.0 ? 255.0 : (t <= 19.0 ? 0.0 : 138.5177312231 * log(t - 10.0) - 305.0447927307);
      return clamp(vec3(r, g, b) / 255.0, 0.0, 1.0);
    }

    // ---- geodesic acceleration: a = -1.5 h^2 x / |x|^5 ----------------------------------------
    vec3 accel(vec3 x, float h2) {
      float r2 = dot(x, x);
      return (-1.5 * h2) * x / (r2 * r2 * sqrt(r2));
    }

    // classic RK4 step of the coupled system dx/dl = v, dv/dl = accel(x)
    void rk4Step(inout vec3 x, inout vec3 v, float h2, float dt) {
      vec3 k1x = v,                    k1v = accel(x, h2);
      vec3 k2x = v + 0.5 * dt * k1v,   k2v = accel(x + 0.5 * dt * k1x, h2);
      vec3 k3x = v + 0.5 * dt * k2v,   k3v = accel(x + 0.5 * dt * k2x, h2);
      vec3 k4x = v + dt * k3v,         k4v = accel(x + dt * k3x, h2);
      x += (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
      v += (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    }

    // ---- accretion disk shading at one plane-crossing point -----------------------------------
    // p: crossing point (y ~ 0). marchDir: the ray's local tangent as WE traced it (camera -> out).
    vec4 diskShade(vec3 p, vec3 marchDir, float rDisk) {
      float edgeFade = smoothstep(DISK_INNER, DISK_INNER + 0.45, rDisk)
                      * smoothstep(DISK_OUTER, DISK_OUTER - 2.2, rDisk);
      if (edgeFade <= 0.002) return vec4(0.0);

      // Procedural turbulence: rotate the hit point BACKWARDS by omega(r)*t around the y axis
      // (a co-rotating frame) and sample noise directly in that Cartesian (x, z, log r) space,
      // instead of in raw (log r, phi). Differential Keplerian shear - inner radii rotate faster
      // - then makes the static noise field appear to spiral as u_diskTime advances. Sampling
      // Cartesian coordinates (rather than the angle phi itself) means there is no branch-cut
      // seam at phi = +-pi: a raw-phi noise coordinate jumps discontinuously there, which shows
      // up as a hard line through the disk; cos/sin-free Cartesian coordinates never jump.
      float omega = OMEGA0 * pow(rDisk, -1.5);
      float ang = -omega * u_diskTime;
      float ca = cos(ang), sa = sin(ang);
      vec2 rot = vec2(ca * p.x - sa * p.z, sa * p.x + ca * p.z);
      vec3 sp = vec3(rot.x, rot.y, log(rDisk) * 4.5);
      // Broad low-frequency layer gives overall coverage/shape; a ridged (folded) detail layer
      // turns the high-frequency component into thin bright filaments instead of soft blotches,
      // matching sheared turbulent gas rather than clouds.
      float base = fbm3(sp * 0.55, 3);
      float detail = fbm3(sp * 1.7 + 23.0, 3);
      float ridged = pow(1.0 - abs(detail * 2.0 - 1.0), 2.0);
      float density = clamp(0.18 + 0.42 * base + 0.55 * ridged, 0.0, 1.0) * edgeFade;

      // Shakura-Sunyaev / Novikov-Thorne style temperature profile with zero-torque inner edge.
      float x = DISK_INNER / rDisk;
      float T = T0 * pow(rDisk / DISK_INNER, -0.75) * pow(max(0.0, 1.0 - sqrt(x)), 0.25);
      // Push the inner disk hotter (white/pale-yellow) and let it fall off harder into the deep
      // orange/amber outer disk, and make the inner edge noticeably more opaque/bright so the
      // near limb reads as a solid bright band rather than a thin translucent haze.
      float innerT = smoothstep(DISK_OUTER * 0.55, DISK_INNER, rDisk);
      float radialBoost = mix(0.55, 2.3, innerT);
      float opacityBoost = mix(0.5, 1.0, innerT);

      float g = 1.0;
      if (u_doppler > 0.5) {
        // Orbital speed of Keplerian gas (beta = 0.5 c at the ISCO by construction).
        float beta = sqrt(0.5 / max(rDisk - 1.0, 0.05));
        float gamma = 1.0 / sqrt(max(1.0 - beta * beta, 1e-4));
        vec3 velDir = normalize(vec3(p.z, 0.0, -p.x));   // tangential orbital direction
        vec3 n = normalize(-marchDir);                    // photon prop. dir, source -> observer
        float cosTheta = dot(velDir, n);
        float dopplerFac = 1.0 / (gamma * (1.0 - beta * cosTheta));
        float gravRedshift = sqrt(max(0.0, 1.0 - 1.0 / rDisk));
        g = dopplerFac * gravRedshift;
      }

      vec3 col = blackbody(T * g);
      // The raw blackbody curve desaturates fast toward white - push it a little further from
      // grey so the hot-white / cool-orange contrast across the disk actually reads on screen.
      float lum = dot(col, vec3(0.299, 0.587, 0.114));
      col = clamp(lum + (col - lum) * 1.5, 0.0, 4.0);
      float intensity = pow(max(g, 0.0001), 4.0);   // relativistic beaming, I_obs ~ g^4 I_emit
      vec3 emit = col * intensity * BASE_BRIGHT * radialBoost * density;
      float alpha = clamp(density * 0.72 * opacityBoost, 0.0, 1.0);
      return vec4(emit, alpha);
    }

    // ---- procedural sky: stars (direction-hashed, smooth discs) + a faint galactic band -------
    // Stars are placed by hashing a cube-face grid (so the hashing itself is O(1) and seam-free
    // in practice), but their SHAPE is measured as a true angular distance on the sphere (dot
    // product), not as Euclidean distance in the gnomonic-projected uv plane - that plane badly
    // stretches shapes away from each face's centre, which is the "distortion" the contract
    // warns about. The only place stars should visibly stretch is near the shadow, where that
    // is real gravitational lensing, not a projection artifact.
    vec3 starLayer(vec3 dir, float density, float thresh) {
      vec3 ad = abs(dir);
      vec2 uv; float faceId; float axisSign;
      if (ad.x >= ad.y && ad.x >= ad.z) {
        uv = dir.yz / ad.x; axisSign = sign(dir.x); faceId = axisSign > 0.0 ? 0.0 : 1.0;
      } else if (ad.y >= ad.x && ad.y >= ad.z) {
        uv = dir.xz / ad.y; axisSign = sign(dir.y); faceId = axisSign > 0.0 ? 2.0 : 3.0;
      } else {
        uv = dir.xy / ad.z; axisSign = sign(dir.z); faceId = axisSign > 0.0 ? 4.0 : 5.0;
      }
      vec2 guv = (uv * 0.5 + 0.5) * density;
      vec2 cellF = floor(guv);
      vec3 acc = vec3(0.0);
      for (int oy = -1; oy <= 1; oy++) {
        for (int ox = -1; ox <= 1; ox++) {
          vec2 c = cellF + vec2(float(ox), float(oy));
          float h = hash31(vec3(c, faceId));
          if (h > thresh) {
            float hBright = hash31(vec3(c + 7.0, faceId + 3.0));   // brightness seed
            float hTemp = hash31(vec3(c + 13.0, faceId + 29.0));   // colour-temperature seed
            vec2 starGrid = c + vec2(hash31(vec3(c, faceId + 11.0)), hash31(vec3(c, faceId + 19.0)));
            vec2 starFaceUv = (starGrid / density) * 2.0 - 1.0; // back to this face's [-1,1] plane
            // Reconstruct the star's true 3D direction from its face coordinate, matching how
            // uv was built above for each axis, then compare with a dot product (angle), not
            // a planar distance - this keeps every star perfectly round anywhere on the sky.
            vec3 starDir = faceId < 1.5 ? vec3(axisSign, starFaceUv.x, starFaceUv.y)
                         : faceId < 3.5 ? vec3(starFaceUv.x, axisSign, starFaceUv.y)
                         : vec3(starFaceUv.x, starFaceUv.y, axisSign);
            starDir = normalize(starDir);
            // Chord (Euclidean) distance between the two unit vectors, not cos(angle): for the
            // tiny sub-degree angles a pinpoint star spans, 1-cos(angle) ~= angle^2/2, which
            // throws away almost all precision in a 32-bit float and made the falloff noisy and
            // asymmetric per pixel (showing up as streaky, elongated "rice grain" shapes instead
            // of round dots). Chord distance is ~= angle for small angles and stays well
            // conditioned, giving a clean, stable, truly round falloff.
            float d = length(dir - starDir);
            // Size directly in units of the CURRENT pixel's angular footprint: roughly half a
            // pixel to two pixels in radius, with about a pixel of soft falloff on top - small,
            // reliably round pinpoints at any render resolution.
            float radiusPx = mix(0.5, 1.5, pow(hBright, 3.0));
            float angRad = u_pixelAngle * radiusPx;
            float shape = 1.0 - smoothstep(angRad, angRad + u_pixelAngle * 1.4, d);
            // Real star fields are dominated by faint stars with only a rare few standing out -
            // a steep power keeps most pinpoints dim and only a handful bright. Kept below the
            // bloom bright-pass threshold so stars stay crisp pinpoints instead of blooming into
            // soft (and, at this render resolution, slightly asymmetric-looking) blobs.
            float brightness = mix(0.05, 0.32, pow(hBright, 4.5));
            vec3 tint = mix(vec3(0.72, 0.80, 1.0), vec3(1.0, 0.86, 0.68), hTemp);
            acc += shape * brightness * tint;
          }
        }
      }
      return acc;
    }

    vec3 sky(vec3 dir) {
      // Space is essentially black - the disk must be the only large light source in frame.
      vec3 col = vec3(0.0015, 0.0016, 0.0035);

      // A very faint Milky-Way-like band around a fixed tilted great circle: its peak luminance
      // is kept to a small fraction of the disk's so it never competes with it.
      vec3 galN = normalize(vec3(0.25, 0.92, -0.30));
      float bandDist = dot(dir, galN);
      float band = exp(-bandDist * bandDist * 16.0);
      float bandNoise = fbm2(vec2(dir.x * 3.0 + dir.z * 2.0, dir.y * 5.0) * 2.2, 3) * 0.5 + 0.5;
      col += band * bandNoise * vec3(0.045, 0.04, 0.045) * 0.4;

      col += starLayer(dir, 45.0, 0.90);
      col += starLayer(dir, 105.0, 0.94);
      return col;
    }

    void main() {
      vec2 uv = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;
      uv.x *= u_aspect;
      vec3 rd = normalize(uv.x * u_tanHalfFov * u_camRight
                         + uv.y * u_tanHalfFov * u_camUp
                         + u_camForward);

      vec3 pos = u_camPos;
      vec3 vel = rd;
      float h2 = dot(cross(pos, vel), cross(pos, vel));

      vec3 colorAccum = vec3(0.0);
      float alphaAccum = 0.0;
      bool captured = false;
      bool escaped = false;
      const float FAR = 80.0; // escape radius - well past the disk (13) and photon sphere (1.5)

      for (int i = 0; i < MAX_STEPS_CONST; i++) {
        if (i >= u_maxSteps) break;
        if (alphaAccum > 0.995) break;

        float r = length(pos);
        if (r < 1.0) { captured = true; break; }
        if (r > FAR && dot(pos, vel) > 0.0) { escaped = true; break; }

        // Adaptive step: small near the hole and disk (where curvature is strong and the disk
        // crossing point needs sub-pixel precision), then MUCH larger once we're well clear of
        // that zone so the ray blasts out to FAR in only a handful of steps. Without this second
        // regime, a small capped step size takes many near-uniform steps to cross r=16..FAR,
        // and slight per-pixel differences in how many steps that takes show up as faint
        // concentric ring banding in the sky once escaped.
        float dt = r > 16.0 ? clamp(r * 0.6, 1.5, 25.0) : clamp(r * 0.14, 0.012, 1.3);

        vec3 prevPos = pos;
        rk4Step(pos, vel, h2, dt);

        // Detect a crossing of the disk plane y = 0 and interpolate the exact crossing point.
        if (prevPos.y * pos.y < 0.0) {
          float t = prevPos.y / (prevPos.y - pos.y);
          vec3 hitP = mix(prevPos, pos, t);
          float rDisk = length(hitP.xz);
          if (rDisk > DISK_INNER && rDisk < DISK_OUTER) {
            vec3 marchDir = normalize(pos - prevPos);
            vec4 s = diskShade(hitP, marchDir, rDisk);
            colorAccum += (1.0 - alphaAccum) * s.rgb * s.a;
            alphaAccum += (1.0 - alphaAccum) * s.a;
          }
        }
      }

      // A ray that neither escaped nor was captured only happens for near-critical impact
      // parameters that spend the whole step budget winding around the photon sphere. If it is
      // still deep in the strong field (r < 3) it is visually indistinguishable from a captured
      // ray, so treat it as one (black) rather than sampling the sky with whatever direction the
      // integration happened to stop at - that stray "special colour" is what produced a sharp,
      // unphysical ring artifact around the shadow.
      vec3 finalColor = colorAccum;
      if (escaped) {
        finalColor += (1.0 - alphaAccum) * sky(normalize(vel));
      } else if (!captured && length(pos) >= 3.0) {
        finalColor += (1.0 - alphaAccum) * sky(normalize(vel));
      }
      outColor = vec4(finalColor, 1.0);
    }
  `;

  const DOWNSAMPLE_FRAG = `#version 300 es
    precision highp float;
    in vec2 vUv;
    uniform sampler2D u_tex;
    uniform vec2 u_texel;
    out vec4 outColor;
    void main() {
      // Proper 4x4 box filter for the 4x downsample - a sparse 2x2 sample (as before) badly
      // undersamples a small bright feature like a 1px star: depending on exact alignment it
      // gets caught by one destination texel but missed by its neighbour, and the asymmetric
      // "who caught the light" pattern survives the later separable blur as a diagonal streak
      // instead of a round glow. Covering the full source block removes that aliasing.
      vec3 c = vec3(0.0);
      for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
          vec2 off = vec2(float(i) - 1.5, float(j) - 1.5);
          c += texture(u_tex, vUv + u_texel * off).rgb;
        }
      }
      c *= (1.0 / 16.0);
      float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
      float thresh = 0.4;
      vec3 bright = c * smoothstep(thresh, thresh * 2.2, lum);
      outColor = vec4(bright, 1.0);
    }
  `;

  const BLUR_FRAG = `#version 300 es
    precision highp float;
    in vec2 vUv;
    uniform sampler2D u_tex;
    uniform vec2 u_dir; // texel-scaled blur direction
    out vec4 outColor;
    void main() {
      const float w0 = 0.2270270270;
      const float w1 = 0.1945945946;
      const float w2 = 0.1216216216;
      const float w3 = 0.0540540541;
      const float w4 = 0.0162162162;
      vec3 sum = texture(u_tex, vUv).rgb * w0;
      sum += (texture(u_tex, vUv + u_dir * 1.0).rgb + texture(u_tex, vUv - u_dir * 1.0).rgb) * w1;
      sum += (texture(u_tex, vUv + u_dir * 2.0).rgb + texture(u_tex, vUv - u_dir * 2.0).rgb) * w2;
      sum += (texture(u_tex, vUv + u_dir * 3.0).rgb + texture(u_tex, vUv - u_dir * 3.0).rgb) * w3;
      sum += (texture(u_tex, vUv + u_dir * 4.0).rgb + texture(u_tex, vUv - u_dir * 4.0).rgb) * w4;
      outColor = vec4(sum, 1.0);
    }
  `;

  const COMPOSITE_FRAG = `#version 300 es
    precision highp float;
    in vec2 vUv;
    uniform sampler2D u_hdr;
    uniform sampler2D u_bloom;
    out vec4 outColor;

    vec3 acesFilm(vec3 x) {
      float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
      return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
    }

    void main() {
      vec3 hdr = texture(u_hdr, vUv).rgb;
      vec3 bloom = texture(u_bloom, vUv).rgb;
      vec3 col = hdr + bloom * 0.9;
      col = acesFilm(col);
      col = pow(col, vec3(1.0 / 2.2));
      vec2 p = vUv * 2.0 - 1.0;
      float vig = 1.0 - dot(p, p) * 0.25;
      col *= clamp(vig, 0.0, 1.0);
      outColor = vec4(col, 1.0);
    }
  `;

  // ---------------------------------------------------------------------------------------------
  // GL helpers
  // ---------------------------------------------------------------------------------------------

  function compileShader(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.error('gargantua shader compile error:', gl.getShaderInfoLog(sh));
      gl.deleteShader(sh);
      return null;
    }
    return sh;
  }

  function linkProgram(gl, vsSrc, fsSrc) {
    const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
    if (!vs || !fs) return null;
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error('gargantua program link error:', gl.getProgramInfoLog(prog));
      gl.deleteProgram(prog);
      return null;
    }
    return prog;
  }

  function uniformLocs(gl, prog, names) {
    const u = {};
    for (const n of names) u[n] = gl.getUniformLocation(prog, n);
    return u;
  }

  // ---------------------------------------------------------------------------------------------
  // Scene registration
  // ---------------------------------------------------------------------------------------------

  SCENES.gargantua = {
    id: 'gargantua',
    name: 'Гаргантюа',
    tagline: 'Чёрная дыра Шварцшильда: каждый пиксель — луч света, изогнутый гравитацией',
    about: 'Свет вблизи горизонта движется по изогнутым геодезическим, поэтому дальняя сторона ' +
      'аккреционного диска видна выгнутой над и под тенью — это и есть гравитационное линзирование ' +
      'в чистом виде. Газ у внутреннего края диска обращается почти со скоростью света, и сторона, ' +
      'летящая навстречу наблюдателю, выглядит ярче и голубее — релятивистский эффект Доплера.',
    spec: [
      ['Метрика', 'Шварцшильд, rs = 1'],
      ['Тень чёрной дыры', '≈ 2,6 rs'],
      ['Фотонная сфера', 'r = 1,5 rs'],
      ['Край диска (ISCO)', 'r = 3 rs'],
      ['Интегратор', 'RK4, до 220 шагов'],
      ['Скорость газа на ISCO', '0,5 c'],
    ],
    hint: 'Тяните — облёт · колесо или щипок — зум',
    accent: '#F2B880',

    mount(host, ui) {
      const canvas = document.createElement('canvas');
      canvas.style.display = 'block';
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      canvas.style.touchAction = 'none';
      host.appendChild(canvas);

      const gl = canvas.getContext('webgl2', { alpha: false, antialias: false, powerPreference: 'high-performance' });
      if (!gl) {
        const fb = document.createElement('div');
        fb.className = 'scene-fallback';
        fb.textContent = 'Для этой сцены нужен WebGL2 — откройте страницу в свежем Chrome, Firefox или Safari.';
        host.appendChild(fb);
        return { unmount() { host.innerHTML = ''; ui.innerHTML = ''; } };
      }

      const floatExt = gl.getExtension('EXT_color_buffer_float');
      const useFloat = !!floatExt;

      // ---- programs -----------------------------------------------------------------------
      const rayProg = linkProgram(gl, VERT_SRC, RAYTRACE_FRAG);
      const downProg = linkProgram(gl, VERT_SRC, DOWNSAMPLE_FRAG);
      const blurProg = linkProgram(gl, VERT_SRC, BLUR_FRAG);
      const compProg = linkProgram(gl, VERT_SRC, COMPOSITE_FRAG);

      const rayU = uniformLocs(gl, rayProg, [
        'u_resolution', 'u_camPos', 'u_camRight', 'u_camUp', 'u_camForward',
        'u_tanHalfFov', 'u_aspect', 'u_diskTime', 'u_doppler', 'u_maxSteps', 'u_pixelAngle',
      ]);
      const downU = uniformLocs(gl, downProg, ['u_tex', 'u_texel']);
      const blurU = uniformLocs(gl, blurProg, ['u_tex', 'u_dir']);
      const compU = uniformLocs(gl, compProg, ['u_hdr', 'u_bloom']);

      // Empty VAO: the full-screen triangle is generated purely from gl_VertexID.
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);

      // ---- render targets (recreated on resize) --------------------------------------------
      const internalFmt = useFloat ? gl.RGBA16F : gl.RGBA8;
      const texType = useFloat ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;

      let hdrTex = null, hdrFbo = null;
      let bloomTexA = null, bloomFboA = null;
      let bloomTexB = null, bloomFboB = null;
      let fullW = 0, fullH = 0, smallW = 0, smallH = 0;

      function makeTex(w, h) {
        const t = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texImage2D(gl.TEXTURE_2D, 0, internalFmt, w, h, 0, gl.RGBA, texType, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        return t;
      }
      function makeFbo(tex) {
        const f = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, f);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        return f;
      }
      function deleteTargets() {
        for (const t of [hdrTex, bloomTexA, bloomTexB]) if (t) gl.deleteTexture(t);
        for (const f of [hdrFbo, bloomFboA, bloomFboB]) if (f) gl.deleteFramebuffer(f);
        hdrTex = hdrFbo = bloomTexA = bloomFboA = bloomTexB = bloomFboB = null;
      }
      function buildTargets(w, h) {
        deleteTargets();
        fullW = Math.max(1, w);
        fullH = Math.max(1, h);
        smallW = Math.max(1, Math.round(fullW / 4));
        smallH = Math.max(1, Math.round(fullH / 4));
        hdrTex = makeTex(fullW, fullH);
        hdrFbo = makeFbo(hdrTex);
        bloomTexA = makeTex(smallW, smallH);
        bloomFboA = makeFbo(bloomTexA);
        bloomTexB = makeTex(smallW, smallH);
        bloomFboB = makeFbo(bloomTexB);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      }

      // ---- camera state ----------------------------------------------------------------------
      const DEFAULT_DIST = 20;
      const DEFAULT_YAW = 0.15;
      const DEFAULT_PITCH = 0.14; // ~8 degrees above the disk plane
      let yaw = DEFAULT_YAW;
      let pitch = DEFAULT_PITCH;
      let dist = DEFAULT_DIST;
      let yawVel = 0, pitchVel = 0; // drag inertia
      const FOV_DEG = 50;

      // ---- interaction state -----------------------------------------------------------------
      const pointers = new Map(); // pointerId -> {x,y}
      let dragLast = null;        // {x,y} of the single-pointer drag
      let pinchStartDist = 0, pinchStartZoom = 0;
      let lastInteraction = performance.now();
      const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

      let paused = false;
      let doppler = true;

      function markInteraction() { lastInteraction = performance.now(); }

      function pointerPos(e) {
        const r = canvas.getBoundingClientRect();
        return { x: e.clientX - r.left, y: e.clientY - r.top };
      }

      function onPointerDown(e) {
        canvas.setPointerCapture(e.pointerId);
        pointers.set(e.pointerId, pointerPos(e));
        markInteraction();
        yawVel = 0; pitchVel = 0;
        if (pointers.size === 1) {
          dragLast = pointerPos(e);
        } else if (pointers.size === 2) {
          const pts = Array.from(pointers.values());
          pinchStartDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y) || 1;
          pinchStartZoom = dist;
          dragLast = null;
        }
      }
      function onPointerMove(e) {
        if (!pointers.has(e.pointerId)) return;
        pointers.set(e.pointerId, pointerPos(e));
        markInteraction();
        if (pointers.size === 1 && dragLast) {
          const p = pointerPos(e);
          const dx = p.x - dragLast.x, dy = p.y - dragLast.y;
          const speed = 0.0055;
          yaw -= dx * speed;
          pitch = Math.max(-1.483, Math.min(1.483, pitch - dy * speed)); // clamp to ~+-85 deg
          yawVel = -dx * speed;
          pitchVel = -dy * speed;
          dragLast = p;
        } else if (pointers.size === 2) {
          const pts = Array.from(pointers.values());
          const d = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y) || 1;
          const ratio = pinchStartDist / d;
          dist = Math.max(6, Math.min(60, pinchStartZoom * ratio));
        }
      }
      function onPointerUp(e) {
        pointers.delete(e.pointerId);
        if (pointers.size < 1) dragLast = null;
        else if (pointers.size === 1) {
          dragLast = Array.from(pointers.values())[0];
        }
        try { canvas.releasePointerCapture(e.pointerId); } catch (err) { /* already released */ }
      }
      function onWheel(e) {
        e.preventDefault();
        markInteraction();
        yawVel = 0; pitchVel = 0;
        const factor = Math.exp(e.deltaY * 0.0012);
        dist = Math.max(6, Math.min(60, dist * factor));
      }
      function onKeyDown(e) {
        const t = e.target;
        if (t && (t.tagName === 'INPUT' || t.tagName === 'BUTTON')) return;
        if (e.code === 'Space') {
          e.preventDefault();
          setPaused(!paused);
        } else if (e.code === 'KeyR') {
          resetView();
        }
      }

      canvas.addEventListener('pointerdown', onPointerDown);
      canvas.addEventListener('pointermove', onPointerMove);
      canvas.addEventListener('pointerup', onPointerUp);
      canvas.addEventListener('pointercancel', onPointerUp);
      canvas.addEventListener('wheel', onWheel, { passive: false });
      window.addEventListener('keydown', onKeyDown);

      function resetView() {
        yaw = DEFAULT_YAW; pitch = DEFAULT_PITCH; dist = DEFAULT_DIST;
        yawVel = 0; pitchVel = 0;
        markInteraction();
      }

      // ---- UI controls -------------------------------------------------------------------------
      const btnDoppler = document.createElement('button');
      btnDoppler.type = 'button';
      btnDoppler.className = 'ctl';
      btnDoppler.textContent = 'Доплер';
      btnDoppler.setAttribute('aria-pressed', 'true');
      btnDoppler.addEventListener('click', () => setDoppler(!doppler));

      const btnPause = document.createElement('button');
      btnPause.type = 'button';
      btnPause.className = 'ctl';
      btnPause.textContent = 'Пауза';
      btnPause.setAttribute('aria-pressed', 'false');
      btnPause.addEventListener('click', () => setPaused(!paused));

      const btnReset = document.createElement('button');
      btnReset.type = 'button';
      btnReset.className = 'ctl';
      btnReset.textContent = 'Сброс вида';
      btnReset.addEventListener('click', resetView);

      ui.appendChild(btnDoppler);
      ui.appendChild(btnPause);
      ui.appendChild(btnReset);

      function setDoppler(v) {
        doppler = v;
        btnDoppler.setAttribute('aria-pressed', String(v));
      }
      function setPaused(v) {
        paused = v;
        btnPause.setAttribute('aria-pressed', String(v));
      }

      // ---- resize / quality ---------------------------------------------------------------------
      let quality = 0.5;
      const fixedQ = host.dataset.fixedQuality ? parseFloat(host.dataset.fixedQuality) : null;
      if (fixedQ) quality = fixedQ;

      let sizeDirty = true;
      const ro = new ResizeObserver(() => { sizeDirty = true; });
      ro.observe(host);

      function applySize() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const cw = Math.max(1, Math.round(host.clientWidth * dpr * quality));
        const ch = Math.max(1, Math.round(host.clientHeight * dpr * quality));
        if (cw !== canvas.width || ch !== canvas.height) {
          canvas.width = cw;
          canvas.height = ch;
          buildTargets(cw, ch);
        }
        sizeDirty = false;
      }
      applySize();

      // ---- frame timing / adaptive quality --------------------------------------------------
      let frameTimes = [];
      let diskTime = 0;
      let lastT = performance.now();
      let raf = 0;

      function drawFullscreen() {
        gl.bindVertexArray(vao);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }

      function frame(now) {
        raf = requestAnimationFrame(frame);
        const dtMs = now - lastT;
        const dt = Math.min(0.1, dtMs / 1000);
        lastT = now;

        // idle auto-drift: gentle yaw rotation resumes after 4s of no interaction
        const idleFor = now - lastInteraction;
        if (idleFor > 4000 && Math.abs(yawVel) < 1e-5 && Math.abs(pitchVel) < 1e-5) {
          const driftSpeed = reducedMotion ? 0.008 : 0.028;
          yaw += driftSpeed * dt;
        } else {
          // drag inertia with damping
          yaw += yawVel;
          pitch = Math.max(-1.483, Math.min(1.483, pitch + pitchVel));
          yawVel *= 0.90;
          pitchVel *= 0.90;
          if (Math.abs(yawVel) < 1e-5) yawVel = 0;
          if (Math.abs(pitchVel) < 1e-5) pitchVel = 0;
        }

        if (!paused) diskTime += dt;

        if (sizeDirty) applySize();

        // ---- build camera basis ---------------------------------------------------------------
        const cy = Math.cos(pitch), sy = Math.sin(pitch);
        const camPos = [
          dist * cy * Math.sin(yaw),
          dist * sy,
          dist * cy * Math.cos(yaw),
        ];
        // forward = normalize(origin - camPos)
        let fx = -camPos[0], fy = -camPos[1], fz = -camPos[2];
        const flen = Math.hypot(fx, fy, fz) || 1;
        fx /= flen; fy /= flen; fz /= flen;
        // right = normalize(forward x worldUp)
        const wux = 0, wuy = 1, wuz = 0;
        let rx = fy * wuz - fz * wuy, ry = fz * wux - fx * wuz, rz = fx * wuy - fy * wux;
        const rlen = Math.hypot(rx, ry, rz) || 1;
        rx /= rlen; ry /= rlen; rz /= rlen;
        // up = right x forward
        const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;

        const aspect = canvas.width / canvas.height;
        const tanHalfFov = Math.tan((FOV_DEG * Math.PI / 180) * 0.5);
        const maxSteps = Math.round(70 + 150 * Math.min(1, Math.max(0, quality)));

        // ================= pass A: ray trace into HDR target =================
        gl.bindFramebuffer(gl.FRAMEBUFFER, hdrFbo);
        gl.viewport(0, 0, fullW, fullH);
        gl.useProgram(rayProg);
        gl.uniform2f(rayU.u_resolution, fullW, fullH);
        gl.uniform3f(rayU.u_camPos, camPos[0], camPos[1], camPos[2]);
        gl.uniform3f(rayU.u_camRight, rx, ry, rz);
        gl.uniform3f(rayU.u_camUp, ux, uy, uz);
        gl.uniform3f(rayU.u_camForward, fx, fy, fz);
        gl.uniform1f(rayU.u_tanHalfFov, tanHalfFov);
        gl.uniform1f(rayU.u_aspect, aspect);
        gl.uniform1f(rayU.u_diskTime, diskTime);
        gl.uniform1f(rayU.u_doppler, doppler ? 1.0 : 0.0);
        gl.uniform1i(rayU.u_maxSteps, maxSteps);
        // Radians subtended by one output pixel, vertically - lets the shader size stars as
        // true pinpoints (a fixed number of pixels) at any render resolution/quality level.
        gl.uniform1f(rayU.u_pixelAngle, (FOV_DEG * Math.PI / 180) / canvas.height);
        drawFullscreen();

        // ================= pass B: bright-pass + downsample to 1/4 res =================
        gl.bindFramebuffer(gl.FRAMEBUFFER, bloomFboA);
        gl.viewport(0, 0, smallW, smallH);
        gl.useProgram(downProg);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, hdrTex);
        gl.uniform1i(downU.u_tex, 0);
        gl.uniform2f(downU.u_texel, 1 / fullW, 1 / fullH);
        drawFullscreen();

        // ================= pass C/D: separable gaussian blur, two ping-pong iterations =========
        gl.useProgram(blurProg);
        for (let i = 0; i < 2; i++) {
          gl.bindFramebuffer(gl.FRAMEBUFFER, bloomFboB);
          gl.bindTexture(gl.TEXTURE_2D, bloomTexA);
          gl.uniform1i(blurU.u_tex, 0);
          gl.uniform2f(blurU.u_dir, 1 / smallW, 0);
          drawFullscreen();

          gl.bindFramebuffer(gl.FRAMEBUFFER, bloomFboA);
          gl.bindTexture(gl.TEXTURE_2D, bloomTexB);
          gl.uniform2f(blurU.u_dir, 0, 1 / smallH);
          drawFullscreen();
        }

        // ================= pass E: composite (tone map + bloom + vignette) to screen ===========
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.useProgram(compProg);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, hdrTex);
        gl.uniform1i(compU.u_hdr, 0);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, bloomTexA);
        gl.uniform1i(compU.u_bloom, 1);
        drawFullscreen();

        // ---- adaptive quality bookkeeping ------------------------------------------------------
        if (!fixedQ) {
          frameTimes.push(performance.now() - now);
          if (frameTimes.length >= 30) {
            const avg = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
            if (avg > 22) quality = Math.max(0.3, quality * 0.85);
            else if (avg < 12) quality = Math.min(1, quality * 1.1);
            frameTimes = [];
            sizeDirty = true; // re-evaluate canvas size at the new quality next frame
          }
        }
      }
      raf = requestAnimationFrame(frame);

      return {
        unmount() {
          cancelAnimationFrame(raf);
          ro.disconnect();
          canvas.removeEventListener('pointerdown', onPointerDown);
          canvas.removeEventListener('pointermove', onPointerMove);
          canvas.removeEventListener('pointerup', onPointerUp);
          canvas.removeEventListener('pointercancel', onPointerUp);
          canvas.removeEventListener('wheel', onWheel);
          window.removeEventListener('keydown', onKeyDown);

          deleteTargets();
          gl.deleteVertexArray(vao);
          for (const p of [rayProg, downProg, blurProg, compProg]) if (p) gl.deleteProgram(p);
          gl.getExtension('WEBGL_lose_context')?.loseContext();

          host.innerHTML = '';
          ui.innerHTML = '';
        },
      };
    },
  };
})();

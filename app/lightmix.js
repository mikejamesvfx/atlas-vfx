// Light room viewer: up to 8 relit takes of one layer, mixed on the GPU (WebGL2) and shown through OCIO's own GPU
// code for the view. Sliders only change uniforms, so a change draws in a frame. Everything follows
// atlas_vfx/comp/mix.py (mix_frame) and frames.blend line for line; tests/test_light_view_browser.py holds them
// together.
//
// Frames come from the server's cache (atlas_vfx/comp/cache.py) as half-float RGBA: the takes, plus "under" (plate
// and lower layers) and "over" (higher layers). Proxy frames (1024 wide) while scrubbing and playing; the full-size
// set replaces them once the playhead rests.

const MAX_TAKES = 8;
const MAGIC = "ATLF16\n";
const BLEND_MODES = { over: 0, plus: 1, multiply: 2, screen: 3 };

const VERT = `#version 300 es
in vec2 p;
out vec2 uv;
void main() { uv = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5); gl_Position = vec4(p, 0.0, 1.0); }`;

function fragment(ocioText) {
  return `#version 300 es
precision highp float;
precision highp sampler2D;
${ocioText}
in vec2 uv;
out vec4 outColor;
uniform sampler2D under_t, over_t;
uniform sampler2D take_t[${MAX_TAKES}];
uniform int n_takes;
uniform float weight[${MAX_TAKES}];      // 0 for takes that are not active (mute / solo / weight 0)
uniform vec3 gain[${MAX_TAKES}];         // 2^stops * tint
uniform int mix_mode;                    // 0 blend, 1 sum
uniform int layer_blend;                 // frames.BLENDS index
uniform float layer_opacity;
uniform float exposure;                  // stops
uniform vec4 wipe;                       // x: on, y: position 0-1 (left of it: the plate and other layers only)

vec4 takeAt(int i) {
  // WebGL2 needs a constant index into a sampler array
  ${Array.from({ length: MAX_TAKES }, (_, i) => `if (i == ${i}) return texture(take_t[${i}], uv);`).join("\n  ")}
  return vec4(0.0);
}

vec3 blendOnto(vec3 b, float ba, vec4 f, int mode, out float aOut) {
  float fa = f.a;
  vec3 o;
  if (mode == 0) o = f.rgb + b * (1.0 - fa);
  else if (mode == 1) o = b + f.rgb;
  else if (mode == 2) o = b * (1.0 - fa) + b * f.rgb;
  else o = b + f.rgb - b * f.rgb;
  aOut = clamp(fa + ba * (1.0 - fa), 0.0, 1.0);
  return o;
}

void main() {
  vec4 u = texture(under_t, uv);
  vec4 ov = texture(over_t, uv);
  vec3 rgb = vec3(0.0);
  float a = 0.0, wsum = 0.0;
  for (int i = 0; i < ${MAX_TAKES}; i++) {
    if (i >= n_takes) break;
    float w = weight[i];
    if (w <= 0.0) continue;
    vec4 t = takeAt(i);
    rgb += t.rgb * gain[i] * w;
    a += t.a * w;
    wsum += w;
  }
  vec4 m = vec4(0.0);
  if (wsum > 0.0) {
    if (mix_mode == 0) m = vec4(rgb / wsum, a / wsum);
    else m = vec4(rgb, clamp(a, 0.0, 1.0));
  }
  bool plateSide = wipe.x > 0.5 && uv.x < wipe.y;
  vec3 c = u.rgb;
  float ca = u.a;
  if (!plateSide && wsum > 0.0) c = blendOnto(c, ca, m * layer_opacity, layer_blend, ca);
  c = blendOnto(c, ca, ov, 0, ca);
  c *= exp2(exposure);
  vec4 d = OCIOView(vec4(c, 1.0));
  outColor = vec4(clamp(d.rgb, 0.0, 1.0), 1.0);
}`;
}

export function parseFrame(buf) {
  const bytes = new Uint8Array(buf);
  const magic = new TextDecoder().decode(bytes.subarray(0, MAGIC.length));
  if (magic !== MAGIC) throw new Error("not an Atlas cache frame");
  const n = new DataView(buf).getUint32(MAGIC.length, true);
  const o = MAGIC.length + 4;
  const head = JSON.parse(new TextDecoder().decode(bytes.subarray(o, o + n)));
  const start = o + n;
  const count = head.width * head.height * 4;
  // a Uint16Array needs an even byte offset; copy when the header made it odd
  const px = start % 2 === 0 ? new Uint16Array(buf, start, count) : new Uint16Array(buf.slice(start, start + count * 2));
  return { head, px };
}

// The mix record (comp/mix.py) -> per take shader inputs: weight 0 for takes that do not go in.
export function takeUniforms(mix, versions) {
  const takes = mix?.takes || {};
  const solo = versions.some((v) => takes[v]?.solo);
  return versions.map((v) => {
    const t = takes[v] || { weight: 0, stops: 0, tint: [1, 1, 1], mute: false, solo: false };
    const on = solo ? !!t.solo : !t.mute;
    const g = 2 ** (t.stops || 0);
    const tint = t.tint || [1, 1, 1];
    return { weight: on && t.weight > 0 ? t.weight : 0, gain: [g * tint[0], g * tint[1], g * tint[2]] };
  });
}

export class LightMixer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", { premultipliedAlpha: false, preserveDrawingBuffer: true, antialias: false });
    if (!gl) throw new Error("WebGL2 is not available in this browser");
    this.gl = gl;
    this.floatLinear = !!gl.getExtension("OES_texture_float_linear");
    this.prog = null;
    this.tex = { under: this._tex(), over: this._tex(), takes: Array.from({ length: MAX_TAKES }, () => this._tex()) };
    this.ocioTex = [];
    this.size = [0, 0];
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    this.state = { versions: [], mix: null, mode: "blend", blend: "over", opacity: 1, exposure: 0, wipe: null };
  }

  _tex() {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, 1, 1, 0, gl.RGBA, gl.HALF_FLOAT, new Uint16Array(4));
    return t;
  }

  // The view: OCIO's GLSL ES 3.0 (from /atlas/api/view_shader) compiled into the mix program, its tables uploaded.
  setView(payload) {
    const gl = this.gl;
    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`shader: ${gl.getShaderInfoLog(s)}`);
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, VERT));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fragment(payload.shader)));
    gl.bindAttribLocation(p, 0, "p");
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`link: ${gl.getProgramInfoLog(p)}`);
    if (this.prog) gl.deleteProgram(this.prog);
    this.prog = p;
    this.ocioTex.forEach((t) => gl.deleteTexture(t.tex));
    this.ocioTex = payload.textures.map((t) => {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      const lin = t.linear && this.floatLinear ? gl.LINEAR : gl.NEAREST;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, lin);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, lin);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      const data = new Float32Array(t.values);
      if (t.channels === 1) gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, t.width, t.height, 0, gl.RED, gl.FLOAT, data);
      else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB32F, t.width, t.height, 0, gl.RGB, gl.FLOAT, data);
      return { tex, sampler: t.sampler };
    });
    if (payload.textures3d?.length) throw new Error("this view needs 3D lookup tables, not handled yet");
  }

  // One frame set: { under, over, takes: [..] } of parseFrame results, all the same size.
  upload(set) {
    const gl = this.gl;
    const put = (tex, f) => {
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      if (!f) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, 1, 1, 0, gl.RGBA, gl.HALF_FLOAT, new Uint16Array(4)); return; }
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, f.head.width, f.head.height, 0, gl.RGBA, gl.HALF_FLOAT, f.px);
    };
    put(this.tex.under, set.under);
    put(this.tex.over, set.over);
    for (let i = 0; i < MAX_TAKES; i++) put(this.tex.takes[i], set.takes[i] || null);
    const any = set.under || set.takes.find(Boolean);
    if (any) this.size = [any.head.width, any.head.height];
  }

  set(patch) { Object.assign(this.state, patch); }

  draw() {
    const gl = this.gl;
    const st = this.state;
    if (!this.prog) return;
    const [w, h] = this.size;
    if (w && (this.canvas.width !== w || this.canvas.height !== h)) { this.canvas.width = w; this.canvas.height = h; }
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.prog);
    let unit = 0;
    const bind = (name, tex) => {
      const loc = gl.getUniformLocation(this.prog, name);
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      if (loc) gl.uniform1i(loc, unit);
      unit++;
    };
    bind("under_t", this.tex.under);
    bind("over_t", this.tex.over);
    this.tex.takes.forEach((t, i) => bind(`take_t[${i}]`, t));
    this.ocioTex.forEach((t) => bind(t.sampler, t.tex));
    const tu = takeUniforms(st.mix, st.versions).slice(0, MAX_TAKES);
    const wts = new Float32Array(MAX_TAKES), gains = new Float32Array(MAX_TAKES * 3);
    tu.forEach((t, i) => { wts[i] = t.weight; gains.set(t.gain, i * 3); });
    const u = (n) => gl.getUniformLocation(this.prog, n);
    gl.uniform1i(u("n_takes"), tu.length);
    gl.uniform1fv(u("weight"), wts);
    gl.uniform3fv(u("gain"), gains);
    gl.uniform1i(u("mix_mode"), (st.mix?.mode || "blend") === "sum" ? 1 : 0);
    gl.uniform1i(u("layer_blend"), BLEND_MODES[st.blend] ?? 0);
    gl.uniform1f(u("layer_opacity"), st.opacity ?? 1);
    gl.uniform1f(u("exposure"), st.exposure || 0);
    gl.uniform4f(u("wipe"), st.wipe == null ? 0 : 1, st.wipe ?? 0, 0, 0);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}

// Frame sets fetched from the cache and kept within a memory budget (bytes), least recently used dropped first.
export class FrameStore {
  constructor(api, budget = 2 * 1024 ** 3) {
    this.api = api;
    this.budget = budget;
    this.map = new Map();            // key -> {bytes, promise | set}
    this.used = 0;
  }

  key(shot, layer, level, frame) { return `${shot}|${layer}|${level}|${frame}`; }

  async get(shot, layer, versions, level, frame) {
    const k = this.key(shot, layer, level, frame);
    const hit = this.map.get(k);
    if (hit) { this.map.delete(k); this.map.set(k, hit); return hit.promise; }
    const promise = (async () => {
      const takes = await Promise.all(versions.map((v) => this.api.cacheTake(shot, layer, v, level, frame)
        .then((b) => (b ? parseFrame(b) : null))));
      const ref = takes.find(Boolean);
      const w = ref ? ref.head.width : 1024, h = ref ? ref.head.height : 540;
      const [under, over] = await Promise.all(["under", "over"].map((side) =>
        this.api.cacheStack(shot, layer, side, level, frame, w, h).then((b) => (b ? parseFrame(b) : null))));
      const set = { under, over, takes };
      const bytes = [under, over, ...takes].reduce((s, f) => s + (f ? f.px.byteLength : 0), 0);
      const entry = this.map.get(k);
      if (entry) { entry.bytes = bytes; this.used += bytes; }
      while (this.used > this.budget && this.map.size > 1) {
        const [old, e] = this.map.entries().next().value;
        this.map.delete(old);
        this.used -= e.bytes || 0;
      }
      return set;
    })();
    this.map.set(k, { bytes: 0, promise });
    promise.catch(() => this.map.delete(k));
    return promise;
  }

  // Stack frames depend on the other layers: drop them all when the comp below / above changes.
  clear() { this.map.clear(); this.used = 0; }
}

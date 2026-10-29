/**
 * Atlas VFX Sun Globe: the Physical Sky node's sun (catalogue 48), drawn and driven on a globe.
 *
 * Ported from Atlas Camera's Gravity Compass (atlas_camera/comfy/web/atlas_gravity_compass.js, d1cb096): the
 * native widgets stay the serialized / API contract and this panel only drives them, so a workflow runs exactly
 * the same with no frontend and there is no second sun path. The panel owns (hides) only OWNED_WIDGETS; heading,
 * fov and tilt stay ordinary widgets and are read, not written.
 *
 * Frame: the node's sky frame, right-handed, Y up, north -Z, east +X; azimuth clockwise from north, elevation up.
 * three.js uses the same axes, so a direction goes straight into the scene. Seen from above, N E S W run clockwise.
 *
 * Drag mapping (node_physical_sky.py has three sun modes):
 *   - dragging the sun from metadata or manual writes MANUAL: sun_azimuth / sun_elevation exactly as dropped.
 *     Manual is the honest mapping: offset is "NOAA plus a nudge" and a big dragged nudge would ride the clock
 *     into elevations the node refuses (it raises outside -90..90).
 *   - from offset it stays offset and rewrites the two offsets against the NOAA sun at the current hour, falling
 *     back to manual when the elevation offset would leave its widget range.
 *   - the time scrubber writes time_offset_h; in manual it first returns to metadata (manual ignores the clock).
 *   - RESET TO METADATA: sun_mode metadata, time_offset_h 0, both offsets 0.
 * Nothing queues while dragging. With AUTO-QUEUE on (a panel toggle kept in node.properties, default off, the sky
 * node usually feeds a heavy graph) one prompt is queued when an interaction ends.
 *
 * The node's ui payload (ui.atlas_sun_globe[0], node_physical_sky.globe_ui) arrives in onExecuted: its sun path is
 * used while latitude / longitude / capture time still match, its shot camera fov / tilt while the shot camera is
 * linked, and its sky preview textures the dome while the widgets still give the sun it rendered. Without a run
 * everything is drawn from the widget values (the NOAA sun below mirrors layers/solar.py; tests/test_sun_globe.py
 * runs both and compares).
 */
import { app } from "../../scripts/app.js";

export const NODE_NAME = "AtlasVFXPhysicalSky";
export const EXTENSION_NAME = "AtlasVFX.SunGlobe";
export const SUN_MODES = ["metadata", "offset", "manual"];
export const OWNED_WIDGETS = ["time_offset_h", "sun_mode", "sun_azimuth_offset", "sun_elevation_offset",
  "sun_azimuth", "sun_elevation"];
export const READ_WIDGETS = ["latitude", "longitude", "capture_utc", "heading_deg", "fov_x", "tilt_deg"];
export const PRAGUE_MIN_ELEVATION = -4.2;
export const AUTO_QUEUE_PROPERTY = "atlas_sun_globe_auto_queue";
export const AUTO_QUEUE_DEFAULT = false;

const DEG = Math.PI / 180;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const mod = (a, n) => ((a % n) + n) % n;              // python's %, never negative
export const wrap180 = (v) => mod(v + 180, 360) - 180;
const round1 = (v) => Math.round(v * 10) / 10;

// ------------------------------------------------------------------------------------------- time and the sun
/** ISO 8601 with its offset ('...Z' or '...+11:00') -> epoch ms, else null (a bare time is ambiguous; solar.py
 * refuses it too). */
export function parseUtc(text) {
  const s = String(text ?? "").trim();
  if (!/^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})$/i.test(s)) return null;
  const ms = Date.parse(s.replace(" ", "T").replace(/z$/, "Z"));
  return Number.isFinite(ms) ? ms : null;
}

/** The clock the capture time was written in: minutes east of UTC, or null for Z (show UTC). */
export function clockOffsetMin(text) {
  const m = /([+-])(\d{2}):(\d{2})$/.exec(String(text ?? "").trim());
  if (!m) return null;
  const v = Number(m[2]) * 60 + Number(m[3]);
  return m[1] === "-" ? -v : v;
}

/** NOAA solar position, a line-for-line mirror of layers/solar.py sun_at: [azimuth, elevation] in degrees
 * (azimuth clockwise from true north, geometric elevation, valid at night too). */
export function sunAt(latDeg, lonDeg, ms) {
  const lat = Number(latDeg), lon = Number(lonDeg);
  const jd = ms / 86400000 + 2440587.5;
  const T = (jd - 2451545.0) / 36525.0;
  const L0 = mod(280.46646 + T * (36000.76983 + T * 0.0003032), 360.0);
  const M = 357.52911 + T * (35999.05029 - 0.0001537 * T);
  const e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T);
  const Mr = M * DEG;
  const C = Math.sin(Mr) * (1.914602 - T * (0.004817 + 0.000014 * T)) + Math.sin(2 * Mr) * (0.019993 - 0.000101 * T)
    + Math.sin(3 * Mr) * 0.000289;
  const om = 125.04 - 1934.136 * T;
  const lam = L0 + C - 0.00569 - 0.00478 * Math.sin(om * DEG);
  const eps0 = 23.0 + (26.0 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60.0) / 60.0;
  const eps = eps0 + 0.00256 * Math.cos(om * DEG);
  const dec = Math.asin(Math.sin(eps * DEG) * Math.sin(lam * DEG));
  const y = Math.tan(eps * DEG / 2.0) ** 2;
  const L0r = L0 * DEG;
  const eqt = 4.0 * (y * Math.sin(2 * L0r) - 2 * e * Math.sin(Mr) + 4 * e * y * Math.sin(Mr) * Math.cos(2 * L0r)
    - 0.5 * y * y * Math.sin(4 * L0r) - 1.25 * e * e * Math.sin(2 * Mr)) / DEG;
  const d = new Date(ms);
  const minutes = d.getUTCHours() * 60.0 + d.getUTCMinutes() + (d.getUTCSeconds() + d.getUTCMilliseconds() * 1e-3) / 60.0;
  const tst = mod(minutes + eqt + 4.0 * lon, 1440.0);
  const ha = (tst / 4.0 - 180.0) * DEG;
  const latr = lat * DEG;
  const cz = Math.sin(latr) * Math.sin(dec) + Math.cos(latr) * Math.cos(dec) * Math.cos(ha);
  const zen = Math.acos(clamp(cz, -1.0, 1.0));
  const el = 90.0 - zen / DEG;
  const az = mod(Math.atan2(Math.sin(ha), Math.cos(ha) * Math.sin(latr) - Math.tan(dec) * Math.cos(latr)) / DEG + 180.0,
    360.0);
  return [az, el];
}

/** Sky-frame unit direction for compass azimuth / elevation (physical_sky.azel_to_dir). */
export function azelToDir(azDeg, elDeg) {
  const a = azDeg * DEG, e = elDeg * DEG;
  return [Math.sin(a) * Math.cos(e), Math.sin(e), -Math.cos(a) * Math.cos(e)];
}

export function dirToAzel(d) {
  const n = Math.hypot(d[0], d[1], d[2]) || 1;
  return [mod(Math.atan2(d[0] / n, -d[2] / n) / DEG, 360), Math.asin(clamp(d[1] / n, -1, 1)) / DEG];
}

/** The sun the node renders for (physical_sky.effective_sun), or {error} where the node would raise. */
export function effectiveSun(mode, lat, lon, captureUtc, timeOffsetH = 0, azOffset = 0, elOffset = 0, manualAz = 0,
  manualEl = 0) {
  const out = { mode, time_offset_h: Number(timeOffsetH), derived: null, capture_ms: null, sun_ms: null, error: null };
  const text = String(captureUtc ?? "").trim();
  if (mode === "metadata" || mode === "offset" || text) {
    const t0 = parseUtc(text);
    if (t0 === null) return { ...out, error: text ? "capture time is not ISO 8601 with an offset" : "no capture time" };
    out.capture_ms = t0;
    out.sun_ms = t0 + Number(timeOffsetH) * 3600000;
    const [az, el] = sunAt(lat, lon, out.sun_ms);
    out.derived = { azimuth_deg: az, elevation_deg: el };
  }
  let az, el;
  if (mode === "metadata") {
    [az, el] = [out.derived.azimuth_deg, out.derived.elevation_deg];
  } else if (mode === "offset") {
    az = out.derived.azimuth_deg + Number(azOffset);
    el = out.derived.elevation_deg + Number(elOffset);
  } else if (mode === "manual") {
    [az, el] = [Number(manualAz), Number(manualEl)];
  } else {
    return { ...out, error: `sun mode ${mode}` };
  }
  if (!(el >= -90 && el <= 90)) return { ...out, error: "sun elevation outside -90..90 (check the elevation offset)" };
  return { ...out, azimuth_deg: mod(az, 360), elevation_deg: el, renderable: el >= PRAGUE_MIN_ELEVATION };
}

/** The capture day's sun, mirroring physical_sky.sun_path: samples every stepMin over the local solar day that holds
 * the capture (keyed by time_offset_h), sunrise / sunset (elevation 0, bisected), solar noon (ternary search) and
 * golden hour (elevation `golden`). Null without a capture time. */
export function sunPath(lat, lon, captureUtc, stepMin = 10.0, golden = 4.0) {
  const t0 = parseUtc(captureUtc);
  if (t0 === null) return null;
  const at = (o) => sunAt(lat, lon, t0 + o * 3600000);
  const d0 = new Date(t0);
  const solarH = mod(d0.getUTCHours() + d0.getUTCMinutes() / 60.0 + d0.getUTCSeconds() / 3600.0 + Number(lon) / 15.0, 24.0);
  const start = -solarH;
  const n = Math.round(24 * 60 / stepMin);
  const offs = Array.from({ length: n + 1 }, (_, i) => start + i * stepMin / 60.0);
  const pts = offs.map(at);
  const az = pts.map((p) => p[0]), el = pts.map((p) => p[1]);
  let iMax = 0;
  for (let i = 1; i < el.length; i++) if (el[i] > el[iMax]) iMax = i;
  let a = offs[Math.max(iMax - 1, 0)], b = offs[Math.min(iMax + 1, n)];
  for (let k = 0; k < 60; k++) {
    const m1 = a + (b - a) / 3.0, m2 = b - (b - a) / 3.0;
    if (at(m1)[1] < at(m2)[1]) a = m1; else b = m2;
  }
  const event = (o) => {
    const [eaz, eel] = at(o);
    return { offset_h: o, utc: new Date(t0 + o * 3600000).toISOString().slice(0, 19) + "Z", azimuth_deg: eaz,
      elevation_deg: eel };
  };
  const noon = event(0.5 * (a + b));
  const crossing = (lo, hi, level) => {
    let fa = at(lo)[1] - level;
    for (let k = 0; k < 50; k++) {
      const m = 0.5 * (lo + hi), fm = at(m)[1] - level;
      if ((fm < 0) === (fa < 0)) { lo = m; fa = fm; } else hi = m;
    }
    return 0.5 * (lo + hi);
  };
  const crossings = (level) => {
    let up = null, down = null;
    for (let i = 0; i < n; i++) {
      const lo = el[i] - level, hi = el[i + 1] - level;
      if (lo < 0 && hi >= 0 && up === null) up = event(crossing(offs[i], offs[i + 1], level));
      else if (lo >= 0 && hi < 0 && down === null) down = event(crossing(offs[i], offs[i + 1], level));
    }
    return [up, down];
  };
  const [sunrise, sunset] = crossings(0.0);
  const [goldenMorning, goldenEvening] = crossings(golden);
  let polar = null;
  if (sunrise === null && sunset === null) polar = Math.min(...el) > 0 ? "day" : (Math.max(...el) < 0 ? "night" : null);
  return { step_min: stepMin, start_offset_h: start, end_offset_h: offs[n], offset_h: offs, azimuth_deg: az,
    elevation_deg: el, sunrise, sunset, noon, golden_morning: goldenMorning, golden_evening: goldenEvening,
    golden_elevation_deg: golden, polar, method: "sun globe JS mirror of layers/solar.py" };
}

// ------------------------------------------------------------------------------------------- camera and drag
/** Camera basis in the sky frame for a level-roll camera at heading / tilt (physical_sky.camera_to_sky columns:
 * r right, u up, f forward; camera z = -f). */
export function cameraBasis(headingDeg, tiltDeg) {
  const f = azelToDir(headingDeg, clamp(tiltDeg, -89, 89));
  let r = [-f[2], 0, f[0]];                         // cross(f, +Y)
  const rn = Math.hypot(r[0], r[2]) || 1;
  r = [r[0] / rn, 0, r[2] / rn];
  const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
  return { r, u, f };
}

/** Where a sky direction lands in the camera (physical_sky.project_uv on the camera-space direction): uv 0..1 from
 * top-left and whether it is in front. */
export function projectToCamera(dir, headingDeg, tiltDeg, fovXDeg, aspect) {
  const { r, u, f } = cameraBasis(headingDeg, tiltDeg);
  const dot = (p, q) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
  const x = dot(dir, r), y = dot(dir, u);
  let z = -dot(dir, f);
  const front = z < 0;
  if (Math.abs(z) < 1e-9) z = -1e-9;
  const tx = Math.tan(fovXDeg * DEG / 2);
  return { u: 0.5 + (x / -z) / (2 * tx), v: 0.5 - (y / -z) * aspect / (2 * tx), front, cam: [x, y, -dot(dir, f)] };
}

/** Is the sun behind, beside or in front of the actor the camera looks at. rel > 0: the sun is to the right. */
export function sunRelation(azDeg, elDeg, headingDeg, tiltDeg, fovXDeg, aspect) {
  const rel = wrap180(azDeg - headingDeg);
  const p = projectToCamera(azelToDir(azDeg, elDeg), headingDeg, tiltDeg, fovXDeg, aspect);
  const side = Math.abs(rel) <= 45 ? "behind_actor" : (Math.abs(rel) >= 135 ? "in_front_of_actor" : "beside");
  return { rel, side, hand: rel > 0.5 ? "right" : (rel < -0.5 ? "left" : "centre"),
    in_frame: p.front && p.u >= 0 && p.u <= 1 && p.v >= 0 && p.v <= 1, uv: [p.u, p.v] };
}

/** The widget values a sun dropped at (az, el) writes (see the header for why manual). */
export function dragWidgets(mode, azDeg, elDeg, derived) {
  const az = mod(round1(azDeg), 360), el = clamp(round1(elDeg), -90, 90);
  if (mode === "offset" && derived) {
    const dAz = round1(wrap180(az - derived.azimuth_deg)), dEl = round1(el - derived.elevation_deg);
    if (dEl >= -90 && dEl <= 90) return { sun_mode: "offset", sun_azimuth_offset: dAz, sun_elevation_offset: dEl };
  }
  return { sun_mode: "manual", sun_azimuth: az, sun_elevation: el };
}

/** time_offset_h for a point x (0..1) along the path's day, snapped to whole `snapMin` minutes of the shown clock. */
export function offsetAtX(x01, path, captureMs, clockMin, snapMin = 5) {
  const o = path.start_offset_h + clamp(x01, 0, 1) * (path.end_offset_h - path.start_offset_h);
  const shift = (clockMin || 0) * 60000, step = snapMin * 60000;
  const snapped = Math.round((captureMs + o * 3600000 + shift) / step) * step;
  return Math.round(((snapped - shift - captureMs) / 3600000) * 1e6) / 1e6;
}

/** The sky dome: three's SphereGeometry arguments (upper half only; the preview's lower half is ground) and the
 * texture repeat / offset that put the node's lat-long on it. Ours: u = ((A - centre + 180) mod 360) / 360 with the
 * top row at the zenith; three's sphere runs u the other way (W S E N), so u_image = 1.25 - centre / 360 - u_sphere
 * (mirror and shift, no resampling), and the half dome's v (1 zenith, 0 horizon) maps to v_image = 0.5 + v / 2. */
export const domeGeometryArgs = (radius) => [radius, 64, 16, 0, Math.PI * 2, 0, Math.PI / 2];
export const domeUvTransform = (centreDeg) => ({ repeat: [-1, 0.5], offset: [1.25 - Number(centreDeg || 0) / 360, 0.5] });

export function fmtClock(ms, offMin) {
  const d = new Date(ms + (offMin || 0) * 60000);
  return `${String(d.getUTCHours()).padStart(2, "0")}:${String(d.getUTCMinutes()).padStart(2, "0")}`;
}

// ------------------------------------------------------------------------------------------- panel helpers
function hideNativeWidget(widget) {
  if (!widget || widget._atlasGlobeHidden) return;
  widget._atlasGlobeHidden = true;
  widget._atlasGlobeOriginalType = widget.type;
  widget._atlasGlobeOriginalComputeSize = widget.computeSize;
  widget.type = "atlas_hidden";
  widget.computeSize = (width) => [width, -4];
  // LiteGraph and the Vue renderer consult different visibility fields; both must be set or a zero-height native
  // value can still paint across the panel.
  widget.hidden = true;
  if (widget.options) widget.options.hidden = true;
}

function setNative(widget, value) {
  if (Object.is(value, -0)) value = 0;
  if (!widget || Object.is(widget.value, value)) return false;
  widget.value = value;
  widget.callback?.(value);
  return true;
}

function button(label, title) {
  const el = document.createElement("button");
  el.textContent = label;
  el.title = title;
  Object.assign(el.style, {
    border: "1px solid #375568", borderRadius: "7px", padding: "7px 6px", flex: "1 1 auto", whiteSpace: "nowrap",
    color: "#d9f7ff", background: "#122430", font: "700 9px system-ui", letterSpacing: ".06em", cursor: "pointer",
  });
  return el;
}

function overlay(style) {
  const el = document.createElement("div");
  Object.assign(el.style, { position: "absolute", zIndex: 2, pointerEvents: "none", ...style });
  return el;
}

const MODE_LOOK = {
  metadata: { text: "METADATA SUN", color: "#9ff3e6", background: "#0f2e33", border: "#2f6f74" },
  offset: { text: "OFFSET SUN", color: "#ffd28b", background: "#4b2d0c", border: "#b87721" },
  manual: { text: "MANUAL SUN", color: "#ffd28b", background: "#5a2a08", border: "#d08a2a" },
  error: { text: "CHECK CAPTURE TIME", color: "#ffb4a0", background: "#4a1610", border: "#a8442e" },
};

// track colour by sun elevation: night, twilight, golden, day
const SKY_STOPS = [[-90, [8, 14, 22]], [-18, [8, 17, 26]], [-6, [27, 41, 80]], [0, [94, 58, 98]], [4, [212, 130, 58]],
  [12, [226, 181, 106]], [30, [95, 159, 190]], [90, [127, 196, 221]]];
function skyColour(el) {
  for (let i = 1; i < SKY_STOPS.length; i++) {
    const [e1, c1] = SKY_STOPS[i];
    if (el <= e1) {
      const [e0, c0] = SKY_STOPS[i - 1], t = (el - e0) / (e1 - e0 || 1);
      return `rgb(${c0.map((c, k) => Math.round(c + (c1[k] - c) * t)).join(",")})`;
    }
  }
  return "rgb(127,196,221)";
}

function labelSprite(THREE, text, color, scale) {
  const c = document.createElement("canvas");
  c.width = 128; c.height = 128;
  const g = c.getContext("2d");
  g.font = "800 80px Inter, system-ui, sans-serif";
  g.textAlign = "center"; g.textBaseline = "middle";
  g.lineWidth = 12; g.strokeStyle = "rgba(4,10,14,.9)"; g.strokeText(text, 64, 68);
  g.fillStyle = color; g.fillText(text, 64, 68);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, depthWrite: false }));
  s.scale.set(scale, scale, 1);
  s.renderOrder = 30;
  return s;
}

function radialTexture(THREE, ring) {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  if (ring) {
    g.strokeStyle = "rgba(255,255,255,1)"; g.lineWidth = 5;
    g.beginPath(); g.arc(32, 32, 26, 0, Math.PI * 2); g.stroke();
  } else {
    const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grd.addColorStop(0, "rgba(255,255,255,1)"); grd.addColorStop(0.25, "rgba(255,255,255,.55)");
    grd.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grd; g.fillRect(0, 0, 64, 64);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

const PANEL_H = 446;

// ------------------------------------------------------------------------------------------- the panel
function buildGlobe(node, THREE) {
  const W = (name) => node.widgets?.find((w) => w.name === name);
  const owned = Object.fromEntries(OWNED_WIDGETS.map((n) => [n, W(n)]));
  const missing = OWNED_WIDGETS.filter((n) => !owned[n]);
  if (missing.length) throw new Error(`Physical Sky widgets missing: ${missing.join(", ")}`);
  const num = (name, d = 0) => { const v = Number(W(name)?.value); return Number.isFinite(v) ? v : d; };
  const str = (name) => String(W(name)?.value ?? "");
  node.properties = node.properties || {};
  const autoQueue = () => Boolean(node.properties[AUTO_QUEUE_PROPERTY] ?? AUTO_QUEUE_DEFAULT);

  // ---------------------------------------------------------------- DOM
  const root = document.createElement("div");
  Object.assign(root.style, {
    width: "100%", height: `${PANEL_H}px`, boxSizing: "border-box", overflow: "hidden",
    border: "1px solid #294a58", borderRadius: "11px", color: "#d9f7ff",
    background: "radial-gradient(circle at 50% 35%, #173b45 0%, #09161e 62%, #060d12 100%)",
    fontFamily: "Inter, system-ui, sans-serif", userSelect: "none", position: "relative",
  });
  const heading = overlay({ left: "15px", top: "13px" });
  heading.innerHTML = '<div style="font-size:10px;letter-spacing:.24em;color:#6cd8e5">ATLAS PHYSICAL SKY</div>' +
    '<div style="font-size:19px;font-weight:800;letter-spacing:.05em;margin-top:2px">SUN GLOBE</div>';
  const badge = overlay({ right: "13px", top: "14px", borderRadius: "99px", padding: "5px 8px",
    font: "800 9px system-ui", letterSpacing: ".11em" });
  const status = overlay({ right: "14px", top: "38px", color: "#6f98a2", font: "600 8.5px system-ui",
    letterSpacing: ".09em", textAlign: "right" });
  const canvas = document.createElement("canvas");
  Object.assign(canvas.style, { position: "absolute", left: "0", top: "52px", width: "100%", height: "260px",
    cursor: "grab", touchAction: "none", zIndex: 1 });
  const legend = overlay({ left: "14px", top: "58px", color: "#5f8f99", font: "600 8.5px ui-monospace, monospace",
    letterSpacing: ".06em" });
  legend.textContent = "N -Z  E +X  Y UP";
  const camLabel = overlay({ right: "14px", top: "58px", color: "#7fc9d4", font: "600 8.5px ui-monospace, monospace",
    letterSpacing: ".04em", textAlign: "right" });
  const flag = overlay({ left: "50%", top: "76px", transform: "translateX(-50%)", borderRadius: "99px",
    padding: "4px 9px", font: "800 9px system-ui", letterSpacing: ".11em", color: "#ffd0c0", background: "#5a1d12",
    border: "1px solid #c0523a", display: "none", whiteSpace: "nowrap" });
  flag.textContent = "BELOW HORIZON · SKY NOT RENDERED";
  const relation = overlay({ left: "14px", right: "14px", top: "290px", color: "#9fe0e8",
    font: "700 9.5px system-ui", letterSpacing: ".08em", whiteSpace: "nowrap", overflow: "hidden" });
  const readout = overlay({ left: "14px", right: "14px", top: "318px", display: "flex",
    justifyContent: "space-between", color: "#86dbe4", font: "600 11px ui-monospace, monospace",
    letterSpacing: ".05em", whiteSpace: "nowrap" });
  const scrub = document.createElement("canvas");
  scrub.title = "Time of day: drag to relight the plate to another hour (writes time_offset_h; shift = 1 minute " +
    "steps). In manual the scrubber switches back to the metadata sun.";
  Object.assign(scrub.style, { position: "absolute", left: "12px", right: "12px", top: "340px",
    width: "calc(100% - 24px)", height: "50px", cursor: "ew-resize", touchAction: "none", zIndex: 1 });
  const controls = document.createElement("div");
  Object.assign(controls.style, { position: "absolute", left: "12px", right: "12px", bottom: "10px",
    display: "flex", gap: "6px", zIndex: 2 });
  const resetBtn = button("RESET TO METADATA", "Back to the sun the GPS and capture time give: metadata mode, " +
    "time offset 0, offsets 0");
  const goldenBtn = button("GOLDEN HOUR", "Move the clock to the golden hour on the capture day (the NOAA sun at " +
    "4 deg): evening after solar noon, morning before");
  const viewBtn = button("VIEW NORTH", "View from the south with north up (east to the right), or from behind the " +
    "camera. Drag empty space to orbit, double-click to reset");
  const queueBtn = button("AUTO-QUEUE OFF", "Queue the graph when a drag, scrub or button ends. Off by default: the " +
    "sky usually feeds a heavy graph");
  controls.append(resetBtn, goldenBtn, viewBtn, queueBtn);
  root.append(canvas, heading, badge, status, legend, camLabel, flag, relation, readout, scrub, controls);

  // ---------------------------------------------------------------- three.js scene
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
  const R = 1.25;
  const ambient = new THREE.AmbientLight(0x9bd8e0, 0.7);
  scene.add(ambient);
  const sunLight = new THREE.DirectionalLight(0xffd29a, 2.6);
  scene.add(sunLight, sunLight.target);

  const V = (d, s = R) => new THREE.Vector3(d[0] * s, d[1] * s, d[2] * s);
  const lineMat = (color, opacity) => new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  const polyline = (pts, mat, loop = false) =>
    new (loop ? THREE.LineLoop : THREE.Line)(new THREE.BufferGeometry().setFromPoints(pts), mat);
  const ringPts = (el, n = 96) => Array.from({ length: n }, (_, i) => V(azelToDir(i * 360 / n, el)));

  // sky dome: the last run's preview, seen from inside (BackSide), only while it is still the sun on the widgets.
  // Upper hemisphere only (the preview's lower half is ground and would grey the whole globe).
  const domeMat = new THREE.MeshBasicMaterial({ side: THREE.BackSide, transparent: true, opacity: 0.62,
    depthWrite: false });
  const dome = new THREE.Mesh(new THREE.SphereGeometry(...domeGeometryArgs(R * 0.995)), domeMat);
  dome.visible = false;
  dome.renderOrder = -1;
  scene.add(dome);
  // ground disc: dims whatever is under the horizon
  const ground = new THREE.Mesh(new THREE.CircleGeometry(R, 96), new THREE.MeshBasicMaterial({ color: 0x061017,
    transparent: true, opacity: 0.58, side: THREE.DoubleSide, depthWrite: false }));
  ground.rotation.x = -Math.PI / 2;
  ground.renderOrder = 2;
  scene.add(ground);
  const polar = new THREE.PolarGridHelper(R, 8, 3, 64, 0x1d4250, 0x16323d);
  polar.position.y = 0.002;
  scene.add(polar);
  scene.add(polyline(ringPts(0), lineMat(0x4fa3b5, 0.95), true));
  [30, 60].forEach((e) => scene.add(polyline(ringPts(e), lineMat(0x397486, 0.5), true)));
  scene.add(polyline(ringPts(-35), lineMat(0x23414c, 0.35), true));
  for (let a = 0; a < 360; a += 45) {
    const up = Array.from({ length: 25 }, (_, i) => V(azelToDir(a, i * 90 / 24)));
    const down = Array.from({ length: 25 }, (_, i) => V(azelToDir(a, -i * 90 / 24)));
    scene.add(polyline(up, lineMat(0x397486, 0.38)), polyline(down, lineMat(0x23414c, 0.22)));
  }
  const ticks = [];
  for (let a = 0; a < 360; a += 15) {
    const d = azelToDir(a, 0), long = a % 90 === 0;
    ticks.push(V(d, R), V(d, R * (long ? 1.09 : 1.045)));
  }
  scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(ticks), lineMat(0x6cd8e5, 0.8)));
  [["N", 0, "#ffb347"], ["E", 90, "#9fe9f2"], ["S", 180, "#9fe9f2"], ["W", 270, "#9fe9f2"]].forEach(([t, a, c]) => {
    const s = labelSprite(THREE, t, c, 0.24);
    s.position.copy(V(azelToDir(a, 0), R * 1.2));
    scene.add(s);
  });

  // the day's sun path (amber above the horizon, dim under it) and hourly dots
  const pathLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true,
    transparent: true, opacity: 0.95 }));
  const pathDots = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 3.5,
    sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.95 }));
  scene.add(pathLine, pathDots);

  // the sun, its halo, the ray to the centre and the NOAA ghost (offset / manual)
  const sunMat = new THREE.MeshBasicMaterial({ color: 0xffc35a });
  const sun = new THREE.Mesh(new THREE.SphereGeometry(0.075, 24, 16), sunMat);
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: radialTexture(THREE, false), color: 0xffb347,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  halo.scale.set(0.55, 0.55, 1);
  const rayGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 1, 0)]);
  const ray = new THREE.Line(rayGeo, new THREE.LineDashedMaterial({ color: 0xffb347, dashSize: 0.05, gapSize: 0.04,
    transparent: true, opacity: 0.6 }));
  const ghost = new THREE.Sprite(new THREE.SpriteMaterial({ map: radialTexture(THREE, true), color: 0x6cd8e5,
    transparent: true, depthWrite: false, depthTest: false }));
  ghost.scale.set(0.17, 0.17, 1);
  ghost.renderOrder = 25;
  scene.add(sun, halo, ray, ghost);

  // the camera: frustum from the centre, a body, and the actor it looks at (lit by the sun, with its shadow)
  const frustumLines = new THREE.LineSegments(new THREE.BufferGeometry(), lineMat(0x6cd8e5, 0.95));
  const frustumFaces = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ color: 0x6cd8e5,
    transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false }));
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.09, 0.17), new THREE.MeshStandardMaterial({
    color: 0x7ee8f1, emissive: 0x123e47, metalness: 0.45, roughness: 0.3 }));
  const actor = new THREE.Group();
  const skin = new THREE.MeshStandardMaterial({ color: 0xdde8ec, roughness: 0.55 });
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.038, 0.19, 4, 12), skin);
  torso.position.y = 0.135;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.045, 16, 12), skin);
  head.position.y = 0.3;
  actor.add(torso, head);
  const shadow = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ color: 0x000000,
    transparent: true, opacity: 0.5, depthWrite: false, side: THREE.DoubleSide }));
  shadow.renderOrder = 3;
  scene.add(frustumLines, frustumFaces, body, actor, shadow);

  // ---------------------------------------------------------------- state
  let payload = null, payloadId = 0, domeSrc = "", domeTex = null, domeReady = false;
  let st = null, builtPath = null, dirty = true, lastKey = "", lastSize = "", disposed = false, warned = false;
  let flashMsg = null;
  const view = { yaw: 0, pitch: 0.55, dist: 6.3, follow: false };
  const pathCache = { key: null, path: null };

  function currentPath(lat, lon, cap) {
    const ins = payload?.inputs;
    if (payload?.path && ins && Math.abs(ins.latitude - lat) < 1e-9 && Math.abs(ins.longitude - lon) < 1e-9
      && ins.capture_utc === cap) return payload.path;
    const key = `${lat}|${lon}|${cap}`;
    if (pathCache.key !== key) { pathCache.key = key; pathCache.path = sunPath(lat, lon, cap); }
    return pathCache.path;
  }

  function readState() {
    const lat = num("latitude"), lon = num("longitude"), cap = str("capture_utc");
    const mode = String(owned.sun_mode.value), toff = num("time_offset_h");
    const sunNow = effectiveSun(mode, lat, lon, cap, toff, num("sun_azimuth_offset"), num("sun_elevation_offset"),
      num("sun_azimuth"), num("sun_elevation"));
    const shotLinked = Boolean(node.inputs?.some((i) => i.name === "shot_camera" && i.link != null));
    const shot = shotLinked && payload?.camera?.source === "shot_camera" ? payload.camera : null;
    const cam = { heading: num("heading_deg"), fov: shot ? Number(shot.fov_x) : num("fov_x", 50),
      tilt: shot ? Number(shot.tilt_deg) : num("tilt_deg"), aspect: shot ? Number(shot.aspect) : 16 / 9,
      source: shot ? "shot" : (shotLinked ? "shot_pending" : "typed") };
    let fresh = false;
    if (payload?.sun && !sunNow.error) {
      fresh = Math.abs(wrap180(payload.sun.azimuth_deg - sunNow.azimuth_deg)) < 0.02
        && Math.abs(payload.sun.elevation_deg - sunNow.elevation_deg) < 0.02;
    }
    return { lat, lon, cap, mode, toff, sun: sunNow, cam, path: currentPath(lat, lon, cap), fresh,
      clock: clockOffsetMin(cap) };
  }

  function onPayload(g) {
    payload = g; payloadId += 1; dirty = true;
    const src = typeof g?.preview === "string" && g.preview.startsWith("data:image/") ? g.preview : "";
    if (src === domeSrc) return;
    domeSrc = src; domeReady = false;
    domeMat.map = null; domeMat.needsUpdate = true;
    domeTex?.dispose(); domeTex = null;
    if (!src) return;
    const img = new Image();
    img.onload = () => {
      if (disposed || domeSrc !== src) return;
      const tex = new THREE.Texture(img);                 // mapping: domeUvTransform (tests/test_sun_globe.py)
      const uvt = domeUvTransform(g.latlong_centre_deg);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.wrapS = THREE.RepeatWrapping;
      tex.repeat.set(...uvt.repeat);
      tex.offset.set(...uvt.offset);
      tex.needsUpdate = true;
      domeTex = tex; domeMat.map = tex; domeMat.needsUpdate = true; domeReady = true; dirty = true;
    };
    img.src = src;
  }

  function flash(text) { flashMsg = { text, until: performance.now() + 2500 }; dirty = true; }

  // ---------------------------------------------------------------- drawing
  function swapGeometry(obj, positions, colours) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    if (colours) geo.setAttribute("color", new THREE.Float32BufferAttribute(colours, 3));
    geo.computeBoundingSphere();
    obj.geometry.dispose();
    obj.geometry = geo;
  }

  function setPath(path) {
    if (path === builtPath) return;
    builtPath = path;
    pathLine.visible = pathDots.visible = Boolean(path);
    if (!path) return;
    const pos = [], col = [], dpos = [], dcol = [];
    const cUp = new THREE.Color(0xffb347), cTw = new THREE.Color(0xa0584c), cDn = new THREE.Color(0x35505a);
    const hourly = Math.max(1, Math.round(60 / (path.step_min || 10)));
    path.azimuth_deg.forEach((az, i) => {
      const el = path.elevation_deg[i], d = azelToDir(az, el);
      const c = el >= 0 ? cUp : (el >= PRAGUE_MIN_ELEVATION ? cTw : cDn);
      pos.push(d[0] * R, d[1] * R, d[2] * R); col.push(c.r, c.g, c.b);
      if (i % hourly === 0) { dpos.push(d[0] * R, d[1] * R, d[2] * R); dcol.push(c.r, c.g, c.b); }
    });
    swapGeometry(pathLine, pos, col);
    swapGeometry(pathDots, dpos, dcol);
  }

  function setCamera(c, hot) {
    const { r, u, f } = cameraBasis(c.heading, c.tilt);
    const L = 0.95 * R, tx = Math.tan(clamp(c.fov, 1, 170) * DEG / 2) * L, ty = tx / (c.aspect || 16 / 9);
    const k = Math.min(1, 1.05 * R / Math.hypot(L, tx, ty));
    const P = (sf, sx, sy) => [(f[0] * L * sf + r[0] * tx * sx + u[0] * ty * sy) * k,
      (f[1] * L * sf + r[1] * tx * sx + u[1] * ty * sy) * k, (f[2] * L * sf + r[2] * tx * sx + u[2] * ty * sy) * k];
    const o = [0, 0, 0], tl = P(1, -1, 1), tr = P(1, 1, 1), br = P(1, 1, -1), bl = P(1, -1, -1);
    const t1 = P(1, -0.3, 1.04), t2 = P(1, 0.3, 1.04), t3 = P(1, 0, 1.45);       // the "up" triangle
    swapGeometry(frustumLines, [o, tl, o, tr, o, br, o, bl, tl, tr, tr, br, br, bl, bl, tl, t1, t2, t2, t3, t3, t1].flat());
    swapGeometry(frustumFaces, [o, tl, tr, o, tr, br, o, br, bl, o, bl, tl].flat());
    const colour = hot ? 0xffb347 : 0x6cd8e5;
    frustumLines.material.color.set(colour);
    frustumFaces.material.color.set(colour);
    body.position.set(-f[0] * 0.06, -f[1] * 0.06, -f[2] * 0.06);
    body.lookAt(f[0] * 10, f[1] * 10, f[2] * 10);
    const h = azelToDir(c.heading, 0);
    actor.position.set(h[0] * 0.62 * R, 0, h[2] * 0.62 * R);
  }

  function setShadow(s) {
    const el = s.sun.error ? -90 : s.sun.elevation_deg;
    shadow.visible = el > 0.5;
    if (!shadow.visible) return;
    const a = s.sun.azimuth_deg * DEG, d = [-Math.sin(a), 0, Math.cos(a)];          // away from the sun
    const len = Math.min(0.33 / Math.tan(el * DEG), 1.0), w = 0.03;
    const b = actor.position, p = [d[2] * w, 0, -d[0] * w], y = 0.004;
    const q0 = [b.x + p[0], y, b.z + p[2]], q1 = [b.x - p[0], y, b.z - p[2]];
    const e = [b.x + d[0] * len, y, b.z + d[2] * len];
    const q2 = [e[0] - p[0] * 0.6, y, e[2] - p[2] * 0.6], q3 = [e[0] + p[0] * 0.6, y, e[2] + p[2] * 0.6];
    swapGeometry(shadow, [q0, q1, q2, q0, q2, q3].flat());
  }

  function updateScene(s) {
    const ok = !s.sun.error;
    sun.visible = ray.visible = ok;
    halo.visible = false;
    let rel = null;
    if (ok) {
      const az = s.sun.azimuth_deg, el = s.sun.elevation_deg, d = azelToDir(az, el);
      sun.position.copy(V(d));
      halo.position.copy(sun.position);
      const below = el < PRAGUE_MIN_ELEVATION, under = el < 0;
      sunMat.color.set(below ? 0x5b7380 : (under ? 0xd27a5a : 0xffc35a));
      halo.visible = !below;
      halo.material.opacity = under ? 0.45 : 1;
      rayGeo.attributes.position.setXYZ(1, sun.position.x, sun.position.y, sun.position.z);
      rayGeo.attributes.position.needsUpdate = true;
      rayGeo.computeBoundingSphere();
      ray.computeLineDistances();
      sunLight.position.copy(V(d, 5));
      sunLight.intensity = el > 0 ? 2.6 : (below ? 0 : 0.5);
      ambient.intensity = el > 0 ? 0.7 : 0.45;
      rel = sunRelation(az, el, s.cam.heading, s.cam.tilt, s.cam.fov, s.cam.aspect);
    }
    const g = s.sun.derived;
    ghost.visible = Boolean(ok && g && s.mode !== "metadata"
      && (Math.abs(wrap180(g.azimuth_deg - s.sun.azimuth_deg)) > 0.05 || Math.abs(g.elevation_deg - s.sun.elevation_deg) > 0.05));
    if (ghost.visible) ghost.position.copy(V(azelToDir(g.azimuth_deg, g.elevation_deg)));
    setPath(s.path);
    setCamera(s.cam, Boolean(rel?.in_frame));
    setShadow(s);
    dome.visible = Boolean(domeReady && s.fresh && payload?.sky_rendered);
    return rel;
  }

  const signed = (v) => `${v >= 0 ? "+" : "-"}${Math.abs(v).toFixed(1)}`;
  const clockLabel = (m) => (m === null ? "UTC" : `${m >= 0 ? "+" : "-"}${String(Math.floor(Math.abs(m) / 60)).padStart(2, "0")}:${String(Math.abs(m) % 60).padStart(2, "0")}`);

  // which sky model drew the last run (textContent only)
  const skyModelTag = () => (payload?.sky_model_used === "hosek-wilkie" ? " · HOSEK-WILKIE" : "");

  function updatePanel(s, rel, flashOn) {
    const look = s.sun.error ? MODE_LOOK.error : (MODE_LOOK[s.mode] || MODE_LOOK.error);
    badge.textContent = look.text;
    Object.assign(badge.style, { color: look.color, background: look.background, border: `1px solid ${look.border}` });
    status.textContent = flashOn ? flashMsg.text : (!payload ? "NOT RUN YET · DRAWN FROM WIDGETS"
      : s.fresh ? (payload.sky_rendered ? `LAST RUN · SKY RENDERED${skyModelTag()}` : "LAST RUN · SKY NOT RENDERED")
        : `CHANGED SINCE LAST RUN${autoQueue() ? "" : " · QUEUE TO RENDER"}`);
    status.style.color = flashOn ? "#ffd28b" : "#6f98a2";
    const src = s.cam.source === "shot" ? "SHOT CAM" : (s.cam.source === "shot_pending" ? "SHOT CAM (RUN TO READ)" : "CAM");
    camLabel.textContent = `${src} ${s.cam.heading.toFixed(1)}° · HFOV ${s.cam.fov.toFixed(1)}° · TILT ${signed(s.cam.tilt)}°`;
    // the last run's model sets the limit (Hosek-Wilkie stops at the horizon); before a run, Prague's
    const minEl = s.fresh && typeof payload?.min_elevation_deg === "number" ? payload.min_elevation_deg : PRAGUE_MIN_ELEVATION;
    flag.style.display = !s.sun.error && s.sun.elevation_deg < minEl ? "block" : "none";
    if (s.sun.error) {
      relation.textContent = s.sun.error.toUpperCase();
      relation.style.color = "#ffb4a0";
    } else {
      const under = s.sun.elevation_deg < 0;                 // no lighting words for a sun under the horizon
      const side = under ? (rel.side === "behind_actor" ? "UNDER THE HORIZON, AHEAD OF THE LENS"
        : rel.side === "in_front_of_actor" ? "UNDER THE HORIZON, BEHIND THE CAMERA"
          : `UNDER THE HORIZON, SCREEN ${rel.hand.toUpperCase()}`)
        : rel.side === "behind_actor" ? "SUN BEHIND THE ACTOR · BACKLIT"
          : rel.side === "in_front_of_actor" ? "SUN IN FRONT OF THE ACTOR · FRONT-LIT"
            : `SUN BESIDE · LIT FROM SCREEN ${rel.hand.toUpperCase()}`;
      const angle = `${Math.abs(rel.rel).toFixed(0)}° ${rel.rel >= 0 ? "RIGHT" : "LEFT"} OF THE LENS`;
      relation.textContent = `${side} · ${angle}${rel.in_frame ? " · IN FRAME" : ""}`;
      relation.style.color = rel.in_frame ? "#ffd28b" : "#9fe0e8";
    }
    let time = "NO CAPTURE TIME";
    if (s.sun.sun_ms !== null && s.sun.sun_ms !== undefined) {
      time = s.mode === "manual" ? "MANUAL · CLOCK UNUSED"
        : `${fmtClock(s.sun.sun_ms, s.clock)} ${clockLabel(s.clock)} · SOLAR ${fmtClock(s.sun.sun_ms, s.lon * 4)}`;
    }
    readout.innerHTML = s.sun.error ? `<span>AZ --</span><span>EL --</span><span>${time}</span>`
      : `<span>AZ ${s.sun.azimuth_deg.toFixed(1)}°</span><span>EL ${signed(s.sun.elevation_deg)}°</span><span>${time}</span>`;
    viewBtn.textContent = view.follow ? "VIEW CAMERA" : "VIEW NORTH";
    queueBtn.textContent = autoQueue() ? "AUTO-QUEUE ON" : "AUTO-QUEUE OFF";
    Object.assign(queueBtn.style, autoQueue() ? { color: "#ffd28b", background: "#4b2d0c", borderColor: "#b87721" }
      : { color: "#d9f7ff", background: "#122430", borderColor: "#375568" });
  }

  function drawScrub(s) {
    const rect = scrub.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(rect.width)), h = Math.max(1, Math.round(rect.height));
    if (scrub.width !== w * dpr || scrub.height !== h * dpr) { scrub.width = w * dpr; scrub.height = h * dpr; }
    const g = scrub.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const top = 8, th = 16;
    const track = () => { g.beginPath(); g.roundRect ? g.roundRect(0.5, top, w - 1, th, 5) : g.rect(0.5, top, w - 1, th); };
    g.font = "600 8.5px ui-monospace, monospace";
    g.textBaseline = "top";
    const p = s.path, cap = s.sun.capture_ms;
    if (!p || cap === null || cap === undefined) {
      track(); g.fillStyle = "#0d1a21"; g.fill(); g.strokeStyle = "#294a58"; g.stroke();
      g.fillStyle = "#6f98a2"; g.textAlign = "center";
      g.fillText("NO CAPTURE TIME: THE SCRUBBER NEEDS ONE", w / 2, top + 4);
      return;
    }
    const n = p.elevation_deg.length, span = p.end_offset_h - p.start_offset_h;
    g.save(); track(); g.clip();
    for (let x = 0; x < w; x++) {
      const t = x / Math.max(1, w - 1) * (n - 1), i = Math.min(n - 2, Math.floor(t));
      g.fillStyle = skyColour(p.elevation_deg[i] + (p.elevation_deg[i + 1] - p.elevation_deg[i]) * (t - i));
      g.fillRect(x, top, 1.5, th);
    }
    const manual = s.mode === "manual";
    if (manual) { g.fillStyle = "rgba(6,13,18,.6)"; g.fillRect(0, top, w, th); }
    g.restore();
    track(); g.strokeStyle = "#375568"; g.stroke();
    const X = (o) => (o - p.start_offset_h) / span * w;
    const label = (o, text, colour) => {
      const x = X(o);
      g.fillStyle = colour; g.fillRect(x - 0.5, top - 1, 1, th + 2);
      g.textAlign = "center"; g.fillText(text, clamp(x, 30, w - 30), top + th + 5);
    };
    const at = (o) => fmtClock(cap + o * 3600000, s.clock);
    if (p.sunrise) label(p.sunrise.offset_h, `RISE ${at(p.sunrise.offset_h)}`, "#d9f7ff");
    if (p.noon) label(p.noon.offset_h, `NOON ${at(p.noon.offset_h)}`, "#86dbe4");
    if (p.sunset) label(p.sunset.offset_h, `SET ${at(p.sunset.offset_h)}`, "#d9f7ff");
    if (p.polar) { g.fillStyle = "#d9f7ff"; g.textAlign = "center"; g.fillText(`POLAR ${p.polar.toUpperCase()}`, w / 2, top + th + 5); }
    const xs = X(0);                                                     // the shot's own moment
    g.fillStyle = "#ffffff";
    g.beginPath(); g.moveTo(xs - 4, 0); g.lineTo(xs + 4, 0); g.lineTo(xs, 6); g.closePath(); g.fill();
    if (manual) {
      g.fillStyle = "#ffd28b"; g.textAlign = "center";
      g.fillText("MANUAL SUN: SCRUB TO GO BACK TO METADATA", w / 2, top + 4);
    } else {
      const inside = s.toff >= p.start_offset_h && s.toff <= p.end_offset_h;
      const xt = clamp(X(s.toff), 1.5, w - 1.5);
      g.fillStyle = inside ? "#ffb347" : "#8a6a40";
      g.fillRect(xt - 1.5, top - 4, 3, th + 8);
      g.beginPath(); g.arc(xt, top + th / 2, 4.5, 0, Math.PI * 2); g.fill();
    }
  }

  // ---------------------------------------------------------------- interaction
  const ndc = new THREE.Vector2(), raycaster = new THREE.Raycaster();
  const sphere = new THREE.Sphere(new THREE.Vector3(), R), hit = new THREE.Vector3();
  function pickAzEl(ev) {
    const rect = canvas.getBoundingClientRect();
    ndc.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    if (!raycaster.ray.intersectSphere(sphere, hit)) raycaster.ray.closestPointToPoint(sphere.center, hit);
    return dirToAzel([hit.x, hit.y, hit.z]);
  }
  function sunClientXY() {
    const v = sun.position.clone().project(camera), rect = canvas.getBoundingClientRect();
    return [rect.left + (v.x + 1) / 2 * rect.width, rect.top + (1 - v.y) / 2 * rect.height];
  }
  const nearSun = (ev) => {
    if (!sun.visible) return false;
    const [sx, sy] = sunClientXY();
    return Math.hypot(ev.clientX - sx, ev.clientY - sy) < 16;
  };
  function applyWidgets(values) {
    let changed = false;
    for (const [k, v] of Object.entries(values)) changed = setNative(owned[k], v) || changed;
    if (changed) { dirty = true; node.setDirtyCanvas?.(true, true); }
    return changed;
  }
  function queueIfAuto() {
    // Once, when an interaction ends, and only when the artist asked for it: never from a pointermove.
    if (autoQueue()) app.queuePrompt(0, 1);
  }
  function endInteraction(changed) {
    dirty = true;
    node.setDirtyCanvas?.(true, true);
    if (changed) queueIfAuto();
  }

  let drag = null;
  canvas.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0 || !st) return;
    canvas.setPointerCapture?.(ev.pointerId);
    drag = { kind: nearSun(ev) ? "sun" : "orbit", x: ev.clientX, y: ev.clientY, yaw: view.yaw, pitch: view.pitch,
      moved: false, changed: false, fine: null, dAz: 0, dEl: 0 };
    if (drag.kind === "sun") {
      const [paz, pel] = pickAzEl(ev);                       // keep the grab point: no jump on the first move
      drag.dAz = wrap180(st.sun.azimuth_deg - paz);
      drag.dEl = st.sun.elevation_deg - pel;
    }
    canvas.style.cursor = "grabbing";
    ev.preventDefault();
    ev.stopPropagation();
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (!drag) { canvas.style.cursor = nearSun(ev) ? "pointer" : "grab"; return; }
    if (Math.abs(ev.clientX - drag.x) + Math.abs(ev.clientY - drag.y) > 1) drag.moved = true;
    if (drag.kind === "orbit") {
      view.follow = false;
      view.yaw = drag.yaw - (ev.clientX - drag.x) * 0.008;
      view.pitch = clamp(drag.pitch + (ev.clientY - drag.y) * 0.006, 0.06, 1.5);
      dirty = true;
      return;
    }
    if (!drag.moved) return;
    let [az, el] = pickAzEl(ev);
    az += drag.dAz; el += drag.dEl;
    if (ev.shiftKey) {                                          // fine: a fifth of the cursor's travel
      if (!drag.fine) drag.fine = { az: st.sun.azimuth_deg, el: st.sun.elevation_deg, paz: az, pel: el };
      az = drag.fine.az + wrap180(az - drag.fine.paz) * 0.2;
      el = drag.fine.el + (el - drag.fine.pel) * 0.2;
    } else {
      drag.fine = null;
    }
    drag.changed = applyWidgets(dragWidgets(String(owned.sun_mode.value), az, el, st.sun.derived)) || drag.changed;
  });
  const endDrag = () => {
    if (!drag) return;
    const d = drag;
    drag = null;
    canvas.style.cursor = "grab";
    if (d.kind === "sun") endInteraction(d.changed);
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("dblclick", () => { Object.assign(view, { yaw: 0, pitch: 0.55, follow: false }); dirty = true; });
  // The loop only renders on change, so a lost and restored context (too many WebGL canvases, a GPU reset) would stay
  // blank: draw again once three.js has the context back.
  canvas.addEventListener("webglcontextrestored", () => { dirty = true; });

  let scrubbing = null;
  function scrubTo(ev) {
    const cap = st?.sun?.capture_ms;
    if (!st?.path || cap === null || cap === undefined) return;
    const rect = scrub.getBoundingClientRect();
    const values = { time_offset_h: offsetAtX((ev.clientX - rect.left) / rect.width, st.path, cap, st.clock,
      ev.shiftKey ? 1 : 5) };
    if (String(owned.sun_mode.value) === "manual") values.sun_mode = "metadata";
    scrubbing.changed = applyWidgets(values) || scrubbing.changed;
  }
  scrub.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0 || !st?.path) return;
    scrub.setPointerCapture?.(ev.pointerId);
    scrubbing = { changed: false };
    scrubTo(ev);
    ev.preventDefault();
    ev.stopPropagation();
  });
  scrub.addEventListener("pointermove", (ev) => { if (scrubbing) scrubTo(ev); });
  const endScrub = () => {
    if (!scrubbing) return;
    const changed = scrubbing.changed;
    scrubbing = null;
    endInteraction(changed);
  };
  scrub.addEventListener("pointerup", endScrub);
  scrub.addEventListener("pointercancel", endScrub);

  resetBtn.onclick = () => endInteraction(applyWidgets({ sun_mode: "metadata", time_offset_h: 0, sun_azimuth_offset: 0,
    sun_elevation_offset: 0 }));
  goldenBtn.onclick = () => {
    const p = st?.path;
    if (!p) { flash("GOLDEN HOUR NEEDS A CAPTURE TIME"); return; }
    const evening = p.noon ? st.toff >= p.noon.offset_h : true;
    const ev = (evening ? p.golden_evening : p.golden_morning) || p.golden_evening || p.golden_morning;
    if (!ev) { flash("NO GOLDEN HOUR THAT DAY"); return; }
    const values = { time_offset_h: Math.round(Math.round(ev.offset_h * 60) / 60 * 1e6) / 1e6 };
    if (String(owned.sun_mode.value) === "manual") values.sun_mode = "metadata";
    endInteraction(applyWidgets(values));
  };
  viewBtn.onclick = () => {
    view.follow = !view.follow;
    Object.assign(view, view.follow ? { pitch: 0.42 } : { yaw: 0, pitch: 0.55 });
    dirty = true;
  };
  queueBtn.onclick = () => { node.properties[AUTO_QUEUE_PROPERTY] = !autoQueue(); dirty = true; };

  // ---------------------------------------------------------------- frame loop (renders only when something changed)
  function tick() {
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return;
    const w = Math.round(rect.width), h = Math.round(rect.height);
    st = readState();
    if (view.follow) view.yaw = -st.cam.heading * DEG;
    const flashOn = Boolean(flashMsg && performance.now() < flashMsg.until);
    const key = JSON.stringify([st.mode, st.toff, st.lat, st.lon, st.cap, st.sun.azimuth_deg, st.sun.elevation_deg,
      st.sun.error, st.cam, st.fresh, payloadId, domeReady, view, w, h, autoQueue(), flashOn]);
    if (!dirty && key === lastKey) return;
    dirty = false;
    lastKey = key;
    if (`${w}x${h}` !== lastSize) {
      lastSize = `${w}x${h}`;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    camera.position.set(view.dist * Math.sin(view.yaw) * Math.cos(view.pitch), view.dist * Math.sin(view.pitch),
      view.dist * Math.cos(view.yaw) * Math.cos(view.pitch));
    camera.lookAt(0, -0.08, 0);
    const rel = updateScene(st);
    updatePanel(st, rel, flashOn);
    drawScrub(st);
    renderer.render(scene, camera);
  }
  function frame() {
    if (disposed) return;
    try {
      tick();
    } catch (err) {
      if (!warned) console.warn("[atlas-vfx] Sun Globe frame failed", err);
      warned = true;
    }
    requestAnimationFrame(frame);
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    scene.traverse((o) => {
      o.geometry?.dispose?.();
      const mats = Array.isArray(o.material) ? o.material : (o.material ? [o.material] : []);
      mats.forEach((m) => { m.map?.dispose?.(); m.dispose?.(); });
    });
    domeTex?.dispose();
    renderer.dispose();
    renderer.forceContextLoss?.();
  }

  // ---------------------------------------------------------------- mount: the panel exists, now hide what it owns
  const domWidget = node.addDOMWidget("sun_globe", "ATLAS_SUN_GLOBE", root, { serialize: false,
    getMinHeight: () => PANEL_H + 8, getMaxHeight: () => PANEL_H + 8 });
  domWidget.serialize = false;
  OWNED_WIDGETS.forEach((n) => hideNativeWidget(owned[n]));
  const size = node.computeSize?.() || [0, 0];
  node.setSize?.([Math.max(node.size?.[0] || 0, size[0], 470), Math.max(size[1], 0)]);

  const priorRemoved = node.onRemoved;
  node.onRemoved = function () {
    dispose();
    priorRemoved?.apply(this, arguments);
  };
  node._atlasSunGlobe = { onPayload, sunClientXY, state: () => st, dispose, isDisposed: () => disposed };
  if (node._atlasSunGlobePayload) onPayload(node._atlasSunGlobePayload);
  frame();
}

app.registerExtension({
  name: EXTENSION_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;
    let THREE;
    try {
      THREE = await import("./lib/atlas-three.bundle.js");
    } catch (err) {
      console.warn("[atlas-vfx] Sun Globe: three.js did not load; Physical Sky keeps its native widgets", err);
      return;
    }
    const originalCreated = nodeType.prototype.onNodeCreated;
    const originalExecuted = nodeType.prototype.onExecuted;

    nodeType.prototype.onExecuted = function (message) {
      originalExecuted?.apply(this, arguments);
      const g = Array.isArray(message?.atlas_sun_globe) ? message.atlas_sun_globe[0] : message?.atlas_sun_globe;
      if (g && typeof g === "object") {
        this._atlasSunGlobePayload = g;
        this._atlasSunGlobe?.onPayload(g);
      }
    };

    nodeType.prototype.onNodeCreated = function () {
      originalCreated?.apply(this, arguments);
      try {
        buildGlobe(this, THREE);
      } catch (err) {
        // Nothing is hidden until the panel is built, so a failure leaves an ordinary, working node.
        console.warn("[atlas-vfx] Sun Globe could not start; Physical Sky keeps its native widgets", err);
      }
    };
  },
});

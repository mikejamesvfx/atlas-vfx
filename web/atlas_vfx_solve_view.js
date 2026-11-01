/**
 * Atlas VFX Solve View (catalogue 69): a solved camera and its points, drawn from the node's ui payload
 * (ui.atlas_solve_view[0], node_solve_view.payload).
 *
 * 3D tab: points coloured by their worst reprojection (green < 0.5 px, yellow < 2 px, red beyond, plate size),
 * the camera path, and the frustum at the scrubbed frame. Drag to orbit, right-drag (or shift-drag) to pan,
 * wheel to zoom. Under it, the median reprojection error per frame; the scrub line marks the current frame.
 * Plate tab: the payload's tracking proxies with the solve's points drawn where the camera reprojects them.
 * Clicking a plate frame appends a seed {label, frame (plate index), xy (plate px)} to the seeds widget, which is
 * the only widget the panel writes. Display only: the node runs the same with no frontend.
 * World axes are the shot camera's (metres, Y up, camera looks down -Z), the same as three.js.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

export const NODE_NAME = "AtlasVFXSolveView";
export const UI_KEY = "atlas_solve_view";
const PANEL_H = 420;
const STRIP_H = 44;

export function errColor(e) {
  if (!(e >= 0)) return [0.6, 0.6, 0.6];
  if (e < 0.5) return [0.1, 0.85, 0.2];
  if (e < 2.0) return [0.95, 0.8, 0.1];
  return [0.95, 0.15, 0.1];
}

/** Next seed label not already in the list. */
export function nextLabel(seeds) {
  const used = new Set(seeds.map((s) => s.label));
  let n = seeds.length + 1;
  while (used.has(`s${n}`)) n += 1;
  return `s${n}`;
}

export function addSeed(text, frame, x, y) {
  let seeds = [];
  try { seeds = JSON.parse(text || "[]"); } catch { seeds = []; }
  if (!Array.isArray(seeds)) seeds = [];
  seeds.push({ label: nextLabel(seeds), frame, xy: [Math.round(x * 10) / 10, Math.round(y * 10) / 10] });
  return JSON.stringify(seeds);
}

function buildPanel(node, THREE) {
  const seedsW = node.widgets?.find((w) => w.name === "seeds");
  const root = document.createElement("div");
  root.style.cssText = "display:flex;flex-direction:column;gap:4px;width:100%;height:100%;font:11px sans-serif;" +
    "color:#ddd;background:#1b1b1b;border-radius:4px;padding:4px;box-sizing:border-box";
  const bar = document.createElement("div");
  bar.style.cssText = "display:flex;gap:6px;align-items:center";
  const mkBtn = (t) => {
    const b = document.createElement("button");
    b.textContent = t;
    b.style.cssText = "background:#333;color:#ddd;border:1px solid #555;border-radius:3px;padding:1px 8px;cursor:pointer";
    bar.appendChild(b);
    return b;
  };
  const tab3d = mkBtn("3D"), tabPlate = mkBtn("Plate"), prevB = mkBtn("<"), playB = mkBtn("Play"), nextB = mkBtn(">");
  const info = document.createElement("span");
  info.style.cssText = "flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;opacity:.8";
  bar.appendChild(info);
  root.appendChild(bar);

  const view = document.createElement("div");
  view.style.cssText = "position:relative;flex:1;min-height:200px";
  const canvas3d = document.createElement("canvas");
  const canvas2d = document.createElement("canvas");
  for (const c of [canvas3d, canvas2d]) {
    c.style.cssText = "position:absolute;inset:0;width:100%;height:100%";
    view.appendChild(c);
  }
  root.appendChild(view);
  const strip = document.createElement("canvas");
  strip.style.cssText = `width:100%;height:${STRIP_H}px;cursor:pointer`;
  root.appendChild(strip);
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = 0; slider.max = 0; slider.value = 0;
  slider.style.cssText = "width:100%";
  root.appendChild(slider);

  // ------------------------------------------------------------------ three
  const renderer = new THREE.WebGLRenderer({ canvas: canvas3d, antialias: true });
  renderer.setClearColor(0x111111);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10000);
  scene.add(new THREE.GridHelper(20, 20, 0x444444, 0x2a2a2a));
  const group = new THREE.Group();
  scene.add(group);
  let frustum = null;
  const orbit = { target: new THREE.Vector3(), r: 10, az: 0.6, el: 0.4 };

  const st = { p: null, idx: 0, tab: "3d", img: 0, images: [], disposed: false, playing: false, due: 0 };

  // the plate image nearest the scrubbed frame (all frames when plate_frames covers the shot)
  function syncImage() {
    if (!st.p || !st.images.length) return;
    const f = st.p.frames[st.idx].f;
    let best = 0;
    st.images.forEach((im, i) => { if (Math.abs(im.frame - f) < Math.abs(st.images[best].frame - f)) best = i; });
    st.img = best;
  }

  function placeCamera() {
    const { target, r, az, el } = orbit;
    camera.position.set(target.x + r * Math.cos(el) * Math.sin(az), target.y + r * Math.sin(el),
      target.z + r * Math.cos(el) * Math.cos(az));
    camera.lookAt(target);
  }

  function frameMatrix(i) {
    const m = st.p.frames[i].m;
    return new THREE.Matrix4().set(...m);
  }

  function buildFrustum(scale) {
    const p = st.p;
    const half = Math.tan((p.fov_x_deg * Math.PI) / 360);
    const a = p.plate[1] / p.plate[0];
    const d = scale;
    const c = [[-half, a * half], [half, a * half], [half, -a * half], [-half, -a * half]]
      .map(([x, y]) => new THREE.Vector3(x * d, y * d, -d));
    const o = new THREE.Vector3();
    const pts = [];
    c.forEach((v, k) => { pts.push(o, v, v, c[(k + 1) % 4]); });
    pts.push(new THREE.Vector3(0, a * half * d, -d), new THREE.Vector3(0, a * half * d * 1.4, -d));  // up tick
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x66ccff }));
    l.matrixAutoUpdate = false;
    return l;
  }

  function onPayload(p) {
    st.p = p;
    group.children.slice().forEach((o) => { group.remove(o); o.geometry?.dispose(); o.material?.dispose(); });
    if (frustum) { scene.remove(frustum); frustum.geometry.dispose(); frustum = null; }
    const centres = p.frames.map((f) => new THREE.Vector3(f.m[3], f.m[7], f.m[11]));
    const box = new THREE.Box3().setFromPoints(centres);
    if (p.points.length) {
      const pos = new Float32Array(p.points);
      const col = new Float32Array(pos.length);
      p.err.forEach((e, i) => col.set(errColor(e), 3 * i));
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.setAttribute("color", new THREE.BufferAttribute(col, 3));
      g.computeBoundingBox();
      group.add(new THREE.Points(g, new THREE.PointsMaterial({ size: 2, sizeAttenuation: false, vertexColors: true })));
      // frame the points' middle 90 %, not stray far points
      const xs = [], ys = [], zs = [];
      for (let i = 0; i < pos.length; i += 3) { xs.push(pos[i]); ys.push(pos[i + 1]); zs.push(pos[i + 2]); }
      const q = (a, t) => { const s = a.slice().sort((u, v) => u - v); return s[Math.floor(t * (s.length - 1))]; };
      box.expandByPoint(new THREE.Vector3(q(xs, 0.05), q(ys, 0.05), q(zs, 0.05)));
      box.expandByPoint(new THREE.Vector3(q(xs, 0.95), q(ys, 0.95), q(zs, 0.95)));
    }
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(centres),
      new THREE.LineBasicMaterial({ color: 0xff66cc })));
    if (p.cube) {                                               // the check cube (orange)
      const v = p.cube.corners_disp.map((c) => new THREE.Vector3(...c));
      const seg = [];
      for (const [a, b] of p.cube_edges) seg.push(v[a], v[b]);
      group.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(seg),
        new THREE.LineBasicMaterial({ color: 0xff9900 })));
    }
    const size = box.getSize(new THREE.Vector3()).length() || 1;
    box.getCenter(orbit.target);
    orbit.r = size * 1.2;
    camera.near = size / 1000; camera.far = size * 100; camera.updateProjectionMatrix();
    frustum = buildFrustum(size * 0.04);
    scene.add(frustum);
    slider.max = String(p.frames.length - 1);
    st.idx = Math.min(st.idx, p.frames.length - 1);
    slider.value = String(st.idx);
    st.images = (p.images || []).map((im) => {
      const el = new Image();
      el.src = api.apiURL(`/view?filename=${encodeURIComponent(im.filename)}&subfolder=${encodeURIComponent(im.subfolder)}` +
        `&type=${encodeURIComponent(im.type)}&t=${Date.now()}`);
      el.onload = () => draw2d();
      return { ...im, el };
    });
    st.img = 0;
    placeCamera();
    update();
  }

  function update() {
    const p = st.p;
    if (!p) { info.textContent = "run the node to see the solve"; return; }
    const f = p.frames[st.idx];
    if (frustum) { frustum.matrix.copy(frameMatrix(st.idx)); frustum.matrixWorldNeedsUpdate = true; }
    const e = p.frame_err[String(f.f)];
    const rep = p.report || {};
    info.textContent = `${p.name} · frame ${f.f} · ${e !== undefined ? e.toFixed(2) + " px" : "-"} · ` +
      `${p.err.length} pts` + (p.cube ? ` · orange cube: ${p.cube.note}` : "") + (rep.warnings && rep.warnings.length ? ` · ${rep.warnings.length} warning(s)` : "");
    info.title = (rep.warnings || []).join("\n") || p.points_note;
    syncImage();
    drawStrip();
    draw2d();
  }

  function drawStrip() {
    const p = st.p;
    const w = strip.clientWidth || 300;
    strip.width = w; strip.height = STRIP_H;
    const c = strip.getContext("2d");
    c.fillStyle = "#151515"; c.fillRect(0, 0, w, STRIP_H);
    if (!p) return;
    const n = p.frames.length;
    const vals = p.frames.map((f) => p.frame_err[String(f.f)]);
    const top = Math.max(2, ...vals.filter((v) => v >= 0));
    vals.forEach((v, i) => {
      if (!(v >= 0)) return;
      const [r, g, b] = errColor(v);
      c.fillStyle = `rgb(${r * 255},${g * 255},${b * 255})`;
      const h = (v / top) * (STRIP_H - 12);
      c.fillRect((i / n) * w, STRIP_H - h, Math.max(1, w / n - 0.5), h);
    });
    c.fillStyle = "#fff";
    c.fillRect(((st.idx + 0.5) / n) * w - 0.5, 0, 1, STRIP_H);
    c.fillText(`median px per frame, top ${top.toFixed(1)}`, 4, 10);
  }

  function draw2d() {
    const vis = st.tab === "plate";
    canvas2d.style.display = vis ? "block" : "none";
    canvas3d.style.display = vis ? "none" : "block";
    if (!vis) return;
    const w = view.clientWidth || 300, h = view.clientHeight || 200;
    canvas2d.width = w; canvas2d.height = h;
    const c = canvas2d.getContext("2d");
    c.fillStyle = "#111"; c.fillRect(0, 0, w, h);
    const im = st.images[st.img];
    if (!im || !im.el.complete || !im.el.naturalWidth) {
      c.fillStyle = "#aaa"; c.fillText(st.images.length ? "loading..." : "no plate frames (plate_frames 0, or not a Solve Plate camera)", 8, 16);
      return;
    }
    const s = Math.min(w / im.el.naturalWidth, h / im.el.naturalHeight);
    const ox = (w - im.el.naturalWidth * s) / 2, oy = (h - im.el.naturalHeight * s) / 2;
    c.drawImage(im.el, ox, oy, im.el.naturalWidth * s, im.el.naturalHeight * s);
    const k = (im.el.naturalWidth / st.p.plate[0]) * s;            // plate px -> canvas px
    for (const [x, y, e] of im.points) {
      const [r, g, b] = errColor(e);
      c.fillStyle = `rgb(${r * 255},${g * 255},${b * 255})`;
      c.fillRect(ox + x * k - 1, oy + y * k - 1, 2, 2);
    }
    if (im.cube && st.p.cube_edges) {                           // the check cube through this frame's camera
      c.strokeStyle = "#ff9900"; c.lineWidth = 2;
      c.beginPath();
      for (const [a, b] of st.p.cube_edges) {
        c.moveTo(ox + im.cube[a][0] * k, oy + im.cube[a][1] * k);
        c.lineTo(ox + im.cube[b][0] * k, oy + im.cube[b][1] * k);
      }
      c.stroke(); c.lineWidth = 1;
    }
    let seeds = [];
    try { seeds = JSON.parse(seedsW?.value || "[]"); } catch { seeds = []; }
    c.strokeStyle = "#0ff";
    for (const sd of Array.isArray(seeds) ? seeds : []) {
      if (sd.frame !== im.frame - (st.p.frame_offset ?? 0)) continue;
      c.beginPath(); c.arc(ox + sd.xy[0] * k, oy + sd.xy[1] * k, 5, 0, 2 * Math.PI); c.stroke();
      c.fillStyle = "#0ff"; c.fillText(sd.label, ox + sd.xy[0] * k + 7, oy + sd.xy[1] * k - 4);
    }
    c.fillStyle = "#fff";
    c.fillText(`frame ${im.frame} (${st.img + 1}/${st.images.length}) · click to add a Track 2D seed`, 6, 12);
    st.geo = { ox, oy, k };
  }

  // ------------------------------------------------------------------ input
  tab3d.onclick = () => { st.tab = "3d"; draw2d(); };
  tabPlate.onclick = () => { st.tab = "plate"; draw2d(); };
  prevB.onclick = () => step(-1);
  nextB.onclick = () => step(1);
  playB.onclick = () => { st.playing = !st.playing; playB.textContent = st.playing ? "Pause" : "Play"; st.due = 0; };
  function step(d) { if (!st.p) return; st.idx = Math.max(0, Math.min(st.p.frames.length - 1, st.idx + d)); slider.value = String(st.idx); update(); }
  slider.oninput = () => { st.idx = Number(slider.value); update(); };
  strip.onclick = (ev) => {
    if (!st.p) return;
    const r = strip.getBoundingClientRect();
    st.idx = Math.max(0, Math.min(st.p.frames.length - 1, Math.floor(((ev.clientX - r.left) / r.width) * st.p.frames.length)));
    slider.value = String(st.idx);
    update();
  };
  canvas2d.onclick = (ev) => {
    const im = st.images[st.img];
    if (!im || !st.geo || !seedsW) return;
    const r = canvas2d.getBoundingClientRect();
    const x = (ev.clientX - r.left - st.geo.ox) / st.geo.k, y = (ev.clientY - r.top - st.geo.oy) / st.geo.k;
    if (x < 0 || y < 0 || x >= st.p.plate[0] || y >= st.p.plate[1]) return;
    // seeds are plate frame indices (Track 2D's frames); the payload's frames are shot frames
    seedsW.value = addSeed(seedsW.value, im.frame - (st.p.frame_offset ?? 0), x, y);
    seedsW.callback?.(seedsW.value);
    node.setDirtyCanvas?.(true, true);
    draw2d();
  };
  let drag = null;
  canvas3d.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas3d.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; canvas3d.setPointerCapture(e.pointerId); });
  canvas3d.addEventListener("pointerup", () => { drag = null; });
  canvas3d.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    if (drag.pan) {
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
      const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
      const s = orbit.r * 0.0015;
      orbit.target.addScaledVector(right, -dx * s).addScaledVector(up, dy * s);
    } else {
      orbit.az -= dx * 0.008;
      orbit.el = Math.max(-1.5, Math.min(1.5, orbit.el + dy * 0.008));
    }
    placeCamera();
  });
  canvas3d.addEventListener("wheel", (e) => { e.preventDefault(); orbit.r *= Math.exp(e.deltaY * 0.001); placeCamera(); }, { passive: false });

  function frame() {
    if (st.disposed) return;
    if (st.playing && st.p) {                                   // 24 fps, looping
      const t = performance.now();
      if (!st.due) st.due = t;
      if (t >= st.due) {
        st.idx = st.idx >= st.p.frames.length - 1 ? 0 : st.idx + 1;
        slider.value = String(st.idx);
        update();
        st.due += 1000 / 24;
        if (t - st.due > 200) st.due = t;                       // fell behind (tab hidden): don't race to catch up
      }
    }
    if (st.tab === "3d") {
      const w = view.clientWidth || 300, h = view.clientHeight || 200;
      if (canvas3d.width !== w || canvas3d.height !== h) {
        renderer.setSize(w, h, false);
        camera.aspect = w / h; camera.updateProjectionMatrix();
      }
      renderer.render(scene, camera);
    }
    requestAnimationFrame(frame);
  }

  const domWidget = node.addDOMWidget("solve_view", "ATLAS_SOLVE_VIEW", root, { serialize: false,
    getMinHeight: () => PANEL_H, getMaxHeight: () => PANEL_H * 2 });
  domWidget.serialize = false;
  const size = node.computeSize?.() || [0, 0];
  node.setSize?.([Math.max(node.size?.[0] || 0, 520), Math.max(size[1], PANEL_H + 120)]);
  const priorRemoved = node.onRemoved;
  node.onRemoved = function () {
    st.disposed = true;
    renderer.dispose();
    renderer.forceContextLoss?.();
    priorRemoved?.apply(this, arguments);
  };
  node._atlasSolveView = { onPayload, state: () => st };
  if (node._atlasSolveViewPayload) onPayload(node._atlasSolveViewPayload);
  update();
  frame();
}

app.registerExtension({
  name: "AtlasVFX.SolveView",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;
    let THREE;
    try {
      THREE = await import("./lib/atlas-three.bundle.js");
    } catch (err) {
      console.warn("[atlas-vfx] Solve View: three.js did not load; the node still runs", err);
      return;
    }
    const originalCreated = nodeType.prototype.onNodeCreated;
    const originalExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      originalExecuted?.apply(this, arguments);
      const p = Array.isArray(message?.[UI_KEY]) ? message[UI_KEY][0] : message?.[UI_KEY];
      if (p && typeof p === "object") {
        this._atlasSolveViewPayload = p;
        this._atlasSolveView?.onPayload(p);
      }
    };
    nodeType.prototype.onNodeCreated = function () {
      originalCreated?.apply(this, arguments);
      try {
        buildPanel(this, THREE);
      } catch (err) {
        console.warn("[atlas-vfx] Solve View panel could not start; the node still runs", err);
      }
    };
  },
});

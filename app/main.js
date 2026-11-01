// Atlas Comp: the shot screen. Layers (left), viewer (centre), inspector (right), timeline (bottom).
// State lives on disk (the server); this page only shows it and asks for changes. DOM is built with
// createElement + textContent: nothing from the server is ever parsed as HTML.
import { api, frameURL, listen } from "/atlas/app/api.js";
import { FrameStore, LightMixer, preload } from "/atlas/app/lightmix.js";

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids) if (c) n.append(c);
  return n;
};

const S = {
  shots: [], shot: null, recipes: [], views: [], view: "", frame: 1001, exposure: 0, a: "", b: "", wipe: 0.5,
  selected: null, jobs: {}, frameBusy: false, frameAgain: false, nonce: 0, playing: false,
  presets: [], presetGroup: "Right sun",
  targets: [], target: "ws", qjobs: [], qTimer: null,
};

function status(msg, bad = false) {
  const s = $("status");
  s.textContent = msg || "";
  s.style.color = bad ? "var(--red)" : "";
}

// ---------------------------------------------------------------------------------------------------- loading
async function loadShots(pick) {
  S.shots = (await api.shots()).shots;
  const sel = $("shot-pick");
  sel.replaceChildren(...S.shots.map((n) => el("option", { value: n, text: n })));
  const name = pick || localGet("shot") || S.shots[0];
  if (name && S.shots.includes(name)) {
    sel.value = name;
    await openShot(name);
  } else {
    S.shot = null;
    render();
  }
}

async function openShot(name) {
  S.shot = await api.shot(name);
  localSet("shot", name);
  const f = Number(localGet(`frame:${name}`));
  S.frame = f >= S.shot.first && f <= S.shot.last ? f : S.shot.first;
  if (!S.shot.layers.some((l) => l.name === S.selected)) S.selected = null;
  render();
}

async function reloadShot() {
  if (S.shot) S.shot = await api.shot(S.shot.name);
  render();
}

function localGet(k) { try { return localStorage.getItem(`atlas-comp:${k}`); } catch { return null; } }
function localSet(k, v) { try { localStorage.setItem(`atlas-comp:${k}`, v); } catch { /* private window */ } }

// ---------------------------------------------------------------------------------------------------- render
function render() {
  const has = !!S.shot;
  const light = has && S.room === "light" && !!S.light;
  $("empty").hidden = has;
  $("frame-img").hidden = !has || light;
  $("light-canvas").hidden = !light;
  $("room-comp").setAttribute("aria-current", light ? "false" : "page");
  $("room-light").setAttribute("aria-current", light ? "page" : "false");
  const sel = has && S.shot.layers.find((x) => x.name === S.selected);
  $("room-light").disabled = !(sel && mixRecord(sel.mix));
  $("add-layer").disabled = !has;
  $("presets-btn").disabled = !has;
  if (!has) {
    renderEmpty();
    $("layer-list").replaceChildren();
    $("inspector-body").replaceChildren(document.createTextNode("Make a shot first."));
    $("bars").replaceChildren();
    $("proxy").textContent = "";
    return;
  }
  const p = S.shot.plate_proxy;
  const proxy = $("proxy");
  if (p === "error") {
    proxy.replaceChildren(el("span", { class: "error", text: `The plate would not decode: ${S.shot.plate_proxy_error}` }),
      document.createTextNode(" "),
      el("button", { type: "button", text: "Retry", onclick: async () => {
        try { S.shot = await api.retryProxy(S.shot.name); render(); } catch (e) { status(e.message, true); } } }));
  } else {
    proxy.textContent = p === "building" ? "Making the plate proxy (once per shot, about a minute)..." :
      p === "none" ? (S.shot.mode === "generate" ? "Create both: no plate, black under the layers" : "No plate") : "";
  }
  const scrub = $("scrub");
  scrub.min = S.shot.first; scrub.max = S.shot.last; scrub.value = S.frame;
  $("frame-out").textContent = S.frame;
  renderLayers();
  renderAB();
  renderInspector();
  renderBars();
  refreshFrame();
  if (p === "building") setTimeout(reloadShot, 4000);
}

function renderEmpty() {
  $("empty").replaceChildren(
    el("span", { class: "eyebrow accent", text: "00 — Start" }),
    el("h2", { text: "Make your first shot" }),
    el("p", { text: "A shot is a plate (any movie file) and a frame range in Nuke numbers. Layers stack on top of it: " +
      "EXR sequences you already have, or recipes (matte, relight, sky) that ComfyUI renders for you." }),
    el("button", { class: "primary", type: "button", text: "New shot", onclick: () => openShotDialog() }));
}

function layerRow(l) {
  const li = el("li", { class: l.name === S.selected ? "sel" : "", onclick: () => { S.selected = l.name; render(); } });
  const eye = el("button", { class: `eye ${l.visible ? "" : "off"}`, type: "button", title: "Show / hide", text: "◉",
    onclick: (e) => { e.stopPropagation(); change(l, { visible: !l.visible }); } });
  const name = el("span", { class: "lname", text: l.name });
  const blend = el("select", { title: "Blend", onclick: (e) => e.stopPropagation(),
    onchange: (e) => change(l, { blend: e.target.value }) },
  ...["over", "plus", "multiply", "screen"].map((b) => el("option", { value: b, text: b })));
  blend.value = l.blend;
  const op = el("input", { type: "range", min: 0, max: 1, step: 0.05, value: l.opacity, title: "Opacity",
    onclick: (e) => e.stopPropagation(), onchange: (e) => change(l, { opacity: Number(e.target.value) }) });
  const row2 = el("div", { class: "lrow" }, el("span", { class: "tag", text: l.kind === "recipe" ? l.recipe : "EXR" }));
  if (l.kind === "recipe") {
    const vs = el("select", { title: "Version", onclick: (e) => e.stopPropagation(),
      onchange: (e) => change(l, { version: e.target.value || null }) },
    el("option", { value: "", text: l.versions.length ? "latest" : "not rendered" }),
    ...(l.has_preview ? [el("option", { value: "preview", text: "preview" })] : []),
    ...l.versions.map((v) => el("option", { value: v, text: v })));
    vs.value = l.version || "";
    row2.append(vs);
  }
  li.append(el("div", { class: "lrow" }, eye, name, blend), el("div", { class: "lrow" }, op), row2);
  return li;
}

function renderLayers() {
  const rows = [...S.shot.layers].sort((x, y) => (y.order ?? 0) - (x.order ?? 0)).map(layerRow);   // top of stack first
  if (S.shot.plate) rows.push(el("li", { class: "plate-row" }, document.createTextNode("plate · from the movie, through its proxy")));
  $("layer-list").replaceChildren(...rows);
}

function renderAB() {
  const names = S.shot.layers.map((l) => l.name);
  const opts = (first) => [el("option", { value: "", text: first }),
    ...(S.shot.plate ? [el("option", { value: "plate", text: "plate" })] : []),
    ...names.map((n) => el("option", { value: n, text: n }))];
  $("a-pick").replaceChildren(...opts("everything"));
  $("b-pick").replaceChildren(...opts("off"));
  $("a-pick").value = names.includes(S.a) || S.a === "plate" ? S.a : "";
  $("b-pick").value = names.includes(S.b) || S.b === "plate" ? S.b : "";
  $("wipe-wrap").hidden = !$("b-pick").value;
}

function abList(v) {
  if (!v) return null;
  if (v === "plate") return "plate";
  return S.shot.plate ? `plate,${v}` : v;     // a layer is seen over the plate
}

// ---------------------------------------------------------------------------------------------------- Light room
// The selected layer's takes mixed on the GPU (app/lightmix.js) from the server's half-float cache: proxy frames
// while scrubbing or playing, the full-size set once the playhead rests. Sliders only redraw; the mix is saved
// to shot.json in the background.
const REST_MS = 250;

async function enterLight(layerName) {
  const l = S.shot?.layers.find((x) => x.name === layerName);
  const m = l && mixRecord(l.mix);
  if (!m) { status("Turn on the light mix for this layer first (Inspector).", true); return; }
  try {
    if (!S.light) S.light = { mixer: new LightMixer($("light-canvas")), store: new FrameStore(api) };
    S.light.mixer.setView(await api.viewShader(S.view));
    S.light.viewLoaded = S.view;
  } catch (e) { status(`Light room: ${e.message}`, true); S.light = null; return; }
  Object.assign(S.light, { layer: l.name, versions: Object.keys(m.takes).sort(), token: 0 });
  S.light.mixer.set({ versions: S.light.versions, mix: m, blend: l.blend || "over", opacity: l.opacity ?? 1,
    exposure: S.exposure });
  S.light.store.clear();
  S.light.mixer.release();
  api.warmCache(S.shot.name, l.name, "proxy").catch((e) => status(e.message, true));
  S.room = "light";
  render();
  refreshFrame();
  startPreload();
}

function leaveLight() {
  if (S.light) { S.light.preloadRun = (S.light.preloadRun || 0) + 1; S.light.mixer.release(); cacheBar(); }
  S.room = "comp";
  render();
  refreshFrame();
}

// Playback cache: every proxy frame set of the shot loaded onto the graphics card, from the playhead on, so play
// only switches textures. The bar under the viewer shows how much is loaded.
function startPreload() {
  const L = S.light, shot = S.shot, run = (L.preloadRun = (L.preloadRun || 0) + 1);
  L.held = 0;
  L.total = shot.last - shot.first + 1;
  cacheBar();
  preload(api, L.mixer, { shot: shot.name, layer: L.layer, versions: L.versions, first: shot.first, last: shot.last,
    from: S.frame,
    onProgress: (n) => { if (run === L.preloadRun) { L.held = n; cacheBar(); } },
    cancelled: () => run !== L.preloadRun || S.room !== "light" || S.shot?.name !== shot.name })
    .then((r) => {
      if (run === L.preloadRun && r.full) status(`Playback cache is full at ${r.held} of ${r.total} frames; the rest plays from the server.`);
    })
    .catch((e) => status(`Playback cache: ${e.message}`, true));
}

function cacheBar() {
  const L = S.light, on = S.room === "light" && !!L && !!L.total;
  $("light-cache-wrap").hidden = !on;
  if (!on) return;
  $("light-cache").value = L.held / L.total;
  $("light-cache-out").textContent = L.held >= L.total ? "all frames on the GPU" : `${L.held} of ${L.total} frames`;
}

function shotFps() {
  const f = S.shot?.fps;
  const v = Array.isArray(f) ? f[0] / (f[1] || 1) : Number(f);
  return v > 0 ? v : 24;
}

function lightDraw() { if (S.room === "light" && S.light) S.light.mixer.draw(); }

async function lightFrame() {
  const L = S.light, shot = S.shot.name, f = S.frame, token = ++L.token;
  if (L.viewLoaded !== S.view) {
    try { L.mixer.setView(await api.viewShader(S.view)); L.viewLoaded = S.view; }
    catch (e) { status(`view: ${e.message}`, true); }
  }
  try {
    if (!L.mixer.show(`proxy|${f}`)) {                     // not in the playback cache (yet): fetch it
      if (S.playing) {                                     // and read a few frames ahead
        for (let k = 1; k <= 4; k++) {
          const n = f + k;
          if (n <= S.shot.last && !L.mixer.isHeld(`proxy|${n}`)) L.store.get(shot, L.layer, L.versions, "proxy", n).catch(() => {});
        }
      }
      const set = await L.store.get(shot, L.layer, L.versions, "proxy", f);
      if (token !== L.token) return;
      L.mixer.upload(set);
    }
    L.mixer.set({ exposure: S.exposure });
    L.mixer.draw();
    $("readout").textContent = `${f} · light mix · proxy · ${S.exposure > 0 ? "+" : ""}${S.exposure.toFixed(2)} EV · ${S.view}`;
  } catch (e) { status(`Light room: ${e.message}`, true); }
  if (S.playing) {                                         // paced to the shot's frame rate
    const step = 1000 / shotFps(), t = performance.now();
    L.due = L.due && t - L.due < step ? L.due + step : t + step;
    setTimeout(() => { if (S.playing && S.room === "light") setFrame(S.frame >= S.shot.last ? S.shot.first : S.frame + 1); },
      Math.max(0, L.due - performance.now()));
    return;
  }
  clearTimeout(L.rest);
  L.rest = setTimeout(async () => {                       // the playhead rests: swap in the full-size frame set
    if (token !== L.token || S.playing) return;
    try {
      const full = await L.store.get(shot, L.layer, L.versions, "full", f);
      if (token !== L.token || S.playing) return;
      L.mixer.upload(full);
      L.mixer.draw();
      $("readout").textContent = $("readout").textContent.replace("proxy", "full");
    } catch (e) { status(`Light room (full size): ${e.message}`, true); }
  }, REST_MS);
}

function refreshFrame() {
  if (!S.shot) return;
  if (S.room === "light" && S.light) { lightFrame(); return; }
  if (S.frameBusy) { S.frameAgain = true; return; }
  S.frameBusy = true;
  const img = $("frame-img");
  const w = Math.min(2048, Math.max(256, Math.round($("stage").clientWidth * (window.devicePixelRatio || 1) / 64) * 64));
  // n: every request its own URL. An <img> given the URL it already shows does not reload (the layers may have
  // changed on disk) and never fires load, which would leave frameBusy stuck.
  const url = frameURL(S.shot.name, { f: S.frame, w, exposure: S.exposure, view: S.view, a: abList(S.a),
    b: abList(S.b), wipe: S.b ? S.wipe : "", n: ++S.nonce });
  $("readout").textContent = `${S.frame} · A ${S.a || "everything"}${S.b ? ` │ B ${S.b}` : ""} · ` +
    `${S.exposure > 0 ? "+" : ""}${S.exposure.toFixed(2)} EV · ${S.view}`;
  const done = () => {
    S.frameBusy = false;
    if (S.frameAgain) { S.frameAgain = false; refreshFrame(); return; }
    if (S.playing) requestAnimationFrame(() => setFrame(S.frame >= S.shot.last ? S.shot.first : S.frame + 1));
  };
  img.onload = done;
  img.onerror = () => { status("could not draw this frame", true); done(); };
  img.src = url;
}

function renderBars() {
  const first = S.shot.first, last = S.shot.last, span = Math.max(1, last - first + 1);
  const pct = (n) => `${((n - first) / span) * 100}%`;
  const bars = S.shot.layers.map((l) => {
    const track = el("div", { class: "track" });
    for (const [a, b] of l.frames || []) {
      const lo = Math.max(a, first), hi = Math.min(b, last);
      if (hi < lo) continue;
      const seg = el("div", { class: `seg ${l.version === "preview" ? "preview" : ""}` });
      seg.style.left = pct(lo);
      seg.style.width = `${((hi - lo + 1) / span) * 100}%`;
      track.append(seg);
    }
    const head = el("div", { class: "head" });
    head.style.left = pct(S.frame);
    track.append(head);
    return el("div", { class: "bar" }, el("span", { class: "lname", text: l.name }), track);
  });
  $("bars").replaceChildren(...bars);
}

// ---------------------------------------------------------------------------------------------------- inspector
function recipeOf(l) { return S.recipes.find((r) => r.name === l.recipe); }

// The 8 sun directions as seen in the shot, laid out like a compass: behind the subject at the top, the camera's
// side at the bottom.
const COMPASS = [["the back-left", "behind", "the back-right"], ["the left", null, "the right"],
  ["the front-left", "the front", "the front-right"]];

function compassField(f, value, onchange) {
  const cells = COMPASS.flat().map((dir) => {
    if (dir === null) {
      return el("span", { class: "centre", "aria-hidden": "true" }, sunIcon());
    }
    return el("button", { type: "button", "aria-pressed": String(dir === value), text: dir.replace(/^the /, ""),
      onclick: () => onchange(dir) });
  });
  return el("fieldset", { class: "field" }, el("legend", { text: f.label }), el("div", { class: "compass" }, ...cells));
}

function segmentedField(f, value, onchange) {
  const buttons = f.choices.map((c) => el("button", { type: "button", "aria-pressed": String(c === value),
    text: String(c), onclick: () => onchange(c) }));
  return el("fieldset", { class: "field" }, el("legend", { text: f.label }), el("div", { class: "segmented" }, ...buttons));
}

function sunIcon() {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  for (const [k, v] of [["width", "22"], ["height", "22"], ["viewBox", "0 0 24 24"], ["fill", "none"],
    ["stroke", "currentColor"], ["stroke-width", "1.2"]]) svg.setAttribute(k, v);
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", "M12 2v5M12 17v5M2 12h5M17 12h5");
  const c = document.createElementNS(ns, "circle");
  c.setAttribute("cx", "12"); c.setAttribute("cy", "12"); c.setAttribute("r", "6");
  svg.append(path, c);
  return svg;
}

function fieldInput(f, value, layerNames, onchange) {
  let input;
  if (f.type === "choice" && f.key === "direction" && f.choices.length === 8) {
    return compassField(f, value, onchange);
  }
  if (f.type === "choice" && f.choices.length <= 4) {
    return segmentedField(f, value, onchange);
  }
  if (f.type === "choice") {
    input = el("select", {}, ...f.choices.map((c) => el("option", { value: String(c), text: String(c) })));
    input.value = String(value);
    input.addEventListener("change", () => onchange(typeof f.choices[0] === "number" ? Number(input.value) : input.value));
  } else if (f.type === "layer") {
    input = el("select", {}, el("option", { value: "", text: "pick a layer" }),
      ...layerNames.map((n) => el("option", { value: n, text: n })));
    input.value = value || "";
    input.addEventListener("change", () => onchange(input.value));
  } else if (f.type === "text") {
    input = el("textarea", {});
    input.value = value ?? "";
    input.addEventListener("change", () => onchange(input.value));
  } else if (f.type === "bool") {
    input = el("input", { type: "checkbox" });
    input.checked = !!value;
    input.addEventListener("change", () => onchange(input.checked));
  } else {
    input = el("input", { type: "number", step: f.type === "int" ? 1 : "any", min: f.min, max: f.max });
    input.value = value;
    input.addEventListener("change", () => onchange(f.type === "int" ? parseInt(input.value, 10) : Number(input.value)));
  }
  return el("label", {}, el("span", { text: f.label }), input);
}

function renderInspector() {
  const body = $("inspector-body");
  const l = S.shot.layers.find((x) => x.name === S.selected);
  if (!l) {
    $("inspector-eyebrow").textContent = "03 — Inspector";
    $("jobbox").hidden = true;
    body.replaceChildren(document.createTextNode("Pick a layer."));
    return;
  }
  $("inspector-eyebrow").textContent = `03 — ${l.kind === "recipe" ? l.recipe : "EXR"} · ${l.name}`;
  const head = el("div", { style: "display: grid; gap: 6px;" },
    el("h2", { text: l.kind === "recipe" ? (recipeOf(l)?.label || l.recipe) : l.name }),
    el("div", { class: "muted", text: l.kind === "recipe" ? (recipeOf(l)?.description || "") : l.source }));
  if (l.kind !== "recipe") { body.replaceChildren(head); return; }
  const r = recipeOf(l);
  if (!r) { body.replaceChildren(head, el("p", { class: "error", text: `unknown recipe ${l.recipe}` })); return; }
  const others = S.shot.layers.filter((x) => x.name !== l.name).map((x) => x.name);
  const params = { ...l.params };
  const fields = r.fields.map((f) => fieldInput(f, params[f.key] ?? f.default, others, async (v) => {
    params[f.key] = v;
    try { await api.patchLayer(S.shot.name, l.name, { params }); l.params = { ...params }; status(""); }
    catch (e) { status(e.message, true); }
    renderInspector();
  }));
  const job = Object.values(S.jobs).find((j) => j.layer === l.name && !j.done);
  const preview = el("button", { type: "button", text: "Preview", title: "A few frames, quick decode", disabled: !!job,
    onclick: () => run(l, "preview", params) });
  const renderBtn = el("button", { class: "primary", type: "button", text: "Render", title: "The full job, a new version",
    disabled: !!job, onclick: () => run(l, "render", params) });
  const graphLink = el("a", { href: "/", target: "_blank", rel: "noopener", text: "Open ComfyUI" });
  const mixBox = lightMix(l);
  const pick = targetPick();
  body.replaceChildren(head, ...(mixBox ? [mixBox] : []), ...fields,
    el("div", { class: "runs" }, preview, renderBtn, ...(pick ? [pick] : [])),
    el("div", { class: "muted" }, document.createTextNode("Advanced: "), graphLink));
  renderJob(l.name);
}

// Light mix v2 (atlas_vfx/comp/mix.py): each render is a take; blend = weights as proportions (relit takes),
// sum = weights add (light passes, and every mix saved before v2). Stops and mute/solo per take.
// Warm/cool and green/magenta as an RGB tint scaled to luminance 1 in ACEScg (comp/mix.py tint_unit), so a tint
// changes the colour of a take, not its exposure.
const LUMA_AP1 = [0.2722287, 0.6740818, 0.0536895];
function tintRGB(warm, green) {
  const rgb = [1 + 0.35 * warm, 1 + 0.25 * green, 1 - 0.35 * warm];
  const y = rgb[0] * LUMA_AP1[0] + rgb[1] * LUMA_AP1[1] + rgb[2] * LUMA_AP1[2];
  return rgb.map((c) => Math.round((c / y) * 1e6) / 1e6);
}

function mixRecord(raw) {
  if (!raw || !Object.keys(raw).length) return null;
  if (raw.takes) return { mode: raw.mode || "blend", takes: structuredClone(raw.takes) };
  return { mode: "sum", takes: Object.fromEntries(Object.entries(raw).map(([v, w]) =>
    [v, { weight: w, stops: 0, tint: [1, 1, 1], mute: false, solo: false }])) };
}

function lightMix(l) {
  const versions = l.versions || [];
  if (versions.length < 2) return null;
  const m = mixRecord(l.mix);
  const on = !!m;
  let timer = null;
  const save = () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      try {
        await api.patchLayer(S.shot.name, l.name, { mix: structuredClone(m) });
        l.mix = structuredClone(m);
        if (!(S.room === "light" && S.light?.layer === l.name)) refreshFrame();
      } catch (e) { status(e.message, true); }
    }, S.room === "light" ? 400 : 120);
    if (S.room === "light" && S.light?.layer === l.name) { S.light.mixer.set({ mix: m }); lightDraw(); }
  };
  const wmax = m && m.mode === "sum" ? 200 : 100;
  const rows = on ? versions.map((v) => {
    const t = m.takes[v] || (m.takes[v] = { weight: 0, stops: 0, tint: [1, 1, 1], mute: false, solo: false });
    const pct = el("output", { class: "num", text: `${Math.round(t.weight * 100)}%` });
    const w = el("input", { type: "range", min: 0, max: wmax, step: 1, value: Math.round(t.weight * 100),
      "aria-label": `${v} weight` });
    w.addEventListener("input", () => { t.weight = Number(w.value) / 100; pct.textContent = `${w.value}%`; save(); });
    const ev = el("output", { class: "num", text: `${t.stops >= 0 ? "+" : ""}${t.stops.toFixed(1)}` });
    const st = el("input", { type: "range", min: -4, max: 4, step: 0.1, value: t.stops, "aria-label": `${v} stops` });
    st.addEventListener("input", () => {
      t.stops = Number(st.value); ev.textContent = `${t.stops >= 0 ? "+" : ""}${t.stops.toFixed(1)}`; save();
    });
    const flag = (key, label) => {
      const b = el("input", { type: "checkbox", "aria-label": `${v} ${key}` });
      b.checked = !!t[key];
      b.addEventListener("change", () => { t[key] = b.checked; save(); });
      return el("label", { class: "flag", title: key }, b, document.createTextNode(label));
    };
    const row = el("div", { class: "mixrow" }, el("span", { class: "num", text: v }), w, pct, st, ev,
      flag("mute", "M"), flag("solo", "S"));
    if (!(S.room === "light" && S.light?.layer === l.name)) return row;
    t.warm = t.warm ?? 0; t.green = t.green ?? 0;
    const tint = () => { t.tint = tintRGB(t.warm, t.green); save(); };
    const warm = el("input", { type: "range", min: -1, max: 1, step: 0.02, value: t.warm, "aria-label": `${v} warm/cool` });
    warm.addEventListener("input", () => { t.warm = Number(warm.value); tint(); });
    const green = el("input", { type: "range", min: -1, max: 1, step: 0.02, value: t.green, "aria-label": `${v} green/magenta` });
    green.addEventListener("input", () => { t.green = Number(green.value); tint(); });
    return el("div", { class: "takebox" }, row, el("div", { class: "tintrow" },
      el("span", { class: "muted", text: "cool" }), warm, el("span", { class: "muted", text: "warm" }),
      el("span", { class: "muted", text: "mag" }), green, el("span", { class: "muted", text: "grn" })));
  }) : [];
  const start = el("button", { type: "button", text: on ? "Mix off" : "Mix these renders", onclick: async () => {
    const next = on ? {} : { mode: "blend", takes: Object.fromEntries(versions.map((v, i) =>
      [v, { weight: i === versions.length - 1 ? 1 : 0, stops: 0, tint: [1, 1, 1], mute: false, solo: false }])) };
    try { await api.patchLayer(S.shot.name, l.name, { mix: next }); await reloadShot(); }
    catch (e) { status(e.message, true); }
  } });
  const bake = el("button", { type: "button", text: "Bake mix", title: "Write the mix as a new version (full size)",
    onclick: async () => {
      try {
        await api.bakeMix(S.shot.name, l.name);
        status("Baking the mix...");
        const poll = setInterval(async () => {
          const b = await api.bakeStatus(S.shot.name, l.name);
          if (b.state === "running") { status(`Baking the mix: ${b.done}/${b.of || "?"}`); return; }
          clearInterval(poll);
          if (b.state === "error") status(b.error, true);
          else { status(`Mix baked as ${b.version}`); await reloadShot(); }
        }, 1000);
      } catch (e) { status(e.message, true); }
    } });
  const box = el("fieldset", { class: "field" }, el("legend", { text: "Light mix" }));
  box.append(el("p", { class: "muted", style: "margin: 0 0 8px;",
    text: on ? (m.mode === "sum" ? "Each render is a light; they add." : "Each render is a relit take; weights blend them, stops push each one.")
      : "Blend this layer's renders like lights." }));
  if (on) box.append(...rows);
  box.append(start);
  if (on) box.append(bake);
  if (S.room === "light" && S.light?.layer === l.name) {
    const usage = el("span", { class: "muted" });
    api.cacheUsage().then((u) => { usage.textContent = ` cache ${(u.bytes / 2 ** 30).toFixed(1)} of ${(u.budget_bytes / 2 ** 30).toFixed(0)} GB`; })
      .catch(() => {});
    box.append(el("button", { type: "button", text: "Clear frame cache", title: "Remove this shot's Light room frames (remade on demand)",
      onclick: async () => {
        try { const r = await api.clearCache(S.shot.name); S.light.store.clear(); status(`Cache cleared: ${(r.freed_bytes / 2 ** 30).toFixed(1)} GB`); }
        catch (e) { status(e.message, true); }
      } }), usage);
  }
  return box;
}

function renderJob(layerName) {
  const job = Object.values(S.jobs).filter((j) => j.layer === layerName).pop();
  $("jobbox").hidden = !job;
  if (!job) return;
  const line = $("job-line");
  line.textContent = job.text;
  line.className = `job ${job.error ? "bad" : ""}`;
  $("job-thumb").hidden = !job.thumb;
  if (job.thumb) $("job-thumb").src = job.thumb;
}

// Render on: ws runs the graph here (live previews on ComfyUI's socket); any other target goes through Atlas Comp's
// queue, which checks the target first and sends only the cached latent.
function targetPick() {
  if (S.targets.length < 2) return null;
  const sel = el("select", { id: "target-pick", "aria-label": "Render on", title: "Which machine samples the job" },
    ...S.targets.map((t) => el("option", { value: t.name, text: t.note ? `${t.name} (${t.note})` : t.name })));
  sel.value = S.targets.some((t) => t.name === S.target) ? S.target : "ws";
  sel.addEventListener("change", () => { S.target = sel.value; localSet("target", S.target); });
  return el("label", { class: "target" }, document.createTextNode("on "), sel);
}

async function run(l, mode, params) {
  if (S.target !== "ws" && S.targets.some((t) => t.name === S.target)) {
    try {
      const j = await api.queueJob({ shot: S.shot.name, layer: l.name, mode, target: S.target, params });
      status(`${l.name}: ${mode} queued on ${j.target}. Open Queue to follow it.`);
      pollQueue();
    } catch (e) {
      status(`${l.name}: ${e.message}`, true);
    }
    return;
  }
  try {
    status(`${mode}: filling the graph...`);
    const g = await api.graph(S.shot.name, l.name, mode, params);
    const q = await api.queue(g.prompt);
    S.jobs[q.prompt_id] = { layer: l.name, mode, version: g.version, text: `${mode} queued`, done: false };
    status(`${l.name}: ${mode} queued (${g.version})`);
    renderInspector();
  } catch (e) {
    status(`${l.name}: ${e.message}`, true);
  }
}

// ---------------------------------------------------------------------------------------------------- comfy events
function onEvent(type, d) {
  const job = d.prompt_id && S.jobs[d.prompt_id];
  if (!job) return;
  if (type === "progress") {
    job.text = `${job.mode}: step ${d.value}/${d.max}`;
  } else if (type === "executing" && d.node) {
    job.text = `${job.mode}: running node ${d.node}`;
  } else if (type === "execution_error" || type === "execution_interrupted") {
    // a failed or cancelled job never switches the layer: its version folder may be empty or half written
    job.text = type === "execution_interrupted" ? `${job.mode} cancelled`
      : `${job.mode} failed: ${d.exception_message || d.exception_type || "error"}`;
    job.error = true; job.done = true;
    $("live-img").hidden = true;
    status(`${job.layer}: ${job.text}`, true);
    reloadShot();
  } else if (type === "execution_success") {
    if (job.done) return;
    job.done = true;
    job.text = `${job.mode} done (${job.version})`;
    $("live-img").hidden = true;
    finish(job);
  }
  if (S.shot && S.selected === job.layer) renderJob(job.layer);
}

async function finish(job) {
  try {
    await api.patchLayer(S.shot.name, job.layer, { version: job.version });
  } catch (e) {
    status(`${job.layer}: ${e.message}`, true);
  }
  status(`${job.layer}: ${job.text}`);
  await reloadShot();
}

function onPreview(url) {
  const job = Object.values(S.jobs).filter((j) => !j.done).pop();
  if (!job) { URL.revokeObjectURL(url); return; }
  if (job.thumb) URL.revokeObjectURL(job.thumb);
  job.thumb = url;
  if (S.selected === job.layer) renderJob(job.layer);
}

// ---------------------------------------------------------------------------------------------------- actions
async function change(l, d) {
  try {
    await api.patchLayer(S.shot.name, l.name, d);
    await reloadShot();
  } catch (e) { status(e.message, true); }
}

function setFrame(f) {
  if (!S.shot) return;
  S.frame = Math.min(S.shot.last, Math.max(S.shot.first, f));
  localSet(`frame:${S.shot.name}`, S.frame);
  $("scrub").value = S.frame;
  $("frame-out").textContent = S.frame;
  renderBars();
  refreshFrame();
}

// Play steps through frames as fast as the server composites them (proxy size, not real time). Frame-accurate
// playback is DJV's job (spec: Accurate playback).
function togglePlay() {
  if (!S.shot) return;
  S.playing = !S.playing;
  $("play").setAttribute("aria-pressed", String(S.playing));
  $("play").setAttribute("aria-label", S.playing ? "Pause" : "Play");
  if (S.playing) setFrame(S.frame >= S.shot.last ? S.shot.first : S.frame + 1);
  else if (S.room === "light" && S.light) { S.light.due = 0; lightFrame(); }   // paused: the full-size swap
}

// ---------------------------------------------------------------------------------------------------- presets
// A preset is a recipe with its settings filled in. A card adds a recipe layer on the chosen element; once a layer
// made from that preset has frames, the card shows one of them.
function presetLayer(p) {
  return S.shot.layers.find((l) => l.kind === "recipe" && l.recipe === p.recipe && (l.frames || []).length &&
    Object.entries(p.params).every(([k, v]) => l.params && l.params[k] === v));
}

function renderPresets() {
  const groups = [...new Set(S.presets.map((p) => p.group))];
  if (!groups.includes(S.presetGroup)) S.presetGroup = groups[0];
  $("preset-groups").replaceChildren(...groups.map((g) => el("button", { type: "button", text: g,
    "aria-pressed": String(g === S.presetGroup), onclick: () => { S.presetGroup = g; renderPresets(); } })));
  // EXR elements first and the default; a render of another recipe can still be picked (a matte to relight, say)
  const exrs = S.shot.layers.filter((l) => l.kind === "sequence").map((l) => l.name);
  const renders = S.shot.layers.filter((l) => l.kind === "recipe" && (l.frames || []).length).map((l) => l.name);
  const sources = [...exrs, ...renders];
  const pick = $("preset-element");
  const keep = pick.value;
  pick.replaceChildren(...sources.map((n) => el("option", { value: n, text: n })));
  pick.value = sources.includes(keep) ? keep : (exrs.includes(S.selected) ? S.selected : exrs[0] || sources[0] || "");
  const cards = S.presets.filter((p) => p.group === S.presetGroup).map((p) => {
    const done = presetLayer(p);
    const thumb = done
      ? el("img", { class: "preset-thumb", alt: `${p.name}, rendered`, src: frameURL(S.shot.name, {
        f: done.frames[0][0], w: 480, a: S.shot.plate ? `plate,${done.name}` : done.name, view: S.view, n: ++S.nonce }) })
      : el("div", { class: "preset-thumb empty", text: "not rendered on this shot yet" });
    return el("article", { class: "preset" }, thumb,
      el("div", { class: "preset-body" },
        el("span", { class: "eyebrow", text: p.recipe }),
        el("h3", { text: p.name }),
        el("p", { class: "muted", text: p.what }),
        el("div", { class: "runs" },
          el("button", { type: "button", text: done ? `Open ${done.name}` : "Add as layer",
            onclick: () => (done ? openPresetLayer(done) : addPreset(p)) }))));
  });
  $("preset-grid").replaceChildren(...cards);
}

function openPresetLayer(l) {
  S.selected = l.name;
  $("presets-dialog").close();
  render();
}

async function addPreset(p) {
  const element = $("preset-element").value;
  const fields = (S.recipes.find((r) => r.name === p.recipe) || { fields: [] }).fields;
  const params = { ...p.params };
  if (fields.some((f) => f.key === "element")) {
    if (!element) { $("preset-error").textContent = "Add an EXR layer first: presets run on a layer."; return; }
    params.element = element;
  }
  const names = new Set(S.shot.layers.map((l) => l.name));
  let name = p.id.replace(/-/g, "_");
  for (let i = 2; names.has(name); i++) name = `${p.id.replace(/-/g, "_")}_${i}`;
  try {
    await api.addLayer(S.shot.name, { layer: name, kind: "recipe", recipe: p.recipe, params });
    S.selected = name;
    $("presets-dialog").close();
    await reloadShot();
    status(`${name}: added from the preset. Preview or Render it in the inspector.`);
  } catch (e) { $("preset-error").textContent = e.message; }
}

// ---------------------------------------------------------------------------------------------------- queue
const ACTIVE = (j) => !["done", "failed", "cancelled"].includes(j.status);

async function pollQueue() {
  clearTimeout(S.qTimer);
  let jobs;
  try { jobs = (await api.jobs()).jobs; } catch (e) { status(`queue: ${e.message}`, true); S.qTimer = setTimeout(pollQueue, 5000); return; }
  const was = Object.fromEntries(S.qjobs.map((j) => [j.id, j.status]));
  const ended = jobs.filter((j) => was[j.id] && ACTIVE({ status: was[j.id] }) && !ACTIVE(j));
  S.qjobs = jobs;
  renderQueue();
  for (const j of ended) status(`${j.layer}: ${j.mode} on ${j.target} ${j.status}${j.message ? ` (${j.message})` : ""}`,
    j.status !== "done");
  if (ended.length) await reloadShot();
  if (jobs.some(ACTIVE)) S.qTimer = setTimeout(pollQueue, 2000);
}

function renderQueue() {
  const active = S.qjobs.filter(ACTIVE).length;
  $("queue-count").textContent = active ? String(active) : "";
  const rows = S.qjobs.slice().reverse().map((j) => el("li", { class: `qjob ${j.status}` },
    el("span", { class: "num", text: `${j.shot} · ${j.layer}` }),
    el("span", { text: `${j.mode} on ${j.target}` }),
    el("span", { class: "muted", text: j.stage }),
    el("span", { class: "num", text: `${j.frames_back}/${j.expected}` }),
    el("span", { class: j.status === "failed" ? "error" : "", text: j.message || j.status }),
    j.status === "confirm" ? confirmBox(j) : null,
    ACTIVE(j) ? el("button", { type: "button", text: "Cancel", onclick: async () => {
      try { await api.cancelJob(j.id); pollQueue(); } catch (e) { status(e.message, true); }
    } }) : el("span")));
  $("queue-list").replaceChildren(...(rows.length ? rows
    : [el("li", { class: "muted", text: "Nothing queued. Pick a farm machine beside Render to send a job there." })]));
}

// A Cloud job waits here: what it costs and what it sends, before anything goes.
function confirmBox(j) {
  const e = j.estimate || {};
  return el("div", { class: "confirm" },
    el("span", { text: `About ${e.credits_low}-${e.credits_high} credits (${e.note || "rough"}). ` +
      `${e.frames} reference frames will leave this machine for Comfy Cloud; the result comes back here.` }),
    el("button", { class: "primary", type: "button", text: "Confirm", onclick: async () => {
      try { await api.confirmJob(j.id); pollQueue(); } catch (err) { status(err.message, true); }
    } }));
}

function openQueue() {
  renderQueue();
  $("queue-dialog").showModal();
  pollQueue();
}

function openPresets() {
  if (!S.shot) return;
  $("preset-error").textContent = "";
  renderPresets();
  $("presets-dialog").showModal();
}

// The choice before anything else (layers/shot_mode.py): what the shot starts from decides the fields asked.
function shotModeFields() {
  const mode = new FormData($("shot-form")).get("mode") || "plate";
  $("screen-wrap").hidden = mode !== "foreground";
  $("frame-wrap").hidden = mode !== "generate";
  $("plate-wrap").hidden = mode === "generate";
  $("plate-label").textContent = mode === "foreground" ? "Screen plate (movie file)" : "Plate (movie file, optional)";
  $("shot-form").elements.plate.required = mode === "foreground";
}

function openShotDialog() {
  $("shot-error").textContent = "";
  shotModeFields();
  $("shot-dialog").showModal();
}

// Recipes offered for the open shot's mode (a recipe without modes is offered everywhere).
function shotRecipes() {
  const mode = S.shot?.mode || "plate";
  return S.recipes.filter((r) => !r.modes || r.modes.includes(mode));
}

function openLayerDialog() {
  const what = $("layer-what");
  what.replaceChildren(el("option", { value: "sequence", text: "An EXR sequence I have" }),
    ...shotRecipes().map((r) => el("option", { value: `recipe:${r.name}`, text: r.label })));
  $("source-wrap").hidden = false;
  $("layer-error").textContent = "";
  $("layer-dialog").showModal();
}

function wire() {
  $("shot-pick").addEventListener("change", (e) => openShot(e.target.value));
  $("new-shot").addEventListener("click", openShotDialog);
  $("shot-mode").addEventListener("change", shotModeFields);
  $("presets-btn").addEventListener("click", openPresets);
  $("queue-btn").addEventListener("click", openQueue);
  $("add-layer").addEventListener("click", openLayerDialog);
  $("layer-what").addEventListener("change", (e) => { $("source-wrap").hidden = e.target.value !== "sequence"; });
  $("view-pick").addEventListener("change", (e) => { S.view = e.target.value; localSet("view", S.view); refreshFrame(); });
  $("a-pick").addEventListener("change", (e) => { S.a = e.target.value; refreshFrame(); });
  $("b-pick").addEventListener("change", (e) => { S.b = e.target.value; $("wipe-wrap").hidden = !S.b; refreshFrame(); });
  $("wipe").addEventListener("input", (e) => { S.wipe = Number(e.target.value); refreshFrame(); });
  $("room-comp").addEventListener("click", leaveLight);
  $("room-light").addEventListener("click", () => enterLight(S.selected));
  $("exposure").addEventListener("input", (e) => {
    S.exposure = Number(e.target.value);
    if (S.room === "light" && S.light) { S.light.mixer.set({ exposure: S.exposure }); lightDraw(); }
    $("exposure-out").textContent = (S.exposure > 0 ? "+" : "") + S.exposure;
    refreshFrame();
  });
  $("scrub").addEventListener("input", (e) => setFrame(Number(e.target.value)));
  $("prev").addEventListener("click", () => setFrame(S.frame - 1));
  $("play").addEventListener("click", togglePlay);
  $("next").addEventListener("click", () => setFrame(S.frame + 1));
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, select")) return;
    if (e.key === " ") { e.preventDefault(); togglePlay(); }
    if (e.key === "ArrowLeft") setFrame(S.frame - 1);
    if (e.key === "ArrowRight") setFrame(S.frame + 1);
  });
  $("shot-form").addEventListener("submit", async (e) => {
    if (e.submitter?.value !== "ok") return;
    e.preventDefault();
    const f = new FormData(e.target);
    try {
      const mode = String(f.get("mode") || "plate");
      const d = { name: f.get("name"), mode, first: Number(f.get("first")), last: Number(f.get("last")) };
      if (mode !== "generate") d.plate = f.get("plate");
      if (mode === "foreground") d.screen = f.get("screen");
      if (mode === "generate") Object.assign(d, { width: Number(f.get("width")), height: Number(f.get("height")),
        fps: String(f.get("fps")) });
      const s = await api.createShot(d);
      $("shot-dialog").close();
      await loadShots(s.name);
    } catch (err) { $("shot-error").textContent = err.message; }
  });
  $("layer-form").addEventListener("submit", async (e) => {
    if (e.submitter?.value !== "ok") return;
    e.preventDefault();
    const f = new FormData(e.target);
    const what = String(f.get("what"));
    const d = what === "sequence" ? { layer: f.get("layer"), kind: "sequence", source: f.get("source") }
      : { layer: f.get("layer"), kind: "recipe", recipe: what.slice(7), params: {} };
    try {
      await api.addLayer(S.shot.name, d);
      $("layer-dialog").close();
      S.selected = d.layer;
      await reloadShot();
    } catch (err) { $("layer-error").textContent = err.message; }
  });
  new ResizeObserver(() => refreshFrame()).observe($("stage"));
}

async function start() {
  wire();
  listen({ onEvent, onPreview });
  try {
    const [r, v, ps] = await Promise.all([api.recipes(), api.views(), api.presets()]);
    S.recipes = r.recipes;
    S.presets = ps.presets;
    S.views = v.views;
    S.view = localGet("view") && S.views.includes(localGet("view")) ? localGet("view") : S.views[0];
    $("view-pick").replaceChildren(...S.views.map((x) => el("option", { value: x, text: x })));
    $("view-pick").value = S.view;
    await loadShots();
  } catch (e) {
    status(`Atlas Comp could not start: ${e.message}`, true);
  }
  try {
    S.targets = (await api.targets()).targets;
    S.target = localGet("target") || "ws";
    if (S.shot) renderInspector();
    pollQueue();
  } catch (e) {
    status(`targets: ${e.message}`, true);       // a bad targets file never stops the app; ws still renders
  }
}

start();

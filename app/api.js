// Atlas Comp: talking to the server. Our JSON API under /atlas/api, and ComfyUI's own /prompt and websocket for
// running graphs (so live previews and progress come back on ComfyUI's socket, addressed to this page's clientId).

export const clientId = (crypto.randomUUID && crypto.randomUUID()) || String(Math.random()).slice(2);

async function call(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  let data = null;
  try { data = await r.json(); } catch { /* not JSON */ }
  if (!r.ok) {
    const msg = (data && (data.error || (data.node_errors && JSON.stringify(data.node_errors)))) || `${r.status} ${r.statusText}`;
    throw new Error(msg);
  }
  return data;
}

const enc = encodeURIComponent;

export const api = {
  shots: () => call("GET", "/atlas/api/shots"),
  createShot: (d) => call("POST", "/atlas/api/shots", d),
  shot: (name) => call("GET", `/atlas/api/shots/${enc(name)}`),
  addLayer: (shot, d) => call("POST", `/atlas/api/shots/${enc(shot)}/layers`, d),
  patchLayer: (shot, layer, d) => call("PATCH", `/atlas/api/shots/${enc(shot)}/layers/${enc(layer)}`, d),
  recipes: () => call("GET", "/atlas/api/recipes"),
  graph: (shot, layer, mode, params) =>
    call("POST", `/atlas/api/shots/${enc(shot)}/layers/${enc(layer)}/graph`, { mode, params }),
  views: () => call("GET", "/atlas/api/views"),
  // ComfyUI's own queue
  // preview_method: ComfyUI previews nothing unless asked; taesd uses the tiny VAE (atlas_vfx ltx_live_preview)
  queue: (prompt) => call("POST", "/prompt", { prompt, client_id: clientId, extra_data: { preview_method: "taesd" } }),
};

export function frameURL(shot, q) {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(q)) if (v !== null && v !== undefined && v !== "") p.set(k, String(v));
  return `/atlas/api/shots/${enc(shot)}/frame?${p.toString()}`;
}

// ComfyUI's websocket: text messages are JSON {type, data}; binary messages are previews (8-byte header, then an
// image). Reconnects on its own.
export function listen({ onEvent, onPreview }) {
  let ws;
  const open = () => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws?clientId=${enc(clientId)}`);
    ws.binaryType = "arraybuffer";
    ws.onmessage = (m) => {
      if (typeof m.data === "string") {
        try { const d = JSON.parse(m.data); onEvent(d.type, d.data || {}); } catch { /* ignore */ }
      } else if (m.data instanceof ArrayBuffer && m.data.byteLength > 8) {
        const view = new DataView(m.data);
        const fmt = view.getUint32(4);                       // 1 = JPEG, 2 = PNG
        const blob = new Blob([m.data.slice(8)], { type: fmt === 2 ? "image/png" : "image/jpeg" });
        onPreview(URL.createObjectURL(blob));
      }
    };
    ws.onclose = () => setTimeout(open, 1500);
  };
  open();
}

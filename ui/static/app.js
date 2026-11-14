// wxrouter Control Room: polls /api/overview once a second, draws the dashboard, runs the chat.
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const BOXES = ["box-a", "box-b"];
const WINDOW_S = 60;
const POLICY_HINT = {
  static: "Everything goes to box-a. The other box is used only if box-a fails.",
  round_robin: "Takes turns. Keeps sending half the traffic even to a slow box.",
  least_loaded: "Sends each request to the box with the fewest in flight.",
  adaptive: "One active box. Switches only when the other is clearly faster, then cools down.",
};

const state = {
  reqs: [], lastT: null, routerNow: 0, clockOffset: 0,
  overview: null, seenSwitches: new Set(), health: {}, events: [],
  chat: [], busy: false, ok: null,
};

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function esc(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function fmtMs(v) { return v == null ? "–" : v >= 1000 ? `${(v / 1000).toFixed(1)}<small>s</small>` : `${Math.round(v)}<small>ms</small>`; }
function clock(ts) { return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }); }

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

// ------------------------------------------------------------------ box cards
function buildBoxes() {
  $("#boxes").innerHTML = BOXES.map(b => `
    <div class="card box" id="card-${b}" style="--c: var(--${b === "box-a" ? "a" : "b"})">
      <div class="box-top">
        <div class="box-name"><span class="dot" data-k="dot"></span>${b}<span class="tag" data-k="pref" hidden>preferred</span></div>
        <div style="display:flex;gap:6px"><span class="tag" data-k="mode">–</span><span class="tag active" data-k="active" hidden>ACTIVE</span></div>
      </div>
      <div class="box-grid">
        <div><div class="share" data-k="share">–</div><div class="share-l">of traffic, last 10 s · p95 <span data-k="p95">–</span></div></div>
        <div class="meters">
          <div><div class="meter-l"><span>Queue + in flight</span><span data-k="q">–</span></div><div class="bar" data-k="qbar"><i></i></div></div>
          <div><div class="meter-l"><span>GPU</span><span data-k="gpu">–</span></div><div class="bar" data-k="gbar"><i></i></div></div>
          <div class="meter-l"><span>Probe compute</span><span data-k="probe">–</span></div>
        </div>
      </div>
      <div class="box-actions">
        <button data-mode="normal">Normal</button><button data-mode="slow">Training (slow)</button><button data-mode="error">Fail</button>
      </div>
    </div>`).join("");
  for (const b of BOXES) {
    $(`#card-${b}`).querySelectorAll(".box-actions button").forEach(btn =>
      btn.addEventListener("click", () => setMode(b, btn.dataset.mode)));
  }
}

async function setMode(box, mode) {
  try {
    await api(`api/box/${box}/mode`, { mode, slow_factor: 3 });
    addEvent(Date.now() / 1000, "demo", `<b>${box}</b> set to ${mode === "slow" ? "training (3× slower)" : mode}`);
    poll();
  } catch (e) { addEvent(Date.now() / 1000, "fail", `Couldn't change ${box}: ${esc(e.message)}`); }
}

function renderBoxes(ov) {
  const st = ov.router, m = ov.metrics;
  const total = m.overall.requests || 0;
  for (const b of BOXES) {
    const card = $(`#card-${b}`), k = n => card.querySelector(`[data-k="${n}"]`);
    const be = st.backends[b] || {}, bx = ov.boxes[b] || {}, pb = m.per_box[b] || {};
    const mode = bx.mode || (be.healthy ? "normal" : "unreachable");
    card.classList.toggle("down", !be.healthy);
    k("dot").className = "dot " + (!be.healthy ? "bad" : mode === "slow" ? "warn" : "ok");
    const modeTag = k("mode");
    modeTag.textContent = !be.healthy ? "down" : mode === "slow" ? "training" : mode;
    modeTag.className = "tag " + (!be.healthy ? "downtag" : mode === "slow" ? "slow" : mode === "error" ? "error" : "");
    k("active").hidden = !(st.policy === "adaptive" && st.adaptive.active === b);
    k("pref").hidden = st.adaptive.preferred !== b;
    k("share").textContent = total ? `${Math.round(100 * (pb.requests || 0) / total)}%` : "–";
    k("p95").innerHTML = pb.p95_ms == null ? "–" : `${Math.round(pb.p95_ms)} ms`;
    const cap = bx.max_concurrency || 4, used = (bx.in_flight || 0) + (bx.queue_depth || 0);
    k("q").textContent = `${bx.in_flight ?? "–"} + ${bx.queue_depth ?? "–"} / ${cap}`;
    k("qbar").firstElementChild.style.width = `${Math.min(100, 100 * used / cap)}%`;
    k("qbar").classList.toggle("hot", used > cap);
    k("gpu").textContent = bx.gpu_util == null ? "–" : `${Math.round(100 * bx.gpu_util)}%`;
    k("gbar").firstElementChild.style.width = `${Math.round(100 * (bx.gpu_util || 0))}%`;
    k("probe").textContent = be.probe_ms == null ? "–" : `${Math.round(be.probe_ms)} ms`;
    card.querySelectorAll(".box-actions button").forEach(btn => btn.classList.toggle("on", btn.dataset.mode === mode));

    if (state.health[b] !== undefined && state.health[b] !== be.healthy) {
      addEvent(Date.now() / 1000, be.healthy ? "switch" : "fail",
        be.healthy ? `<b>${b}</b> is healthy again and rejoined` : `<b>${b}</b> marked down; traffic fails over`);
    }
    state.health[b] = be.healthy;
  }
}

// ------------------------------------------------------------------ KPIs + controls
function renderKpis(ov) {
  const o = ov.metrics.overall, st = ov.router;
  $("#k-rps").textContent = o.rps.toFixed(1);
  $("#k-p95").innerHTML = fmtMs(o.p95_ms);
  $("#k-p50").textContent = `p50 ${o.p50_ms == null ? "–" : Math.round(o.p50_ms) + " ms"}`;
  $("#k-p99").innerHTML = fmtMs(o.p99_ms);
  $("#k-err").textContent = o.errors;
  $("#k-err").style.color = o.errors ? "var(--err)" : "";
  $("#k-fo").textContent = `failovers ${o.failovers} · total ${st.failovers}`;
  $("#k-active").textContent = st.policy === "adaptive" ? st.adaptive.active : "spread";
  $("#k-pref").textContent = st.policy === "adaptive" ? `preferred ${st.adaptive.preferred || "none"}` : `rule ${st.policy}`;
  $("#pill-policy").textContent = `policy ${st.policy}`;
  const agent = ov.agent_mode === "offline" ? "agent offline (rules)" : `agent ${ov.agent_mode.replace("llm:", "")}`;
  $("#pill-agent").textContent = agent;
  $("#chat-mode").textContent = ov.agent_mode === "offline" ? "rule-based, no LLM configured" : ov.agent_mode.replace("llm:", "LLM: ");

  document.querySelectorAll("#seg-policy button").forEach(b => {
    b.classList.toggle("on", b.dataset.v === st.policy);
    b.setAttribute("aria-checked", b.dataset.v === st.policy);
  });
  $("#policy-hint").textContent = POLICY_HINT[st.policy] || "";
  const sel = $("#sel-pref");
  if (document.activeElement !== sel) sel.value = st.adaptive.preferred || "";
  sel.disabled = st.policy !== "adaptive";
  if (document.activeElement !== $("#rng-load")) { $("#rng-load").value = ov.load_rps; $("#load-val").textContent = ov.load_rps; }

  for (const s of st.adaptive.switches) {
    const key = `${s.t}-${s.to}`;
    if (state.seenSwitches.has(key)) continue;
    state.seenSwitches.add(key);
    addEvent(state.clockOffset + s.t, "switch", `Switched <b>${esc(s.from)} → ${esc(s.to)}</b>: ${esc(s.reason)}`);
  }
}

function addEvent(ts, kind, html) {
  state.events.unshift({ ts, kind, html });
  state.events = state.events.slice(0, 60);
  $("#events").innerHTML = state.events.map(e =>
    `<li class="k-${e.kind}"><time>${clock(e.ts)}</time><span>${e.html}</span></li>`).join("");
  $("#ev-count").textContent = `${state.events.length}`;
}

// ------------------------------------------------------------------ charts
function setupCanvas(c) {
  // Keep the CSS (logical) height in data-h: setting c.height below rewrites the attribute.
  if (!c.dataset.h) c.dataset.h = c.getAttribute("height");
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = +c.dataset.h;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.height = h + "px"; }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return [g, w, h];
}

function drawChart() {
  const [g, w, h] = setupCanvas($("#chart"));
  const padL = 44, padB = 6, padT = 6, plotW = w - padL - 6, plotH = h - padT - padB;
  const lo = Math.log10(10), hi = Math.log10(10000), t0 = state.routerNow - WINDOW_S;
  const X = t => padL + (t - t0) / WINDOW_S * plotW;
  const Y = ms => padT + plotH - (Math.log10(Math.max(10, Math.min(10000, ms))) - lo) / (hi - lo) * plotH;
  g.font = "11px -apple-system, sans-serif"; g.textAlign = "right"; g.textBaseline = "middle";
  for (const v of [10, 30, 100, 300, 1000, 3000, 10000]) {
    g.strokeStyle = css("--line"); g.lineWidth = 1; g.beginPath(); g.moveTo(padL, Y(v)); g.lineTo(w - 6, Y(v)); g.stroke();
    g.fillStyle = css("--muted"); g.fillText(v >= 1000 ? `${v / 1000}s` : `${v}ms`, padL - 6, Y(v));
  }
  const col = { "box-a": css("--a"), "box-b": css("--b") };
  for (const r of state.reqs) {
    const x = X(r.t), y = Y(r.ms);
    if (r.status >= 500) {
      g.strokeStyle = css("--err"); g.lineWidth = 2; g.beginPath();
      g.moveTo(x - 4, y - 4); g.lineTo(x + 4, y + 4); g.moveTo(x + 4, y - 4); g.lineTo(x - 4, y + 4); g.stroke();
      continue;
    }
    g.globalAlpha = 0.75; g.fillStyle = col[r.box] || css("--muted");
    g.beginPath(); g.arc(x, y, 2.6, 0, Math.PI * 2); g.fill(); g.globalAlpha = 1;
    if (r.failover) { g.strokeStyle = css("--fo"); g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, 5.5, 0, Math.PI * 2); g.stroke(); }
  }
  // share strip, one column per second
  const [s, sw, sh] = setupCanvas($("#share"));
  const bins = Array.from({ length: WINDOW_S }, () => ({ a: 0, b: 0, e: 0 }));
  for (const r of state.reqs) {
    const i = Math.floor(r.t - t0);
    if (i < 0 || i >= WINDOW_S) continue;
    if (r.status >= 500) bins[i].e++; else if (r.box === "box-a") bins[i].a++; else if (r.box === "box-b") bins[i].b++;
  }
  const bw = plotW / WINDOW_S;
  s.fillStyle = css("--panel-2"); s.fillRect(padL, 0, plotW, sh);
  bins.forEach((bn, i) => {
    const n = bn.a + bn.b + bn.e; if (!n) return;
    let y = 0; const x = padL + i * bw;
    for (const [v, c] of [[bn.a, css("--a")], [bn.b, css("--b")], [bn.e, css("--err")]]) {
      const hh = sh * v / n; s.fillStyle = c; s.fillRect(x + 0.5, y, Math.max(1, bw - 1), hh); y += hh;
    }
  });
  s.font = "11px -apple-system, sans-serif"; s.fillStyle = css("--muted"); s.textAlign = "right"; s.textBaseline = "middle";
  s.fillText("share", padL - 6, sh / 2);
}

// ------------------------------------------------------------------ polling
async function poll() {
  try {
    const q = state.lastT == null ? "" : `?since=${state.lastT}`;
    const ov = await api(`api/overview${q}`);
    state.overview = ov;
    state.routerNow = ov.metrics.now;
    state.clockOffset = ov.ts - ov.metrics.now;
    if (state.lastT == null) state.lastT = Math.max(0, ov.metrics.now - WINDOW_S);
    const fresh = ov.metrics.requests || [];
    if (fresh.length) { state.reqs.push(...fresh); state.lastT = fresh[fresh.length - 1].t; }
    state.reqs = state.reqs.filter(r => r.t >= state.routerNow - WINDOW_S);
    setRouterPill(true);
    renderKpis(ov); renderBoxes(ov); drawChart();
  } catch (e) {
    setRouterPill(false, e.message);
  }
}
function setRouterPill(ok, msg) {
  if (state.ok === ok) return;
  state.ok = ok;
  $("#pill-router").innerHTML = `<span class="dot ${ok ? "ok" : "bad"}"></span><span>${ok ? "Router connected" : "Router unreachable"}</span>`;
  if (!ok) $("#pill-router").title = msg || "";
}

// ------------------------------------------------------------------ chat
function md(s) {
  return esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\*(.+?)\*/g, "<i>$1</i>").replace(/\n/g, "<br>");
}

// ------------------------------------------------------------------ weather card
function loadUnits() { try { return localStorage.getItem("wx-units") === "F" ? "F" : "C"; } catch { return "C"; } }
function saveUnits(u) { try { localStorage.setItem("wx-units", u); } catch { /* storage blocked: keep in memory */ } }
state.units = loadUnits();
state.forecasts = new Map();  // card id -> forecast result, so a unit switch can re-render
let fcSeq = 0;

const T = c => c == null ? "–" : `${Math.round(state.units === "F" ? c * 9 / 5 + 32 : c)}°`;
const Wind = ms => ms == null ? "–" : state.units === "F" ? `${Math.round(ms * 2.237)} mph` : `${Math.round(ms * 3.6)} km/h`;
const Pct = v => v == null ? "–" : `${Math.round(v)}%`;
const Rain = mm => mm == null ? "–" : `${(+mm).toFixed(1)} mm`;

// Local time from the UTC valid time and longitude (solar time; good enough for day/night and labels).
function local(vt, lon) { return new Date(Date.parse(vt) + lon / 15 * 3600e3); }
const DAY = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const hhmm = d => `${String(d.getUTCHours()).padStart(2, "0")}:00`;

function condition(p, lon) {
  const h = local(p.vt, lon).getUTCHours(), night = h < 6 || h >= 18;
  if (p.p != null && p.p >= 0.3) {
    if (p.t <= 0.5) return { k: "snow", label: "Snow" };
    return { k: "rain", label: p.p < 1 ? "Light rain" : "Rain" };
  }
  const cl = p.cl ?? 0;
  if (cl >= 75) return { k: "cloud", label: "Cloudy" };
  if (cl >= 35) return { k: night ? "pmoon" : "psun", label: "Partly cloudy" };
  return { k: night ? "moon" : "sun", label: night ? "Clear" : "Sunny" };
}

function icon(k, size = 28) {
  const sun = `<circle cx="12" cy="12" r="4.5" fill="#f5b301"/><g stroke="#f5b301" stroke-width="1.8" stroke-linecap="round">${[0, 45, 90, 135, 180, 225, 270, 315].map(a => { const r = a * Math.PI / 180; return `<line x1="${12 + 7 * Math.cos(r)}" y1="${12 + 7 * Math.sin(r)}" x2="${12 + 9.5 * Math.cos(r)}" y2="${12 + 9.5 * Math.sin(r)}"/>`; }).join("")}</g>`;
  const moon = `<path d="M15.5 3.5a8.5 8.5 0 1 0 5 15 7 7 0 0 1-5-15z" fill="#c9cfdb"/>`;
  const cloud = (dx = 0, dy = 0, fill = "#aeb6c4") => `<path transform="translate(${dx} ${dy})" d="M7 18h10a4 4 0 0 0 .5-7.97A5.5 5.5 0 0 0 7 9.5 4.25 4.25 0 0 0 7 18z" fill="${fill}"/>`;
  const drops = `<g stroke="#4c9bff" stroke-width="1.8" stroke-linecap="round"><line x1="9" y1="20" x2="8" y2="23"/><line x1="13" y1="20" x2="12" y2="23"/><line x1="17" y1="20" x2="16" y2="23"/></g>`;
  const flakes = `<g fill="#dfe8ff"><circle cx="8.5" cy="21.5" r="1.2"/><circle cx="12.5" cy="22.5" r="1.2"/><circle cx="16.5" cy="21.5" r="1.2"/></g>`;
  const body = {
    sun, moon, cloud: cloud(),
    psun: `<g transform="translate(-3 -3) scale(.8)">${sun}</g>${cloud(2, 2)}`,
    pmoon: `<g transform="translate(-2 -3) scale(.75)">${moon}</g>${cloud(2, 2)}`,
    rain: `${cloud(0, -2, "#8f9aac")}${drops}`, snow: `${cloud(0, -2, "#aeb6c4")}${flakes}`,
  }[k] || cloud();
  return `<svg class="wx-ico" width="${size}" height="${size}" viewBox="0 0 24 24" aria-hidden="true">${body}</svg>`;
}

function unitToggle() {
  return `<div class="units" role="group" aria-label="Units">${["C", "F"].map(u =>
    `<button data-units="${u}" class="${state.units === u ? "on" : ""}" aria-pressed="${state.units === u}">°${u}</button>`).join("")}</div>`;
}

function stat(label, value) { return `<div class="wx-stat"><span>${label}</span><b>${value}</b></div>`; }

function weatherCard(r, id) {
  const pts = r.points || []; if (!pts.length) return "";
  const now = pts[0], c = condition(now, r.lon), s = r.summary;
  const steps = pts.filter(p => !p.i).slice(0, 12);
  const hasRain = s.precip_mm != null;
  return `<div class="wx" data-fc="${id}">
    <div class="wx-head">
      <div><div class="wx-place">${esc(r.place || `${r.lat.toFixed(2)}, ${r.lon.toFixed(2)}`)}</div>
        <div class="wx-sub">next ${r.hours_ahead} h · served by ${esc(r.served_by || "?")}</div></div>
      <div class="wx-tools">${unitToggle()}<button class="wx-max" data-open="${id}" aria-label="Open full forecast" title="Open full forecast"><svg viewBox="0 0 24 24" width="16" height="16"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg></button></div>
    </div>
    <div class="wx-now">
      ${icon(c.k, 52)}
      <div><div class="wx-temp">${T(now.t)}</div><div class="wx-cond">${c.label}</div><div class="wx-cond">H ${T(s.max_c)} · L ${T(s.min_c)}</div></div>
      <div class="wx-stats">${stat("Wind", Wind(now.w))}${stat("Humidity", Pct(now.hum))}${hasRain ? stat("Rain", Rain(s.precip_mm)) : ""}</div>
    </div>
    <div class="wx-strip">${steps.map(p => {
      const d = local(p.vt, r.lon), k = condition(p, r.lon).k;
      return `<div class="wx-step"><span>${d.getUTCHours() === 0 ? DAY[d.getUTCDay()] : hhmm(d)}</span>${icon(k, 26)}<b>${T(p.t)}</b></div>`;
    }).join("")}</div>
    ${s.ens_min_c != null ? `<div class="wx-note">Range across ${r.ensemble_members} ensemble members: ${T(s.ens_min_c)} to ${T(s.ens_max_c)}</div>` : ""}
    ${r.low_confidence ? `<div class="wx-warn">Beyond 72 h the model is less reliable.</div>` : ""}
  </div>`;
}

function dailyRows(r) {
  const days = new Map();
  r.points.forEach((p, i) => {
    const d = local(p.vt, r.lon), key = d.toISOString().slice(0, 10);
    if (!days.has(key)) days.set(key, { d, pts: [], idx: [] });
    days.get(key).pts.push(p); days.get(key).idx.push(i);
  });
  return [...days.values()].map(({ d, pts, idx }) => {
    const ts = pts.map(p => p.t), rain = pts.map(p => p.p).filter(v => v != null), hum = pts.map(p => p.hum).filter(v => v != null);
    const noon = pts.filter(p => !p.i).reduce((b, p) => Math.abs(local(p.vt, r.lon).getUTCHours() - 12) < Math.abs(local(b.vt, r.lon).getUTCHours() - 12) ? p : b, pts[0]);
    const worst = pts.some(p => condition(p, r.lon).k === "rain" || condition(p, r.lon).k === "snow") ? condition(pts.find(p => ["rain", "snow"].includes(condition(p, r.lon).k)), r.lon) : condition({ ...noon, vt: noon.vt }, r.lon);
    const ens = r.members_temp ? idx.flatMap(i => r.members_temp.map(m => m[i])) : null;
    return { d, lo: Math.min(...ts), hi: Math.max(...ts), wind: Math.max(...pts.map(p => p.w)),
      rain: rain.length ? rain.reduce((a, b) => a + b, 0) : null, hum: hum.length ? hum.reduce((a, b) => a + b, 0) / hum.length : null,
      c: worst, ens: ens ? [Math.min(...ens), Math.max(...ens)] : null, hours: pts.length };
  });
}

function fullForecast(r) {
  const days = dailyRows(r), s = r.summary, hasRain = s.precip_mm != null;
  const gLo = Math.min(...days.map(d => d.ens ? d.ens[0] : d.lo)), gHi = Math.max(...days.map(d => d.ens ? d.ens[1] : d.hi));
  const pos = v => `${(100 * (v - gLo) / Math.max(1, gHi - gLo)).toFixed(1)}%`;
  const now = r.points[0], c = condition(now, r.lon);
  const steps = r.points.filter(p => !p.i);
  return `<div class="wxm-head">
      <div><h3>${esc(r.place || "Forecast")}</h3>
        <div class="wx-sub">${r.lat.toFixed(2)}, ${r.lon.toFixed(2)} · next ${r.hours_ahead} h · ${esc(r.model_version || "")} · served by <b>${esc(r.served_by || "?")}</b> in ${Math.round(r.latency_ms)} ms${r.failover ? " (failover)" : ""}</div></div>
      <div class="wx-tools">${unitToggle()}<button class="wxm-close" aria-label="Close">✕</button></div>
    </div>
    ${r.low_confidence ? `<div class="wx-warn">Beyond 72 h the model is less reliable. Treat later days as a rough guide.</div>` : ""}
    <div class="wxm-now">
      ${icon(c.k, 72)}
      <div><div class="wx-temp big">${T(now.t)}</div><div class="wx-cond">${c.label} · next hour</div></div>
      <div class="wx-stats">${stat("Wind", Wind(now.w))}${stat("Humidity", Pct(now.hum))}${stat("Cloud", Pct(now.cl))}${hasRain ? stat("Rain total", Rain(s.precip_mm)) : stat("Rain", "not provided")}</div>
    </div>
    <h4>By day <span class="muted">(local solar time)</span></h4>
    <div class="wxm-days">${days.map(d => `
      <div class="wxm-day">
        <span class="dname">${DAY[d.d.getUTCDay()]}<small>${d.hours < 24 ? ` · ${d.hours} h` : ""}</small></span>
        ${icon(d.c.k, 30)}<span class="dcond">${d.c.label}</span>
        <span class="dlo">${T(d.lo)}</span>
        <span class="drange">${d.ens ? `<i class="ens" style="left:${pos(d.ens[0])};right:calc(100% - ${pos(d.ens[1])})"></i>` : ""}<i style="left:${pos(d.lo)};right:calc(100% - ${pos(d.hi)})"></i></span>
        <span class="dhi">${T(d.hi)}</span>
        <span class="dmeta">${Wind(d.wind)} · ${Pct(d.hum)}${hasRain ? ` · ${Rain(d.rain)}` : ""}</span>
      </div>`).join("")}</div>
    ${r.members_temp ? `<p class="muted">Lighter bar = range across ${r.ensemble_members} ensemble members.</p>` : ""}
    <h4>Every ${r.model_step_hours || 6} hours <span class="muted">(model steps)</span></h4>
    <div class="wxm-table-wrap"><table class="wxm-table">
      <thead><tr><th>Time</th><th>Conditions</th><th>Temp</th><th>Wind</th><th>Humidity</th><th>Cloud</th>${hasRain ? "<th>Rain</th>" : ""}</tr></thead>
      <tbody>${steps.map(p => { const d = local(p.vt, r.lon), k = condition(p, r.lon); return `<tr>
        <td>${DAY[d.getUTCDay()]} ${hhmm(d)}</td><td class="cond">${icon(k.k, 22)} ${k.label}</td><td><b>${T(p.t)}</b></td>
        <td>${Wind(p.w)}</td><td>${Pct(p.hum)}</td><td>${Pct(p.cl)}</td>${hasRain ? `<td>${Rain(p.p)}</td>` : ""}</tr>`; }).join("")}</tbody>
    </table></div>
    <p class="muted">The model computes a value every ${r.model_step_hours || 6} hours. Hourly values in between are straight-line interpolations, so this table shows only the model steps.${hasRain ? "" : " This model run provides no rain data."}</p>`;
}

function openForecast(id) {
  const r = state.forecasts.get(id); if (!r) return;
  let m = $("#wx-modal");
  if (!m) {
    m = document.createElement("div"); m.id = "wx-modal"; m.className = "wxm"; m.setAttribute("role", "dialog"); m.setAttribute("aria-modal", "true");
    document.body.appendChild(m);
    m.addEventListener("click", e => { if (e.target === m || e.target.closest(".wxm-close")) closeForecast(); });
  }
  m.dataset.fc = id;
  m.innerHTML = `<div class="wxm-panel">${fullForecast(r)}</div>`;
  m.hidden = false; document.body.classList.add("modal-open");
  m.querySelector(".wxm-close").focus();
}
function closeForecast() { const m = $("#wx-modal"); if (m) { m.hidden = true; document.body.classList.remove("modal-open"); } }

function setUnits(u) {
  if (u === state.units) return;
  state.units = u; saveUnits(u);
  document.querySelectorAll(".wx[data-fc]").forEach(el => { el.outerHTML = weatherCard(state.forecasts.get(el.dataset.fc), el.dataset.fc); });
  const m = $("#wx-modal"); if (m && !m.hidden) m.querySelector(".wxm-panel").innerHTML = fullForecast(state.forecasts.get(m.dataset.fc));
}

document.addEventListener("click", e => {
  const u = e.target.closest("[data-units]"); if (u) return setUnits(u.dataset.units);
  const o = e.target.closest("[data-open]"); if (o) return openForecast(o.dataset.open);
});
document.addEventListener("keydown", e => { if (e.key === "Escape") closeForecast(); });

function renderSteps(steps) {
  if (!steps?.length) return "";
  return `<div class="steps">${steps.map(s => {
    const r = s.result || {};
    if (s.tool === "get_forecast" && r.served_by) {
      return `<span class="step ${esc(r.served_by)}">forecast · <b>${esc(r.served_by)}</b> · ${Math.round(r.latency_ms)} ms${r.failover ? " · failover" : ""}</span>`;
    }
    return `<span class="step">${esc(s.tool)}${r.error ? " · error" : ""} · ${Math.round(s.ms)} ms</span>`;
  }).join("")}</div>`;
}

function addMsg(role, html) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = html;
  $("#msgs").appendChild(el);
  $("#msgs").scrollTop = $("#msgs").scrollHeight;
  return el;
}

async function send(text) {
  text = text.trim();
  if (!text || state.busy) return;
  state.busy = true; $("#send").disabled = true;
  state.chat.push({ role: "user", content: text });
  addMsg("user", `<p>${esc(text)}</p>`);
  const pending = addMsg("bot", `<span class="typing"><i></i><i></i><i></i></span>`);
  try {
    const res = await api("api/chat", { messages: state.chat.slice(-12), units: state.units });
    state.chat.push({ role: "assistant", content: res.reply });
    const fc = (res.steps || []).find(s => s.tool === "get_forecast" && s.result?.points);
    let card = "";
    if (fc) { const id = `fc${++fcSeq}`; state.forecasts.set(id, fc.result); card = weatherCard(fc.result, id); }
    pending.innerHTML = `<p>${md(res.reply)}</p>${card}${renderSteps(res.steps)}${res.note ? `<div class="note">${esc(res.note)}</div>` : ""}`;
    for (const s of res.steps || []) {
      if (s.tool === "simulate_box" && !s.result?.error) addEvent(Date.now() / 1000, "demo", `Assistant set <b>${esc(s.args.box)}</b> to ${esc(s.args.mode)}`);
      if (s.tool === "set_routing_policy" && !s.result?.error) addEvent(Date.now() / 1000, "demo", `Assistant changed rule to <b>${esc(s.args.policy)}</b>`);
    }
    poll();
  } catch (e) {
    pending.innerHTML = `<p>Something went wrong: ${esc(e.message)}</p>`;
  } finally {
    state.busy = false; $("#send").disabled = false;
    $("#msgs").scrollTop = $("#msgs").scrollHeight;
  }
}

// ------------------------------------------------------------------ wiring
function wire() {
  document.querySelectorAll("#seg-policy button").forEach(b => b.addEventListener("click", async () => {
    const preferred = b.dataset.v === "adaptive" ? ($("#sel-pref").value || "box-b") : undefined;
    await api("api/router/config", { policy: b.dataset.v, ...(preferred !== undefined ? { preferred } : {}) });
    state.seenSwitches.clear();
    addEvent(Date.now() / 1000, "demo", `Rule changed to <b>${b.dataset.v}</b>`);
    poll();
  }));
  $("#sel-pref").addEventListener("change", async e => {
    await api("api/router/config", { policy: "adaptive", preferred: e.target.value });
    addEvent(Date.now() / 1000, "demo", `Preferred box set to <b>${esc(e.target.value || "none")}</b>`);
    poll();
  });
  let t;
  const setLoad = v => { $("#load-val").textContent = v; clearTimeout(t); t = setTimeout(() => api("api/load", { rps: +v }), 200); };
  $("#rng-load").addEventListener("input", e => setLoad(e.target.value));
  document.querySelectorAll(".quick button").forEach(b => b.addEventListener("click", () => {
    $("#rng-load").value = b.dataset.rps; setLoad(b.dataset.rps);
  }));
  $("#composer").addEventListener("submit", e => { e.preventDefault(); const v = $("#input").value; $("#input").value = ""; send(v); });
  $("#input").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } });
  $("#input").addEventListener("input", e => { e.target.style.height = "auto"; e.target.style.height = Math.min(120, e.target.scrollHeight) + "px"; });
  $("#chips").querySelectorAll("button").forEach(b => b.addEventListener("click", () => send(b.textContent)));
  window.addEventListener("resize", drawChart);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawChart);
}

buildBoxes();
wire();
poll();
setInterval(poll, 1000);

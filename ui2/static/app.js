// wxrouter Control Room (ui2): polls /api/overview once a second, draws the dashboard, runs the chat.
// A revised copy of ui/static/app.js: it also says what is simulated and what is real, runs the real
// fine-tune job and the real kill on boxes whose supervisor is known, and draws no rain when rain is unknown.
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const BOXES = ["box-a", "box-b"];
const WINDOW_S = 60;
const POLICY_HINT = {
  static: "Everything goes to box-a. The other box is used only if box-a fails.",
  round_robin: "Takes turns. Keeps sending half the traffic even to a slow box.",
  least_loaded: "Sends each request to the box with the fewest in flight.",
  adaptive: "One active box. Switches only when the other is clearly faster, then cools down.",
  balanced: "Both boxes in play. Each request goes by health score: GPU time, active requests, health-check ms, errors, recent work.",
};
const SIM_NOTE = "Simulated boxes: forecast numbers are made up by the mock.";

const state = {
  reqs: [], lastT: null, routerNow: 0, clockOffset: 0,
  overview: null, seenSwitches: new Set(), health: {}, events: [],
  chat: [], busy: false, ok: null,
  sim: {},      // box -> the last answer of its /health that told mock from real (kept while the box is down)
  job: {},      // box -> the last state seen of its real fine-tune job
  pending: {},  // "box:train" / "box:kill" -> true while that request is on its way
};

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function esc(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function fmtMs(v) { return v == null ? "–" : v >= 1000 ? `${(v / 1000).toFixed(1)}<small>s</small>` : `${Math.round(v)}<small>ms</small>`; }
function clock(ts) { return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }); }
function isNum(v) { return typeof v === "number" && Number.isFinite(v); }
function nowS() { return Date.now() / 1000; }
function shortVersion(v) { v = String(v ?? ""); return v.length > 18 ? v.slice(0, 17) + "…" : v; }

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
        <div class="box-tags"><span class="tag" data-k="sim" hidden></span><span class="tag" data-k="mode">–</span><span class="tag active" data-k="active" hidden>ACTIVE</span></div>
      </div>
      <div class="box-grid">
        <div><div class="share" data-k="share">–</div><div class="share-l">of traffic, last 10 s · p95 <span data-k="p95">–</span></div></div>
        <div class="meters">
          <div><div class="meter-l"><span>Queue + in flight</span><span data-k="q">–</span></div><div class="bar" data-k="qbar"><i></i></div></div>
          <div><div class="meter-l"><span>GPU</span><span data-k="gpu">–</span></div><div class="bar" data-k="gbar"><i></i></div></div>
          <div class="meter-l"><span>Probe compute</span><span data-k="probe">–</span></div>
          <div class="meter-l" data-k="scorebox" title=""><span>Health score <small>(lower = better)</small></span><b data-k="score">–</b></div>
          <div class="score-parts" data-k="parts"></div>
        </div>
      </div>
      <div class="job-line" data-k="job" hidden></div>
      <div class="box-actions" data-k="sim-actions">
        <button data-mode="normal">Normal</button><button data-mode="slow" title="Simulated: the box's test hook makes it 3× slower. No job runs.">Training (simulated)</button><button data-mode="error" title="Simulated: the box's test hook answers 503. Nothing is killed.">Fail (simulated)</button>
      </div>
      <div class="box-actions" data-k="real-actions" hidden>
        <button data-act="train" title="Real job: the fine-tune run on this box, through its supervisor">Start fine-tune</button><button data-act="kill" title="Real: kills the forecast server process. Its supervisor starts it again by itself.">Kill forecast server</button>
      </div>
    </div>`).join("");
  for (const b of BOXES) {
    const card = $(`#card-${b}`);
    card.querySelectorAll("button[data-mode]").forEach(btn =>
      btn.addEventListener("click", () => setMode(b, btn.dataset.mode)));
    card.querySelector('button[data-act="train"]').addEventListener("click", () => trainClick(b));
    card.querySelector('button[data-act="kill"]').addEventListener("click", () => killClick(b));
  }
}

// The supervisor's answer for a box: undefined = the box has no supervisor (simulated buttons),
// null = it has one but it did not answer, else {finetune, forecast}.
function stackOf(box) {
  const stacks = (state.overview && state.overview.stacks) || {};
  return Object.prototype.hasOwnProperty.call(stacks, box) ? stacks[box] : undefined;
}

// Simulated: the box's /admin/mode test hook.
async function setMode(box, mode) {
  try {
    const res = await api(`api/box/${box}/mode`, { mode, slow_factor: 3 });
    if (res && res.error) addEvent(nowS(), "fail", `Simulated: couldn't change <b>${box}</b>: ${esc(res.error)}`);
    else addEvent(nowS(), "demo", `Simulated: <b>${box}</b> set to ${mode === "slow" ? "training (3× slower)" : mode}`);
    poll();
  } catch (e) { addEvent(nowS(), "fail", `Couldn't change ${box}: ${esc(e.message)}`); }
}

// Real job: start the fine-tune run of a box, or stop it when it runs.
async function trainClick(box) {
  const key = `${box}:train`;
  if (state.pending[key]) return;
  const sk = stackOf(box), ft = sk && sk.finetune;
  const action = ft && ft.state === "running" ? "stop" : "start";
  state.pending[key] = true;
  try {
    const res = await api(`api/box/${box}/train`, { action });
    if (res && res.error) {
      addEvent(nowS(), "fail", `Real job: fine-tune on <b>${box}</b> did not ${action}: ${esc(res.error)}`);
    } else {
      const st = res && res.status && res.status.state;
      if (st) state.job[box] = st;
      addEvent(nowS(), "demo", action === "start"
        ? `Real job: fine-tune started on <b>${box}</b>`
        : `Real job: fine-tune on <b>${box}</b> stopped`);
    }
  } catch (e) {
    addEvent(nowS(), "fail", `Real job: couldn't ${action} the fine-tune on ${box}: ${esc(e.message)}`);
  } finally {
    state.pending[key] = false;
    poll();
  }
}

// Real: SIGKILL to the forecast server of a box. Its supervisor starts it again by itself.
async function killClick(box) {
  const key = `${box}:kill`;
  if (state.pending[key]) return;
  state.pending[key] = true;
  try {
    const res = await api(`api/box/${box}/kill`, {});
    if (res && res.error) addEvent(nowS(), "fail", `Real job: forecast server on <b>${box}</b> was not killed: ${esc(res.error)}`);
    else addEvent(nowS(), "fail", `Real job: forecast server on <b>${box}</b> killed (a real crash, not simulated). Its supervisor restarts it by itself.`);
  } catch (e) {
    addEvent(nowS(), "fail", `Real job: couldn't kill the forecast server on ${box}: ${esc(e.message)}`);
  } finally {
    state.pending[key] = false;
    poll();
  }
}

// The line under the meters of a box with a supervisor.
function jobText(sk, ft) {
  if (!sk) return "supervisor not reachable";
  if (!ft) return "fine-tune: this supervisor has no job named finetune";
  const prog = isNum(ft.step) && isNum(ft.total) ? `step ${ft.step}/${ft.total}` : null;
  const secs = isNum(ft.seconds) ? `${Math.round(ft.seconds)} s` : null;
  if (ft.state === "running") {
    const what = prog || (ft.last_line ? String(ft.last_line).slice(0, 60) : "starting");
    return "fine-tune: " + [what, prog ? ft.speed : null, secs].filter(Boolean).join(" · ");
  }
  if (ft.state === "done") return "fine-tune: done" + (prog ? ` (${prog})` : "") + (secs ? ` in ${secs}` : "");
  if (ft.state === "failed") {
    // The supervisor calls every run that did not exit with 0 "failed". A negative exit code is a signal: a stop.
    const stopped = isNum(ft.exit) && ft.exit < 0;
    return "fine-tune: " + (stopped ? "stopped" : `failed${isNum(ft.exit) ? ` (exit ${ft.exit})` : ""}`)
      + (prog ? ` at ${prog}` : "") + (ft.error ? ` · ${ft.error}` : "");
  }
  return "fine-tune: not started (real job, idle)";
}

function renderBoxes(ov) {
  const st = ov.router, m = ov.metrics;
  const total = m.overall.requests || 0;
  const health = ov.health || {};
  for (const b of BOXES) {
    const card = $(`#card-${b}`), k = n => card.querySelector(`[data-k="${n}"]`);
    const be = st.backends[b] || {}, bx = (ov.boxes || {})[b] || {}, pb = m.per_box[b] || {};
    const mode = bx.mode || (be.healthy ? "normal" : "unreachable");
    const training = mode === "slow" || mode === "training";   // the mock says "slow", the contract "training"
    card.classList.toggle("down", !be.healthy);
    k("dot").className = "dot " + (!be.healthy ? "bad" : training ? "warn" : "ok");
    const modeTag = k("mode");
    modeTag.textContent = !be.healthy ? "down" : training ? "training" : mode;
    modeTag.className = "tag " + (!be.healthy ? "downtag" : training ? "slow" : mode === "error" ? "error" : "");
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
    const hs = be.health_score;
    if (hs) {
      const scores = BOXES.map(n => st.backends[n]?.health_score?.score).filter(v => v != null);
      k("score").textContent = Math.round(hs.score);
      k("score").style.color = be.healthy && hs.score === Math.min(...scores) ? "var(--ok)" : "";
      k("parts").textContent = `GPU ${hs.unit_ms} ms/unit · active ${hs.active} · health ${hs.health_ms} ms · errors ${Math.round(100 * hs.err_rate)}% · recent work ${Math.round(100 * hs.work_share)}%`;
    }
    card.querySelectorAll("button[data-mode]").forEach(btn => btn.classList.toggle("on", btn.dataset.mode === mode));

    // Simulated or real: from the box's own /health. Keep the last answer while the box is down.
    const hl = health[b];
    if (hl && hl.simulated != null) state.sim[b] = hl;
    const known = state.sim[b], simTag = k("sim");
    simTag.hidden = !known;
    if (known) {
      simTag.className = "tag " + (known.simulated ? "simtag" : "realtag");
      simTag.textContent = known.simulated ? "SIMULATED" : `REAL MODEL ${shortVersion(known.model_version)}`;
      simTag.title = `model_version: ${known.model_version ?? "?"}`;
    }

    // With a supervisor: the real fine-tune job and the real kill. Without one: the simulated buttons.
    const sk = stackOf(b), hasStack = sk !== undefined;
    const ft = sk && sk.finetune ? sk.finetune : null;
    const running = !!ft && ft.state === "running";
    k("sim-actions").hidden = hasStack;
    k("real-actions").hidden = !hasStack;
    const jobEl = k("job");
    jobEl.hidden = !hasStack;
    if (hasStack) {
      jobEl.textContent = jobText(sk, ft);
      jobEl.title = ft && ft.last_line ? String(ft.last_line) : "";
      const failed = !!ft && ft.state === "failed" && !(isNum(ft.exit) && ft.exit < 0);
      jobEl.className = "job-line" + (running ? " run" : failed || !sk ? " bad" : "");
      const trainBtn = card.querySelector('button[data-act="train"]'), killBtn = card.querySelector('button[data-act="kill"]');
      trainBtn.textContent = running ? "Stop fine-tune" : "Start fine-tune";
      trainBtn.classList.toggle("on", running);
      trainBtn.disabled = !ft || !!state.pending[`${b}:train`];
      killBtn.disabled = !sk || !!state.pending[`${b}:kill`];

      // A change of the job's state that this page did not ask for (it ended, or someone else started it).
      // While this page's own start or stop is on its way, that click writes the line.
      const prev = state.job[b], cur = ft ? ft.state : null;
      if (cur && prev !== undefined && prev !== cur && !state.pending[`${b}:train`]) {
        const stopped = isNum(ft.exit) && ft.exit < 0;
        if (cur === "running") addEvent(nowS(), "demo", `Real job: fine-tune is running on <b>${b}</b>`);
        else if (cur === "done") addEvent(nowS(), "demo", `Real job: fine-tune on <b>${b}</b> is done`);
        else if (cur === "failed") addEvent(nowS(), stopped ? "demo" : "fail", `Real job: fine-tune on <b>${b}</b> ${stopped ? "stopped" : "failed"}`);
      }
      if (cur) state.job[b] = cur;
    }

    if (state.health[b] !== undefined && state.health[b] !== be.healthy) {
      addEvent(nowS(), be.healthy ? "switch" : "fail",
        be.healthy ? `<b>${b}</b> is healthy again and rejoined` : `<b>${b}</b> marked down; traffic fails over`);
    }
    state.health[b] = be.healthy;
  }
  const pill = $("#pill-sim");
  if (pill) pill.hidden = !BOXES.some(b => state.sim[b] && state.sim[b].simulated);
}

// ------------------------------------------------------------------ model card
function renderModel(m) {
  const note = $("#model-note"), badge = $("#model-badge"), grid = $("#model-grid");
  if (!m || !note || !badge || !grid) return;
  const set = (id, v) => { const el = $(id); if (el) el.textContent = v; };
  badge.hidden = !(m.available || m.simulated);
  badge.className = "tag " + (m.simulated ? "simtag" : "realtag");
  badge.textContent = m.simulated ? "SIMULATED" : "REAL MODEL";
  grid.hidden = !m.available;
  if (!m.available) {
    note.textContent = m.simulated ? SIM_NOTE
      : m.reason === "no_v1_info" ? "These boxes have no /v1/info, so the model is not known here."
      : "No box has answered /v1/info yet.";
    return;
  }
  set("#m-name", m.model ?? "–");
  set("#m-params", isNum(m.params_m) && m.params_m > 0 ? `${m.params_m.toFixed(1)} M` : "–");
  set("#m-device", m.device ?? "–");
  set("#m-latest", typeof m.latest === "string" && m.latest ? `${m.latest.replace("T", " ")}:00 UTC` : "–");
  set("#m-speed", isNum(m.ms_per_unit) ? `${m.ms_per_unit} ms per member-step${m.simulated ? " (emulated)" : ""}` : "not measured yet");
  note.textContent = (m.simulated ? SIM_NOTE + " " : "") + `From ${m.box ?? "a box"} /v1/info.`;
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
  $("#k-active").textContent = st.policy === "adaptive" ? st.adaptive.active : st.policy === "balanced" ? "both" : "spread";
  $("#k-pref").textContent = st.policy === "adaptive" ? `preferred ${st.adaptive.preferred || "none"}`
    : st.policy === "balanced" ? "by health score" : `rule ${st.policy}`;
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
  // "Both" = the balanced rule; a box = adaptive with that home box. Other rules: show "Both".
  if (document.activeElement !== sel) sel.value = st.policy === "adaptive" ? (st.adaptive.preferred || "") : "";
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
    try { renderModel(ov.model); } catch (e) { console.error("model card:", e); }   // never blames the router
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

function forecastChart(r) {
  // Points without a temperature are left out. Rain may be missing on every point (our model has no
  // rain output): then there are no bars and no "rain bars" label.
  const rows = (r.points || []).map((p, i) => ({ p, i })).filter(x => x.p && isNum(x.p.t) && isNum(x.p.h));
  const pts = rows.map(x => x.p); if (!pts.length) return "";
  const W = 340, H = 120, L = 30, R = 8, T = 10, B = 22;
  const members = Array.isArray(r.members_temp) && r.members_temp.length ? r.members_temp : null;
  // per drawn point: the members' temperatures that are numbers
  const mvals = members ? rows.map(x => members.map(m => (Array.isArray(m) ? m[x.i] : null)).filter(isNum)) : null;
  const hasBand = !!mvals && mvals.every(v => v.length > 0);
  let lo = Math.min(...pts.map(p => p.t)), hi = Math.max(...pts.map(p => p.t));
  if (hasBand) { lo = Math.min(lo, ...mvals.flat()); hi = Math.max(hi, ...mvals.flat()); }
  lo = Math.floor(lo - 1); hi = Math.ceil(hi + 1);
  const hasRain = pts.some(p => isNum(p.p));
  const maxH = pts[pts.length - 1].h, maxP = Math.max(1, ...pts.map(p => (isNum(p.p) ? p.p : 0)));
  const X = h => L + (h - pts[0].h) / Math.max(1, maxH - pts[0].h) * (W - L - R);
  const Y = t => T + (hi - t) / (hi - lo) * (H - T - B);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${X(p.h).toFixed(1)},${Y(p.t).toFixed(1)}`).join("");
  let band = "";
  if (hasBand) {
    const mins = mvals.map(v => Math.min(...v)), maxs = mvals.map(v => Math.max(...v));
    band = `<path d="${pts.map((p, i) => `${i ? "L" : "M"}${X(p.h).toFixed(1)},${Y(maxs[i]).toFixed(1)}`).join("")}${pts.map((p, i) => `L${X(pts[pts.length - 1 - i].h).toFixed(1)},${Y(mins[pts.length - 1 - i]).toFixed(1)}`).join("")}Z" fill="var(--accent)" opacity=".15"/>`;
  }
  const bw = Math.max(1, (W - L - R) / pts.length - 0.5);
  const bars = !hasRain ? "" : pts.map(p => isNum(p.p) && p.p > 0 ? `<rect x="${(X(p.h) - bw / 2).toFixed(1)}" y="${(H - B - p.p / maxP * 26).toFixed(1)}" width="${bw.toFixed(1)}" height="${(p.p / maxP * 26).toFixed(1)}" fill="var(--b)" opacity=".35"/>` : "").join("");
  const v72 = maxH > 72 ? `<line x1="${X(72)}" x2="${X(72)}" y1="${T}" y2="${H - B}" stroke="var(--warn)" stroke-dasharray="3 3"/><text x="${X(72) + 3}" y="${T + 8}" font-size="11" fill="var(--warn)">72 h</text>` : "";
  const ticks = [0, Math.round(maxH / 2), maxH].map(h => `<text x="${X(Math.max(pts[0].h, h))}" y="${H - 6}" font-size="11" text-anchor="middle" fill="var(--muted)">+${h}h</text>`).join("");
  return `<div class="fc">
    <div class="fc-l"><span>Temperature °C${hasBand ? " (band = ensemble range)" : ""}</span>${hasRain ? "<span>rain bars</span>" : ""}</div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Temperature forecast">
      <text x="${L - 4}" y="${Y(hi) + 3}" font-size="11" text-anchor="end" fill="var(--muted)">${hi}°</text>
      <text x="${L - 4}" y="${Y(lo) + 3}" font-size="11" text-anchor="end" fill="var(--muted)">${lo}°</text>
      <line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="var(--line)"/>
      ${bars}${band}${v72}<path d="${line}" fill="none" stroke="var(--accent)" stroke-width="1.8"/>${ticks}
    </svg></div>`;
}

// Is this forecast from a mock? The answer's own model_version decides. null = not known.
function forecastSimulated(r) {
  if (typeof r.simulated === "boolean") return r.simulated;
  return typeof r.model_version === "string" && r.model_version ? /mock/i.test(r.model_version) : null;
}

const REAL_TOOLS = ["start_finetune", "stop_finetune", "kill_forecast_server"];

function renderSteps(steps) {
  if (!steps?.length) return "";
  return `<div class="steps">${steps.map(s => {
    const r = s.result || {};
    if (s.tool === "get_forecast" && r.served_by) {
      const sim = forecastSimulated(r);
      const chip = sim == null ? "" : `<span class="step ${sim ? "sim" : "real"}" title="model_version: ${esc(r.model_version)}">${sim ? "simulated data" : "real model"}</span>`;
      return `<span class="step ${esc(r.served_by)}">forecast · <b>${esc(r.served_by)}</b> · ${Math.round(r.latency_ms)} ms${r.failover ? " · failover" : ""}</span>${chip}`;
    }
    const kind = s.tool === "simulate_box" ? " · simulated" : REAL_TOOLS.includes(s.tool) ? " · real job" : "";
    return `<span class="step">${esc(s.tool)}${kind}${r.error ? " · error" : ""} · ${Math.round(s.ms)} ms</span>`;
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
    const res = await api("api/chat", { messages: state.chat.slice(-12) });
    state.chat.push({ role: "assistant", content: res.reply });
    const fc = (res.steps || []).find(s => s.tool === "get_forecast" && s.result?.points);
    pending.innerHTML = `<p>${md(res.reply)}</p>${fc ? forecastChart(fc.result) : ""}${renderSteps(res.steps)}${res.note ? `<div class="note">${esc(res.note)}</div>` : ""}`;
    for (const s of res.steps || []) {
      if (s.result?.error) continue;
      const box = s.args?.box, jobState = s.result?.status?.state;
      if (s.tool === "simulate_box") addEvent(nowS(), "demo", `Simulated: assistant set <b>${esc(box)}</b> to ${esc(s.args?.mode)}`);
      if (s.tool === "start_finetune") { if (jobState) state.job[box] = jobState; addEvent(nowS(), "demo", `Real job: assistant started the fine-tune on <b>${esc(box)}</b>`); }
      if (s.tool === "stop_finetune") { if (jobState) state.job[box] = jobState; addEvent(nowS(), "demo", `Real job: assistant stopped the fine-tune on <b>${esc(box)}</b>`); }
      if (s.tool === "kill_forecast_server") addEvent(nowS(), "fail", `Real job: assistant killed the forecast server on <b>${esc(box)}</b> (a real crash, not simulated). Its supervisor restarts it by itself.`);
      if (s.tool === "set_routing_policy") addEvent(nowS(), "demo", `Assistant changed rule to <b>${esc(s.args?.policy)}</b>`);
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
    const preferred = b.dataset.v === "adaptive" ? ($("#sel-pref").value || "box-b") : b.dataset.v === "balanced" ? "" : undefined;
    await api("api/router/config", { policy: b.dataset.v, ...(preferred !== undefined ? { preferred } : {}) });
    state.seenSwitches.clear();
    addEvent(nowS(), "demo", `Rule changed to <b>${b.dataset.v}</b>`);
    poll();
  }));
  $("#sel-pref").addEventListener("change", async e => {
    const v = e.target.value;  // "" = Both → balanced rule; a box → adaptive with that home box
    await api("api/router/config", v ? { policy: "adaptive", preferred: v } : { policy: "balanced", preferred: "" });
    addEvent(nowS(), "demo", v ? `Preferred box set to <b>${esc(v)}</b> (adaptive)` : `Both boxes: <b>balanced</b> by health score`);
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

// wxrouter Control Room: polls /api/overview once a second, draws the dashboard, runs the chat.
// A revised copy of ui/static/app.js: it also says what is simulated and what is real, runs the real
// fine-tune job and the real kill on boxes whose supervisor is known, and draws no rain when rain is unknown.
"use strict";

const $ = (s, el = document) => el.querySelector(s);
// The router's boxes. Read from GET api/router/config at start; these two are the fallback.
let BOXES = ["box-a", "box-b"];
// The router's settings and what may be set (GET api/router/config); refreshed from every overview.
let ROUTING = null;
const WINDOW_S = 60;
const POLICY_HINT = {
  static: "Everything goes to the home box (or the first box). The other one is used only if it fails.",
  round_robin: "Takes turns. Keeps sending half the traffic even to a slow box.",
  least_loaded: "Sends each request to the box with the fewest in flight.",
  adaptive: "One active box. Switches only when the other is clearly faster, then cools down.",
  balanced: "Both boxes in play. Each request goes by health score: GPU time, active requests, health-check ms, errors, recent work.",
};
const SIM_NOTE = "Simulated boxes: forecast numbers are made up by the mock.";

const state = {
  reqs: [], lastT: null, routerNow: 0, clockOffset: 0,
  overview: null, seenSwitches: new Set(), health: {}, events: [],
  evFilter: "all",        // Events card: all, routing, telemetry, fuzzball or incident
  evSeq: 0,               // last /api/events seq this page has
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
// A run that was stopped on request, not one that broke. The supervisor calls every exit that is not 0
// "failed". Below zero: the stop signal ended the process. 3: weather/finetune.py caught the signal, ended
// its step and left ("3 = stopped early by a signal"). The server sends the same judgement as ft.stopped.
function jobStopped(ft) {
  if (!ft || ft.state !== "failed") return false;
  if (typeof ft.stopped === "boolean") return ft.stopped;
  return isNum(ft.exit) && (ft.exit < 0 || ft.exit === 3);
}
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
    <div class="card box" id="card-${b}" style="--c: var(--${boxColor(b)})">
      <div class="box-top">
        <div class="box-name"><span class="dot" data-k="dot"></span>${b}<span class="tag" data-k="pref" hidden>home</span></div>
        <div class="box-tags"><span class="tag" data-k="sim" hidden></span><span class="tag" data-k="mode">–</span><span class="tag active" data-k="active" hidden>ACTIVE</span><span class="tag drainedtag" data-k="drained" hidden>DRAINED</span><button class="drain-btn" data-act="drain" title="Drain: no new requests go to this box (it stays the last resort if the other box fails). Click again to put it back.">Drain</button></div>
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
      <div class="box-actions five">
        <button data-mode="normal" title="Turn off the simulated slow or fail mode">Normal</button><button data-mode="slow" title="Simulated: the box's test hook makes it 3× slower. No job runs.">Slow (sim)</button><button data-mode="error" title="Simulated: the box's test hook answers 503. Nothing is killed.">Fail (sim)</button><button data-act="train" class="real" title="Real job: the fine-tune run on this box, through its supervisor">Fine-tune</button><button data-act="kill" class="real" title="Real: kills the forecast server process. Its supervisor starts it again by itself.">Kill server</button>
      </div>
    </div>`).join("");
  for (const b of BOXES) {
    const card = $(`#card-${b}`);
    card.querySelectorAll("button[data-mode]").forEach(btn =>
      btn.addEventListener("click", () => setMode(b, btn.dataset.mode)));
    card.querySelector('button[data-act="train"]').addEventListener("click", () => trainClick(b));
    card.querySelector('button[data-act="kill"]').addEventListener("click", () => killClick(b));
    card.querySelector('button[data-act="drain"]').addEventListener("click", () => drainClick(b));
  }
}

// The first box is drawn in --a, the second in --b.
function boxColor(b) { return BOXES.indexOf(b) === 1 ? "b" : "a"; }

// Router setting: drain a box, or put it back.
async function drainClick(box) {
  const on = !(ROUTING && ROUTING.drained.includes(box));
  try {
    await setRouting({ drain: { [box]: on } });
    addEvent(nowS(), "demo", on ? `<b>${box}</b> drained: no new requests` : `<b>${box}</b> back in service`);
  } catch (e) { addEvent(nowS(), "fail", `Couldn't change ${box}: ${esc(e.message)}`); }
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
    const hook = state.sim[box] && !state.sim[box].simulated && /untrained/i.test(state.sim[box].model_version ?? "");
    if (res && res.error) addEvent(nowS(), "fail", `${hook ? "Test hook" : "Simulated"}: couldn't change <b>${box}</b>: ${esc(res.error)}`);
    else if (hook) addEvent(nowS(), "demo", `Test hook on a real server: <b>${box}</b> set to ${mode === "slow" ? "slow (a real second load on its GPU)" : mode === "error" ? "error (answers 503, process not killed)" : mode}`);
    else addEvent(nowS(), "demo", `Simulated: <b>${box}</b> set to ${mode === "slow" ? "training (3× slower)" : mode}`);
    pollNow();
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
    pollNow();
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
    pollNow();
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
    const stopped = jobStopped(ft);   // a stop on request also arrives as "failed"
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
    // The server's "training" only means: other work on its GPU. With a supervisor the page knows whether the
    // fine-tune job really runs; when it does not, the tag says "GPU busy" (the LLM answering, probes), not "training".
    const sup = stackOf(b);
    const busyOnly = training && !!sup && !(sup.finetune && sup.finetune.state === "running");
    modeTag.textContent = !be.healthy ? "down" : busyOnly ? "GPU busy" : training ? "training" : mode;
    modeTag.title = busyOnly ? "The forecast server sees other work on its GPU (the LLM answering, probes). No fine-tune job is running." : "";
    modeTag.className = "tag " + (!be.healthy ? "downtag" : training ? "slow" : mode === "error" ? "error" : "");
    k("active").hidden = !(st.policy === "adaptive" && st.adaptive.active === b);
    k("pref").hidden = st.adaptive.preferred !== b || !["adaptive", "static"].includes(st.policy);
    const drained = !!be.drained;
    k("drained").hidden = !drained;
    const db = card.querySelector('button[data-act="drain"]');
    db.textContent = drained ? "Undrain" : "Drain";
    db.classList.toggle("on", drained);
    card.classList.toggle("drained", drained);
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
      // A real server with random weights (the inference stand-in) computes for real but is not the trained model.
      const standIn = !known.simulated && /untrained/i.test(known.model_version ?? "");
      simTag.textContent = known.simulated ? "SIMULATED" : standIn ? "REAL SERVER · STAND-IN MODEL" : `REAL MODEL ${shortVersion(known.model_version)}`;
      simTag.title = `model_version: ${known.model_version ?? "?"}` + (standIn ? " (random weights: the compute is real, the forecast values are not weather)" : "");
      // On the inference stand-in server the same test hook is not a simulation: slow mode runs a real second load on its GPU.
      const slowBtn = card.querySelector('button[data-mode="slow"]'), failBtn = card.querySelector('button[data-mode="error"]');
      slowBtn.textContent = standIn ? "Slow (real)" : "Slow (sim)";
      failBtn.textContent = standIn ? "Fail (503)" : "Fail (sim)";
      if (standIn) {
        slowBtn.title = "Test hook on a real server: starts a real second load on this box's GPU. No training job runs.";
        failBtn.title = "Test hook on a real server: it answers every forecast with 503. The process is not killed.";
      }
    }
    // A box running the real model gets no simulated controls: what the page shows there is live.
    // (The stand-in server keeps them: there they start real load.) Mock boxes are for laptop development.
    const simOnly = !!known && !known.simulated && !/untrained/i.test(known.model_version ?? "");
    card.querySelectorAll("button[data-mode]").forEach(btn => { btn.hidden = simOnly; });

    // With a supervisor: the real fine-tune job and the real kill. Without one: the simulated buttons.
    const sk = stackOf(b), hasStack = sk !== undefined;
    const ft = sk && sk.finetune ? sk.finetune : null;
    const running = !!ft && ft.state === "running";
    // All five buttons are always shown. The two real ones need the box's supervisor (--stack).
    const trainBtn = card.querySelector('button[data-act="train"]'), killBtn = card.querySelector('button[data-act="kill"]');
    if (!hasStack) {
      trainBtn.disabled = killBtn.disabled = true;
      trainBtn.title = killBtn.title = "Needs this box's supervisor (serve/node_stack.py, given with --stack)";
    }
    const jobEl = k("job");
    jobEl.hidden = !hasStack;
    if (hasStack) {
      jobEl.textContent = jobText(sk, ft);
      jobEl.title = ft && ft.last_line ? String(ft.last_line) : "";
      const failed = !!ft && ft.state === "failed" && !jobStopped(ft);
      jobEl.className = "job-line" + (running ? " run" : failed || !sk ? " bad" : "");
      trainBtn.title = "Real job: the fine-tune run on this box, through its supervisor";
      killBtn.title = "Real: kills the forecast server process. Its supervisor starts it again by itself.";
      trainBtn.textContent = running ? "Stop tune" : "Fine-tune";
      trainBtn.classList.toggle("on", running);
      trainBtn.disabled = !ft || !!state.pending[`${b}:train`];
      killBtn.disabled = !sk || !!state.pending[`${b}:kill`];

      // A change of the job's state is logged by the server's telemetry (the WhatIf recorder reads the same
      // supervisor /status), so it is not logged here again. This page logs only its own clicks.
      const cur = ft ? ft.state : null;
      if (cur) state.job[b] = cur;
    }

    // A box marked down or healthy again is logged by the server's telemetry ("router marked box-a down"),
    // not here, so the Events card shows it once.
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
      : m.reason === "no_v1_info" ? "These boxes have no /v1/info, so there is no model card. That is expected for the inference stand-in servers; the weather team's forecast server fills this in."
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

// ------------------------------------------------------------------ routing config
const SETTING_LABEL = {
  margin: "Switch margin", window_s: "Must hold for", cooldown_s: "Cooldown", return_margin: "Return margin",
  stall_ms: "Stall after",
};

function fmtSetting(key, v, unit) {
  if (unit === "fraction") return `${Math.round(v * 100)}%`;
  return unit === "ms" ? `${Math.round(v)} ms` : `${+v.toFixed(1)} s`;
}

// Changes this page makes are logged where they are made. Until this time (ms), a change seen in the
// router's config is taken as ours and not logged again as "changed elsewhere".
let expectRoutingUntil = 0;
function expectRouting() { expectRoutingUntil = Date.now() + 3000; }

async function setRouting(update) {
  expectRouting();
  const res = await api("api/router/config", update);
  if (res.config) renderRouting(res.config);
  pollNow();
  return res;
}

// Home box choices and the Advanced sliders, built once from the router's own lists and ranges.
function buildRouting() {
  $("#sel-pref").innerHTML = `<option value="">None (first box)</option>` +
    BOXES.map(b => `<option value="${esc(b)}">${esc(b)}</option>`).join("");
  const limits = (ROUTING && ROUTING.limits) || {};
  $("#adv-fields").innerHTML = Object.entries(limits).map(([k, l]) => `
    <div class="field adv-field">
      <label for="rng-${k}" title="${esc(l.help)}">${esc(SETTING_LABEL[k] || k)} <b id="val-${k}">–</b></label>
      <input type="range" id="rng-${k}" data-key="${k}" data-unit="${l.unit}" min="${l.min}" max="${l.max}" step="${l.step}">
      <p class="hint">${esc(l.help)}</p>
    </div>`).join("") || `<p class="hint">The router did not send its settings.</p>`;
  let t;
  $("#adv-fields").querySelectorAll("input[type=range]").forEach(inp => {
    inp.addEventListener("input", () => {
      const key = inp.dataset.key, v = +inp.value;   // keep it: the 1 s refresh must not overwrite it before it is sent
      inp.dataset.pending = "1";
      $(`#val-${key}`).textContent = fmtSetting(key, v, inp.dataset.unit);
      clearTimeout(t);
      t = setTimeout(async () => {
        try {
          await setRouting({ [key]: v });
          addEvent(nowS(), "demo", `${esc(SETTING_LABEL[key] || key)} set to <b>${fmtSetting(key, v, inp.dataset.unit)}</b>`);
        } catch (e) { addEvent(nowS(), "fail", `Setting not changed: ${esc(e.message)}`); }
        finally { delete inp.dataset.pending; }
      }, 300);
    });
  });
}

// What differs between two router configs, in words; [] when nothing does.
function routingDiff(a, b) {
  const out = [];
  if (a.policy !== b.policy) out.push(`rule <b>${esc(a.policy)} → ${esc(b.policy)}</b>`);
  if ((a.preferred || "") !== (b.preferred || "")) out.push(`home box <b>${esc(a.preferred || "none")} → ${esc(b.preferred || "none")}</b>`);
  for (const k of Object.keys(b.settings || {})) {
    const unit = b.limits?.[k]?.unit;
    if ((a.settings || {})[k] !== b.settings[k])
      out.push(`${esc(SETTING_LABEL[k] || k)} <b>${fmtSetting(k, a.settings[k], unit)} → ${fmtSetting(k, b.settings[k], unit)}</b>`);
  }
  for (const box of b.drained.filter(x => !a.drained.includes(x))) out.push(`<b>${esc(box)}</b> drained`);
  for (const box of a.drained.filter(x => !b.drained.includes(x))) out.push(`<b>${esc(box)}</b> back in service`);
  return out;
}

function renderRouting(cfg) {
  if (ROUTING && ROUTING.settings && Date.now() > expectRoutingUntil) {
    // Changed by something other than this page: another tab, the assistant elsewhere, a script, the demo.
    const diff = routingDiff(ROUTING, cfg);
    if (diff.length) addEvent(nowS(), "demo", `Routing changed elsewhere: ${diff.join(", ")}`);
  }
  ROUTING = cfg;
  if (!$("#adv-fields input") && cfg.limits) buildRouting();   // the router was not up when the page loaded
  const homeUsed = cfg.policy === "adaptive" || cfg.policy === "static";
  $("#field-pref").hidden = !homeUsed;
  $("#pref-hint").textContent = cfg.policy === "static" ? "All traffic goes here while it is healthy."
    : "Traffic starts here and comes back here once it recovers.";
  const sel = $("#sel-pref");
  if (document.activeElement !== sel) sel.value = cfg.preferred || "";
  for (const [k, v] of Object.entries(cfg.settings || {})) {
    const inp = $(`#rng-${k}`);
    if (!inp || document.activeElement === inp || inp.dataset.pending) continue;
    inp.value = v;
    $(`#val-${k}`).textContent = fmtSetting(k, v, inp.dataset.unit);
  }
  const d = cfg.defaults || {}, s = cfg.settings || {};
  const changed = Object.keys(s).filter(k => d[k] !== undefined && d[k] !== s[k]).length
    + (d.policy && d.policy !== cfg.policy) + ((d.preferred || null) !== (cfg.preferred || null)) + cfg.drained.length;
  $("#adv-changed").textContent = changed ? `${changed} changed from start-up` : "start-up settings";
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
  $("#k-active").textContent = st.policy === "adaptive" ? st.adaptive.active : st.policy === "balanced" ? "both"
    : st.policy === "static" ? (st.adaptive.preferred || BOXES[0]) : "spread";
  $("#k-pref").textContent = st.policy === "adaptive" ? `home ${st.adaptive.preferred || "none"}`
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
  if (st.config) renderRouting(st.config);
  if (document.activeElement !== $("#rng-load")) { $("#rng-load").value = ov.load_rps; $("#load-val").textContent = ov.load_rps; }

  for (const s of st.adaptive.switches) {
    const key = `${s.t}-${s.to}`;
    if (state.seenSwitches.has(key)) continue;
    state.seenSwitches.add(key);
    addEvent(state.clockOffset + s.t, "switch", `Switched <b>${esc(s.from)} → ${esc(s.to)}</b>: ${esc(s.reason)}`);
  }
}

// One live log. src "routing": the router's switches, routing changes and this page's actions (logged here);
// "telemetry", "fuzzball" and "incident" come from the page server (GET api/events): changes the WhatIf
// recorder saw, the Fuzzball workflow's events, and diagnoses. Kept newest first by time.
const SRC_LABEL = { telemetry: "telemetry", fuzzball: "fuzzball", incident: "incident" };
function addEvent(ts, kind, html, src = "routing", extra = null) {
  const e = { ts: ts ?? nowS(), kind, html, src, extra };
  let i = 0;
  while (i < state.events.length && state.events[i].ts > e.ts) i++;
  state.events.splice(i, 0, e);
  state.events = state.events.slice(0, 400);
  renderEvents();
}
function incidentExtra(x) {
  if (!x) return "";
  const trail = (x.trail || []).length ? `<details><summary>Evidence trail (${x.trail.length} lines)</summary><pre>${esc(x.trail.join("\n"))}</pre></details>` : "";
  const apply = x.can_apply ? `<button class="ev-apply" data-incident="${x.incident}">Apply remedy</button>` : "";
  return trail + apply;
}
function renderEvents() {
  const open = new Set([...document.querySelectorAll("#events details[open]")].map(d => d.closest("li")?.dataset.key));
  const shown = state.events.filter(e => state.evFilter === "all" || e.src === state.evFilter);
  $("#events").innerHTML = shown.length
    ? shown.map(e => { const key = `${e.src}-${e.ts}`;
        return `<li class="k-${e.kind}" data-key="${key}"><time>${clock(e.ts)}</time><span>${SRC_LABEL[e.src] ? `<i class="src">${SRC_LABEL[e.src]}</i> ` : ""}${e.html}${incidentExtra(e.extra)}</span></li>`; }).join("")
    : `<li class="muted">No ${state.evFilter === "all" ? "" : state.evFilter + " "}events yet.</li>`;
  document.querySelectorAll("#events li").forEach(li => { if (open.has(li.dataset.key)) li.querySelector("details")?.setAttribute("open", ""); });
  $("#ev-count").textContent = `${shown.length}`;
}

// The server-side stream: telemetry changes, Fuzzball workflow events, incidents. Every 2 s.
async function pollEvents() {
  let d;
  try { d = await api(`api/events?after=${state.evSeq}`); } catch (e) { $("#ev-feed").textContent = "Events: page server unreachable"; return; }
  const f = d.fuzzball || {}, t = d.telemetry || {};
  const wf = f.workflow ? ` ${String(f.workflow).slice(0, 8)}…` : "";
  $("#ev-feed").innerHTML = `Telemetry: ${t.on ? `<span class="ok">recording</span> (${t.seconds} s)` : "off"} · `
    + `Fuzzball${wf}: <span class="${f.status === "ok" ? "ok" : f.status === "unreachable" ? "bad" : ""}">${esc(f.status === "ok" ? "live" : f.status || "off")}</span>${f.reason ? ` (${esc(f.reason)})` : ""}`;
  for (const e of d.events || []) {
    let html = esc(e.text);
    if (e.src === "fuzzball") html = `${e.stage_kind === "workflow" || !e.stage ? "<b>workflow</b>" : `<b>${esc(e.stage)}</b> <span class="muted">${esc(e.stage_kind)}</span>`}: ${esc(e.text)}`;
    else if (e.box) html = esc(e.text).replace(esc(e.box), `<b>${esc(e.box)}</b>`);
    else if (e.src === "incident") html = `<b>${esc(e.text.split(":")[0])}</b>:${esc(e.text.slice(e.text.indexOf(":") + 1))}`;
    else html = `<b>${esc(e.text)}</b>`;
    addEvent(e.ts, (e.src === "fuzzball" ? "wf-" : "t-") + e.level, html, e.src,
      e.src === "incident" && (e.trail || e.can_apply) ? { trail: e.trail, can_apply: e.can_apply, incident: e.incident } : null);
  }
  state.evSeq = Math.max(state.evSeq, d.last_seq || 0);
}

async function diagnoseClick() {
  const btn = $("#btn-diagnose"), policy = $("#diag-policy").value;
  btn.disabled = true; btn.textContent = "Diagnosing…";
  try {
    const r = await fetch("api/incident/diagnose", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ policy }) });
    const d = await r.json();
    if (!r.ok) addEvent(nowS(), "fail", `<b>Diagnosis not run</b>: ${esc(d.detail || r.status)}`);
    pollEvents();
  } catch (e) { addEvent(nowS(), "fail", `<b>Diagnosis failed</b>: ${esc(e.message)}`); }
  finally { btn.disabled = false; btn.textContent = "Diagnose last 60 s"; }
}

async function applyRemedy(id, btn) {
  btn.disabled = true; btn.textContent = "Applying…";
  try {
    const r = await fetch("api/incident/remediate", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ incident: +id }) });
    const d = await r.json();
    if (!r.ok) addEvent(nowS(), "fail", `<b>Remedy not applied</b>: ${esc(d.detail || r.status)}`);
    pollEvents();
  } catch (e) { addEvent(nowS(), "fail", `<b>Remedy failed</b>: ${esc(e.message)}`); }
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
  const col = Object.fromEntries(BOXES.map(b => [b, css(`--${boxColor(b)}`)]));
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
    if (r.status >= 500) bins[i].e++; else if (r.box === BOXES[0]) bins[i].a++; else if (r.box === BOXES[1]) bins[i].b++;
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
// Overviews are numbered as they are asked for, and drawn in that order. An answer is dropped when it was
// asked for before an overview that is drawn already, or before the last action of this page was answered:
// it shows an older state. Drawn late, it would flip a card back and be read as news ("fine-tune stopped").
let pollSent = 0, pollFloor = 0;
function pollNow() { pollFloor = pollSent; return poll(); }   // after an action's answer: a fresh overview

async function poll() {
  const n = ++pollSent;
  let drawing = false;
  try {
    const q = state.lastT == null ? "" : `?since=${state.lastT}`;
    const ov = await api(`api/overview${q}`);
    if (n <= pollFloor) return;
    pollFloor = n; drawing = true;
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
    if (!drawing) {
      if (n <= pollFloor) return;   // an old request that failed late says nothing about now
      pollFloor = n;
    }
    setRouterPill(false, e.message);
    showNoData();
  }
}
// Router unreachable: show no data at all. Old numbers would look live, so every value the router
// gave is cleared and the controls are disabled until it answers again.
function showNoData() {
  document.body.classList.add("no-router");
  $("#no-router").hidden = false;
  state.overview = null; state.reqs = []; state.lastT = null;
  for (const [id, v] of [["#k-rps", "–"], ["#k-p95", "–"], ["#k-p50", "p50 –"], ["#k-p99", "–"], ["#k-err", "–"],
                         ["#k-fo", "failovers –"], ["#k-active", "–"], ["#k-pref", "–"], ["#pill-policy", "policy –"]])
    $(id).textContent = v;
  $("#k-err").style.color = "";
  $("#pill-sim").hidden = true;
  for (const b of BOXES) {
    const card = $(`#card-${b}`);
    if (!card) continue;
    const k = n => card.querySelector(`[data-k="${n}"]`);
    for (const n of ["share", "p95", "q", "gpu", "probe", "score"]) if (k(n)) k(n).textContent = "–";
    if (k("parts")) k("parts").textContent = "";
    for (const n of ["sim", "active", "pref", "drained", "job"]) if (k(n)) k(n).hidden = true;
    k("mode").textContent = "no data"; k("mode").className = "tag";
    k("dot").className = "dot";
    for (const n of ["qbar", "gbar"]) if (k(n)) k(n).firstElementChild.style.width = "0%";
    card.classList.remove("down", "drained");
  }
  $("#model-badge").hidden = true; $("#model-grid").hidden = true;
  $("#model-note").textContent = "Router unreachable: no live data.";
  drawChart();
}

function setRouterPill(ok, msg) {
  if (state.ok === ok) return;
  state.ok = ok;
  if (ok) { document.body.classList.remove("no-router"); $("#no-router").hidden = true; }
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

// What this forecast actually carries. The real AFNO model has no rain, humidity or cloud
// output, so the card shows only what exists and never guesses the sky.
function facts(r) {
  const any = k => (r.points || []).some(p => p[k] != null);
  return { cl: any("cl"), hum: any("hum"), pr: any("pr"), rain: r.summary?.precip_mm != null, sky: any("cl") || any("p") };
}
const Hpa = v => v == null ? "–" : `${Math.round(v)} hPa`;

function condition(p, lon) {
  const h = local(p.vt, lon).getUTCHours(), night = h < 6 || h >= 18;
  if (p.cl == null && p.p == null) return { k: night ? "nighto" : "dayo", label: night ? "Night" : "Day", nosky: true };
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
    // Outline sun/moon: time of day only, used when the model gives no sky data.
    dayo: `<circle cx="12" cy="12" r="4.5" fill="none" stroke="#9aa3b2" stroke-width="1.6"/><g stroke="#9aa3b2" stroke-width="1.4" stroke-linecap="round">${[0, 90, 180, 270].map(a => { const r = a * Math.PI / 180; return `<line x1="${12 + 7 * Math.cos(r)}" y1="${12 + 7 * Math.sin(r)}" x2="${12 + 9 * Math.cos(r)}" y2="${12 + 9 * Math.sin(r)}"/>`; }).join("")}</g>`,
    nighto: `<path d="M15.5 4a8 8 0 1 0 4.5 13.5 6.5 6.5 0 0 1-4.5-13.5z" fill="none" stroke="#9aa3b2" stroke-width="1.6"/>`,
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
  const now = pts[0], c = condition(now, r.lon), s = r.summary, f = facts(r);
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
      <div><div class="wx-temp">${T(now.t)}</div><div class="wx-cond">${c.nosky ? "Temperature & wind (no sky data)" : c.label}</div><div class="wx-cond">H ${T(s.max_c)} · L ${T(s.min_c)}</div></div>
      <div class="wx-stats">${stat("Wind", Wind(now.w))}${f.hum ? stat("Humidity", Pct(now.hum)) : f.pr ? stat("Pressure", Hpa(now.pr)) : ""}${hasRain ? stat("Rain", Rain(s.precip_mm)) : ""}</div>
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
    const ens = r.members_temp ? idx.flatMap(i => r.members_temp.map(m => m[i])).filter(v => v != null) : null;
    return { d, lo: Math.min(...ts), hi: Math.max(...ts), wind: Math.max(...pts.map(p => p.w)),
      rain: rain.length ? rain.reduce((a, b) => a + b, 0) : null, hum: hum.length ? hum.reduce((a, b) => a + b, 0) / hum.length : null,
      c: worst, ens: ens && ens.length ? [Math.min(...ens), Math.max(...ens)] : null, hours: pts.length };
  });
}

function fullForecast(r) {
  const days = dailyRows(r), s = r.summary, hasRain = s.precip_mm != null, f = facts(r);
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
      <div><div class="wx-temp big">${T(now.t)}</div><div class="wx-cond">${c.nosky ? "No sky data from this model" : c.label} · next hour</div></div>
      <div class="wx-stats">${stat("Wind", Wind(now.w))}${f.hum ? stat("Humidity", Pct(now.hum)) : ""}${f.cl ? stat("Cloud", Pct(now.cl)) : ""}${f.pr ? stat("Pressure", Hpa(now.pr)) : ""}${hasRain ? stat("Rain total", Rain(s.precip_mm)) : stat("Rain", "not provided")}</div>
    </div>
    <h4>By day <span class="muted">(local solar time)</span></h4>
    <div class="wxm-days">${days.map(d => `
      <div class="wxm-day">
        <span class="dname">${DAY[d.d.getUTCDay()]}<small>${d.hours < 24 ? ` · ${d.hours} h` : ""}</small></span>
        ${icon(d.c.k, 30)}<span class="dcond">${d.c.label}</span>
        <span class="dlo">${T(d.lo)}</span>
        <span class="drange">${d.ens ? `<i class="ens" style="left:${pos(d.ens[0])};right:calc(100% - ${pos(d.ens[1])})"></i>` : ""}<i style="left:${pos(d.lo)};right:calc(100% - ${pos(d.hi)})"></i></span>
        <span class="dhi">${T(d.hi)}</span>
        <span class="dmeta">${Wind(d.wind)}${f.hum ? ` · ${Pct(d.hum)}` : ""}${hasRain ? ` · ${Rain(d.rain)}` : ""}</span>
      </div>`).join("")}</div>
    ${r.members_temp ? `<p class="muted">Lighter bar = range across ${r.ensemble_members} ensemble members.</p>` : ""}
    <h4>Every ${r.native_step_hours || 6} hours <span class="muted">(model steps)</span></h4>
    <div class="wxm-table-wrap"><table class="wxm-table">
      <thead><tr><th>Time</th><th>${f.sky ? "Conditions" : "Time of day"}</th><th>Temp</th><th>Wind</th>${f.hum ? "<th>Humidity</th>" : ""}${f.cl ? "<th>Cloud</th>" : ""}${f.pr ? "<th>Pressure</th>" : ""}${hasRain ? "<th>Rain</th>" : ""}</tr></thead>
      <tbody>${steps.map(p => { const d = local(p.vt, r.lon), k = condition(p, r.lon); return `<tr>
        <td>${DAY[d.getUTCDay()]} ${hhmm(d)}</td><td class="cond">${icon(k.k, 22)} ${k.label}</td><td><b>${T(p.t)}</b></td>
        <td>${Wind(p.w)}</td>${f.hum ? `<td>${Pct(p.hum)}</td>` : ""}${f.cl ? `<td>${Pct(p.cl)}</td>` : ""}${f.pr ? `<td>${Hpa(p.pr)}</td>` : ""}${hasRain ? `<td>${Rain(p.p)}</td>` : ""}</tr>`; }).join("")}</tbody>
    </table></div>
    <p class="muted">The model computes a value every ${r.native_step_hours || 6} hours. Hourly values in between are straight-line interpolations, so this table shows only the model steps.${hasRain ? "" : " This model run provides no rain data."}</p>`;
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
    expectRouting();   // the assistant's routing changes are logged below, from its steps
    const res = await api("api/chat", { messages: state.chat.slice(-12), units: state.units });
    expectRouting();
    state.chat.push({ role: "assistant", content: res.reply });
    // One card for each forecast of this answer (a question about two places has two), three at most.
    // No card means: no forecast from the weather model came with this answer.
    let card = "";
    for (const fc of (res.steps || []).filter(s => s.tool === "get_forecast" && s.result?.points?.length).slice(0, 3)) {
      try {
        const id = `fc${fcSeq + 1}`, html = weatherCard(fc.result, id);
        fcSeq += 1; state.forecasts.set(id, fc.result); card += html;
      } catch (e) { /* a result the card cannot draw: the text and the chips below still show it */ }
    }
    pending.innerHTML = `<p>${md(res.reply)}</p>${card}${renderSteps(res.steps)}${res.note ? `<div class="note">${esc(res.note)}</div>` : ""}`;
    for (const s of res.steps || []) {
      if (s.result?.error) continue;
      const box = s.args?.box, jobState = s.result?.status?.state;
      if (s.tool === "simulate_box") addEvent(nowS(), "demo", `Simulated: assistant set <b>${esc(box)}</b> to ${esc(s.args?.mode)}`);
      if (s.tool === "start_finetune") { if (jobState) state.job[box] = jobState; addEvent(nowS(), "demo", `Real job: assistant started the fine-tune on <b>${esc(box)}</b>`); }
      if (s.tool === "stop_finetune") { if (jobState) state.job[box] = jobState; addEvent(nowS(), "demo", `Real job: assistant stopped the fine-tune on <b>${esc(box)}</b>`); }
      if (s.tool === "kill_forecast_server") addEvent(nowS(), "fail", `Real job: assistant killed the forecast server on <b>${esc(box)}</b> (a real crash, not simulated). Its supervisor restarts it by itself.`);
      if (s.tool === "set_routing_config" || s.tool === "set_routing_policy") {
        const what = Object.entries(s.args || {}).map(([k, v]) => `${k} ${typeof v === "object" ? JSON.stringify(v) : v}`).join(", ");
        addEvent(nowS(), "demo", `Assistant changed routing: <b>${esc(what)}</b>`);
      }
    }
    pollNow();
  } catch (e) {
    pending.innerHTML = `<p>Something went wrong: ${esc(e.message)}</p>`;
  } finally {
    state.busy = false; $("#send").disabled = false;
    $("#msgs").scrollTop = $("#msgs").scrollHeight;
  }
}

// ------------------------------------------------------------------ wiring
function wire() {
  // The rule buttons are the routing switch. The home box is its own setting: changing it never changes the rule.
  document.querySelectorAll("#seg-policy button").forEach(b => b.addEventListener("click", async () => {
    try {
      await setRouting({ policy: b.dataset.v });
      state.seenSwitches.clear();
      addEvent(nowS(), "demo", `Rule changed to <b>${b.dataset.v}</b>`);
    } catch (e) { addEvent(nowS(), "fail", `Rule not changed: ${esc(e.message)}`); }
  }));
  $("#sel-pref").addEventListener("change", async e => {
    const v = e.target.value;
    try {
      await setRouting({ preferred: v });
      addEvent(nowS(), "demo", v ? `Home box set to <b>${esc(v)}</b>` : "Home box cleared");
    } catch (err) { addEvent(nowS(), "fail", `Home box not changed: ${esc(err.message)}`); }
  });
  $("#btn-reset").addEventListener("click", async () => {
    try {
      const res = await setRouting({ reset: true });
      state.seenSwitches.clear();
      addEvent(nowS(), "demo", `Routing reset to start-up settings (rule <b>${esc(res.policy)}</b>, no box drained)`);
    } catch (e) { addEvent(nowS(), "fail", `Routing not reset: ${esc(e.message)}`); }
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
  document.querySelectorAll("#ev-filter button").forEach(b => b.addEventListener("click", () => {
    state.evFilter = b.dataset.f;
    document.querySelectorAll("#ev-filter button").forEach(x => x.classList.toggle("on", x === b));
    renderEvents();
  }));
  $("#btn-diagnose").addEventListener("click", diagnoseClick);
  $("#events").addEventListener("click", e => { const b = e.target.closest(".ev-apply"); if (b) applyRemedy(b.dataset.incident, b); });
  window.addEventListener("resize", drawChart);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawChart);
}

async function start() {
  try {
    ROUTING = await api("api/router/config");
    if (Array.isArray(ROUTING.backends) && ROUTING.backends.length) BOXES = ROUTING.backends;
  } catch (e) { /* router not up yet: keep the fallback names; the overview poll fills ROUTING in */ }
  buildBoxes();
  buildRouting();
  wire();
  if (ROUTING) renderRouting(ROUTING);
  poll();
  setInterval(poll, 1000);
  pollEvents();
  setInterval(pollEvents, 2000);
}

start();

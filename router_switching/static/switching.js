// Control Room layout and live GPU data.
//
// One place per fact:
//   GPUs                  one panel per box: state, CUDA, model, GPU readings, inference timing, counters, box actions
//   Live performance      the router's request rate, latency and errors, the request chart, and the load test
//   Routing and switching the rule and its settings, where traffic goes, what the router sees of each box,
//                         who changed the rule, and the event list
//   Forecast assistant    the one place to send a forecast and read the model's answer
//
// Every value comes from /api/switching/live (the boxes' and the router's own answers) or
// /api/switching/fuzzball (the fuzzball CLI). A value nobody reported is N/A; a box that does not answer is OFFLINE.
(function () {
  "use strict";
  const dash = document.querySelector(".dash");
  if (!dash) return;
  const css = document.createElement("link");
  css.rel = "stylesheet"; css.href = "switching/static/switching.css";
  document.head.appendChild(css);

  const $ = s => document.querySelector(s);
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (v, d = 0) => v == null || Number.isNaN(v) ? "N/A" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
  const ms = v => v == null ? "N/A" : v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${num(v, v < 10 ? 1 : 0)} ms`;
  const clock = ts => ts == null ? "never" : new Date(ts * 1000).toLocaleTimeString();
  const boxColor = (n, names) => (names ? names.indexOf(n) : (n === "box-a" ? 0 : 1)) === 0 ? "var(--a)" : "var(--b)";
  const na = why => `<span class="gx-na" title="${esc(why)}">N/A</span>`;

  // ------------------------------------------------------------------ layout: move the page's own parts into sections
  const section = (id, title, sub) => {
    const s = document.createElement("section");
    s.className = "gx-sec"; s.id = id;
    s.innerHTML = `<div class="gx-sec-head"><h2>${title}</h2><span class="muted" id="${id}-sub">${sub || ""}</span></div>`;
    return s;
  };
  const gpus = section("gx-gpus", "GPUs"), perf = section("gx-perf", "Live performance", "measured by the router, last 10 s"),
    routing = section("gx-routing", "Routing and switching", "from the router's own status");
  const banner = document.createElement("div"); banner.id = "gx-banner"; banner.hidden = true;
  const fbStrip = document.createElement("div"); fbStrip.id = "gx-fb"; fbStrip.className = "gx-fb"; fbStrip.textContent = "Asking Fuzzball…";
  const kpis = $(".kpis"), boxes = $("#boxes"), chart = $(".chart-card"), row2 = $(".row2"), loadField = $("#rng-load") && $("#rng-load").closest(".field");
  gpus.append(fbStrip); if (boxes) gpus.append(boxes);
  if (kpis) perf.append(kpis);
  if (loadField) {   // the one traffic control, out of the Routing card; its listeners move with it
    const card = document.createElement("div");
    card.className = "card gx-load";
    card.innerHTML = `<div class="card-head"><h2>Load test</h2><span class="muted">real forecast requests through the router to both GPUs · off until you turn it on</span></div>`;
    card.append(loadField); perf.append(card);
  }
  if (chart) perf.append(chart);
  if (row2) {
    const read = document.createElement("div"); read.id = "gx-route"; read.className = "gx-route muted"; read.textContent = "Waiting for the router…";
    const routingCard = row2.querySelector(".card"); if (routingCard) routingCard.querySelector(".card-head").after(read);
    routing.append(row2);
  }
  const anchor = $("#no-router") || dash.firstChild;
  anchor.after(banner, gpus, perf, routing);

  // ------------------------------------------------------------------ real only, Fuzzball
  let fb = null;
  const stageOf = n => fb && fb.available ? fb.stages.find(s => s.name === n) : null;
  async function pollGuard() {
    let r, d = null;
    try { r = await fetch("api/switching/guard"); d = r.ok ? await r.json() : null; } catch (e) { return; }
    if (r.status === 404) {   // started with --dev: mock and CPU boxes are allowed, and the page says so
      banner.hidden = false; banner.className = "gx-banner dev";
      banner.innerHTML = "<b>DEVELOPMENT MODE.</b> Started with --dev: mock servers and CPU boxes are allowed. Anything tagged SIMULATED below is not a GPU measurement.";
    } else if (d && d.detail) {
      banner.hidden = false; banner.className = "gx-banner refused";
      banner.innerHTML = `<b>Nothing is shown.</b> ${esc(d.detail)}`;
    } else banner.hidden = true;
  }
  const pill = (word, cls) => `<span class="gx-state ${cls}">${esc(word)}</span>`;
  const stageWord = s => s === "STARTED" ? "RUNNING" : s;
  async function pollFuzzball() {
    try { fb = await (await fetch("api/switching/fuzzball")).json(); } catch (e) { fb = null; }
    if (!fb || !fb.available) { fbStrip.innerHTML = `<b>Fuzzball workflow</b> ${na(fb ? fb.reason : "the page's server did not answer")} <span class="muted">${esc(fb ? fb.reason : "")}</span>`; return; }
    fbStrip.innerHTML = `<b>Fuzzball workflow</b> ${esc(fb.name ?? "")} <code>${esc(fb.id)}</code> ${pill(stageWord(fb.status), fb.status === "STARTED" ? "on" : "off")}
      ${fb.stages.filter(s => s.kind === "Service").map(s => `<span class="gx-stage">${esc(s.name)} ${pill(stageWord(s.status), s.status === "STARTED" ? "on" : "off")}${s.node ? ` <span class="muted">${esc(s.node)}</span>` : ""}</span>`).join("")}
      <span class="muted">fuzzball CLI · ${clock(fb.checked_ts)}</span>`;
  }

  // ------------------------------------------------------------------ GPU panels
  const KIND = { "real GPU": "real", "CPU": "cpu", "simulated": "sim", "unknown": "unknown" };
  function tile(label, value, sub, title) { return `<div class="gx-tile"${title ? ` title="${esc(title)}"` : ""}><div class="l">${label}</div><div class="v">${value}</div>${sub ? `<div class="s">${sub}</div>` : ""}</div>`; }
  function panel(name, b, r, names) {
    const st = stageOf(name);
    // OFFLINE is this page getting no telemetry; when Fuzzball says the service is not running, its word is used (PENDING, ERRORED…)
    const word = !b ? (st && st.status !== "STARTED" ? stageWord(st.status) : "OFFLINE") : b.state === "OFFLINE" && st && st.status !== "STARTED" ? stageWord(st.status) : b.state;
    const on = word === "ONLINE";
    const head = `<div class="gx-head"><span class="gx-node">${st && st.node ? esc(st.node) : `node ${na("the node is known only when the page is started with --workflow and the fuzzball CLI answers")}`}</span>
      ${pill(word, on ? "on" : "off")}${b && b.kind ? `<span class="sw-kind ${KIND[b.kind] || "unknown"}">${esc(b.kind)}</span>` : ""}
      <span class="muted">telemetry ${b ? clock(b.last_ok_ts) : "never"}</span></div>`;
    if (!b || !on) return head + `<div class="gx-off">No live values: ${esc(word)}. Nothing is shown until this box answers again.</div>`;
    const rb = (r && r.per_box && r.per_box[name]) || {};
    const cuda = b.device_type == null ? na("the box did not report its device") : String(b.device_type).startsWith("cuda") ? "CUDA" : `<span class="gx-bad">${esc(String(b.device_type).toUpperCase())}</span>`;
    const model = `${esc(b.model_version ?? "N/A")}${b.trained_model === false ? ' <span class="sw-kind sim" title="Random weights: the GPU work and timings are real, the forecast values are not weather">untrained stand-in</span>' : ""}`;
    const infer = b.compute_p50_ms != null ? [`${ms(b.compute_p50_ms)} / ${ms(b.compute_p95_ms)} / ${ms(b.compute_p99_ms)}`, "GPU call p50 / p95 / p99", "The server's own timing of each GPU call (/v1/metrics)"]
      : b.latency_p50_ms != null ? [`${ms(b.latency_p50_ms)} / ${ms(b.latency_p95_ms)}`, `request p50 / p95${b.ms_per_unit != null ? ` · ${num(b.ms_per_unit, 1)} ms per model step` : ""}`, "The server's own request latency (/stats). It reports no per-call GPU percentiles."]
      : [na("no request in the server's window yet"), "no requests yet"];
    const thr = b.requests_per_s != null ? [`${num(b.requests_per_s, 1)} req/s`, "counted by the server", "The server's own throughput (/v1/metrics)"]
      : rb.rps != null ? [`${num(rb.rps, 1)} req/s`, "counted by the router, last 10 s", "This server does not report throughput; the router's count of requests it sent here is shown"] : [na("neither the server nor the router reported it"), ""];
    return head + `<div class="gx-grid">
      ${tile("Device", cuda, esc(b.device ?? ""), "From the box: device type and name")}
      ${tile("Model", model, b.model_params != null ? `${num(b.model_params / 1e6, 1)} M parameters${b.model_latest ? ` · starts from ${esc(b.model_latest)} UTC` : ""}` : esc(b.gpu_uuid ?? ""), "model_version from the box's /health")}
      ${tile("GPU utilization", b.gpu_util == null ? na(b.gpu_util_source ?? "not reported") : `${num(100 * b.gpu_util)} %<div class="gx-bar"><i style="width:${Math.min(100, 100 * b.gpu_util)}%"></i></div>`, "", b.gpu_util_source)}
      ${tile("GPU memory", b.gpu_proc_mem_mb != null ? `${num(b.gpu_proc_mem_mb / 1024, 1)} GB` : b.gpu_memory_mb != null ? `${num(b.gpu_memory_mb, 0)} MB` : na("not reported by this server"),
        b.gpu_proc_mem_mb != null ? "this box's processes · total N/A (unified memory)" : b.gpu_memory_mb != null ? `model tensors · peak ${num(b.gpu_memory_peak_mb, 0)} MB · total N/A` : "",
        "nvidia-smi process memory, or torch's allocated tensors. A GB10 shares the machine's memory, so nvidia-smi gives no device total.")}
      ${tile("Temperature / power", b.gpu_temp_c == null && b.gpu_power_w == null ? na("not reported by this server") : `${b.gpu_temp_c == null ? "N/A" : num(b.gpu_temp_c) + " °C"} / ${b.gpu_power_w == null ? "N/A" : num(b.gpu_power_w, 1) + " W"}`, "", "nvidia-smi, via the server")}
      ${tile("Inference latency", infer[0], infer[1], infer[2])}
      ${tile("Throughput", thr[0], thr[1], thr[2])}
      ${tile("Batch size", b.mean_batch_rows == null ? na("this server does not report batch sizes") : `${num(b.mean_batch_rows, 1)} rows`, b.max_batch_rows != null ? `mean per GPU call · max ${num(b.max_batch_rows)}` : "", "Rows per GPU call, from the server's /v1/metrics")}
      ${tile("Requests served", b.requests_total == null ? na("not reported") : num(b.requests_total), `${num(b.errors_total)} errors · includes the router's probes`, "The box's own counter. The router probes each box with a small forecast every 0.5 s.")}
      ${tile("Queue / in flight", `${num(b.queue_depth)} / ${num(b.in_flight)}`, b.mode && b.mode !== "normal" ? `mode: ${esc(b.mode)}` : "", "From the box's /stats")}
    </div>`;
  }
  function renderPanels(d) {
    const names = [...document.querySelectorAll("#boxes .card.box")].map(c => c.id.replace("card-", ""));
    for (const name of names) {
      const card = document.getElementById(`card-${name}`);
      let el = card.querySelector(".gx");
      if (!el) { el = document.createElement("div"); el.className = "gx"; card.querySelector(".box-top").after(el); }
      el.innerHTML = d ? panel(name, d.boxes[name], d.router, names) : `<div class="gx-off">No live values: the page's server did not answer.</div>`;
    }
  }

  // ------------------------------------------------------------------ routing readout
  function renderRoute(d) {
    const el = document.getElementById("gx-route"); if (!el) return;
    const r = d && d.router;
    if (!r) { el.className = "gx-route muted"; el.textContent = `Router OFFLINE: no routing data. Last answer: ${d && d.router_seen ? clock(d.router_seen.last_ok_ts) : "never"}.`; return; }
    const names = Object.keys(r.backends), sh = r.share_pct || {}, th = r.thresholds, pct = v => v == null ? "N/A" : `${num(100 * v)}%`;
    const adaptive = r.policy === "adaptive";
    el.className = "gx-route";
    el.innerHTML = `
      <div class="gx-dist-l"><span>Where requests went, last ${num(r.window_s)} s</span><span>${names.map(n => `<b style="color:${boxColor(n, names)}">${esc(n)} ${sh[n] == null ? "no requests" : num(sh[n]) + "%"}</b>`).join(" · ")}</span></div>
      <div class="gx-dist">${names.map(n => `<i style="width:${sh[n] ?? 0}%;background:${boxColor(n, names)}"></i>`).join("")}</div>
      <table class="sw-table"><tr><th>What the router sees</th>${names.map(n => `<th style="color:${boxColor(n, names)}">${esc(n)}</th>`).join("")}</tr>
        <tr><td>Healthy</td>${names.map(n => `<td>${r.backends[n].healthy ? "yes" : pill("DOWN", "off")}</td>`).join("")}</tr>
        <tr><td title="The router's own score per box; lower is better">Health score</td>${names.map(n => `<td>${num(r.backends[n].health_score, 1)}</td>`).join("")}</tr>
        <tr><td title="GPU time of the router's probe forecast">Probe</td>${names.map(n => `<td>${ms(r.backends[n].probe_ms)}</td>`).join("")}</tr>
        <tr><td>Recent error share</td>${names.map(n => `<td>${pct(r.backends[n].err_rate)}</td>`).join("")}</tr>
      </table>
      <p class="gx-line">${adaptive ? `Adaptive is routing: all requests go to <b style="color:${boxColor(r.active, names)}">${esc(r.active)}</b>.` : `Rule <b>${esc(r.policy)}</b> decides per request; the adaptive rule is not routing.`}
        Switch thresholds: other box <b>${pct(th.margin)}</b> faster for <b>${num(th.window_s, 1)} s</b>, at most one switch per <b>${num(th.cooldown_s)} s</b>, return home within <b>${pct(th.return_margin)}</b>. Failovers since the router started: <b>${num(r.failovers_total)}</b>.</p>
      <details class="gx-changes"><summary>Who changed the rule <span class="muted">(${(d.config_changes || []).length} recorded by this page's server)</span></summary>
        <ol class="sw-list">${(d.config_changes || []).slice().reverse().map(c => `<li><b>${clock(c.ts)}</b> · rule is <b>${esc(c.policy)}</b> · ${esc(c.source)}${c.asked ? ` · asked: <code>${esc(JSON.stringify(c.asked))}</code>` : ""}</li>`).join("") || "<li>nothing yet</li>"}</ol></details>`;
  }

  async function pollLive() {
    let d = null;
    try { const r = await fetch("api/switching/live"); d = r.ok ? await r.json() : null; } catch (e) { d = null; }
    renderPanels(d); renderRoute(d);
  }
  pollGuard(); setInterval(pollGuard, 5000); pollFuzzball(); setInterval(pollFuzzball, 10000);
  pollLive(); setInterval(pollLive, 2000);
})();

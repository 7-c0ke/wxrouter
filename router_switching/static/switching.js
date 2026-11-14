// Inference and switching panels for the wxrouter Control Room.
// Adds three cards under the existing dashboard. Reads /api/switching/live every 2 s and
// /api/switching/results once. A value nobody reported is shown as a dash, never guessed.
(function () {
  "use strict";
  const dash = document.querySelector(".dash");
  if (!dash) return;
  const css = document.createElement("link");
  css.rel = "stylesheet"; css.href = "switching/static/switching.css";
  document.head.appendChild(css);

  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (v, d = 0) => v == null || Number.isNaN(v) ? "–" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
  const ms = v => v == null ? "–" : v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${num(v, v < 10 ? 1 : 0)} ms`;
  const KIND = { "real GPU": "real", "CPU": "cpu", "simulated": "sim", "unknown": "unknown" };
  const badge = k => `<span class="sw-kind ${KIND[k] || "unknown"}">${esc(k)}</span>`;
  const boxColor = n => n === "box-a" ? "var(--a)" : "var(--b)";

  const wrap = document.createElement("div");
  wrap.innerHTML = `
    <div class="card" id="sw-demo-card"><div class="card-head"><h2>Switching demo</h2><span class="muted" id="sw-demo-kind"></span></div>
      <div class="sw-form">
        <button type="button" class="sw-btn primary" id="sw-demo-run">Run Switching Demo</button>
        <span class="muted" id="sw-demo-status">One click: slow GPU A, restore it, fail it, restore it, with forecasts flowing through the router. About a minute. Every number is read back from the router.</span>
      </div>
      <div id="sw-demo-out"></div>
    </div>
    <div class="sw-grid">
      <div class="card"><div class="card-head"><h2>GPU inference</h2><span class="muted" id="sw-live-age"></span></div><div id="sw-boxes" class="muted">Waiting for the boxes…</div></div>
      <div class="card"><div class="card-head"><h2>Intelligent switching</h2><span class="muted">from the router's own status</span></div><div id="sw-router" class="muted">Waiting for the router…</div></div>
    </div>
    <div class="card"><div class="card-head"><h2>Run inference</h2><span class="muted">a real request through the router to a GPU</span></div>
      <form id="sw-run-form" class="sw-form">
        <label>Latitude <input name="lat" type="number" step="any" min="-90" max="90" value="41.98" required></label>
        <label>Longitude <input name="lon" type="number" step="any" min="-180" max="180" value="-87.9" required></label>
        <label>Hours ahead <input name="hours_ahead" type="number" min="1" max="120" value="48" required></label>
        <label>Ensemble members <input name="ensemble_members" type="number" min="1" max="20" value="1" required></label>
        <button type="submit" class="sw-btn primary">Run inference</button>
        <button type="button" class="sw-btn" id="sw-burst">Load test: 300 requests</button>
      </form>
      <div id="sw-run-out" class="muted">Run a request to see which GPU serves it and what the model returns.</div>
      <ol class="sw-list" id="sw-run-log" style="max-height:150px"></ol>
    </div>
    <div class="card"><div class="card-head"><h2>Benchmarks and reliability tests</h2><span class="muted" id="sw-res-age"></span></div><div id="sw-results" class="muted">Loading results…</div></div>`;
  while (wrap.firstChild) dash.appendChild(wrap.firstChild);

  // ------------------------------------------------------------------ live
  function renderBoxes(boxes) {
    const names = Object.keys(boxes);
    const row = (label, f) => `<tr><td>${label}</td>${names.map(n => `<td>${f(boxes[n], n)}</td>`).join("")}</tr>`;
    document.getElementById("sw-boxes").innerHTML = `<table class="sw-table">
      <tr><th></th>${names.map(n => `<th style="color:${boxColor(n)}">${esc(n === "box-a" ? "GPU A · box-a" : n === "box-b" ? "GPU B · box-b" : n)}</th>`).join("")}</tr>
      ${row("Runs on", b => badge(b.kind))}
      ${row("Device", b => esc(b.device ?? "–"))}
      ${row("Available", b => !b.reachable ? '<span class="tag downtag">unreachable</span>' : esc(b.status ?? "–") + (b.mode && b.mode !== "normal" ? ` · ${esc(b.mode)}` : ""))}
      ${row("GPU utilization", b => b.gpu_util == null || b.kind === "simulated" || b.kind === "CPU" ? `<span title="${esc(b.gpu_util_source ?? "not reported")}">unavailable</span>` : `${num(100 * b.gpu_util)} %`)}
      ${row("Active requests", b => num(b.in_flight))}
      ${row("Queue depth", b => num(b.queue_depth))}
      ${row("Inference p50 / p95", b => b.compute_p50_ms == null ? "–" : `${ms(b.compute_p50_ms)} / ${ms(b.compute_p95_ms)}`)}
      ${row("Request mean / p50 / p95", b => b.latency_p50_ms == null ? "–" : `${b.latency_mean_ms == null ? "N/A" : ms(b.latency_mean_ms)} / ${ms(b.latency_p50_ms)} / ${ms(b.latency_p95_ms)}`)}
      ${row("Requests served / errors", b => b.requests_total == null ? "–" : `${num(b.requests_total)} / ${num(b.errors_total)}${b.timeouts ? ` (${num(b.timeouts)} timeouts)` : ""}`)}
      ${row("Batch size (mean / max)", b => b.mean_batch_rows == null ? "–" : `${num(b.mean_batch_rows, 1)} / ${num(b.max_batch_rows)}`)}
      ${row("Throughput", b => b.requests_per_s == null ? "–" : `${num(b.requests_per_s, 1)} req/s · ${num(b.rows_per_s, 1)} rows/s`)}
      ${row("GPU memory (model, now / peak)", b => b.gpu_memory_mb == null ? '<span title="Only a box running on CUDA reports this">N/A</span>' : `${num(b.gpu_memory_mb, 1)} / ${num(b.gpu_memory_peak_mb, 1)} MB`)}
      ${row("GPU memory (whole device)", b => b.gpu_memory_device_mb == null ? '<span title="nvidia-smi reports no memory figure on a GB10: the GPU shares the machine\u2019s unified memory">N/A</span>' : `${num(b.gpu_memory_device_mb)} MB`)}
      ${row("Model", b => `<span title="${esc(b.gpu_uuid ?? "")}">${esc(b.model_version ?? "–")}</span>${b.trained_model === false ? ' <span class="sw-kind sim" title="Seeded random weights: real GPU work, but the forecast values mean nothing">stand-in</span>' : ""}`)}
    </table>`;
  }

  function renderRouter(r) {
    const el = document.getElementById("sw-router");
    if (!r) { el.textContent = "Router unreachable."; return; }
    const th = r.thresholds, names = Object.keys(r.backends), sh = r.share_pct || {};
    const pct = v => v == null ? "–" : `${num(100 * v)}%`;
    const last = r.last_switch;
    el.className = "";
    el.innerHTML = `
      <div class="sw-facts">
        <div class="sw-fact"><div class="l">Policy</div><div class="v">${esc(r.policy)}</div></div>
        <div class="sw-fact"><div class="l">Preferred backend</div><div class="v">${esc(r.preferred || "none")}</div></div>
        <div class="sw-fact"><div class="l">Active (adaptive)</div><div class="v" style="color:${boxColor(r.active)}">${esc(r.active ?? "–")}</div></div>
      </div>
      <div class="muted" style="margin-bottom:4px">Traffic, last ${num(r.window_s)} s: ${names.map(n => `<b style="color:${boxColor(n)}">${esc(n)} ${sh[n] == null ? "–" : num(sh[n]) + "%"}</b>`).join(" · ")}</div>
      <div class="sw-share">${names.map(n => `<i style="width:${sh[n] ?? 0}%;background:${boxColor(n)}" title="${esc(n)} ${sh[n] ?? 0}%"></i>`).join("")}</div>
      <table class="sw-table" style="margin-top:10px">
        <tr><th></th>${names.map(n => `<th style="color:${boxColor(n)}">${esc(n)}</th>`).join("")}</tr>
        <tr><td>Healthy</td>${names.map(n => `<td>${r.backends[n].healthy ? "yes" : '<span class="tag downtag">down</span>'}</td>`).join("")}</tr>
        <tr><td>Health score (lower is better)</td>${names.map(n => `<td>${num(r.backends[n].health_score, 1)}</td>`).join("")}</tr>
        <tr><td>Probe (GPU time)</td>${names.map(n => `<td>${ms(r.backends[n].probe_ms)}</td>`).join("")}</tr>
        <tr><td>Recent error share</td>${names.map(n => `<td>${pct(r.backends[n].err_rate)}</td>`).join("")}</tr>
      </table>
      <div class="muted" style="margin-top:10px">Adaptive thresholds: leave when the other box is <b>${pct(th.margin)}</b> faster for <b>${num(th.window_s, 1)} s</b>,
        at most one switch per <b>${num(th.cooldown_s)} s</b>; return home within <b>${pct(th.return_margin)}</b>. Failovers so far: <b>${num(r.failovers_total)}</b>.</div>
      <div class="sw-last">${last ? `<b>Last switch</b> ${last.age_s != null ? num(last.age_s) + " s ago" : ""}: ${esc(last.from)} → ${esc(last.to)} <span class="muted">(${esc(last.reason)})</span>` : "<b>No switch yet</b> since this policy was set."}</div>
      <ol class="sw-list">${r.switches.slice().reverse().map(s => `<li><b>${esc(s.from)} → ${esc(s.to)}</b> at ${num(s.t, 1)} s · ${esc(s.reason)}</li>`).join("")}</ol>`;
  }

  async function pollLive() {
    try {
      const d = await (await fetch("api/switching/live")).json();
      renderBoxes(d.boxes); renderRouter(d.router);
      document.getElementById("sw-live-age").textContent = "live · every 2 s";
    } catch (e) { document.getElementById("sw-live-age").textContent = "not connected"; }
  }

  // ------------------------------------------------------------------ run inference
  const runOut = document.getElementById("sw-run-out"), runLog = document.getElementById("sw-run-log");
  const post = async (path, body) => (await fetch(path, { method: "POST", headers: { "content-type": "application/json" }, body: body ? JSON.stringify(body) : undefined })).json();
  const standIn = v => String(v || "").includes("untrained");
  function logLine(html) { const li = document.createElement("li"); li.innerHTML = `${new Date().toLocaleTimeString()} · ${html}`; runLog.prepend(li); while (runLog.children.length > 30) runLog.lastChild.remove(); }
  document.getElementById("sw-run-form").addEventListener("submit", async e => {
    e.preventDefault();
    const f = new FormData(e.target), body = { lat: +f.get("lat"), lon: +f.get("lon"), hours_ahead: +f.get("hours_ahead"), ensemble_members: +f.get("ensemble_members") };
    runOut.className = ""; runOut.textContent = "Sending to the router…";
    try {
      const d = await post("api/forecast", body);
      if (d.error) { runOut.innerHTML = `<span class="tag downtag">failed</span> ${esc(d.error)}`; logLine(`<b>failed</b> ${esc(d.error)}`); return; }
      const pts = d.points.slice(0, 6);
      runOut.innerHTML = `
        <div class="sw-facts" style="grid-template-columns:repeat(4,minmax(0,1fr));margin-top:10px">
          <div class="sw-fact"><div class="l">Served by</div><div class="v" style="color:${boxColor(d.served_by)}">${esc(d.served_by === "box-a" ? "GPU A · box-a" : d.served_by === "box-b" ? "GPU B · box-b" : d.served_by)}</div></div>
          <div class="sw-fact"><div class="l">Round trip from this Mac</div><div class="v">${ms(d.latency_ms)}</div></div>
          <div class="sw-fact"><div class="l">On the server</div><div class="v">${ms(d.server_ms)}</div></div>
          <div class="sw-fact"><div class="l">Policy${d.failover ? " · failover" : ""}</div><div class="v">${esc(d.policy)}</div></div>
        </div>
        ${standIn(d.model_version) ? `<div class="sw-note"><b>Stand-in model (${esc(d.model_version)}).</b> The GPU work and the timings are real. The numbers below come from random weights and are not a weather forecast.</div>` : `<p class="sw-src">Model: ${esc(d.model_version)}</p>`}
        <table class="sw-table"><tr><th>lead</th><th>valid</th><th>temp °C</th><th>wind m/s</th><th>pressure hPa</th></tr>
        ${pts.map(p => `<tr><td>+${p.h} h</td><td>${esc(String(p.vt).slice(5, 16).replace("T", " "))}</td><td>${num(p.t, 1)}</td><td>${num(p.w, 1)}</td><td>${num(p.pr, 1)}</td></tr>`).join("")}</table>
        <p class="sw-src">${d.points.length} points returned${d.members_temp ? `, ${d.members_temp.length} ensemble members` : ""}; first ${pts.length} shown.</p>`;
      logLine(`<b style="color:${boxColor(d.served_by)}">${esc(d.served_by)}</b> · ${body.hours_ahead} h × ${body.ensemble_members} · ${ms(d.latency_ms)} round trip, ${ms(d.server_ms)} on the server${d.failover ? " · <b>failover</b>" : ""}`);
    } catch (err) { runOut.textContent = "Request failed: " + err; }
  });
  document.getElementById("sw-burst").addEventListener("click", async e => {
    e.target.disabled = true; runOut.className = ""; runOut.textContent = "Sending 300 real requests through the router, 16 at a time…";
    try {
      const d = await post("api/switching/burst?n=300&concurrency=16");
      const who = Object.entries(d.served_by).map(([n, c]) => `<b style="color:${boxColor(n)}">${esc(n)} ${c}</b>`).join(" · ");
      runOut.innerHTML = `<div class="sw-facts" style="grid-template-columns:repeat(4,minmax(0,1fr));margin-top:10px">
        <div class="sw-fact"><div class="l">Served</div><div class="v">${d.ok} / ${d.requests}</div></div>
        <div class="sw-fact"><div class="l">Throughput</div><div class="v">${num(d.requests_per_s, 1)} req/s</div></div>
        <div class="sw-fact"><div class="l">Round trip mean / p95</div><div class="v">${ms(d.client_ms.mean)} / ${ms(d.client_ms.p95)}</div></div>
        <div class="sw-fact"><div class="l">GPU call p50 / p95</div><div class="v">${ms(d.gpu_call_ms.p50)} / ${ms(d.gpu_call_ms.p95)}</div></div></div>
        <p class="sw-src">Who served them: ${who || "nobody"} · errors ${d.errors}${d.never_reached_router ? ` (${d.never_reached_router} never reached the router: the connection from this machine dropped)` : ""} · failovers ${d.failovers} · ${esc(d.note)}</p>`;
      logLine(`load test: ${d.ok}/${d.requests} ok, ${num(d.requests_per_s, 1)} req/s · ${who}`);
    } catch (err) { runOut.textContent = "Load test failed: " + err; }
    e.target.disabled = false;
  });

  // ------------------------------------------------------------------ charts
  const W = 460, H = 210, L = 44, R = 8, T = 14, B = 46;
  function niceMax(v) { if (!(v > 0)) return 1; const m = Math.pow(10, Math.floor(Math.log10(v))); for (const k of [1, 2, 2.5, 5, 10]) if (v <= k * m) return k * m; return 10 * m; }
  function frame(max, unit) {
    let s = "";
    for (let k = 0; k <= 4; k++) { const y = T + (H - T - B) * (1 - k / 4); s += `<line class="g" x1="${L}" x2="${W - R}" y1="${y}" y2="${y}"/><text x="${L - 6}" y="${y + 4}" text-anchor="end">${num(max * k / 4, max < 10 ? 1 : 0)}</text>`; }
    return s + `<text x="${L + 4}" y="${T + 2}">${esc(unit)}</text>`;
  }
  function bars(title, rows, series, unit) {
    const vals = rows.flatMap(r => series.map(s => r[s.key])).filter(v => v != null);
    if (!vals.length) return "";
    const max = niceMax(Math.max(...vals)), pw = W - L - R, ph = H - T - B, gw = pw / rows.length;
    const bw = Math.min(26, (gw - 10) / series.length - 2);
    let s = frame(max, unit);
    rows.forEach((r, i) => {
      const x0 = L + gw * i + (gw - (bw + 2) * series.length) / 2;
      series.forEach((se, j) => {
        const v = r[se.key]; if (v == null) return;
        const h = Math.max(1, ph * v / max), x = x0 + (bw + 2) * j, y = T + ph - h;
        s += `<rect x="${x}" y="${y}" width="${bw}" height="${h}" rx="3" fill="${se.color}"><title>${esc(r.label)} · ${esc(se.name)}: ${num(v, v < 100 ? 1 : 0)} ${esc(unit)}</title></rect>`;
        if (series.length === 1 || rows.length <= 4) s += `<text class="v" x="${x + bw / 2}" y="${y - 3}" text-anchor="middle">${num(v, v < 10 ? 1 : 0)}</text>`;
      });
      const lab = r.label.length > 16 ? r.label.slice(0, 15) + "…" : r.label;
      s += `<text x="${L + gw * (i + .5)}" y="${H - B + 14}" text-anchor="end" transform="rotate(-28 ${L + gw * (i + .5)} ${H - B + 14})">${esc(lab)}</text>`;
    });
    const legend = series.length > 1 ? `<div class="sw-legend">${series.map(se => `<span><i style="background:${se.color}"></i>${esc(se.name)}</span>`).join("")}</div>` : "";
    return `<div class="sw-chart"><h3>${esc(title)}</h3>${legend}<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">${s}</svg></div>`;
  }
  function timeline(c) {
    const tl = c.timeline; if (!tl.length) return "";
    const t0 = tl[0].t, t1 = tl[tl.length - 1].t, max = niceMax(Math.max(...tl.map(p => Math.max(p.a, p.b)))), pw = W - L - R, ph = H - T - B;
    const x = t => L + pw * (t - t0) / Math.max(1, t1 - t0), y = v => T + ph * (1 - v / max);
    let s = frame(max, "req/s served");
    c.events.forEach(e => { s += `<line class="ev" x1="${x(e.t)}" x2="${x(e.t)}" y1="${T}" y2="${T + ph}"><title>${esc(e.action)} ${esc(e.box)} at ${e.t} s</title></line><text x="${x(e.t) + 3}" y="${T + 22 + 11 * (c.events.indexOf(e) % 2)}">${esc(e.action)}</text>`; });
    [["a", "var(--a)", "box-a"], ["b", "var(--b)", "box-b"]].forEach(([k, col, name]) => {
      s += `<path d="${tl.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p[k]).toFixed(1)}`).join(" ")}" fill="none" stroke="${col}" stroke-width="2"/>`;
      tl.forEach(p => { s += `<circle cx="${x(p.t)}" cy="${y(p[k])}" r="6" fill="transparent"><title>${name} at ${p.t} s: ${p[k]} req/s · errors ${p.errors} · p95 ${p.p95_ms ?? "–"} ms</title></circle>`; });
    });
    tl.filter(p => p.errors > 0).forEach(p => { s += `<rect x="${x(p.t) - 2}" y="${T + ph + 2}" width="4" height="5" fill="var(--err)"><title>${p.errors} errors at ${p.t} s</title></rect>`; });
    for (let k = 0; k <= 4; k++) { const t = t0 + (t1 - t0) * k / 4; s += `<text x="${x(t)}" y="${H - B + 18}" text-anchor="${k === 4 ? "end" : k === 0 ? "start" : "middle"}">${num(t)} s</text>`; }
    return `<div class="sw-chart"><h3>${esc(c.policy)}: who served the requests</h3><div class="sw-legend"><span><i style="background:var(--a)"></i>box-a</span><span><i style="background:var(--b)"></i>box-b</span><span><i style="background:var(--err)"></i>errors</span></div><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Requests per second by box, ${esc(c.policy)}">${s}</svg></div>`;
  }
  function steps(c) {
    const f = c.fault_report || {}, sl = f.slow, st = f.stop;
    if (!sl && !st) return "";
    const v = x => x == null ? "–" : `${num(x)}%`;
    const items = [
      ["Normal operation", sl ? `box-a serves ${v(sl.share_before_pct)}` : "–"],
      ["GPU A slowdown", sl ? `${sl.from_s}–${sl.to_s} s: box-a share ${v(sl.share_during_pct)}, ${sl.errors_during} errors` : "not in this run"],
      ["GPU A failure", st ? `stopped at ${st.stopped_s} s` : "not in this run"],
      ["Failover to GPU B", st ? `${st.errors_while_down} errors while down` : "–"],
      ["GPU A recovery", st ? `restart at ${st.restart_s} s; serving again after ${st.seconds_to_serve_again ?? "–"} s` : "–"],
      ["Traffic rebalancing", st ? `box-a share in the last 5 s: ${v(st.share_last_5s_pct)} · ${c.switches.length} switches` : "–"]];
    return `<div class="sw-steps">${items.map((it, i) => `<div class="sw-step"><div class="n">${i + 1}</div><div class="t">${it[0]}</div><div class="d">${esc(it[1])}</div></div>`).join("")}</div>`;
  }
  const LAT = [{ key: "p50_ms", name: "p50", color: "color-mix(in srgb, var(--accent) 40%, transparent)" }, { key: "p95_ms", name: "p95", color: "color-mix(in srgb, var(--accent) 70%, transparent)" }, { key: "p99_ms", name: "p99", color: "var(--accent)" }];
  function table(rows) {
    const cols = [["label", "case"], ["requests", "requests"], ["error_pct", "errors %"], ["p50_ms", "p50 ms"], ["p95_ms", "p95 ms"], ["p99_ms", "p99 ms"], ["rps", "req/s"], ["mean_batch_rows", "batch"], ["share_a_pct", "A %"], ["share_b_pct", "B %"], ["failovers", "failovers"], ["switches", "switches"]].filter(c => rows.some(r => r[c[0]] != null));
    return `<details style="margin-top:8px"><summary class="muted">Table</summary><table class="sw-table"><tr>${cols.map(c => `<th>${c[1]}</th>`).join("")}</tr>${rows.map(r => `<tr>${cols.map(c => `<td>${typeof r[c[0]] === "number" ? num(r[c[0]], r[c[0]] < 100 ? 1 : 0) : esc(r[c[0]] ?? "–")}</td>`).join("")}</tr>`).join("")}</table></details>`;
  }
  function renderRun(run, all) {
    let h = `<p class="sw-src">${badge(run.kind)} ${esc(run.device ?? "")} · ${esc(run.when ?? "")} · <code>${esc(run.source)}</code></p>`;
    if (run.kind !== "real GPU") h += `<div class="sw-note">${run.kind === "simulated" ? "Simulated: the boxes were mock servers with a made-up cost model." : "CPU only: not a GPU measurement."} Do not read these as GPU results.</div>`;
    if (run.placement) h += `<p class="sw-src">${Object.entries(run.placement).map(([n, p]) => `${esc(n)}: ${esc(p.gpu_uuid ?? p.address ?? "?")}`).join(" · ")} · on different GPUs: <b>${esc(run.distinct_nodes)}</b>${run.calibration ? ` · slow mode measured at ${esc(run.calibration.measured_slowdown)}×` : ""}</p>`;
    if (run.compare) h += `<p class="sw-src">Same requests to both servers: ${num(run.compare.points_compared)} points, largest difference ${run.compare.max_abs_diff.temp_c} °C · agree: <b>${run.compare.agree}</b></p>`;
    let charts = bars("Latency", run.rows, LAT, "ms");
    if (run.rows.some(r => r.rps != null)) charts += bars("Requests per second", run.rows, [{ key: "rps", name: "req/s", color: "var(--accent)" }], "req/s");
    if (run.rows.some(r => r.error_pct != null && r.error_pct > 0)) charts += bars("Error rate", run.rows, [{ key: "error_pct", name: "errors", color: "var(--err)" }], "%");
    if (run.rows.some(r => r.share_a_pct != null)) charts += bars("Traffic share", run.rows, [{ key: "share_a_pct", name: "box-a", color: "var(--a)" }, { key: "share_b_pct", name: "box-b", color: "var(--b)" }], "%");
    if (run.type === "batching" && all.cpu_vs_gpu && run.kind === "real GPU") {
      const c = all.cpu_vs_gpu;
      charts += bars(`CPU vs GPU, batched, ${c.gpu.clients} clients`, [{ label: "laptop CPU", rps: c.cpu.rps }, { label: c.gpu.device, rps: c.gpu.rps }], [{ key: "rps", name: "req/s", color: "var(--accent)" }], "req/s");
    }
    h += `<div class="sw-charts">${charts}</div>`;
    if (run.type === "batching" && all.cpu_vs_gpu && run.kind === "real GPU") h += `<p class="sw-src">${esc(all.cpu_vs_gpu.note)}</p>`;
    if (run.cases) h += run.cases.map(c => `<h3 style="font-size:13px;margin:16px 0 0">Reliability test · ${esc(c.policy)}</h3>${steps(c)}<div class="sw-charts">${timeline(c)}${bars("p95 per second", c.timeline.filter((_, i) => i % Math.ceil(c.timeline.length / 20) === 0).map(p => ({ label: p.t + " s", p95_ms: p.p95_ms })), [{ key: "p95_ms", name: "p95", color: "var(--accent)" }], "ms")}</div>`).join("");
    return h + table(run.rows);
  }
  async function loadResults() {
    const el = document.getElementById("sw-results");
    let d;
    try { d = await (await fetch("api/switching/results")).json(); } catch (e) { el.textContent = "Results unavailable."; return; }
    if (!d.runs.length) { el.textContent = d.note || "No results yet."; return; }
    document.getElementById("sw-res-age").textContent = `collected ${d.generated_at}`;
    const blocked = d.not_measured_on_gpu.length ? `<div class="sw-note"><b>Not measured on a GPU yet:</b> ${d.not_measured_on_gpu.map(esc).join("; ")}.</div>` : "";
    el.className = "";
    el.innerHTML = `${blocked}<div class="sw-tabs">${d.runs.map((r, i) => `<button data-i="${i}">${badge(r.kind)} ${esc(r.title)}</button>`).join("")}</div><div id="sw-run"></div>`;
    const show = i => { el.querySelectorAll(".sw-tabs button").forEach(b => b.classList.toggle("on", +b.dataset.i === i)); document.getElementById("sw-run").innerHTML = renderRun(d.runs[i], d); };
    el.querySelector(".sw-tabs").addEventListener("click", e => { const b = e.target.closest("button"); if (b) show(+b.dataset.i); });
    show(0);
  }


  // ------------------------------------------------------------------ switching demo
  const demoBtn = document.getElementById("sw-demo-run"), demoOut = document.getElementById("sw-demo-out");
  const demoStatus = document.getElementById("sw-demo-status"), demoKind = document.getElementById("sw-demo-kind");
  let demoTimer = null;
  const secs = v => v == null ? "–" : `${num(v, 1)} s`;
  function renderDemo(d) {
    if (!d || d.status === "idle") return;
    const run = d.status === "running", a = d.a, b = d.b;
    demoBtn.disabled = run;
    demoBtn.textContent = run ? "Demo running…" : "Run Switching Demo";
    demoKind.innerHTML = d.kind ? `${badge(d.kind)}${d.kind === "simulated" ? " mock boxes: not GPU numbers" : ""}` : "";
    const cur = (d.phases || []).find(p => p.state === "running");
    demoStatus.innerHTML = run ? `<b>${esc(cur ? cur.title : "Starting")}</b> · ${num(d.elapsed_s)} s · router active box: <b style="color:${boxColor(d.active)}">${esc(d.active ?? "–")}</b>${d.phase === "cleanup" ? " · putting box A and the router policy back" : ""}`
      : d.status === "done" && d.policy_restored && !d.cleanup_error ? `Finished in ${num(d.elapsed_s)} s. Router is back on <b>${esc(d.policy_restored ? d.policy_restored.policy : "?")}</b>${d.saved_to ? ` · saved to <code>${esc(d.saved_to)}</code>` : ""}`
      : `<span class="tag downtag">failed</span> ${esc(d.error || "")}${d.cleanup_error ? ` · <b>not put back:</b> ${esc(d.cleanup_error)}` : d.policy_restored ? " · box A and the router policy were put back" : ""}`;
    if (!d.timeline) { demoOut.innerHTML = ""; return; }
    const now = d.now || {}, tm = d.timings || {};
    const stepHtml = d.phases.map((p, i) => `<div class="sw-step ${p.state}"><div class="n">${i + 1} · ${esc(p.state)}</div><div class="t">${esc(p.title)}</div>
      <div class="d">${p.requests == null ? (p.state === "running" ? "measuring…" : "–") : `A ${num(p.share_a_pct)}% · B ${num(p.share_b_pct)}%<br>p95 ${ms(p.p95_ms)} · ${num(p.errors)} errors`}</div></div>`).join("");
    const facts = [
      ["GPU A share, last 3 s", now.share_a_pct == null ? "–" : `${num(now.share_a_pct)}%`, "var(--a)"],
      ["GPU B share, last 3 s", now.share_b_pct == null ? "–" : `${num(now.share_b_pct)}%`, "var(--b)"],
      ["p50 / p95, last 3 s", `${ms(now.p50_ms)} / ${ms(now.p95_ms)}`],
      ["Requests / errors so far", `${num(d.timeline.reduce((n, p) => n + p.a + p.b, 0))} / ${num(d.timeline.reduce((n, p) => n + p.errors, 0))}`],
      ["Slowdown → switch to B", secs(tm.slow_to_switch_s)],
      ["Failure → switch to B", secs(tm.fail_to_switch_s)],
      ["Errors while A was failed", tm.errors_while_failed == null ? "–" : `${num(tm.errors_while_failed)} (${num(tm.failovers_while_failed)} retried on B)`],
      ["Restore → A serving again", secs(tm.restore_to_serving_s)]];
    const rows = d.phases.filter(p => p.requests != null).map(p => ({ label: p.title, requests: p.requests, error_pct: p.error_pct, p50_ms: p.p50_ms, p95_ms: p.p95_ms, p99_ms: p.p99_ms, rps: p.rps, share_a_pct: p.share_a_pct, share_b_pct: p.share_b_pct, failovers: p.failovers }));
    if (d.overall) rows.push({ label: "whole run", requests: d.overall.requests, error_pct: d.overall.error_pct, p50_ms: d.overall.p50_ms, p95_ms: d.overall.p95_ms, p99_ms: d.overall.p99_ms, rps: d.overall.rps, share_a_pct: d.overall.share_a_pct, share_b_pct: d.overall.share_b_pct, failovers: d.overall.failovers });
    const tl = d.timeline.length > 1 ? `<div class="sw-charts">${timeline({ policy: "adaptive, home " + a, timeline: d.timeline, events: d.marks })}${bars("p95 per second", d.timeline.filter((_, i) => i % Math.ceil(d.timeline.length / 12) === 0).map(p => ({ label: p.t + " s", p95_ms: p.p95_ms })), [{ key: "p95_ms", name: "p95", color: "var(--accent)" }], "ms")}</div>` : "";
    demoOut.innerHTML = `
      ${d.kind === "simulated" ? '<div class="sw-note" style="margin-top:10px"><b>SIMULATED.</b> These boxes are mock servers. The router and its decisions are real; the latencies are not GPU measurements.</div>' : ""}
      <div class="sw-steps" style="grid-template-columns:repeat(5,minmax(0,1fr))">${stepHtml}</div>
      <div class="sw-facts" style="grid-template-columns:repeat(4,minmax(0,1fr))">${facts.map(f => `<div class="sw-fact"><div class="l">${f[0]}</div><div class="v"${f[2] ? ` style="color:${f[2]}"` : ""}>${f[1]}</div></div>`).join("")}</div>
      ${tl}
      <div class="sw-last"><b>Switching events</b> <span class="muted">(seconds since the demo started; switches and health changes are the router's own)</span></div>
      <ol class="sw-list" style="max-height:170px">${d.events.slice().reverse().map(e => `<li><b>${num(e.t, 1)} s</b> · <span class="sw-ev-${esc(e.kind)}">${esc(e.kind)}</span> · ${esc(e.text)}</li>`).join("") || "<li>none yet</li>"}</ol>
      ${rows.length ? table(rows) : ""}
      <p class="sw-src">Demo traffic: ${num(d.traffic_rps)} forecasts/s through the router (${num(d.sent)} sent${d.unanswered ? `, ${num(d.unanswered)} got no answer from the router` : ""}) · slow mode ×${num(d.slow_factor)} · switch margin ${d.thresholds && d.thresholds.margin_during_demo != null ? num(100 * d.thresholds.margin_during_demo) + "% during the demo" : "–"} · ${Object.entries(d.boxes || {}).map(([n, x]) => `${esc(n)}: ${esc(x.device ?? x.kind)}${x.gpu_uuid ? " " + esc(String(x.gpu_uuid).slice(0, 12)) + "…" : ""}`).join(" · ")}</p>`;
  }
  async function pollDemo() {
    let d;
    try { d = await (await fetch("api/switching/demo")).json(); } catch (e) { return; }
    renderDemo(d);
    if (d.status === "running" && !demoTimer) demoTimer = setInterval(pollDemo, 1000);
    if (d.status !== "running" && demoTimer) { clearInterval(demoTimer); demoTimer = null; pollLive(); }
  }
  demoBtn.addEventListener("click", async () => {
    demoBtn.disabled = true;
    try { renderDemo(await post("api/switching/demo")); } catch (e) { demoStatus.textContent = "Could not start the demo: " + e; demoBtn.disabled = false; return; }
    pollDemo();
  });

  pollLive(); setInterval(pollLive, 2000); loadResults(); pollDemo();
})();

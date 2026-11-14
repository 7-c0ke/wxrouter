"""Turn benchmark JSON into one self-contained HTML report with charts and tables.

  python -m inference.report --bench results/inference/<run>/bench.json \\
      [--cluster results/cluster/<run>/cluster_bench.json] [--afno results/afno/<run>/afno_check.json] \\
      --out results/report.html

Standard library only. Every chart is inline SVG with a hover tooltip on each
mark and the same numbers in a table underneath. Sections whose input file is
not given are left out, so the report never shows a chart without data.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

# Two categorical slots, validated for colour-blind separation on both surfaces (dataviz palette).
CSS = """
:root { --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9;
        --axis:#c3c2b7; --s1:#2a78d6; --s2:#eb6834; --event:#898781; }
@media (prefers-color-scheme: dark) {
  :root { --surface:#1a1a19; --page:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a;
          --axis:#383835; --s1:#3987e5; --s2:#d95926; --event:#898781; } }
body { background:var(--page); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
       margin:0; padding:32px; max-width:980px; margin-inline:auto; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:40px 0 4px; } h3 { font-size:15px; margin:0 0 2px; }
p, li { color:var(--ink2); } p.lead { color:var(--ink); }
.tag { display:inline-block; font-size:12px; padding:1px 8px; border:1px solid var(--axis); border-radius:10px;
       color:var(--ink2); margin-left:6px; vertical-align:middle; }
figure { background:var(--surface); border:1px solid var(--grid); border-radius:8px; padding:16px; margin:16px 0; }
figcaption { color:var(--ink2); font-size:13px; margin-bottom:8px; }
svg { width:100%; height:auto; display:block; } svg text { fill:var(--ink2); font-size:12px; }
svg .val { fill:var(--ink); } svg .axis { stroke:var(--axis); } svg .grid { stroke:var(--grid); }
.legend { display:flex; gap:16px; font-size:13px; color:var(--ink2); margin:4px 0 8px; }
.legend i { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:16px 0; }
.tile { background:var(--surface); border:1px solid var(--grid); border-radius:8px; padding:12px 14px; }
.tile b { display:block; font-size:24px; color:var(--ink); } .tile span { font-size:13px; color:var(--ink2); }
table { border-collapse:collapse; width:100%; font-size:13px; margin-top:8px; }
th, td { text-align:right; padding:4px 8px; border-bottom:1px solid var(--grid); color:var(--ink2); }
th:first-child, td:first-child { text-align:left; } th { color:var(--ink); font-weight:600; }
details summary { cursor:pointer; color:var(--ink2); font-size:13px; }
"""

W, H, L, R, T, B = 720, 300, 56, 16, 16, 40


def esc(v) -> str:
    return html.escape(str(v))


def fmt(v: float) -> str:
    if v is None:
        return ""
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.1f}" if abs(v) >= 10 else f"{v:,.2f}"


def nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    mag = 10 ** (len(str(int(v))) - 1)
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * mag:
            return m * mag
    return 10 * mag


def frame(ymax: float, unit: str) -> list[str]:
    out = []
    for k in range(5):
        y = T + (H - T - B) * (1 - k / 4)
        out.append(f'<line class="{"axis" if k == 0 else "grid"}" x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end">{fmt(ymax * k / 4)}</text>')
    out.append(f'<text x="{L + 6}" y="{T + 4}" text-anchor="start">{esc(unit)}</text>')
    return out


def table(head: list[str], rows: list[list]) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<details><summary>Data table</summary><table><tr>{th}</tr>{body}</table></details>"


def legend(names: list[str]) -> str:
    if len(names) < 2:
        return ""
    return '<div class="legend">' + "".join(
        f'<span><i style="background:var(--s{i + 1})"></i>{esc(n)}</span>' for i, n in enumerate(names)) + "</div>"


def bars(title: str, caption: str, cats: list[str], series: dict[str, list[float]], unit: str, xlabel: str) -> str:
    """Grouped bars: one group per category, one bar per series, a value label on every bar."""
    names = list(series)
    ymax = nice_max(max(max(v) for v in series.values()))
    plot_w, plot_h = W - L - R, H - T - B
    group = plot_w / len(cats)
    bw = min(46, (group - 24) / len(names) - 2)
    svg = frame(ymax, unit)
    for ci, cat in enumerate(cats):
        x0 = L + group * ci + (group - (bw + 2) * len(names)) / 2
        for si, name in enumerate(names):
            v = series[name][ci]
            h = plot_h * v / ymax
            x, y = x0 + (bw + 2) * si, T + plot_h - h
            r = min(4, h)
            path = (f"M{x:.1f},{T + plot_h:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + bw - r:.1f} "
                    f"Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} V{T + plot_h:.1f} Z")
            svg.append(f'<path d="{path}" fill="var(--s{si + 1})"><title>{esc(name)}, {esc(cat)} {esc(xlabel)}: '
                       f'{fmt(v)} {esc(unit)}</title></path>')
            svg.append(f'<text class="val" x="{x + bw / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle">{fmt(v)}</text>')
        svg.append(f'<text x="{L + group * (ci + 0.5):.1f}" y="{H - B + 16}" text-anchor="middle">{esc(cat)}</text>')
    svg.append(f'<text x="{L + plot_w / 2:.1f}" y="{H - 6}" text-anchor="middle">{esc(xlabel)}</text>')
    rows = [[c] + [fmt(series[n][i]) for n in names] for i, c in enumerate(cats)]
    return (f"<figure><h3>{esc(title)}</h3><figcaption>{esc(caption)}</figcaption>{legend(names)}"
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">{"".join(svg)}</svg>'
            f"{table([xlabel] + [f'{n} ({unit})' for n in names], rows)}</figure>")


def lines(title: str, caption: str, xs: list[float], series: dict[str, list[float | None]], unit: str, xlabel: str,
          events: list[tuple[float, str]] = (), tick_labels: list[str] | None = None) -> str:
    """Lines over a numeric x axis, a marker with a tooltip at every point, optional event markers."""
    names = list(series)
    ymax = nice_max(max((v for s in series.values() for v in s if v is not None), default=1))
    xmin, xmax = min(xs), max(xs)
    plot_w, plot_h = W - L - R, H - T - B

    def px(x: float) -> float:
        return L + plot_w * (x - xmin) / max(xmax - xmin, 1e-9)

    def py(v: float) -> float:
        return T + plot_h * (1 - v / ymax)

    svg = frame(ymax, unit)
    for t, label in events:
        svg.append(f'<line x1="{px(t):.1f}" x2="{px(t):.1f}" y1="{T}" y2="{T + plot_h}" stroke="var(--event)" '
                   f'stroke-dasharray="3 3"><title>{esc(label)} at {t:g} s</title></line>')
        svg.append(f'<text x="{px(t) + 4:.1f}" y="{T + 12}" text-anchor="start">{esc(label)}</text>')
    for si, name in enumerate(names):
        pts = [(px(x), py(v), x, v) for x, v in zip(xs, series[name]) if v is not None]
        if not pts:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y, _, _) in enumerate(pts))
        svg.append(f'<path d="{d}" fill="none" stroke="var(--s{si + 1})" stroke-width="2"/>')
        for x, y, xv, v in pts:
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="transparent"><title>{esc(name)} at {esc(tick_labels[int(xv)]) if tick_labels else f"{xv:g}"} '
                       f'{esc(xlabel)}: {fmt(v)} {esc(unit)}</title></circle>')
            if len(pts) <= 12:
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var(--s{si + 1})" stroke="var(--surface)" '
                           f'stroke-width="2" pointer-events="none"/>')
        if len(names) > 1:
            x, y, _, _ = pts[-1]
            svg.append(f'<text class="val" x="{min(x + 6, W - R - 4):.1f}" y="{y - 8:.1f}" text-anchor="end">{esc(name)}</text>')
    ticks = xs if len(xs) <= 8 else [xmin + (xmax - xmin) * k / 6 for k in range(7)]
    for i, x in enumerate(ticks):
        label = tick_labels[i] if tick_labels else f"{x:g}"
        svg.append(f'<text x="{px(x):.1f}" y="{H - B + 16}" text-anchor="middle">{esc(label)}</text>')
    svg.append(f'<text x="{L + plot_w / 2:.1f}" y="{H - 6}" text-anchor="middle">{esc(xlabel)}</text>')
    rows = [[tick_labels[i] if tick_labels else f"{x:g}"] + [fmt(series[n][i]) for n in names] for i, x in enumerate(xs)]
    return (f"<figure><h3>{esc(title)}</h3><figcaption>{esc(caption)}</figcaption>{legend(names)}"
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">{"".join(svg)}</svg>'
            f"{table([xlabel] + [f'{n} ({unit})' for n in names], rows)}</figure>")


def tiles(items: list[tuple[str, str]]) -> str:
    return '<div class="tiles">' + "".join(f'<div class="tile"><b>{esc(v)}</b><span>{esc(k)}</span></div>'
                                           for k, v in items) + "</div>"


def kind(env: dict) -> str:
    return "real GPU" if env.get("device") == "cuda" else "CPU only"


def bench_section(b: dict) -> str:
    env, cold = b["environment"], b["cold_start"]
    serving = b["serving"]
    clients = sorted({r["clients"] for r in serving})

    def pick(mode: str, key) -> list[float]:
        return [key(next(r for r in serving if r["clients"] == c and r["mode"] == mode)) for c in clients]

    top = max(clients)
    base = next(r for r in serving if r["clients"] == top and r["mode"] == "baseline")
    bat = next(r for r in serving if r["clients"] == top and r["mode"] == "batched")
    out = [f'<h2>One GPU: batching <span class="tag">{kind(env)}</span></h2>',
           (f"<p>{esc(env['device_name'])}, {esc(env['backend'])} {esc(env.get('torch', ''))}, {esc(env['precision'])}, "
           f"model <code>{esc(env['model_version'])}</code> (a seeded stand-in, not a trained weather model). "
           f"{b['params']['hours_ahead']} h forecasts. Run at {esc(b['created_at'])}.</p>"),
           tiles([(f"req/s at {top} clients, batched", fmt(bat["requests_per_s"])),
                  (f"req/s at {top} clients, no batching", fmt(base["requests_per_s"])),
                  ("p95 latency, batched", f"{fmt(bat['latency_ms']['p95'])} ms"),
                  ("p95 latency, no batching", f"{fmt(base['latency_ms']['p95'])} ms"),
                  ("cold start: load + first inference", f"{fmt(cold['load_ms'] + cold['first_inference_ms'])} ms"),
                  ("warm inference (12 h, 1 row)", f"{fmt(cold['warm_inference_ms'])} ms")]),
           bars("Throughput by number of clients", "Requests completed per second. Higher is better.",
                [str(c) for c in clients], {"no batching": pick("baseline", lambda r: r["requests_per_s"]),
                                            "batched": pick("batched", lambda r: r["requests_per_s"])},
                "req/s", "concurrent clients"),
           bars("p95 latency by number of clients", "What the slowest 5% of requests waited. Lower is better.",
                [str(c) for c in clients], {"no batching": pick("baseline", lambda r: r["latency_ms"]["p95"]),
                                            "batched": pick("batched", lambda r: r["latency_ms"]["p95"])},
                "ms", "concurrent clients"),
           lines("Rows per second by batch size", "One model call per batch; a row is one ensemble member.",
                 list(range(len(b["batch_sweep"]))), {"rows/s": [r["rows_per_s"] for r in b["batch_sweep"]]},
                 "rows/s", "rows per call", tick_labels=[str(r["batch_rows"]) for r in b["batch_sweep"]])]
    if b.get("precision"):
        rows = [[r["precision"], r["batch_rows"], fmt(r["latency_ms"]["p50"]), fmt(r["rows_per_s"]),
                 r["gpu_memory"].get("peak_allocated_mb", ""), r.get("max_abs_diff_vs_fp32", ""),
                 r.get("numerically_valid", "")] for r in b["precision"]]
        th = "".join(f"<th>{h}</th>" for h in ["precision", "rows", "p50 ms", "rows/s", "peak GPU MB",
                                                "max diff vs fp32", "valid"])
        out.append("<figure><h3>fp32 vs fp16 autocast</h3><figcaption>Outputs are in [-1, 1]; fp16 counts as valid only "
                   f"within 0.05 of fp32.</figcaption><table><tr>{th}</tr>"
                   + "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
                   + "</table></figure>")
    return "".join(out)


def cluster_section(c: dict) -> str:
    place = c["placement"]
    cuda = all((p.get("device") or {}).get("type") == "cuda" for p in place.values())
    out = [f'<h2>Two GPUs: routing policies <span class="tag">{"real GPU" if cuda else "CPU only"}</span></h2><ul>']
    for name, p in place.items():
        node, dev = p.get("node") or {}, p.get("device") or {}
        out.append(f"<li><b>{esc(name)}</b>: {esc(dev.get('name'))}, node {esc((node.get('addresses') or ['?'])[0])}, "
                   f"GPU {esc(node.get('gpu_uuid'))}</li>")
    out.append(f"<li>On different GPU nodes: <b>{esc(c['distinct_nodes'])}</b></li></ul>")
    cal = c.get("calibration")
    if cal:
        out.append(f"<p>Slow mode is a real second load on the box's GPU. On an idle box one 48 h forecast went from "
                   f"{cal['compute_ms_normal']} ms to {cal['compute_ms_slow']} ms ({cal['measured_slowdown']}x).</p>")
    cases = c["cases"]
    [x["label"] for x in cases]
    head = ["case", "requests", "errors", "p50 ms", "p95 ms", "p99 ms", "ok/s", "box A %", "box B %", "failovers", "switches"]
    rows = [[x["label"], x["summary"]["requests"], x["summary"]["errors"], x["summary"]["p50_ms"], x["summary"]["p95_ms"],
             x["summary"]["p99_ms"], x["summary"]["ok_per_s"], x["summary"]["share_a_pct"], x["summary"]["share_b_pct"],
             x["summary"]["failovers"], len(x["switches"])] for x in cases]
    th = "".join(f"<th>{h}</th>" for h in head)
    out.append(f"<figure><h3>Every case, same seeded trace ({c['params']['rps']} req/s offered)</h3><table><tr>{th}</tr>"
               + "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in r) + "</tr>" for r in rows) + "</table></figure>")
    for prefix, title in (("single_gpu", "one GPU"), ("two_gpu", "two GPUs, no faults"), ("fault", "two GPUs with faults")):
        group = [x for x in cases if x["label"].startswith(prefix)]
        if not group:
            continue
        names = [x["label"].replace(prefix + "_", "") for x in group]
        out.append(bars(f"Latency, {title}", "p50 and p95 of successful requests, as the client saw them.",
                        names, {"p50": [x["summary"]["p50_ms"] for x in group],
                                "p95": [x["summary"]["p95_ms"] for x in group]}, "ms", "case"))
        if prefix != "single_gpu":
            out.append(bars(f"Where requests went, {title}", "Share of successful requests served by each box.",
                            names, {"box A": [x["summary"]["share_a_pct"] for x in group],
                                    "box B": [x["summary"]["share_b_pct"] for x in group]}, "%", "case"))
    for x in cases:
        if not x["label"].startswith("fault"):
            continue
        tl = x["timeline"]
        xs = [s["t"] for s in tl]
        ev = [(e["t"], f"{e['action']} {e['box']}") for e in x["events"]]
        out.append(lines(f"Failure recovery timeline: {x['policy']}",
                         "Requests served per second by each box. Dashed lines are the faults.", xs,
                         {"box A": [s["served_by"].get("box-a", 0) for s in tl],
                          "box B": [s["served_by"].get("box-b", 0) for s in tl]}, "req/s", "seconds", ev))
        out.append(lines(f"p95 latency over time: {x['policy']}", "Per second of send time.", xs,
                         {"p95": [s["p95_ms"] for s in tl]}, "ms", "seconds", ev))
        fr = x.get("fault_report") or {}
        notes = []
        if "slow" in fr:
            f = fr["slow"]
            notes.append(f"While {f['box']} was slowed its share went from {f['share_before_pct']}% to "
                         f"{f['share_during_pct']}%, with {f['errors_during']} errors.")
        if "stop" in fr:
            f = fr["stop"]
            notes.append(f"While {f['box']} was stopped there were {f['errors_while_down']} errors; it served again "
                         f"{f['seconds_to_serve_again']} s after its restart. {len(x['switches'])} switches in total.")
        if notes:
            out.append("<p>" + " ".join(esc(n) for n in notes) + "</p>")
    return "".join(out)


def afno_section(a: dict) -> str:
    c = a["compare"]
    out = ['<h2>The trained weather model (AFNO) <span class="tag">real GPU</span></h2>',
           (f"<p>{esc(a['gpu'])}. The team's server and the engine loaded the same weights "
           f"(<code>{esc(a['servers'][0]['model_version'])}</code>). {c['points_compared']} forecast points compared: "
           f"largest difference {c['max_abs_diff']['temp_c']} C, {c['max_abs_diff']['wind_ms']} m/s, "
           f"{c['max_abs_diff']['pressure_hpa']} hPa. Agree: <b>{c['agree']}</b>.</p>")]
    clients = [r["clients"] for r in a["servers"][0]["load"]]
    by = {s["server"]: s["load"] for s in a["servers"]}
    names = {"team": "team server", "engine": "engine (batched)"}
    out.append(bars("AFNO throughput by number of clients", "One 48 h forecast per request.", [str(k) for k in clients],
                    {names[k]: [r["requests_per_s"] for r in v] for k, v in by.items()}, "req/s", "concurrent clients"))
    out.append(bars("AFNO p95 latency by number of clients", "Lower is better.", [str(k) for k in clients],
                    {names[k]: [r["p95_ms"] for r in v] for k, v in by.items()}, "ms", "concurrent clients"))
    return "".join(out)


def build(bench: list[dict], cluster: dict | None, afno: dict | None) -> str:
    body = ["<h1>wxrouter inference report</h1>",
            '<p class="lead">Every number below was measured. Each section says whether it ran on a real GPU or on a CPU.</p>']
    body += [bench_section(b) for b in bench]
    if cluster:
        body.append(cluster_section(cluster))
    if afno:
        body.append(afno_section(afno))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>wxrouter inference report</title><style>{CSS}</style>'
            f'</head><body>{"".join(body)}</body></html>')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", action="append", default=[], help="bench.json; repeat for several runs")
    ap.add_argument("--cluster", help="cluster_bench.json")
    ap.add_argument("--afno", help="afno_check.json")
    ap.add_argument("--out", default="results/report.html")
    a = ap.parse_args()

    def load(p):
        return json.loads(Path(p).read_text()) if p else None

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(build([load(p) for p in a.bench], load(a.cluster), load(a.afno)))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

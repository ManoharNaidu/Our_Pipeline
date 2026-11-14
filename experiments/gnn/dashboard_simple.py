"""Simple results dashboard for the GNN experiment (stdlib only).

Put it next to dashboard.py (experiments/gnn/). It imports helpers from that file and never changes it.

Usage:
    python experiments/gnn/dashboard_simple.py --check                      # verify every path it needs, then exit
    python experiments/gnn/dashboard_simple.py [--host 127.0.0.1] [--port 8766] [--root /workspace/Our_Pipeline]
                                               [--skip reddit] [--full-url http://...] [--banner "SAMPLE DATA"]

Pages: `/` (simple view, auto-refreshes every 15 s), `/api/state` (same data as JSON), `/decisions` (decisions log as
text), `/full` (the complete technical dashboard from dashboard.py, unchanged) and `/api/full-state` (its JSON).

This file is the entry point: it replaces `python dashboard.py` in start_dashboard.sh. It starts the same background
sampler thread (GPU history, summary-store growth) that dashboard.py's own main() starts, so `/full` keeps working.

What it shows, nothing else:
  1. one-sentence verdict per dataset (from the test-set numbers and, if present, the significance file)
  2. what the CLI is waiting on (only when progress.json has a `needs_you` block)
  3. phase strip and GPU idle/busy
  4. one results table per dataset: DGP vs Ours K=1 vs the PMP arm, with differences
  5. per-seed numbers, collapsed

Every number comes from files on disk. A missing file shows as "not found", never as a guess.

Significance (CI, Holm p) is NOT computed here. It is read from experiments/gnn/runs/compare/significance.json, written
from scripts/compare_arms.py output, in this shape:
    {"comparisons": [{"dataset": "instagram", "arm": "instagram_pmp_l1", "baseline": "ours_k1" | "dgp",
                      "metric": "auroc" | "macro_f1" | "auprc", "delta": -0.29, "ci_low": -1.49, "ci_high": 0.90,
                      "p_holm": 0.89}, ...]}
Without that file the tables still show every mean and difference, with "p not loaded".

progress.json may carry an optional block that the CLI writes when it stops at a gate and removes afterwards:
    "needs_you": {"title": "...", "facts": [{"k": "Cost", "v": "..."}], "reply": "Tell the CLI: ..."}
"""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
import sys
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import dashboard as D  # the existing dashboard in the same folder; only read, never edited

METRICS = (("macro_f1", "Macro-F1"), ("auroc", "AUROC"), ("auprc", "AUPRC"))
PRIMARY = "auroc"
ALPHA = 0.05
SKIP: set[str] = {"reddit"}
FULL_URL: str | None = None
BANNER: str | None = None
BAD_STATES = {"failed", "gave_up", "stalled"}
TAIL = {"better": "significantly better", "worse": "significantly worse", "same": "no clear difference",
        "unknown": "significance not loaded"}
e = html.escape


# ------------------------------------------------------------------ data
def short_model(cfg: dict) -> str:
    m = (cfg.get("classifier") or {}).get("model_id") or (cfg.get("summarizer") or {}).get("model_id") or ""
    return str(m).split("/")[-1]


def method_info(path: Path) -> dict:
    """Config facts for one method: hash, results folder, seeds, K and model. Never raises."""
    info = {"path": D.rel(path), "exists": path.exists()}
    try:
        cfg = D.load_config(path)
        info.update(dataset=cfg["dataset"]["name"], hash=D.config_hash(cfg), seeds=list(cfg.get("seeds") or []),
                    results=(cfg.get("paths") or {}).get("results", "runs/results"),
                    K=(cfg.get("method") or cfg.get("dgp") or {}).get("K"), model=short_model(cfg))
    except Exception as exc:
        info["error"] = repr(exc)
    return info


# Original DGP runs were made by the DGP-Reproduce code, so their hash is not this repo's config hash. Recipe (K, model)
# comes from configs/datasets/<ds>.yaml (checked identical); runs are the copies in experiments/gnn/runs/dgp_reference.
DGP_REFERENCE = {"instagram": "9dcd4dad8e"}


def dgp_info_for(ds: str) -> dict:
    info = method_info(D.ROOT / "configs" / "datasets" / f"{ds}.yaml")
    if ds in DGP_REFERENCE:
        info.update(hash=DGP_REFERENCE[ds], results="experiments/gnn/runs/dgp_reference")
    return info


def read_runs(info: dict, ds: str, seeds) -> dict:
    """{seed: {metric: test value}} for the finished runs of one method."""
    out: dict = {}
    if not info.get("hash"):
        return out
    for s in seeds:
        m = D.read_json(D.ROOT / info["results"] / "raw" / f"{ds}_seed{s}_{info['hash']}" / "metrics.json")
        t = m.get("test") if isinstance(m, dict) else None
        if isinstance(t, dict):
            vals = {k: D.num(t.get(k)) for k, _ in METRICS}
            if any(v is not None for v in vals.values()):
                out[s] = vals
    return out


def values(runs: dict, seeds, metric: str) -> list:
    return [runs[s][metric] for s in seeds if s in runs and runs[s].get(metric) is not None]


def mean_of(runs: dict, seeds, metric: str):
    v = values(runs, seeds, metric)
    return statistics.mean(v) if v else None


def std_of(runs: dict, seeds, metric: str):
    v = values(runs, seeds, metric)
    return statistics.pstdev(v) if len(v) > 1 else None


def arm_label(stem: str, ds: str) -> str:
    name = stem[len(ds) + 1:] if stem.startswith(ds + "_") else stem
    return name.replace("_", "-").upper()


def load_sig() -> list:
    d = D.read_json(D.gnn_dir() / "runs" / "compare" / "significance.json")
    rows = d.get("comparisons") if isinstance(d, dict) else None
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def sig_for(sig: list, stem: str, base: str, metric: str):
    for r in sig:
        if r.get("arm") == stem and r.get("baseline") == base and r.get("metric") == metric:
            return r
    return None


def collect_dataset(c: dict, sig: list) -> dict:
    ds, stem = c["dataset"], c["stem"]
    label = arm_label(stem, ds)
    arm_runs = read_runs({"hash": c["hash"], "results": c.get("results", "runs/results")}, ds, c.get("seeds") or [])
    seeds = sorted(arm_runs)
    ours_info = method_info(D.ROOT / c["baseline_path"]) if c.get("baseline_path") else {"error": "no baseline path"}
    dgp_info = dgp_info_for(ds)
    bases = {"ours_k1": (ours_info, read_runs(ours_info, ds, seeds)), "dgp": (dgp_info, read_runs(dgp_info, ds, seeds))}

    def sub(info: dict) -> str:
        bits = []
        if info.get("K") is not None:
            bits.append(f"K={info['K']}")
        if info.get("model"):
            bits.append(info["model"])
        return " · ".join(bits)

    methods = []
    for key, name in (("dgp", "DGP (original)"), ("ours_k1", "Ours K=1")):
        info, runs = bases[key]
        common = [s for s in seeds if s in runs]
        methods.append({"key": key, "name": name, "sub": sub(info), "n": len(common), "seeds": common,
                        "error": info.get("error"), "missing_file": not info.get("exists", True),
                        "mean": {m: mean_of(runs, common, m) for m, _ in METRICS},
                        "std": {m: std_of(runs, common, m) for m, _ in METRICS}, "runs": runs})
    methods.append({"key": "arm", "name": f"Ours + {label}", "sub": "cosine on GNN embeddings",
                    "n": len(seeds), "seeds": seeds,
                    "mean": {m: mean_of(arm_runs, seeds, m) for m, _ in METRICS},
                    "std": {m: std_of(arm_runs, seeds, m) for m, _ in METRICS}, "runs": arm_runs})
    diffs = {}
    for key in ("ours_k1", "dgp"):
        runs = bases[key][1]
        common = [s for s in seeds if s in runs]
        row = {}
        for m, _ in METRICS:
            a, b = mean_of(arm_runs, common, m), mean_of(runs, common, m)
            row[m] = {"delta": (a - b) if a is not None and b is not None else None,
                      "sig": sig_for(sig, stem, key, m)}
        diffs[key] = row
    return {"dataset": ds, "stem": stem, "label": label, "n_done": len(seeds), "n_planned": len(c.get("seeds") or []),
            "methods": methods, "diffs": diffs}


def verdict_word(delta, s):
    """('better'|'worse'|'same'|'unknown', p) from the significance record."""
    p = D.num((s or {}).get("p_holm"))
    if delta is None or p is None:
        return "unknown", p
    if p < ALPHA:
        return ("better" if delta > 0 else "worse"), p
    return "same", p


def headline(b: dict) -> dict:
    ds, label, n = b["dataset"].capitalize(), b["label"], b["n_done"]
    if not n:
        return {"tone": "muted", "title": f"{ds}: no finished runs yet.", "text": ""}
    d1 = b["diffs"]["ours_k1"]
    word, _ = verdict_word(d1[PRIMARY]["delta"], d1[PRIMARY]["sig"])
    title = {"better": f"{label} beat Ours K=1 on {ds}.", "worse": f"{label} was worse than Ours K=1 on {ds}.",
             "same": f"{label} did not beat Ours K=1 on {ds}.",
             "unknown": f"{label} vs Ours K=1 on {ds}: significance not loaded."}[word]
    tone = {"better": "good", "worse": "bad", "same": "warn", "unknown": "muted"}[word]
    by = {m["key"]: m for m in b["methods"]}
    parts = []
    for m, name in (("auroc", "AUROC"), ("macro_f1", "Macro-F1")):
        a, base = by["arm"]["mean"][m], by["ours_k1"]["mean"][m]
        if a is None or base is None:
            continue
        w, _ = verdict_word(d1[m]["delta"], d1[m]["sig"])
        parts.append(f"{name} <b>{a:.2f}</b> vs {base:.2f}: {TAIL[w]}")
    dg = b["diffs"]["dgp"][PRIMARY]
    if dg["delta"] is not None:
        w, _ = verdict_word(dg["delta"], dg["sig"])
        parts.append(f"Against original DGP, AUROC {fnum(dg['delta'], True)}: {TAIL[w]}")
    note = "" if n >= b["n_planned"] else f" ({n} of {b['n_planned']} seeds finished)"
    return {"tone": tone, "title": title, "text": (". ".join(parts) + "." + note) if parts else ""}


def short_name(name) -> str:
    s = re.split(r"[(,:]| and ", str(name or ""))[0].strip()
    return s if len(s) <= 24 else s[:22].rstrip() + "…"


def get_status(progress) -> dict:
    phases = [p for p in (progress or {}).get("phases", []) if isinstance(p, dict)]
    needs = (progress or {}).get("needs_you")
    bad = [p for p in phases if str(p.get("status")) in BAD_STATES]
    waiting = [p for p in phases if str(p.get("status")) == "waiting_gate"]
    running = [p for p in phases if str(p.get("status")) == "running"]
    if not progress:
        chip = ("muted", "No progress file")
    elif bad:
        chip = ("bad", "Needs attention")
    elif isinstance(needs, dict) or waiting:
        chip = ("warn", "Waiting for you")
    elif running:
        chip = ("info", f"Running: {short_name(running[0].get('name'))}")
    elif phases and all(str(p.get("status")) in ("done", "skipped") for p in phases):
        chip = ("good", "Finished")
    else:
        chip = ("muted", "Idle")
    return {"chip": chip, "phases": phases, "bad": bad, "needs": needs if isinstance(needs, dict) else None}


def gpu_line() -> str:
    try:
        with D.LOCK:
            last = D.GPU_HISTORY[-1] if D.GPU_HISTORY else None
        if not last or time.time() - last.get("t", 0) > 60:  # sampler not running or stale: ask nvidia-smi directly
            last = D.sample_gpu()
        g = ((last or {}).get("gpus") or [None])[0]
    except Exception:
        g = None
    if not g:
        return "GPU: not available"
    used, total, util = (g.get("mem_used") or 0) / 1024, (g.get("mem_total") or 0) / 1024, g.get("util") or 0
    state = "idle" if util < 5 and used < 1 else f"busy {util:.0f}%"
    return f"GPU {state} · {used:.0f} / {total:.0f} GB"


def build_state() -> dict:
    sig = load_sig()
    arms = [c for c in D.arms() if c.get("dataset")]
    progress = D.get_progress()
    return {"generated": datetime.now(UTC).strftime("%H:%M UTC"), "status": get_status(progress), "gpu": gpu_line(),
            "datasets": [collect_dataset(c, sig) for c in arms if c["dataset"] not in SKIP],
            "skipped": sorted({c["dataset"] for c in arms if c["dataset"] in SKIP}), "sig_loaded": bool(sig),
            "updated": (progress or {}).get("updated")}


# ------------------------------------------------------------------ rendering
def fnum(v, signed: bool = False) -> str:
    if v is None:
        return "—"
    return (f"{v:+.2f}" if signed else f"{v:.2f}").replace("-", "−")


def fp(p: float) -> str:
    return f"p {p:.3f}" if p < 0.1 else f"p {p:.2f}"


def delta_cell(rec: dict) -> str:
    dv, s = rec["delta"], rec["sig"]
    if dv is None:
        return '<td class="r muted">—</td>'
    word, p = verdict_word(dv, s)
    cls = {"better": "pos", "worse": "neg"}.get(word, "")
    subs = [fp(p) if p is not None else "p not loaded"]
    lo, hi = D.num((s or {}).get("ci_low")), D.num((s or {}).get("ci_high"))
    if lo is not None and hi is not None:
        subs.append(f"CI {fnum(lo, True)} to {fnum(hi, True)}")
    return (f'<td class="r"><span class="dv {cls}">{fnum(dv, True)}</span>'
            + "".join(f'<span class="sub">{e(x)}</span>' for x in subs) + "</td>")


def results_table(b: dict) -> str:
    best = {}
    for m, _ in METRICS:
        vals = [x["mean"][m] for x in b["methods"] if x["mean"][m] is not None]
        best[m] = max(vals) if vals else None
    rows = []
    for x in b["methods"]:
        is_arm = x["key"] == "arm"
        if x["n"] == 0:
            why = "config not found" if x.get("missing_file") else (x.get("error") or "no finished runs for these seeds")
            rows.append(f'<tr><td><b>{e(x["name"])}</b><span class="sub">{e(x["sub"])}</span></td>'
                        f'<td colspan="3" class="muted">not found · {e(str(why))}</td></tr>')
            continue
        cells = []
        for m, _ in METRICS:
            v, sd = x["mean"][m], x["std"][m]
            bold = v is not None and best[m] is not None and abs(v - best[m]) < 1e-9
            sdt = f'<span class="sub">± {sd:.2f}</span>' if sd is not None else ""
            cells.append(f'<td class="r">{"<b>" if bold else ""}{fnum(v)}{"</b>" if bold else ""}{sdt}</td>')
        sub = x["sub"] if is_arm else f"{x['sub']} · {x['n']} seed(s)"
        rows.append(f'<tr class="{"me" if is_arm else ""}"><td><b>{e(x["name"])}</b><span class="sub">{e(sub)}</span>'
                    "</td>" + "".join(cells) + "</tr>")
    for key, name in (("ours_k1", "Ours K=1"), ("dgp", "original DGP")):
        rows.append(f'<tr class="dif"><td>{e(b["label"])} minus {e(name)}</td>'
                    + "".join(delta_cell(b["diffs"][key][m]) for m, _ in METRICS) + "</tr>")
    head = "<th>Method</th>" + "".join(f'<th class="r">{e(n)}</th>' for _, n in METRICS)
    return f'<div class="tw"><table class="res"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def per_seed_table(b: dict) -> str:
    by = {m["key"]: m for m in b["methods"]}
    seeds = by["arm"]["seeds"]
    if not seeds:
        return ""
    cols = [("arm", b["label"]), ("ours_k1", "Ours"), ("dgp", "DGP")]
    h1 = '<th rowspan="2">Seed</th>' + "".join(f'<th colspan="3" class="c">{e(n)}</th>' for _, n in METRICS)
    h2 = "".join(f'<th class="r">{e(c)}</th>' for _ in METRICS for _, c in cols)
    body = []
    for s in seeds:
        cells = "".join(f'<td class="r">{fnum((by[key]["runs"].get(s) or {}).get(m))}</td>'
                        for m, _ in METRICS for key, _c in cols)
        body.append(f"<tr><td>{s}</td>{cells}</tr>")
    return (f'<details data-k="seeds:{e(b["stem"])}"><summary>Per-seed numbers</summary><div class="tw">'
            f'<table class="res seeds"><thead><tr>{h1}</tr><tr>{h2}</tr></thead><tbody>{"".join(body)}</tbody></table>'
            "</div></details>")


def render_main(st: dict) -> str:
    out = []
    tone, label = st["status"]["chip"]
    banner = f'<div class="banner">{e(BANNER)}</div>' if BANNER else ""
    links = [f'<a href="{e(FULL_URL)}">Full technical dashboard</a>'] if FULL_URL else []
    links += ['<a href="/decisions">Decisions log</a>', '<a href="/api/state">Raw data (JSON)</a>']
    out.append(banner + f'<header><h1>GNN Exp1</h1><span class="chip {tone}"><i></i>{e(label)}</span>'
               f'<span class="stamp">Updated {e(st["generated"])} · refreshes every 15 s</span></header>')
    for p in st["status"]["bad"]:
        out.append(f'<section class="card alert"><p class="label">Needs attention · {e(short_name(p.get("name")))}'
                   f'</p><p>{e(str(p.get("note", "")))}</p></section>')
    for b in st["datasets"]:
        h = headline(b)
        out.append(f'<section class="card verdict {h["tone"]}"><p class="label">Answer so far</p>'
                   f'<h2>{e(h["title"])}</h2><p>{h["text"]}</p></section>')
    n = st["status"]["needs"]
    if n:
        facts = "".join(f'<li><span class="k">{e(str(f.get("k", "")))}</span><span>{e(str(f.get("v", "")))}</span></li>'
                        for f in (n.get("facts") or []) if isinstance(f, dict))
        reply = f'<div class="reply">{e(str(n["reply"]))}</div>' if n.get("reply") else ""
        out.append(f'<section class="card decision"><p class="label">Decision needed</p>'
                   f'<h2>{e(str(n.get("title", "")))}</h2><ul class="facts">{facts}</ul>{reply}</section>')
    steps = []
    for ph in st["status"]["phases"]:
        s = str(ph.get("status"))
        cls = ("done" if s in ("done", "skipped") else "now" if s == "running" else "gate" if s == "waiting_gate"
               else "bad" if s in BAD_STATES else "")
        mark = {"done": "✓", "bad": "!", "gate": "!", "now": "…"}.get(cls) or e(str(ph.get("id", "")))
        steps.append(f'<li class="step {cls}" title="{e(str(ph.get("note", "")))}"><span class="dot">{mark}</span>'
                     f'<span class="n">{e(short_name(ph.get("name")))}</span></li>')
    if steps:
        out.append(f'<section class="card"><p class="label">Progress</p><ol class="steps">{"".join(steps)}</ol>'
                   f'<p class="run">{e(st["gpu"])}</p></section>')
    for b in st["datasets"]:
        seeds_txt = f'{b["n_done"]} of {b["n_planned"]} seeds' if b["n_done"] < b["n_planned"] \
            else f'mean of {b["n_done"]} seeds'
        sig = "" if st["sig_loaded"] else ' <span class="warnchip">p-values not loaded yet</span>'
        body = results_table(b) if b["n_done"] else '<p class="muted">No finished runs yet.</p>'
        out.append(f'<section class="card"><p class="label">{e(b["dataset"].capitalize())} · test set · {seeds_txt}{sig}'
                   f'</p>{body}{per_seed_table(b)}<p class="note">Bold = best in column. p-values are '
                   "Holm-corrected, from scripts/compare_arms.py. Differences use the same seeds on both sides.</p>"
                   "</section>")
    for ds in st["skipped"]:
        out.append(f'<section class="card"><p class="label">{e(ds.capitalize())}</p>'
                   '<p class="muted">Skipped by decision. Not run.</p></section>')
    out.append(f'<section class="card"><p class="label">More</p><div class="more">{"".join(links)}</div></section>')
    return "".join(out)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>GNN Exp1</title><style>
:root{--bg:#111418;--card:#181c22;--text:#e8eaed;--muted:#9aa3ad;--line:#2a3038;--good:#3fb950;--bad:#f85149;
--warn:#d29922;--info:#58a6ff;--good-bg:rgba(63,185,80,.14);--bad-bg:rgba(248,81,73,.14);--warn-bg:rgba(210,153,34,.16);
--info-bg:rgba(88,166,255,.14);--row:rgba(88,166,255,.08)}
@media (prefers-color-scheme:light){:root{--bg:#f6f7f9;--card:#fff;--text:#1a1d21;--muted:#5b6570;--line:#e1e5ea;
--good:#1a7f37;--bad:#cf222e;--warn:#9a6700;--info:#0969da;--good-bg:rgba(26,127,55,.10);--bad-bg:rgba(207,34,46,.09);
--warn-bg:rgba(154,103,0,.12);--info-bg:rgba(9,105,218,.09);--row:rgba(9,105,218,.06)}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:880px;margin:0 auto;padding:20px 16px 48px}
header{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;margin-bottom:16px}
h1{font-size:18px;margin:0;font-weight:650}
.chip{display:inline-flex;align-items:center;gap:8px;font-size:14px;padding:3px 10px;border-radius:999px;font-weight:600;
background:var(--line);color:var(--muted)}.chip i{width:8px;height:8px;border-radius:50%;background:currentColor;display:block}
.chip.warn{background:var(--warn-bg);color:var(--warn)}.chip.good{background:var(--good-bg);color:var(--good)}
.chip.bad{background:var(--bad-bg);color:var(--bad)}.chip.info{background:var(--info-bg);color:var(--info)}
.stamp{margin-left:auto;font-size:13px;color:var(--muted)}
.banner{background:var(--warn-bg);color:var(--warn);border-radius:8px;padding:8px 12px;margin-bottom:14px;font-size:14px;
font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin-bottom:14px;min-width:0}
.label{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600;margin:0 0 8px}
.verdict{border-left:4px solid var(--muted)}.verdict.bad{border-left-color:var(--bad)}
.verdict.good{border-left-color:var(--good)}.verdict.warn{border-left-color:var(--warn)}
.verdict h2{font-size:24px;line-height:1.25;margin:0 0 6px;font-weight:650}
.verdict p:not(.label){margin:0;color:var(--muted)}.verdict b{color:var(--text);font-weight:600}
.decision{border-left:4px solid var(--warn)}.decision h2{font-size:19px;margin:0 0 12px;font-weight:650}
.alert{border-left:4px solid var(--bad)}.alert p:not(.label){margin:0}
.facts{list-style:none;margin:0 0 14px;padding:0;display:grid;gap:10px}
.facts li{display:grid;grid-template-columns:120px 1fr;gap:12px}.facts .k{color:var(--muted);font-size:14px}
.reply{background:var(--info-bg);border-radius:8px;padding:10px 12px;font-size:14px}
.steps{list-style:none;margin:0;padding:0;display:grid;grid-auto-flow:column;grid-auto-columns:1fr;gap:8px}
.step{text-align:center;font-size:13px;min-width:0}
.dot{width:28px;height:28px;margin:0 auto 6px;border-radius:50%;display:grid;place-items:center;font-size:14px;
font-weight:700;border:2px solid var(--line);color:var(--muted)}
.step.done .dot{background:var(--good-bg);border-color:var(--good);color:var(--good)}
.step.now .dot{background:var(--info-bg);border-color:var(--info);color:var(--info)}
.step.gate .dot{background:var(--warn-bg);border-color:var(--warn);color:var(--warn)}
.step.bad .dot{background:var(--bad-bg);border-color:var(--bad);color:var(--bad)}
.n{font-weight:600;display:block;overflow-wrap:break-word}
.run{margin:14px 0 0;padding-top:12px;border-top:1px solid var(--line);font-size:14px;color:var(--muted)}
.tw{overflow-x:auto;max-width:100%}
table.res{width:100%;border-collapse:collapse;font-size:15px;font-variant-numeric:tabular-nums}
.res th{text-align:left;font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);font-weight:600;
padding:0 10px 8px 0;white-space:nowrap}
.res th.r{text-align:right}.res th.c{text-align:center;border-bottom:1px solid var(--line)}
.res td{padding:10px 10px 10px 0;border-top:1px solid var(--line);vertical-align:top}
.res tr.me td{background:var(--row)}.res tr.me td:first-child{padding-left:8px;border-radius:6px 0 0 6px}
.res tr.dif td{font-size:14px}.res tr.dif td:first-child{color:var(--muted)}
.r{text-align:right}.sub{display:block;font-size:12px;color:var(--muted);font-weight:400}
.dv{font-weight:600}.dv.pos{color:var(--good)}.dv.neg{color:var(--bad)}
.muted{color:var(--muted)}.note{margin:12px 0 0;font-size:12.5px;color:var(--muted)}
.warnchip{background:var(--warn-bg);color:var(--warn);border-radius:999px;padding:1px 8px;text-transform:none;
letter-spacing:0;margin-left:6px}
details{margin-top:12px}summary{cursor:pointer;font-size:14px;color:var(--info);padding:4px 0}
.seeds td,.seeds th{font-size:13px;padding-right:8px}
.more{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:14px}.more a{color:var(--info);text-decoration:none}
@media (max-width:560px){.stamp{margin-left:0;width:100%}.facts li{grid-template-columns:1fr;gap:2px}
.verdict h2{font-size:21px}.card{padding:16px}table.res{font-size:14px}.res td{padding-right:6px}
.steps{grid-auto-flow:row;grid-auto-columns:auto;gap:10px}.step{display:flex;align-items:center;gap:10px;text-align:left;
font-size:14px}.dot{margin:0;flex:none}}
</style></head><body><div class="wrap" id="main">{{MAIN}}</div><script>
(function(){var K="gnnSimple";
function load(){try{return JSON.parse(sessionStorage.getItem(K)||"{}");}catch(x){return {};}}
function save(m){try{sessionStorage.setItem(K,JSON.stringify(m));}catch(x){}}
function apply(){var m=load();document.querySelectorAll("details[data-k]").forEach(function(d){
var k=d.getAttribute("data-k");if(k in m){d.open=!!m[k];}});}
document.addEventListener("toggle",function(ev){var d=ev.target;if(!d||!d.getAttribute||!d.getAttribute("data-k"))return;
var m=load();m[d.getAttribute("data-k")]=d.open;save(m);},true);
apply();
setInterval(function(){fetch("/?partial=1",{cache:"no-store"}).then(function(r){if(!r.ok)throw 0;return r.text();})
.then(function(t){var y=window.scrollY;var doc=new DOMParser().parseFromString(t,"text/html");
document.getElementById("main").replaceChildren.apply(document.getElementById("main"),Array.from(doc.body.childNodes));
apply();window.scrollTo(0,y);})
.catch(function(){var s=document.querySelector(".stamp");if(s&&s.textContent.indexOf("[stale")<0){s.textContent+=" [stale]";}});
},15000);})();
</script></body></html>"""


def render(st: dict) -> str:
    return PAGE.replace("{{MAIN}}", render_main(st))


# ------------------------------------------------------------------ check + server
def check() -> int:
    """Print what the page will read and whether it exists. Exit code 1 if anything it needs is missing."""
    problems = 0
    print(f"root: {D.ROOT}")
    prog = D.get_progress()
    print(f"progress.json: {'ok' if prog else 'MISSING'}")
    problems += 0 if prog else 1
    sig = load_sig()
    print(f"significance.json: {'ok (' + str(len(sig)) + ' rows)' if sig else 'not found (page will say p not loaded)'}")
    arms = [c for c in D.arms() if c.get("dataset")]
    if not arms:
        print("no arm configs found in experiments/gnn/configs")
        problems += 1
    for c in arms:
        if c["dataset"] in SKIP:
            print(f"- {c['stem']}: skipped by decision")
            continue
        ds = c["dataset"]
        seeds = sorted(read_runs({"hash": c["hash"], "results": c.get("results", "runs/results")}, ds,
                                 c.get("seeds") or []))
        print(f"- {c['stem']} [{ds}] hash {c['hash']}: finished seeds {seeds} of {c.get('seeds')}")
        problems += 0 if seeds else 1
        for name, info in (("Ours K=1", method_info(D.ROOT / c["baseline_path"])),
                           ("DGP original", dgp_info_for(ds))):
            runs = read_runs(info, ds, seeds)
            ok = bool(seeds) and all(s in runs for s in seeds)
            print(f"    {name}: {info['path']} hash {info.get('hash')} K={info.get('K')} model={info.get('model')} "
                  f"seeds found {sorted(runs)} -> {'ok' if ok else 'INCOMPLETE'}"
                  + (f" error={info['error']}" if info.get("error") else ""))
            problems += 0 if ok else 1
    print("RESULT:", "all inputs found" if not problems else f"{problems} problem(s) above")
    return 1 if problems else 0


def full_page(partial: bool) -> str:
    """The complete dashboard from dashboard.py, re-pointed so its self-refresh and JSON link stay on /full."""
    st = D.build_state()
    page = D.render_main(st) if partial else D.render(st)
    page = page.replace('href="/api/state"', 'href="/api/full-state"')
    if not partial:
        page = page.replace('fetch("/?partial=1"', 'fetch("/full?partial=1"')
        page = page.replace("<nav><b>GNN Exp1</b>", '<nav><b>GNN Exp1</b><a href="/">&larr; Simple view</a>', 1)
    return page


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, ctype: str, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path, _, query = self.path.partition("?")
        try:
            if path == "/api/state":
                self._send(200, "application/json", json.dumps(build_state(), indent=1, default=str))
            elif path in ("/", "/index.html"):
                st = build_state()
                self._send(200, "text/html; charset=utf-8", render_main(st) if "partial=1" in query else render(st))
            elif path in ("/full", "/full/"):
                self._send(200, "text/html; charset=utf-8", full_page("partial=1" in query))
            elif path == "/api/full-state":
                self._send(200, "application/json", json.dumps(D.build_state(), indent=1, default=str))
            elif path == "/decisions":
                files = sorted(D.gnn_dir().glob("*decisions*.md"))
                text = "\n\n".join(D.read_text(p) or "" for p in files) or "no decisions log found"
                self._send(200, "text/plain; charset=utf-8", text)
            else:
                self._send(404, "text/plain", "not found")
        except Exception as exc:  # never crash the server
            try:
                self._send(500, "text/plain", f"error: {exc!r}")
            except Exception:
                pass

    def log_message(self, fmt: str, *args) -> None:
        pass


def main() -> int:
    global FULL_URL, BANNER
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--root", default=str(D.ROOT))
    ap.add_argument("--skip", default="reddit", help="comma-separated datasets skipped by decision")
    ap.add_argument("--full-url", default="/full", help="link to the full technical dashboard (default: /full)")
    ap.add_argument("--banner", default=None, help="banner text, e.g. SAMPLE DATA")
    ap.add_argument("--check", action="store_true", help="check inputs and exit")
    a = ap.parse_args()
    D.ROOT = Path(a.root).resolve()
    SKIP.clear()
    SKIP.update(x.strip() for x in a.skip.split(",") if x.strip())
    FULL_URL, BANNER = a.full_url, a.banner
    if a.check:
        return check()
    threading.Thread(target=D.sampler, daemon=True, name="sampler").start()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"Simple dashboard on http://{a.host}:{a.port} (root={D.ROOT})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

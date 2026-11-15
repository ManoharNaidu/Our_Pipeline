"""Dashboard card: which neighbours MD (diffusion), PMP and BWGNN picked, read from compare_neighbours.py output.

Used by dashboard_simple.py. Reads <gnn>/runs/compare/*-neighbour-comparison-<dataset>.json (latest per dataset) and
renders one card per dataset and metapath with the same CSS classes as the simple page. Missing file -> no card.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

e = html.escape


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _num(x, nd=1) -> str:
    return "—" if x is None else f"{x:,.{nd}f}"


def load(compare_dir: Path) -> list[dict]:
    latest: dict[str, Path] = {}
    for p in sorted(compare_dir.glob("*-neighbour-comparison-*.json")):     # dated names sort by date
        latest[p.stem.split("-neighbour-comparison-")[-1]] = p
    out = []
    for p in latest.values():
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")) | {"_file": p.name})
        except Exception:
            continue
    return out


def _shared_bar(hist: list[int], m: int) -> str:
    total = sum(hist) or 1
    cells = "".join(f'<span title="{k} of {m} shared: {n:,} targets" style="flex:{n};background:'
                    f'color-mix(in srgb, var(--info) {15 + 85 * k // m}%, transparent)"></span>'
                    for k, n in enumerate(hist) if n)
    legend = " · ".join(f"{k}: {100 * n / total:.0f}%" for k, n in enumerate(hist))
    return (f'<div style="display:flex;height:10px;border-radius:5px;overflow:hidden;min-width:120px">{cells}</div>'
            f'<span class="sub">shared {legend}</span>')


def card(compare_dir: Path) -> str:
    out = []
    for res in load(compare_dir):
        names = list(res["methods"])
        m = res["M"]
        for pname, rec in res["metapaths"].items():
            pairs = "".join(
                f'<tr><td>{e(p["a"])} <span class="muted">vs</span> {e(p["b"])}</td>'
                f'<td class="r"><b>{p["jaccard_mean"]:.2f}</b></td><td class="r">{_pct(p["identical_share"])}</td>'
                f'<td>{_shared_bar(p["shared_histogram"], m)}</td></tr>' for p in rec["pairs"].values())
            rows = []
            for n in names + ["Random M", "All candidates"]:
                pm, ag = rec["per_method"].get(n), rec["agreement"].get(n)
                cells = [_num(pm["distinct_neighbours_used"], 0) if pm else "", _num(pm["mean_neighbour_degree"]) if pm else "",
                         _pct(pm["fraud_share_of_known"]) if pm else ""]
                cells += [_pct(ag[g]["agreement"]) if ag else "" for g in ("all", "fraud", "benign")]
                rows.append(f'<tr><td><b>{e(n)}</b></td>' + "".join(f'<td class="r">{c}</td>' for c in cells) + "</tr>")
            ex = []
            for x in rec.get("examples", []):
                picks = "".join(
                    "<td>" + " ".join(
                        f'<span class="dv {"neg" if t == "F" else ""}" title="{"fraud" if t == "F" else "benign" if t == "B" else "no train/val label"}">'
                        f'{u}<sup>{e(t)}</sup></span>' for u, t in x["picks"][n]) + "</td>" for n in names)
                ex.append(f'<tr><td>{x["target"]} <span class="sub">{e(x["target_label"])} · {x["candidates"]} cand.</span>'
                          f"</td>{picks}</tr>")
            vt = rec["val_targets_with_headroom"]
            out.append(
                f'<section class="card"><p class="label">{e(res["dataset"].capitalize())} · neighbours picked · {e(pname)}, '
                f'M = {m}</p>'
                f'<p style="margin:0 0 12px">{rec["headroom_targets"]:,} of {rec["targets"]:,} targets '
                f'({_pct(rec["headroom_share"])}) have more than {m} candidates; only there can the methods differ. '
                f'All three pick the same {m} for <b>{_pct(rec["all_methods_identical_share"])}</b> of them; on average '
                f'<b>{rec["mean_picked_by_all"]:.2f} of {m}</b> neighbours are picked by all three.</p>'
                '<div class="tw"><table class="res"><thead><tr><th>Pair</th><th class="r">Mean Jaccard</th>'
                f'<th class="r">Identical</th><th>Targets by neighbours shared (0–{m})</th></tr></thead><tbody>{pairs}'
                "</tbody></table></div>"
                '<div class="tw" style="margin-top:14px"><table class="res"><thead><tr><th>Method</th>'
                '<th class="r">Distinct nbrs</th><th class="r">Mean degree</th><th class="r">Fraud among labelled</th>'
                '<th class="r">Same label: all</th><th class="r">fraud</th><th class="r">benign</th></tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table></div>'
                f'<details data-k="nbrs:{e(res["dataset"])}:{e(pname)}"><summary>Examples: what each method picked for '
                f'{len(rec.get("examples", []))} validation targets</summary><div class="tw"><table class="res seeds">'
                "<thead><tr><th>Target</th>" + "".join(f"<th>{e(n)}</th>" for n in names) +
                f'</tr></thead><tbody>{"".join(ex)}</tbody></table></div>'
                '<p class="note">Superscript: F fraud, B benign (train/validation labels), ? no train/validation label '
                "(test or unlabelled nodes are never revealed).</p></details>"
                '<p class="note">Every row, including MD (Ours K=1, the pipeline\'s own selection), Random M and All '
                f'candidates, is computed over the same {rec["headroom_targets"]:,} targets with more than {m} candidates. '
                '"Same label" = share of picked neighbours with a train/validation label that match the '
                f'validation target\'s label ({vt["all"]} validation targets with headroom; only {vt["fraud"]} fraud, so '
                f"the fraud column is noisy). Test labels are never read. Source: {e(res['_file'])} "
                "(experiments/gnn/compare_neighbours.py, from the pipeline's own trims caches).</p></section>")
    return "".join(out)

"""Compare the neighbours each selection method actually gave the pipeline: MD (diffusion), PMP and BWGNN.

    python experiments/gnn/compare_neighbours.py --dataset instagram \
        --method "MD (Ours K=1)=configs/ours/instagram_k1.yaml" \
        --method "PMP-L1=experiments/gnn/configs/instagram_pmp_l1.yaml" \
        --method "BWGNN-C1=experiments/gnn/configs/instagram_bwgnn_c1.yaml" \
        --out experiments/gnn/runs/compare

Each method's neighbour sets are read from the pipeline's own trims cache (the folder is found from the config the same
way the pipeline names it, and its metadata.json is checked), so this compares exactly what went into the prompts.
Nothing is written to the cache.

Labels: train and validation labels only. Test labels are never read; test and unlabelled neighbours count as
"unknown". Outputs <out>/<date>-neighbour-comparison-<dataset>.{json,md} (date: --date, default today UTC); the
simple dashboard reads the latest JSON per dataset.

All per-method statistics (distinct neighbours, degree, label shares, agreement) are over the same targets: those with
more than M candidates, the only ones where the methods can differ. MD (diffusion) is the pipeline's own Ours K=1
selection; "Random M" and "All candidates" are reference rows over the same targets.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_selection import agreement  # noqa: E402  (same label-agreement measure as Phase C)
from our_pipeline.config import load_config  # noqa: E402
from our_pipeline.graph.data import load_graph, load_split  # noqa: E402
from our_pipeline.graph.trimming import enumerate_metapaths, load_trim, metapath_name, metapath_neighbours  # noqa: E402


def trims_dir(cfg: dict, split) -> tuple[Path, dict]:
    """The trims folder the pipeline uses for this config (same key as pipeline.trims_for), without creating it."""
    method = cfg["method"]
    targets = split.targets()
    meta = {"dataset": cfg["dataset"]["name"], "split_seed": split.seed, "K": method["K"], "M": method["M"],
            "targets": hashlib.sha1(np.sort(targets).astype(np.int64).tobytes()).hexdigest()[:16],
            "embedder": cfg["embedder"].get("model_id", "hashing")}
    if method.get("selection", "diffusion") == "embedding_cosine":
        sha = hashlib.sha256((ROOT / method["selection_embeddings"]).read_bytes()).hexdigest()[:16]
        meta.update({"selection": "embedding_cosine", "selection_embeddings_sha": sha})
    key = hashlib.sha1(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:16]
    folder = ROOT / cfg["paths"]["cache"] / "trims" / key
    if not (folder / "metadata.json").exists():
        raise FileNotFoundError(f"trims not cached yet: {folder} (run the pipeline's trim stage first)")
    stored = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    wrong = {k: (v, stored.get(k)) for k, v in meta.items() if stored.get(k) != v}
    if wrong:
        raise ValueError(f"{folder} metadata does not match the config: {wrong}")
    return folder, meta


def as_sets(trim: dict, targets) -> dict[int, frozenset]:
    return {int(v): frozenset(int(u) for u in trim[int(v)]) for v in targets}


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def compare(dataset: str, methods: list[tuple[str, str]]) -> dict:
    cfgs = {name: load_config(ROOT / path) for name, path in methods}
    first = next(iter(cfgs.values()))
    data_dir = ROOT / first["paths"]["data"] / dataset
    graph, split = load_graph(data_dir), load_split(data_dir)
    M, K = first["method"]["M"], first["method"]["K"]
    for name, cfg in cfgs.items():
        assert (cfg["method"]["M"], cfg["method"]["K"]) == (M, K), f"{name}: M and K must match"
    targets = np.asarray(split.targets(), dtype=np.int64)
    labels = np.asarray(graph.labels).astype(np.int64)
    known = np.zeros(graph.num_nodes, dtype=bool)                  # train and validation only, never test
    known[split.train] = True
    known[split.val] = True
    val = set(np.asarray(split.val).tolist())
    names = [n for n, _ in methods]

    out = {"dataset": dataset, "M": M, "K": K, "labels_used": "train and validation only; test labels never read",
           "methods": {}, "metapaths": {}}
    for name, cfg in cfgs.items():
        folder, meta = trims_dir(cfg, split)
        out["methods"][name] = {"config": methods[[n for n, _ in methods].index(name)][1], "trims": folder.name,
                                "selection": meta.get("selection", "diffusion"),
                                "embeddings_sha256_16": meta.get("selection_embeddings_sha")}

    for path in enumerate_metapaths(first["dataset"]["relations"], K):
        pname = metapath_name(path)
        cand = dict(zip(targets.tolist(), metapath_neighbours(graph.relations, path, targets)))
        counts = np.array([len(cand[int(v)]) for v in targets])
        room = [int(v) for v, c in zip(targets, counts) if c > M]
        sets = {n: as_sets(load_trim(ROOT / first["paths"]["cache"] / "trims" / out["methods"][n]["trims"] /
                                     f"{pname}.npz"), targets) for n in names}
        for n in names:                                            # sanity: selections are drawn from the candidates
            assert all(sets[n][v] <= set(cand[v].tolist()) for v in room), f"{n}: neighbour outside candidates"
        degree = np.asarray(sum((r != 0).astype(np.int64) for r in graph.relations.values()).sum(1)).ravel()

        rng = np.random.default_rng(0)
        rand = {v: (rng.choice(cand[v], M, replace=False) if len(cand[v]) > M else cand[v]) for v in cand}

        def stats(picked: dict) -> dict:                           # every row over the same headroom targets
            chosen = [int(u) for v in room for u in picked[v]]
            usable = [u for u in chosen if known[u]]
            return {"distinct_neighbours_used": len(set(chosen)),
                    "mean_neighbour_degree": float(np.mean(degree[chosen])) if chosen else None,
                    "known_label_share": len(usable) / len(chosen) if chosen else None,
                    "fraud_share_of_known": float(np.mean(labels[usable] == 1)) if usable else None}

        per_method = {n: stats(sets[n]) for n in names}
        per_method["Random M"] = stats(rand)
        cand_all = [u for v in room for u in cand[v]]
        cand_known = [u for u in cand_all if known[u]]
        per_method["All candidates"] = {
            "distinct_neighbours_used": len(set(cand_all)), "mean_neighbour_degree": float(np.mean(degree[cand_all])),
            "known_label_share": len(cand_known) / len(cand_all), "fraud_share_of_known": float(np.mean(labels[cand_known] == 1))}

        pairs = {}
        for a, b in itertools.combinations(names, 2):
            shared = [len(sets[a][v] & sets[b][v]) for v in room]
            pairs[f"{a} | {b}"] = {"a": a, "b": b,
                                   "jaccard_mean": float(np.mean([jaccard(sets[a][v], sets[b][v]) for v in room])),
                                   "identical_share": float(np.mean([sets[a][v] == sets[b][v] for v in room])),
                                   "shared_histogram": [int(np.sum(np.array(shared) == k)) for k in range(M + 1)]}
        all_same = float(np.mean([len({sets[n][v] for n in names}) == 1 for v in room]))
        common = [len(frozenset.intersection(*[sets[n][v] for n in names])) for v in room]
        union = [len(frozenset.union(*[sets[n][v] for n in names])) for v in room]

        val_room = [v for v in room if v in val]
        groups = {"all": np.array(val_room), "fraud": np.array([v for v in val_room if labels[v] == 1]),
                  "benign": np.array([v for v in val_room if labels[v] == 0])}
        agree = {n: {g: agreement({v: np.array(sorted(s)) for v, s in sets[n].items()}, t, known, labels)
                     for g, t in groups.items()} for n in names}
        agree["Random M"] = {g: agreement(rand, t, known, labels) for g, t in groups.items()}

        def tag(u: int) -> str:
            return ("F" if labels[u] == 1 else "B") if known[u] else "?"

        examples = []
        for lab, k in ((1, 4), (0, 4)):                            # first validation targets with headroom, by id
            for v in [v for v in sorted(val_room) if labels[v] == lab][:k]:
                examples.append({"target": v, "target_label": "fraud" if lab else "benign", "candidates": int(len(cand[v])),
                                 "picks": {n: [[u, tag(u)] for u in sorted(sets[n][v])] for n in names}})

        out["metapaths"][pname] = {
            "targets": int(len(targets)), "headroom_targets": len(room), "headroom_share": len(room) / len(targets),
            "no_candidate_share": float(np.mean(counts == 0)), "per_method": per_method, "pairs": pairs,
            "all_methods_identical_share": all_same, "mean_picked_by_all": float(np.mean(common)),
            "mean_union_size": float(np.mean(union)), "val_targets_with_headroom": {g: int(len(t)) for g, t in groups.items()},
            "agreement": agree, "examples": examples,
            "per_method_scope": "targets with more than M candidates (headroom_targets), for every row"}
    return out


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def markdown(res: dict) -> str:
    lines = [f"# Neighbour comparison on {res['dataset']}: " + ", ".join(res["methods"]), "",
             f"K = {res['K']}, M = {res['M']}. Labels: {res['labels_used']}. Statistics over targets with more than M "
             "candidates (the only ones where methods can differ).", ""]
    for pname, rec in res["metapaths"].items():
        lines += [f"## {pname}: {rec['headroom_targets']} of {rec['targets']} targets with headroom "
                  f"({pct(rec['headroom_share'])}%)", "", "| Pair | Mean Jaccard | Identical sets | Targets sharing 0/1/2/3/4 |",
                  "|---|---|---|---|"]
        for p in rec["pairs"].values():
            lines.append(f"| {p['a']} vs {p['b']} | {p['jaccard_mean']:.3f} | {pct(p['identical_share'])}% | "
                         + " / ".join(str(x) for x in p["shared_histogram"]) + " |")
        lines += ["", f"All methods pick the same set: {pct(rec['all_methods_identical_share'])}%; neighbours picked by "
                  f"all: {rec['mean_picked_by_all']:.2f} of {res['M']}; union per target: {rec['mean_union_size']:.2f}.", "",
                  "| Method | Distinct neighbours | Mean degree | Known label | Fraud among known | Val agreement all | fraud | benign |",
                  "|---|---|---|---|---|---|---|---|"]
        for n, m in rec["per_method"].items():
            a = rec["agreement"].get(n)
            ag = (f"{pct(a['all']['agreement'])} | {pct(a['fraud']['agreement'])} | {pct(a['benign']['agreement'])}"
                  if a else " | | ")
            lines.append(f"| {n} | {m['distinct_neighbours_used']} | {m['mean_neighbour_degree']:.1f} | "
                         f"{pct(m['known_label_share'])}% | {pct(m['fraud_share_of_known'])}% | {ag} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", action="append", required=True, metavar="NAME=CONFIG")
    ap.add_argument("--out", default="experiments/gnn/runs/compare")
    ap.add_argument("--date", default=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
                    help="date prefix of the output files (default: today, UTC)")
    a = ap.parse_args()
    methods = [tuple(x.rsplit("=", 1)) for x in a.method]   # names may contain "=", paths do not
    res = compare(a.dataset, methods)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{a.date}-neighbour-comparison-{a.dataset}"
    if stem.with_suffix(".json").exists():
        raise SystemExit(f"{stem}.json exists; pass another --date (files are never overwritten)")
    stem.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    stem.with_suffix(".md").write_text(markdown(res), encoding="utf-8")
    print(markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

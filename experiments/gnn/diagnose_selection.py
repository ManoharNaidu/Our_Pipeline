"""Phase C diagnostics: does GNN-cosine selection pick different, more label-consistent neighbours than diffusion?

    python experiments/gnn/diagnose_selection.py --datasets instagram reddit --variants pmp_l1 pmp_l2 \
        --out experiments/gnn/runs/diagnostics

For each dataset (metapath K = 1, M from the Ours K=1 config) and GNN variant:
  1. GNN validation AUROC / Macro-F1 and train/val/test label counts, read from the embedding manifest.
  2. Headroom: share of targets with more than M candidates (otherwise every method returns the same set).
  3. Overlap@M: mean Jaccard between the GNN-cosine set and the diffusion set, over targets with more than M candidates.
  4. Neighbour label agreement on validation targets: share of selected neighbours whose label equals the target's,
     counting only neighbours with a train or validation label, for GNN cosine, diffusion and a random M (seed 0).
     Reported over all validation targets with headroom, and separately for fraud and benign targets.

Test labels are never read. The diffusion sets come from the cached baseline trims, which must already exist; nothing is
written to the pipeline cache. GNN sets are computed in memory with the same code the pipeline uses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from our_pipeline.config import load_config
from our_pipeline.graph.data import load_graph, load_split
from our_pipeline.graph.trimming import enumerate_metapaths, load_trim, metapath_name, metapath_neighbours, top_m_cosine
from our_pipeline.text.cache import artifact_dir  # noqa: F401  (documents where the key format comes from)

ROOT = Path(__file__).resolve().parents[2]


def baseline_trim_dir(cfg: dict, graph, split) -> Path:
    """The existing diffusion trims folder of a config, without creating anything."""
    method = cfg["method"]
    targets = split.targets()
    meta = {"dataset": graph.name, "split_seed": split.seed, "K": method["K"], "M": method["M"],
            "targets": hashlib.sha1(np.sort(targets).astype(np.int64).tobytes()).hexdigest()[:16],
            "embedder": cfg["embedder"].get("model_id", "hashing")}
    key = hashlib.sha1(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:16]
    directory = ROOT / cfg["paths"]["cache"] / "trims" / key
    if not directory.exists():
        raise FileNotFoundError(f"baseline trims not cached: {directory}")
    return directory


def agreement(selected: dict[int, np.ndarray], targets: np.ndarray, known: np.ndarray, labels: np.ndarray) -> dict:
    """Share of selected, label-known neighbours that share the target's label, and the share of selections known."""
    same = total = chosen = 0
    for v in targets:
        found = selected[int(v)]
        chosen += len(found)
        usable = found[known[found]]
        total += len(usable)
        same += int((labels[usable] == labels[v]).sum())
    return {"agreement": same / total if total else float("nan"), "coverage": total / chosen if chosen else float("nan"),
            "neighbours_known": total}


def diagnose(dataset: str, variants: list[str], embeddings_root: Path) -> dict:
    cfg = load_config(ROOT / "configs" / "ours" / f"{dataset}_k1.yaml")
    data_dir = ROOT / cfg["paths"]["data"] / dataset
    graph, split = load_graph(data_dir), load_split(data_dir)
    M, K = cfg["method"]["M"], cfg["method"]["K"]
    targets = np.asarray(split.targets(), dtype=np.int64)
    labels = np.asarray(graph.labels).astype(np.int64)
    known = np.zeros(graph.num_nodes, dtype=bool)                  # train and validation labels only, never test
    known[split.train] = True
    known[split.val] = True
    val = np.asarray(split.val, dtype=np.int64)
    trim_dir = baseline_trim_dir(cfg, graph, split)
    rng = np.random.default_rng(0)

    out = {"dataset": dataset, "M": M, "K": K, "baseline_trims": trim_dir.name, "metapaths": {}}
    for path in enumerate_metapaths(cfg["dataset"]["relations"], K):
        name = metapath_name(path)
        diffusion = load_trim(trim_dir / f"{name}.npz")
        candidates = dict(zip(targets.tolist(), metapath_neighbours(graph.relations, path, targets)))
        counts = np.array([len(candidates[int(v)]) for v in targets])
        room = targets[counts > M]
        room_val = np.intersect1d(val, room)
        random_sets = {int(v): (rng.choice(c, M, replace=False) if len(c) > M else c) for v, c in candidates.items()}
        groups = {"all": room_val, "fraud": room_val[labels[room_val] == 1], "benign": room_val[labels[room_val] == 0]}
        record = {"targets": int(len(targets)), "headroom_share": float((counts > M).mean()),
                  "zero_candidates_share": float((counts == 0).mean()), "median_candidates": float(np.median(counts)),
                  "val_targets_with_headroom": {g: int(len(t)) for g, t in groups.items()},
                  "diffusion": {g: agreement(diffusion, t, known, labels) for g, t in groups.items()},
                  "random": {g: agreement(random_sets, t, known, labels) for g, t in groups.items()}, "variants": {}}
        for variant in variants:
            folder = embeddings_root / f"{dataset}_{variant}"
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            embeddings = np.load(folder / "embeddings.npy")
            gnn = {int(v): (top_m_cosine(candidates[int(v)], embeddings, int(v), M) if len(candidates[int(v)])
                            else candidates[int(v)]) for v in targets}
            jaccard = [len(set(gnn[int(v)]) & set(diffusion[int(v)])) / len(set(gnn[int(v)]) | set(diffusion[int(v)]))
                       for v in room]
            chosen = manifest["chosen_config"]                     # the grid row chosen on validation AUROC
            record["variants"][variant] = {
                "val_auroc": chosen["val_auroc"], "val_macro_f1": chosen["val_macro_f1"],
                "chosen_config": manifest["chosen"], "label_counts": manifest.get("label_counts"),
                "overlap_jaccard_mean": float(np.mean(jaccard)) if jaccard else float("nan"),
                "identical_set_share": float(np.mean([j == 1.0 for j in jaccard])) if jaccard else float("nan"),
                "agreement": {g: agreement(gnn, t, known, labels) for g, t in groups.items()}}
        out["metapaths"][name] = record
    return out


def pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def markdown(results: list[dict]) -> str:
    lines = ["# Phase C diagnostics: GNN-cosine against diffusion neighbour selection", "",
             "Metapath K = 1. Agreement = share of selected neighbours with the target's label, over neighbours with a "
             "train or validation label, on validation targets with more than M candidates. Coverage = share of "
             "selected neighbours with such a label. Test labels are never read.", ""]
    for res in results:
        for name, rec in res["metapaths"].items():
            lines += [f"## {res['dataset']} ({name}), M = {res['M']}", "",
                      f"Targets {rec['targets']}; headroom (more than M candidates) {pct(rec['headroom_share'])}%; "
                      f"no candidates {pct(rec['zero_candidates_share'])}%; median candidates {rec['median_candidates']:.0f}. "
                      f"Validation targets with headroom: {rec['val_targets_with_headroom']}.", "",
                      "| Selection | Val AUROC | Val Macro-F1 | Chosen config | Jaccard vs diffusion | Identical sets | "
                      "Agreement all | fraud | benign | Coverage |", "|---|---|---|---|---|---|---|---|---|---|"]
            for label, agree in (("Diffusion (Ours K=1)", rec["diffusion"]), ("Random M", rec["random"])):
                lines.append(f"| {label} | | | | | | {pct(agree['all']['agreement'])} | {pct(agree['fraud']['agreement'])} "
                             f"| {pct(agree['benign']['agreement'])} | {pct(agree['all']['coverage'])} |")
            for variant, v in rec["variants"].items():
                a = v["agreement"]
                auroc, f1 = f"{v['val_auroc']:.2f}", f"{v['val_macro_f1']:.2f}"     # already percentages
                lines.append(f"| {variant} | {auroc} | {f1} | {v['chosen_config']} | {v['overlap_jaccard_mean']:.3f} | "
                             f"{pct(v['identical_set_share'])}% | {pct(a['all']['agreement'])} | {pct(a['fraud']['agreement'])} "
                             f"| {pct(a['benign']['agreement'])} | {pct(a['all']['coverage'])} |")
            lines.append("")
    return "\n".join(lines)


def main():
    made = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    made.add_argument("--datasets", nargs="+", default=["instagram", "reddit"])
    made.add_argument("--variants", nargs="+", default=["pmp_l1", "pmp_l2"])
    made.add_argument("--embeddings-root", default="experiments/gnn/runs/embeddings")
    made.add_argument("--out", default="experiments/gnn/runs/diagnostics")
    args = made.parse_args()
    results = [diagnose(d, args.variants, ROOT / args.embeddings_root) for d in args.datasets]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "2026-11-14-phase-c-diagnostics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "2026-11-14-phase-c-diagnostics.md").write_text(markdown(results), encoding="utf-8")
    print(markdown(results))


if __name__ == "__main__":
    main()

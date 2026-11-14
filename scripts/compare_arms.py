"""Compare arms against a reference over paired seeds, in the layout of docs/RESULTS.md and docs/results.json.

    python scripts/compare_arms.py --reference "Ours K=2"=configs/ours/instagram_k2.yaml \\
        --arm "Ours K=1"=configs/ours/instagram_k1.yaml --json out.json --markdown out.md

Each --reference and --arm is NAME=CONFIG. A config's finished runs are read (never written) from
<paths.results>/raw/<dataset>_seed*_<config_hash>/, with paths.results resolved against the working directory, as
in our_pipeline.runs. Several datasets go in one call: give one --reference per dataset, and each arm is compared with
the reference of its own dataset.

Per arm: seeds, test mean ± population std (from metrics.json) and the mean prompt tokens (from run_manifest.json).
Per arm, metric: Δ = arm − reference averaged over the seeds both have, seeds where the arm is better, the paired
bootstrap 95% CI over test nodes (evaluate.paired_bootstrap on predictions.csv, Macro-F1 at each run's validation
threshold), the paired t-test p over seeds, and Holm over the family chosen by --holm-family (each dataset, or all
tests together). Holm values that survive at 0.05 are bold in the markdown.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

import _common  # noqa: F401  (puts the package on the path)
from our_pipeline.config import config_hash, load_config
from our_pipeline.evaluate import METRICS, compute_metrics, holm, paired_bootstrap

LABELS = {"macro_f1": "Macro-F1", "auroc": "AUROC", "auprc": "AUPRC"}
ALPHA = 0.05


def named_config(text: str) -> tuple[str, str]:
    name, sep, path = text.rpartition("=")      # split at the last "=": names such as "Ours K=2" contain one
    if not sep or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"expected NAME=CONFIG.yaml, got {text!r}")
    return name.strip(), path.strip()


def read_test_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, labels, probabilities = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                nodes.append(int(row["node"]))
                labels.append(int(row["label"]))
                probabilities.append(float(row["p_fraud"]))
    return np.array(nodes), np.array(labels), np.array(probabilities, dtype=np.float64)


def load_arm(name: str, config_path: str) -> dict:
    """Every finished run of one config, keyed by seed."""
    cfg = load_config(config_path)
    dataset = cfg["dataset"]["name"]
    raw = Path(cfg["paths"]["results"]) / "raw"
    runs = {}
    for run_dir in sorted(raw.glob(f"{dataset}_seed*_{config_hash(cfg)}")):
        if not (run_dir / "metrics.json").exists():
            continue                                    # unfinished run
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        nodes, labels, probabilities = read_test_predictions(run_dir / "predictions.csv")
        rescored = compute_metrics(labels, probabilities, metrics["test"]["threshold"])
        off = {m: rescored[m] - metrics["test"][m] for m in METRICS if abs(rescored[m] - metrics["test"][m]) > 1e-6}
        if off:     # e.g. rounded probabilities: ties change AUROC/AUPRC, so the bootstrap CI would drift too
            print(f"warning: {run_dir.name}/predictions.csv does not reproduce metrics.json: {off}", file=sys.stderr)
        runs[int(metrics["seed"])] = {"test": metrics["test"], "threshold": float(metrics["test"]["threshold"]),
                                      "tokens": float(manifest["prompt_tokens"]["mean"]),
                                      "nodes": nodes, "labels": labels, "p": probabilities}
    if not runs:
        raise SystemExit(f"{name}: no finished runs under {raw} for {config_path}")
    return {"name": name, "config": config_path, "dataset": dataset, "runs": runs}


def arm_row(arm: dict) -> dict:
    runs = [arm["runs"][s] for s in sorted(arm["runs"])]
    stats = {m: (float(np.mean([r["test"][m] for r in runs])), float(np.std([r["test"][m] for r in runs])))
             for m in METRICS}
    return {"name": arm["name"], "seeds": len(runs), "stats": stats, "tokens": float(np.mean([r["tokens"] for r in runs]))}


def compare(arm: dict, reference: dict) -> list[dict]:
    seeds = sorted(set(arm["runs"]) & set(reference["runs"]))
    if not seeds:
        raise SystemExit(f"{arm['name']} and {reference['name']} share no seeds")
    a = [arm["runs"][s] for s in seeds]
    b = [reference["runs"][s] for s in seeds]
    nodes, labels = a[0]["nodes"], a[0]["labels"]
    for run in a + b:      # one label vector serves every seed in paired_bootstrap, so all test sets must agree
        assert np.array_equal(run["nodes"], nodes), f"{arm['name']} vs {reference['name']}: test nodes differ"
        assert np.array_equal(run["labels"], labels), f"{arm['name']} vs {reference['name']}: test labels differ"
    ci = paired_bootstrap(labels, [r["p"] for r in a], [r["p"] for r in b],
                          [r["threshold"] for r in a], [r["threshold"] for r in b])
    records = []
    for metric in METRICS:
        x = np.array([r["test"][metric] for r in a])
        y = np.array([r["test"][metric] for r in b])
        records.append({"arm": arm["name"], "reference": reference["name"], "metric": metric, "seeds": seeds,
                        "delta": float(np.mean(x - y)), "better": int(np.sum(x > y)),
                        "p": float(ttest_rel(x, y).pvalue), "ci95": ci[metric], "holm": None})
    return records


def apply_holm(records: list[dict]) -> None:
    adjusted = holm({i: r["p"] for i, r in enumerate(records)})
    for i, r in enumerate(records):
        r["holm"] = adjusted[i]


def markdown(dataset: str, rows: list[dict], records: list[dict]) -> str:
    lines = [f"## {dataset}", "", "| Arm | Seeds | Macro-F1 | AUROC | AUPRC | Mean prompt tokens |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        cells = " | ".join(f"{row['stats'][m][0]:.2f} ± {row['stats'][m][1]:.2f}" for m in METRICS)
        lines.append(f"| {row['name']} | {row['seeds']} | {cells} | {row['tokens']:.1f} |")
    lines += ["", "| Arm | Against | Metric | Δ (seeds better) | 95% CI | p | Holm |", "|---|---|---|---|---|---|---|"]
    for r in records:
        held = f"{r['holm']:.4f}"
        held = f"**{held}**" if r["holm"] < ALPHA else held
        lines.append(f"| {r['arm']} | {r['reference']} | {LABELS[r['metric']]} | {r['delta']:+.2f} "
                     f"({r['better']}/{len(r['seeds'])}) | [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | {r['p']:.4f} "
                     f"| {held} |")
    return "\n".join(lines)


def main():
    made = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    made.add_argument("--reference", action="append", required=True, type=named_config, metavar="NAME=CONFIG",
                      help="one per dataset")
    made.add_argument("--arm", action="append", required=True, type=named_config, metavar="NAME=CONFIG")
    made.add_argument("--holm-family", choices=("dataset", "all"), default="dataset")
    made.add_argument("--json", type=Path, help="write records keyed by dataset, as docs/results.json")
    made.add_argument("--markdown", type=Path, help="write the tables, as docs/RESULTS.md")
    args = made.parse_args()

    references = {}
    for name, path in args.reference:
        ref = load_arm(name, path)
        if ref["dataset"] in references:
            raise SystemExit(f"two references for dataset {ref['dataset']}")
        references[ref["dataset"]] = ref
    arms: dict[str, list[dict]] = {d: [] for d in references}
    for name, path in args.arm:
        arm = load_arm(name, path)
        if arm["dataset"] not in references:
            raise SystemExit(f"{name}: no --reference for dataset {arm['dataset']}")
        arms[arm["dataset"]].append(arm)

    results = {d: [r for arm in arms[d] for r in compare(arm, references[d])] for d in references}
    families = list(results.values()) if args.holm_family == "dataset" else [[r for rs in results.values() for r in rs]]
    for family in families:
        if family:
            apply_holm(family)

    text = "\n\n".join(markdown(d, [arm_row(references[d])] + [arm_row(a) for a in arms[d]], results[d])
                       for d in references)
    note = f"Holm over each {'dataset' if args.holm_family == 'dataset' else 'call (all datasets)'}; bold: < {ALPHA}."
    text = f"{text}\n\n{note}\n"
    print(text)
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

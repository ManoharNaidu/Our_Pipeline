"""Print a results table for one or more configs: mean ± std over seeds, beside the reference number.

    python scripts/evaluate.py configs/datasets/*.yaml
"""

import argparse

import _common  # noqa: F401  (puts the package on the path)
from our_pipeline.config import load_config
from our_pipeline.evaluate import METRICS
from our_pipeline.runs import summarise_runs


def main():
    made = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    made.add_argument("configs", nargs="+")
    args = made.parse_args()
    print("| Dataset | Seeds | Macro-F1 | AUROC | AUPRC | Reference |")
    print("|---|---|---|---|---|---|")
    for path in args.configs:
        cfg = load_config(path)
        summary = summarise_runs(cfg)
        reference = cfg["dataset"].get("reference", {})
        known = " / ".join(f"{reference[m]:.2f}" if m in reference else "n/a" for m in METRICS)
        cells = [f"{summary['test'][m]['mean']:.2f} ± {summary['test'][m]['std']:.2f}" for m in METRICS] if summary else ["no runs"] * 3
        seeds = len(summary["seeds"]) if summary else 0
        print(f"| {cfg['dataset']['name']} | {seeds} | {' | '.join(cells)} | {reference.get('source', '')} {known} |")


if __name__ == "__main__":
    main()

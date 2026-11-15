"""Train PMP or BWGNN over its hyperparameter grid, select on validation AUROC, and write all-node embeddings.

    python experiments/gnn/train_gnn.py --config configs/ours/instagram_k1.yaml --model pmp --layers 1 \
        --out experiments/gnn/runs/embeddings/instagram_pmp_l1 [--device cpu] [--grid A1,A2] [--max-epochs N]

Writes embeddings.npy (num_nodes x hid_dim, float32, from the chosen config) and manifest.json. No test-set metric is
ever computed or recorded; test labels are used only to count positives/negatives for the manifest. A finished run with
matching settings is skipped. Paths in the config are relative to the repository root, which is the working directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bwgnn_embed as B  # noqa: E402
import gnn_embed as G  # noqa: E402

MODELS = {"pmp": G, "bwgnn": B}   # each exposes GRID, TRAIN, SELECTION and EMBEDDING_WHAT


def sha256_16(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def split_hash(split) -> str:
    payload = {name: sorted(int(v) for v in getattr(split, name)) for name in ("train", "val", "test")}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def label_counts(labels: np.ndarray, split) -> dict:
    out = {}
    for name in ("train", "val", "test"):
        y = np.asarray(labels)[np.asarray(getattr(split, name), dtype=np.int64)]
        out[name] = {"positives": int((y == 1).sum()), "negatives": int((y == 0).sum()), "total": int(len(y))}
    return out


def versions() -> dict:
    import scipy
    import sklearn
    import torch

    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="pmp", choices=sorted(MODELS))
    parser.add_argument("--layers", type=int, required=True, choices=[1, 2],
                        help="PMP: number of layers; BWGNN: beta-wavelet order C")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--relation-mode", default="merge", choices=["merge", "cat"])
    parser.add_argument("--grid", default=None, help="comma-separated subset of the model's grid (default: all)")
    parser.add_argument("--max-epochs", type=int, default=None, help="cap on epochs (smoke tests only)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="rerun even if a matching manifest exists")
    args = parser.parse_args(argv)

    from our_pipeline.config import load_config
    from our_pipeline.evaluate import macro_f1_at, select_threshold
    from our_pipeline.graph.data import load_graph, load_split
    from our_pipeline.runs import git_commit

    M = MODELS[args.model]
    grid_names = [g.strip() for g in (args.grid or ",".join(M.GRID)).split(",") if g.strip()]
    unknown = set(grid_names) - set(M.GRID)
    if unknown or not grid_names:
        parser.error(f"unknown grid configs {sorted(unknown)}; choose from {list(M.GRID)}")

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data"]) / cfg["dataset"]["name"]
    graph, split = load_graph(data_dir), load_split(data_dir)
    features, emb_path = G.load_text_embeddings(graph, cfg)
    emb_sha = sha256_16(emb_path)
    s_hash = split_hash(split)

    settings = {"dataset": graph.name, "model": args.model, "layers": args.layers, "seed": args.seed,
                "relation_mode": args.relation_mode, "grid": {name: M.GRID[name] for name in grid_names},
                "max_epochs": args.max_epochs, "split_sha1": s_hash, "embedding_sha256_16": emb_sha,
                "num_nodes": graph.num_nodes}
    out = Path(args.out)
    manifest_path, emb_out = out / "manifest.json", out / "embeddings.npy"
    if not args.force and manifest_path.exists() and emb_out.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("settings") == settings:
            print(f"{manifest_path} exists with matching settings; skipping")
            return 0
        print(f"{manifest_path} exists with different settings; rerunning")

    start = time.time()
    nodes = G.labelled_nodes(graph.labels, split)
    results, rows = {}, []
    for name in grid_names:
        t0 = time.time()
        res = M.TRAIN(features, graph.relations, nodes, M.GRID[name], args.layers, args.relation_mode,
                      seed=args.seed, device=args.device, early_stopping=True, max_epochs=args.max_epochs)
        threshold = select_threshold(nodes.val_y, res.val_scores)
        results[name] = res
        rows.append({"name": name, "hyperparameters": M.GRID[name], "val_auroc": 100 * res.val_auroc,
                     "val_macro_f1": macro_f1_at(nodes.val_y, res.val_scores, threshold), "val_threshold": threshold,
                     "best_epoch_0based": res.best_epoch, "epochs_run": res.epochs_run,
                     "val_auroc_curve": [round(100 * v, 4) for v in res.val_auroc_curve],
                     "train_loss_curve": [round(v, 6) for v in res.train_loss_curve],
                     "seconds": round(time.time() - t0, 2)})
        print(f"{name}: val AUROC {100 * res.val_auroc:.2f}  best epoch {res.best_epoch}  "
              f"epochs {res.epochs_run}  {time.time() - t0:.1f}s", flush=True)
    chosen = G.select_config(results, grid_names)

    out.mkdir(parents=True, exist_ok=True)
    np.save(emb_out, results[chosen].embeddings)
    manifest = {
        "settings": settings,
        "dataset": graph.name, "model": args.model, "layers": args.layers, "seed": args.seed,
        "relation_mode": args.relation_mode, "device": args.device, "config_file": args.config,
        "selection": M.SELECTION,
        "chosen": chosen, "chosen_config": next(r for r in rows if r["name"] == chosen), "grid": rows,
        "embedding_output": {"path": str(emb_out), "shape": list(results[chosen].embeddings.shape),
                             "dtype": "float32", "what": M.EMBEDDING_WHAT},
        "split": {"sha1_sorted_ids": s_hash, "seed": split.seed, "sizes": split.sizes()},
        "label_counts": label_counts(graph.labels, split),
        "features": {"source": "DeBERTa text embeddings", "model_id": cfg["embedder"].get("model_id"),
                     "revision": cfg["embedder"].get("revision"), "max_length": cfg["embedder"].get("max_length"),
                     "backend": cfg["embedder"].get("backend"), "path": str(emb_path), "sha256_16": emb_sha,
                     "numeric_attributes_used": False, "numeric_attributes_available": int(graph.numeric.shape[1]),
                     **G.feature_checks(features)},
        "graph_checks": G.graph_checks(graph.relations),
        "git_commit": git_commit(), "versions": versions(),
        "runtime_seconds": round(time.time() - start, 2),
        "test_metrics": "never computed",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"chosen {chosen}: val AUROC {100 * results[chosen].val_auroc:.2f}; wrote {emb_out} and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Experiment 1 report: GNN embeddings for neighbour selection

Date: 2026-11-14. Scope: Instagram (run), Reddit (diagnostics only, full run skipped), Amazon Instruments (not run).
Every decision referred to by ID (S, H, D, E, G, O) is in [`2026-11-14-exp1-decisions.md`](2026-11-14-exp1-decisions.md).

## Summary

Replacing diffusion ranking with cosine similarity on PMP-L1 embeddings (metapath K = 1, M = 4) **does not improve
Ours on Instagram**. AUROC, the primary metric, changes by −0.29 points (95% CI [−1.49, +0.90], Holm p = 0.89).
Macro-F1 is worse on all five seeds, by −1.96 points (95% CI [−2.85, −1.04], Holm p = 0.033). AUPRC changes by −0.53
(n.s.). Prompt length is unchanged (70.6 vs 70.5 tokens). Following decision G3 and the user's go-ahead for Phase E,
Reddit and Amazon Instruments were not run (G6).

## What was compared

| Arm | Config | Neighbour ranking |
|---|---|---|
| Ours K=1 (reference) | `configs/ours/instagram_k1.yaml` (hash `1994e80176`) | Diffusion (unchanged pipeline) |
| Ours + PMP-L1 cosine | `experiments/gnn/configs/instagram_pmp_l1.yaml` (hash `e29d17327e`) | Cosine on PMP 1-layer embeddings, top M = 4, ties by node id (H3) |

The two configs differ only in `method.selection`, `method.selection_embeddings` and the results folder (G4). Writer
and classifier (Gemma-2-9B-it, LoRA), prompts, epoch rule (`auroc_es3`), split (seed 0) and classifier seeds 0–4 are
identical. Node summaries are shared: only the 67 nodes that cosine selected and diffusion had not were newly
summarised (H4); all 6,522 metapath summaries were rewritten for the new neighbour sets.

## GNN (Phase B/C)

PMP reimplemented in plain PyTorch from the official code (commit `3f7629f`; deviations D1–D11 and the 12-item list in
`gnn_embed.py`). Train labels only for training and for the label partition (D3, D4); validation labels only for early
stopping and grid selection (D9, D2); test labels never read. Tests check that shuffling test labels leaves the
embeddings bit-identical, and that with fixed epochs shuffling validation and test labels does too
(`experiments/gnn/tests/test_gnn_embed.py`; 75/75 tests pass).

| Variant | Chosen config | GNN val AUROC | GNN val Macro-F1 | Best epoch (0-based) / epochs run |
|---|---|---|---|---|
| Instagram PMP-L1 (used) | Y3 | 62.86 | 57.73 | 15 / 116 |
| Instagram PMP-L2 | Y3 | 61.21 | 52.14 | 4 / 105 |
| Reddit PMP-L1 | Y3 | 67.66 | 58.14 | 0 / 101 (G5) |
| Reddit PMP-L2 | Y3 | 63.64 | 49.24 | 28 / 129 |

Training labels: Instagram 72 fraud / 722 benign (validation the same sizes); Reddit 167 / 1,672.

Y3 = official Yelp widths, weighted loss, raw features. The official feature normalisation explodes on DeBERTa
features (D1), which is why the raw-feature configs won.

## Selection diagnostics (Phase C, validation labels only)

From `runs/diagnostics/2026-11-14-phase-c-diagnostics.md`, metapath UFU (Instagram) / URU (Reddit), M = 4:

| Dataset | Headroom (> M candidates) | Jaccard PMP-L1 vs diffusion | Agreement: diffusion / random / PMP-L1 (all) | Fraud targets: diffusion / random / PMP-L1 |
|---|---|---|---|---|
| Instagram | 51.8% | 0.309 | 90.9 / 91.8 / 90.4 | 21.1 / 5.9 / 4.3 (28 targets) |
| Reddit | 12.1% | 0.511 | 69.8 / 69.4 / 71.8 | 24.1 / 20.4 / 32.6 (48 targets) |

On Instagram, cosine selection changed most neighbour sets (Jaccard 0.31; only 3.6% identical) but did not choose
neighbours that agree with the target's label more often than diffusion or random. The fraud-target columns rest on
28 and 48 validation targets and few labelled neighbours, so they are noisy. This matched the final result.

## Results (Phase E, test set, 5 paired seeds)

Produced by `scripts/compare_arms.py` (paired bootstrap 2,000 rounds over test nodes, paired t-test over seeds, Holm
over the three metrics; E1, E2). Files: `runs/eval/2026-11-14-instagram-pmp-l1-vs-ours-k1.{md,json}`.

| Arm | Macro-F1 | AUROC | AUPRC | Mean prompt tokens |
|---|---|---|---|---|
| Ours K=1 | 53.86 ± 0.87 | 58.86 ± 1.19 | 14.18 ± 1.26 | 70.5 |
| Ours + PMP-L1 cosine | 51.90 ± 0.72 | 58.57 ± 1.14 | 13.65 ± 0.75 | 70.6 |

| Metric | Δ (seeds better) | 95% CI | p | Holm |
|---|---|---|---|---|
| Macro-F1 | −1.96 (0/5) | [−2.85, −1.04] | 0.0109 | **0.0326** |
| AUROC | −0.29 (3/5) | [−1.49, +0.90] | 0.7943 | 0.8891 |
| AUPRC | −0.53 (1/5) | [−1.38, +0.31] | 0.4445 | 0.8891 |

Mean ± population std over seeds. Holm over this comparison's three tests (one dataset); bold: < 0.05.

### Per seed

| Seed | Arm | Best epoch | Val AUROC | Val Macro-F1 | Val threshold | Test AUROC | Test Macro-F1 | Test AUPRC |
|---|---|---|---|---|---|---|---|---|
| 0 | Ours K=1 | 3 | 59.76 | 56.61 | 0.10 | 59.18 | 54.31 | 15.15 |
| 0 | PMP-L1 | 3 | 60.80 | 54.01 | 0.10 | 57.71 | 52.40 | 13.61 |
| 1 | Ours K=1 | 3 | 62.98 | 55.14 | 0.30 | 58.79 | 54.63 | 14.83 |
| 1 | PMP-L1 | 4 | 61.99 | 51.14 | 0.05 | 59.30 | 51.35 | 14.81 |
| 2 | Ours K=1 | 2 | 62.01 | 54.51 | 0.15 | 56.66 | 52.99 | 11.70 |
| 2 | PMP-L1 | 10 | 61.77 | 51.88 | 0.90 | 59.45 | 50.82 | 13.21 |
| 3 | Ours K=1 | 3 | 62.62 | 52.52 | 0.10 | 60.18 | 52.65 | 14.67 |
| 3 | PMP-L1 | 6 | 60.80 | 53.95 | 0.05 | 56.76 | 52.10 | 12.57 |
| 4 | Ours K=1 | 3 | 61.67 | 55.38 | 0.15 | 59.48 | 54.72 | 14.55 |
| 4 | PMP-L1 | 5 | 63.76 | 53.69 | 0.05 | 59.63 | 52.81 | 14.03 |

The validation set gives the same picture as test: mean validation AUROC 61.82 (PMP-L1) vs 61.81 (Ours K=1), mean
validation Macro-F1 52.93 vs 54.83. Four of five PMP-L1 runs chose an extreme validation threshold (0.05 or 0.90)
against 0.10–0.30 for Ours, which is consistent with weaker class separation at the chosen epoch; this is an
observation, not a tested explanation of the Macro-F1 drop.

## Runtime and memory (this arm, Instagram)

| Job | Seconds |
|---|---|
| Trim (cosine, UFU) | earlier run, CPU, under a minute |
| Node summaries (67 new) | 30 |
| Metapath summaries (6,522) | 645 |
| Prompts | 15 |
| Classifier seeds 0–4 | 1,095 / 1,215 / 1,245 / 1,185 / 1,320 |
| **Total GPU wall time** | **6,750 (≈ 1.9 h)** |

Peak GPU memory 19.35 GB and peak CPU RSS 5.87 GB per seed (A100-SXM4-80GB). The PMP training itself ran on CPU in
seconds per grid config. Seed run times (1,073–1,305 s) are longer than the stored Ours K=1 runs (885–1,019 s), with
the same epoch count (10) and library versions, but the Ours runs were made on 2026-11-06 on a different host and
kernel, so the times are not comparable.

## Limitations

- One GNN per variant (seed 0) shared by all five classifier seeds (D10): the seed variance above is classifier
  variance only, not GNN variance.
- PMP is trained on few labels: 72 fraud training nodes on Instagram and 167 on Reddit, and the label-agreement diagnostics rest on 28 (Instagram) and 48 (Reddit) fraud validation targets.
- Headroom: only 51.8% of Instagram targets have more than M = 4 candidates; for the rest the selection cannot
  change. Reddit's headroom is 12.1%.
- Only PMP, only metapath K = 1, only M = 4, only cosine on the last hidden layer (S2, S3, H3). BWGNN, other
  similarity measures and K = 2 were not tried.
- The PMP implementation deviates from the official code where needed to run (D6, D8) and in feature normalisation
  (D1); results are for this reimplementation.
- Reddit and Amazon Instruments full runs were not made, so there is no test evidence for them (G6).

## Reproduction

```
# GNN embeddings (CPU)
.venv-exp1/bin/python experiments/gnn/train_gnn.py --config configs/ours/instagram_k1.yaml --model pmp --layers 1 \
    --out experiments/gnn/runs/embeddings/instagram_pmp_l1
# Diagnostics
.venv-exp1/bin/python experiments/gnn/diagnose_selection.py --datasets instagram reddit --variants pmp_l1 pmp_l2
# Pipeline (GPU, resumable)
HF_HOME=/workspace/.hf_home_exp1 HF_HUB_OFFLINE=1 .venv-exp1/bin/python scripts/run_queue.py \
    experiments/gnn/configs/instagram_pmp_l1.yaml --gpus 0 --shards 1 --out experiments/gnn/runs/queue
# Comparison
.venv-exp1/bin/python scripts/compare_arms.py --reference "Ours K=1"=configs/ours/instagram_k1.yaml \
    --arm "Ours + PMP-L1 cosine"=experiments/gnn/configs/instagram_pmp_l1.yaml --holm-family all \
    --json experiments/gnn/runs/eval/2026-11-14-instagram-pmp-l1-vs-ours-k1.json \
    --markdown experiments/gnn/runs/eval/2026-11-14-instagram-pmp-l1-vs-ours-k1.md
```

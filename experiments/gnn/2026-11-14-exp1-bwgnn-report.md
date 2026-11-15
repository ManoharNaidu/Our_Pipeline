# Experiment 1 report: BWGNN neighbour selection vs Ours K=1 (Instagram)

Date: 2026-11-14. Decision IDs refer to [`2026-11-14-exp1-decisions.md`](2026-11-14-exp1-decisions.md) (BWGNN: BW1–BW9;
statistics: E1, E5). The PMP-L1 arm is reported in [`2026-11-14-exp1-report.md`](2026-11-14-exp1-report.md).

## Result

**Choosing neighbours by cosine similarity on BWGNN-C1 embeddings does not improve Ours on Instagram.** AUROC, the
primary metric, is unchanged within noise (+0.18, 95% CI −1.05 to +1.44, Holm p 1.0; higher on 2 of 5 seeds).
Macro-F1 is lower on average (−1.98, bootstrap CI −2.72 to −1.16) but not significantly over seeds (paired t-test
p 0.29, Holm p 1.0), because two seeds collapsed to a low threshold (see Per seed). AUPRC is unchanged (−0.17, Holm
p 1.0).

## Instagram, test set, mean ± population std over seeds 0–4

| Method | Macro-F1 | AUROC | AUPRC | Mean prompt tokens |
|---|---|---|---|---|
| Ours K=1 (diffusion) | **53.86** ± 0.87 | 58.86 ± 1.19 | **14.18** ± 1.26 | 70.5 |
| Ours + PMP-L1 cosine | 51.90 ± 0.72 | 58.57 ± 1.14 | 13.65 ± 0.75 | 70.6 |
| Ours + BWGNN-C1 cosine | 51.88 ± 2.66 | **59.04** ± 0.68 | 14.01 ± 0.88 | 70.6 |

| Arm minus Ours K=1 | Metric | Δ (seeds better) | 95% CI | p | Holm |
|---|---|---|---|---|---|
| BWGNN-C1 | Macro-F1 | −1.98 (2/5) | [−2.72, −1.16] | 0.2907 | 1.0000 |
| BWGNN-C1 | AUROC | +0.18 (2/5) | [−1.05, +1.44] | 0.8335 | 1.0000 |
| BWGNN-C1 | AUPRC | −0.17 (2/5) | [−0.93, +0.61] | 0.7230 | 1.0000 |
| PMP-L1 | Macro-F1 | −1.96 (0/5) | [−2.85, −1.04] | 0.0109 | 0.0651 |
| PMP-L1 | AUROC | −0.29 (3/5) | [−1.49, +0.90] | 0.7943 | 1.0000 |
| PMP-L1 | AUPRC | −0.53 (1/5) | [−1.38, +0.31] | 0.4445 | 1.0000 |

- `scripts/compare_arms.py`: paired bootstrap 95% CI over test nodes (2,000 rounds), paired t-test over seeds, Holm
  over all 6 tests against Ours K=1 (E1, E5). Files: `runs/compare/2026-11-14-compare-arms-instagram-pmp-bwgnn.{md,json}`,
  converted into `runs/compare/significance.json` (old version kept as `2026-11-14-significance.json.bak`).
- With the 6-test family, PMP-L1's Macro-F1 drop (raw p 0.011) is no longer significant after Holm (0.065; it was
  0.033 in the 3-test family of E3).
- No test against the original DGP (E4). BWGNN-C1 was not compared directly with PMP-L1.

## Per seed

| Seed | Best epoch | Val AUROC | Val Macro-F1 | Val threshold | Test AUROC | Test Macro-F1 | Test AUPRC | Ours K=1 test Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 63.26 | 56.06 | 0.10 | 60.01 | 54.60 | 15.47 | 54.31 |
| 1 | 2 | 62.06 | 47.63 | 0.05 | 58.22 | 47.62 | 13.67 | 54.63 |
| 2 | 4 | 61.42 | 56.06 | 0.10 | 59.64 | 52.20 | 12.92 | 52.99 |
| 3 | 3 | 61.92 | 54.62 | 0.10 | 58.72 | 54.60 | 14.47 | 52.65 |
| 4 | 4 | 63.57 | 49.98 | 0.05 | 58.59 | 50.39 | 13.52 | 54.72 |

Observation (not used for any choice): on seeds 1 and 4 the validation threshold is the lowest of the grid (0.05) and
Macro-F1 falls to 47.6 and 50.4; these two seeds account for the Macro-F1 gap. The AUROC epoch rule (`auroc_es3`) chose
the epoch; test AUROC on those seeds (58.22, 58.59) is close to the other three (58.72 to 60.01).

## Why it does not help (neighbour comparison)

From `runs/compare/2026-11-14-neighbour-comparison-instagram.{md,json}` (also on the dashboard, "neighbours picked"):
over 4,117 targets with more than M = 4 candidates, BWGNN-C1 picks different neighbours from diffusion (mean Jaccard
0.302) and from PMP-L1 (0.288); all three agree on the full set for 0.9% of targets. But the picked neighbours do not
share the target's label more often: validation label agreement 89.4% (diffusion 90.9%, PMP 90.4%, random 91.8%).
BWGNN-C1's own validation AUROC as a fraud detector was 57.98, below PMP-L1's 62.86 (BW5).

## Setup

1. **What changed:** the M = 4 neighbours per metapath (K = 1) are ranked by cosine similarity of 64-dim BWGNN
   (wavelet order C = 1) embeddings instead of by diffusion. Config `configs/instagram_bwgnn_c1.yaml` (hash
   `771aa150b9`) differs from the PMP-L1 config only in the embedding file (BW6).
2. **What stayed the same:** Gemma-2-9B-it summaries and LoRA classifier, prompts, `auroc_es3`, M, K and all
   hyperparameters of Ours K=1. Unchanged node summaries were reused; 37 new node summaries and 6,522 metapath
   summaries were written.
3. **GNN:** homogeneous BWGNN with official settings and the official epoch rule (BW1, BW2); C = 1 chosen over C = 2
   on validation AUROC (BW4).
4. **Runtime:** queue launched 05:17 UTC, last seed finished 07:05 UTC (about 1.8 GPU-h on one A100 80 GB); each seed
   ~16 min of training.

## Files

- Runs: `experiments/gnn/runs/results/raw/instagram_seed{0..4}_771aa150b9/`
- Comparison: `runs/compare/2026-11-14-compare-arms-instagram-pmp-bwgnn.{md,json}`, `runs/compare/significance.json`
- Embeddings and diagnostics: `runs/embeddings/instagram_bwgnn_c1/`, `runs/diagnostics/bwgnn/`
- Log: `logs/2026-11-14-queue-instagram-bwgnn.log`

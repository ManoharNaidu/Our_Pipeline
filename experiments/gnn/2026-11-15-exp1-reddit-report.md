# Experiment 1 report: PMP-L1 and BWGNN-C1 neighbour selection on Instagram and Reddit

Date: 2026-11-15. Decision IDs refer to [`2026-11-14-exp1-decisions.md`](2026-11-14-exp1-decisions.md). Earlier reports:
[`2026-11-14-exp1-report.md`](2026-11-14-exp1-report.md) (PMP-L1, Instagram) and
[`2026-11-14-exp1-bwgnn-report.md`](2026-11-14-exp1-bwgnn-report.md) (BWGNN-C1, Instagram). Those quote Holm values
over 6 tests; this report uses the final 12-test family, which is what `runs/compare/significance.json` and the
dashboard now hold.

## Result

**Neither GNN improves Ours K=1 on either dataset.** No comparison is significant after Holm over the 12 tests
(2 arms × 3 metrics × 2 datasets against Ours K=1, E1).

- **Reddit, PMP-L1:** AUROC is lower on all 5 seeds (Δ −1.05, 95% CI −1.73 to −0.35, raw p 0.009, Holm p 0.107).
- **Reddit, BWGNN-C1:** best mean on all three metrics, but within noise. AUROC +0.46 (CI −0.32 to +1.16, Holm p 1.0,
  higher on 2 of 5 seeds), Macro-F1 +0.73 (bootstrap CI +0.22 to +1.24, seed t-test p 0.71, Holm p 1.0).
- **Instagram:** unchanged from the earlier reports. PMP-L1's Macro-F1 drop (raw p 0.011) has Holm p 0.119 with 12
  tests (0.065 with 6).

## Test set, mean ± population std over seeds 0–4

| Dataset | Method | Macro-F1 | AUROC | AUPRC | Mean prompt tokens |
|---|---|---|---|---|---|
| Instagram | Ours K=1 (diffusion) | **53.86** ± 0.87 | 58.86 ± 1.19 | **14.18** ± 1.26 | 70.5 |
| | Ours + PMP-L1 cosine | 51.90 ± 0.72 | 58.57 ± 1.14 | 13.65 ± 0.75 | 70.6 |
| | Ours + BWGNN-C1 cosine | 51.88 ± 2.66 | **59.04** ± 0.68 | 14.01 ± 0.88 | 70.6 |
| Reddit | Ours K=1 (diffusion) | 53.58 ± 3.20 | 65.37 ± 1.89 | 15.17 ± 1.23 | 213.8 |
| | Ours + PMP-L1 cosine | 53.62 ± 2.77 | 64.32 ± 1.80 | 14.56 ± 0.94 | 213.8 |
| | Ours + BWGNN-C1 cosine | **54.31** ± 1.58 | **65.82** ± 1.33 | **15.20** ± 0.74 | 213.8 |

| Dataset | Arm minus Ours K=1 | Metric | Δ (seeds better) | 95% CI | p | Holm (12 tests) |
|---|---|---|---|---|---|---|
| Instagram | PMP-L1 | Macro-F1 | −1.96 (0/5) | [−2.85, −1.04] | 0.0109 | 0.1194 |
| | PMP-L1 | AUROC | −0.29 (3/5) | [−1.49, +0.90] | 0.7943 | 1.0000 |
| | PMP-L1 | AUPRC | −0.53 (1/5) | [−1.38, +0.31] | 0.4445 | 1.0000 |
| | BWGNN-C1 | Macro-F1 | −1.98 (2/5) | [−2.72, −1.16] | 0.2907 | 1.0000 |
| | BWGNN-C1 | AUROC | +0.18 (2/5) | [−1.05, +1.44] | 0.8335 | 1.0000 |
| | BWGNN-C1 | AUPRC | −0.17 (2/5) | [−0.93, +0.61] | 0.7230 | 1.0000 |
| Reddit | PMP-L1 | Macro-F1 | +0.04 (2/5) | [−0.48, +0.60] | 0.9352 | 1.0000 |
| | PMP-L1 | AUROC | −1.05 (0/5) | [−1.73, −0.35] | 0.0089 | 0.1074 |
| | PMP-L1 | AUPRC | −0.62 (1/5) | [−1.15, −0.02] | 0.0459 | 0.4590 |
| | BWGNN-C1 | Macro-F1 | +0.73 (3/5) | [+0.22, +1.24] | 0.7105 | 1.0000 |
| | BWGNN-C1 | AUROC | +0.46 (2/5) | [−0.32, +1.16] | 0.4694 | 1.0000 |
| | BWGNN-C1 | AUPRC | +0.03 (2/5) | [−0.51, +0.59] | 0.9375 | 1.0000 |

- `scripts/compare_arms.py`: paired bootstrap 95% CI over test nodes (2,000 rounds), paired t-test over seeds (p),
  Holm over all 12 tests against Ours K=1 (E1, E6). Files:
  `runs/compare/2026-11-15-compare-arms-instagram-reddit-pmp-bwgnn.{md,json}`, converted into
  `runs/compare/significance.json` (the 6-test version is kept as `runs/compare/2026-11-15-significance.json.bak`).
- The bootstrap CI and the seed t-test can disagree: the CI treats test nodes as the unit, the t-test the 5 seeds.
  BWGNN-C1's Reddit Macro-F1 gain has a CI above zero but p 0.71, because most of it comes from one seed (see below).
- No test against the original DGP (E4). BWGNN-C1 was not compared directly with PMP-L1 (BW9).

## Against the pipeline's other results (context, not paired tests)

From `docs/RESULTS.md`, test AUROC, mean ± std over 5 seeds:

| Dataset | DGP reproduced | Ours K=1 | Ours K=2 | Ours + PMP-L1 | Ours + BWGNN-C1 |
|---|---|---|---|---|---|
| Instagram | 58.27 ± 2.64 | 58.86 ± 1.19 | **60.20** ± 1.76 | 58.57 ± 1.14 | 59.04 ± 0.68 |
| Reddit | 62.26 ± 1.52 | 65.37 ± 1.89 | 65.01 ± 1.00 | 64.32 ± 1.80 | **65.82** ± 1.33 |

On Instagram, Ours K=2 is still the highest. On Reddit, BWGNN-C1 has the highest mean, but its gap to Ours K=1 is not
significant (above).

## Reddit per seed (test set; validation values chose the epoch and threshold)

| Seed | Arm | Best epoch | Val AUROC | Val Macro-F1 | Threshold | Test Macro-F1 | Test AUROC | Test AUPRC |
|---|---|---|---|---|---|---|---|---|
| 0 | Ours K=1 | 1 | 66.01 | 47.62 | 0.10 | 47.62 | 62.31 | 13.05 |
| | PMP-L1 | 3 | 65.57 | 48.14 | 0.05 | 48.14 | 61.38 | 13.08 |
| | BWGNN-C1 | 2 | 66.22 | 55.53 | 0.10 | 54.16 | 63.32 | 13.90 |
| 1 | Ours K=1 | 2 | 70.93 | 55.89 | 0.05 | 56.49 | 66.69 | 16.26 |
| | PMP-L1 | 2 | 68.63 | 55.64 | 0.05 | 55.72 | 65.78 | 15.35 |
| | BWGNN-C1 | 2 | 69.75 | 57.23 | 0.05 | 55.42 | 66.23 | 15.84 |
| 2 | Ours K=1 | 2 | 67.99 | 56.59 | 0.10 | 54.96 | 64.05 | 14.63 |
| | PMP-L1 | 2 | 68.27 | 57.91 | 0.10 | 54.45 | 63.41 | 14.26 |
| | BWGNN-C1 | 3 | 69.09 | 54.25 | 0.10 | 55.53 | 66.49 | 15.29 |
| 3 | Ours K=1 | 2 | 68.23 | 53.00 | 0.10 | 52.99 | 66.49 | 15.54 |
| | PMP-L1 | 3 | 69.38 | 56.19 | 0.05 | 54.80 | 64.58 | 14.32 |
| | BWGNN-C1 | 3 | 69.44 | 55.98 | 0.05 | 55.15 | 65.85 | 14.98 |
| 4 | Ours K=1 | 2 | 70.74 | 55.96 | 0.05 | 55.84 | 67.30 | 16.36 |
| | PMP-L1 | 2 | 69.48 | 56.44 | 0.05 | 55.01 | 66.45 | 15.76 |
| | BWGNN-C1 | 3 | 71.79 | 52.33 | 0.05 | 51.29 | 67.23 | 15.97 |

Observations (not used for any choice):

- PMP-L1's test AUROC is below Ours K=1 on every seed, by 0.64 to 1.91 points.
- BWGNN-C1's test AUROC is above Ours K=1 on seeds 0 and 2 (+1.01, +2.44) and below on seeds 1, 3 and 4 (−0.46, −0.64,
  −0.07).
- On seed 0, Ours K=1 scored test Macro-F1 47.62 (it was 47.62 on validation too), while BWGNN-C1 scored 54.16. This
  one seed accounts for most of BWGNN-C1's mean Macro-F1 gain; on seed 4, BWGNN-C1 fell to 51.29 against 55.84.
- Instagram per-seed tables are in the two earlier reports.

## Why the effect is small (neighbour comparison)

From `runs/compare/2026-11-14-neighbour-comparison-reddit.{md,json}` and
`runs/diagnostics/bwgnn_reddit/2026-11-14-phase-c-diagnostics.md` (also on the dashboard, "neighbours picked"):

- Only 2,218 of 18,389 Reddit targets (12.1%) have more than M = 4 candidate neighbours (Instagram: 51.8%). Selection
  can change the prompt only for those targets; for the rest all three methods give the same neighbours.
- Among those targets, the GNN methods do pick different neighbours: mean Jaccard against diffusion 0.511 (PMP-L1) and
  0.489 (BWGNN-C1); all three agree on the full set for 3.6% of them.
- Validation label agreement of the picked neighbours (share with the target's label, train/validation labels only):
  BWGNN-C1 73.8%, PMP-L1 71.8%, diffusion 69.8%, random 69.4% (206 validation targets, only 48 fraud).
- GNN validation AUROC as a fraud detector: BWGNN-C1 68.19, PMP-L1 67.66 (Instagram: 57.98 and 62.86).
- PMP-L1 also had higher label agreement than diffusion yet gave lower test AUROC, so these diagnostics did not predict
  the result on their own. Reddit PMP-L1's best GNN epoch is 0 (G5).

## Setup

1. **What changed:** the M = 4 neighbours per metapath (K = 1) are ranked by cosine similarity of GNN embeddings
   instead of by diffusion. Reddit configs `configs/reddit_pmp_l1.yaml` (hash `22ce4ff044`) and
   `configs/reddit_bwgnn_c1.yaml` (hash `98520ee79e`) differ only in `method.selection_embeddings`.
2. **What stayed the same:** Gemma-2-9B-it summaries and LoRA classifier, prompts, `auroc_es3`, M, K and all
   hyperparameters of Ours K=1 (`configs/ours/reddit_k1.yaml`).
3. **GNNs:** PMP-L1 as in D1–D11 and G1; BWGNN with official settings (BW1, BW2), C = 1 chosen over C = 2 on GNN
   validation AUROC, 68.19 against 67.36 (BW10).
4. **Runtime:** queue started 09:28 UTC 2026-11-14 and ended 07:24 UTC 2026-11-15 (about 21.9 h on one A100 80 GB),
   all jobs rc 0. Each Reddit seed took 6,705–7,935 s.
5. **Scope:** the Reddit runs were made at the user's request ("run BWGNN and PMP on reddit as well, same as
   instagram"), reversing S5 and G6 (S7). Amazon Instruments was not run.

## Files

- Runs: `runs/results/raw/reddit_seed{0..4}_22ce4ff044/` (PMP-L1), `runs/results/raw/reddit_seed{0..4}_98520ee79e/`
  (BWGNN-C1); baseline Ours K=1 in the repo's `runs/results/raw/reddit_seed{0..4}_5810c90d43/`
- Comparison: `runs/compare/2026-11-15-compare-arms-instagram-reddit-pmp-bwgnn.{md,json}`, `runs/compare/significance.json`
- Neighbours: `runs/compare/2026-11-14-neighbour-comparison-reddit.{md,json}`
- Embeddings and diagnostics: `runs/embeddings/reddit_bwgnn_c{1,2}/`, `runs/diagnostics/bwgnn_reddit/`
- Logs: `runs/queue/status.jsonl`, `logs/2026-11-14-train-gnn-reddit-bwgnn-c{1,2}.log`

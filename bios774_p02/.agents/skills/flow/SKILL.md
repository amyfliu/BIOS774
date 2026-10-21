---
name: flow
description: Configure, train, check and use a RealNVP normalizing flow (tools/train.py --model flow), mainly for exact likelihoods / anomaly scoring.
---

# Model (tools/models/flow.py)
Affine coupling layers + ActNorm (L16). Invertible, so log p(x) is exact
via the change-of-variables formula; latent z has the same dimension as x
(no dimension reduction).
- `mlp` backbone: tabular / low-dimensional data. Keep D modest (≤ ~100):
  use `pca:K` or `hvg` in the prep.
- `conv` backbone: images; checkerboard masks; the input is dequantized
  8-bit pixels mapped through a logit — REQUIRES `--prep scale01`.
  Likelihoods are reported in bits/dim (lower = higher likelihood).

# Hyperparameters — starting points
`--params '{"n_layers": 6-8, "hidden": 32-64}'` (conv) or
`'{"n_layers": 8, "hidden": 256}'` (mlp); `--lr 1e-3`; `--batch-size 64`.
Images: train on a subsample (e.g. 10k–20k) within the budget.

# Health checks
- `nll_per_dim` / `bits_per_dim` decreasing; typical 28×28 RGB natural
  textures land around 3–5 bits/dim for small flows.
- `final_diagnostics.latent_looks_gaussian`: training data mapped to z
  should have mean ≈ 0 and sd ≈ 1.

# Anomaly / OOD use (the main reason to pick a flow)
1. Train on the NORMAL data only (`--include-classes` / `--exclude-classes`).
2. `python tools/score.py --run R --split test`
3. `python tools/evaluate.py --run R --tasks anomaly --anomaly-classes ...`
4. **Known failure mode — likelihood tracks complexity, not semantics**
   (Nalisnick et al., 2019): flows often give HIGHER likelihood to simple,
   smooth inputs (e.g. blank background) than to the training data itself.
   Always read `anomaly.complexity_check`:
   - strong negative `spearman_logp_vs_compressed_size` and
     `auroc_of_complexity_alone` close to the model's AUROC → the
     likelihood mostly measures image complexity;
   - `auroc_complexity_corrected_score` (model code length minus a generic
     compressor's code length; Serrà et al., 2020) often fixes this — if it
     is clearly better, recommend the corrected score and say why.
   - Compare against `baseline_pca_recon_error_auroc` too. A flow that does
     not beat trivial baselines should not be recommended as-is.
5. `mean_logp_per_dim_by_class` shows WHICH classes look typical/atypical.

# Sampling
Possible (`--n`, `--temperature 0.7–0.9` gives cleaner samples) but flow
samples on images are usually poor; don't use them for synthetic release.

---
name: preprocessing
description: Choose the --prep spec, i.e. the space in which the generative model is trained and in which synthetic samples will live.
---

# When to apply
After interpreting the objective, before choosing the model family (the two
interact — re-check this once the family is known).

# Key idea
The generative model learns the distribution of the PREPROCESSED data. The
synthetic samples therefore live in that "model space" (e.g. log-normalized
expression of 2,000 genes), not in the raw format. Choose the space the
downstream user actually needs, and state it in the recommendation.

# Available steps (`tools/common.py::Prep`; comma-separated, applied in order)
`libnorm[:T]`, `log1p`, `hvg:N`, `standardize`, `pca:K`, `scale01`,
`scale_pm1`, `none`. All statistics are fit on the train split.

# Tabular / count data (e.g. single-cell RNA-seq)
Evidence to check in `profile.json`: `integer_valued`, `fraction_zeros`,
`row_sum.max_over_min`, `skewness_median_feature`,
`variance_share_top{N}_features`, `after_log1p_variance_share_top2000`,
`samples_per_feature`.

- Integer counts with very different row sums (max/min ≫ 1) → `libnorm`
  first, otherwise the model mostly learns sequencing depth.
- Heavy right skew → `log1p` (the Gaussian decoders/noise used by every
  family here assume roughly symmetric features).
- Many more features than samples (`samples_per_feature` ≪ 1) → reduce:
  `hvg:N` keeps interpretable features (N ≈ 1,000–3,000 for scRNA-seq;
  check how much log-variance the top N retain). Add `pca:K` (K ≈ 30–50)
  when the family needs a low dimension relative to n — normalizing flows
  and GANs on small n.
- `standardize` puts features on one scale (helps optimization; with a
  learned per-feature variance in the VAE it is optional). Synthetic values
  are then in z-score units — say so.

# Images
- DDPM / WGAN with conv backbone: `scale_pm1` (output layer uses tanh;
  DDPM clips its x0 prediction to [-1, 1]).
- Conv normalizing flow: `scale01` is REQUIRED (the flow models 8-bit
  pixels with dequantization; likelihoods are then in bits/dim).
- VAE conv: `scale01` or `scale_pm1` both work.
- Do not flatten images for conv models; use `--backbone conv`. Use `mlp`
  on images only for a deliberately cheap baseline.

# Subsampling (`--max-train`)
A compute decision, not a preprocessing one, but log it here too: when n
is large relative to the budget, a stratified subsample (e.g. 10k–30k
images) usually costs little quality. Never subsample the rare classes you
are trying to model — `--max-train` is stratified, but check
`run_config.json → train_class_counts`.

---
name: preprocessing
description: Decide what preprocessing (log-transform, standardization) to apply before running dimension reduction.
---

# When to apply this skill
After `inspect_data.py`, before `run_dr.py` — the chosen `--preprocessing`
value is one of the decisions this skill governs.

# Options available in run_dr.py
`none`, `standardize`, `log1p`, `log1p_standardize`

# Decision guidance
- **High `sparsity_fraction` with a wide `value_range`** (e.g. raw count
  data like scRNA-seq): count distributions are typically right-skewed with
  a long tail of highly-expressed features dominating variance under raw
  Euclidean distance. `log1p` (or `log1p_standardize`) is the standard choice
  here — state this explicitly rather than defaulting silently.
- **High `feature_scale_ratio_max_over_min_std`**: features on very different
  scales will dominate any Euclidean-distance-based method (PCA, MDS, k-NN
  graphs) purely due to units, not signal. `standardize` (z-score per
  feature) is appropriate unless the raw scale is itself meaningful (e.g.
  all features already share one physical unit, like pixel intensities all
  in [0,255] — there, standardizing is a real choice with tradeoffs, not a
  default).
- **`n_constant_columns` > 0**: drop these before fitting anything — they
  contribute nothing and can break some solvers (division by zero variance).
- **Missing values (`missing_fraction` > 0)**: run_dr.py's models generally
  assume no NaNs. Decide and log an explicit imputation or removal strategy
  before calling run_dr.py — don't let a tool silently drop rows.

# What to record
Whatever preprocessing choice is made, `run_dr.py`'s `run_config.json`
output records exactly which mode was used — make sure the final report
quotes this rather than re-describing it from memory, so the report and the
reproducible artifact can't drift apart.

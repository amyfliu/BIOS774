---
name: evaluation
description: How to interpret the metrics evaluate.py produces, and when to decide a result is or isn't good enough.
---

# When to apply this skill
After `run_dr.py` has produced an embedding, before writing the final report
section for that run.

# Metrics evaluate.py produces
- `trustworthiness` (0-1, higher better): measures how well the embedding
  preserves each point's high-dimensional nearest neighbors. This is the
  primary *general-purpose* quality metric — it doesn't require labels.
  Rough calibration: >0.9 strong, 0.7-0.9 acceptable for visualization
  purposes, <0.7 suggests the embedding is distorting local structure a lot.
  It only checks *local* neighborhoods: a high value says nothing about
  whether distances between groups, or the global layout, are faithful.
- `silhouette_score_on_labels` (-1 to 1, higher better; only computed if the
  dataset has labels): measures how well-separated the *given* label groups
  are in the embedding. This is NOT a general DR-quality metric — a method
  can have poor silhouette on labels that don't correspond to the data's
  actual geometric structure (e.g. labels reflecting a variable the
  embedding was never meant to separate). Interpret it as "does this
  embedding visually support the known grouping," not as ground truth.
  It is computed in the embedding Z, not in the original data, so on its
  own it cannot tell you whether the groups are separated in the original
  space — that is what the next two metrics are for.
- `silhouette_input_space`: the same silhouette, on the same points, computed
  in the preprocessed input data (before dimension reduction).
- `silhouette_gap_embedding_minus_input`: embedding minus input. Positive
  means the groups look more separated in the plot than in the input data.
- `silhouette_n_points`: both silhouettes use the same random subsample
  (default 5000 points, `--silhouette_sample_size`, `--seed`) because
  silhouette is O(n^2). Report the subsample size.

  **How to read the gap — relative, never absolute.** In high-dimensional
  input, Euclidean distances are dominated by the many dimensions that
  don't carry the group signal, so `silhouette_input_space` is usually
  small even when groups are genuinely distinct. Any method that
  concentrates the signal into 2 dimensions (PCA included) will therefore
  show a large positive gap. Example from testing: three clearly distinct
  groups in 20-D gave input-space silhouette 0.08 but PCA and UMAP
  silhouettes around 0.6. So:
  - Compare each method's gap to **PCA's gap on the same dataset**. PCA
    is linear and does not create separation, so its gap is the baseline
    for "signal concentration."
  - A t-SNE/UMAP gap clearly larger than PCA's is the warning sign: the
    extra separation may come from the method, not the data.
  - A gap similar to PCA's means the method shows about the same
    separation a linear projection finds.

# Interpreting separation in t-SNE / UMAP plots (read before writing)
t-SNE and UMAP outputs are visualizations of particular relationships in the
data, not faithful low-dimensional reconstructions. Distances between groups,
cluster sizes, densities and the apparent separation of groups in Z need not
match the same quantities in the original data. Both methods tend to pull
groups apart visually, so a clean-looking plot and a high silhouette score
are expected even when the groups overlap in the original space.

A cautionary example from lecture (L11): a genomic analysis from the All of
Us Research Program showed a UMAP plot colored by self-identified race and
ethnicity. The visual separation of groups was widely read as evidence of
discrete genetic differences between groups, which the plot could not
support. Coloring an embedding by labels invites exactly this reading.

What to do:
- **Compare against the PCA baseline.** Use the silhouette gaps as
  described above. If a t-SNE/UMAP gap is clearly larger than PCA's, treat
  the extra separation as possibly created by the method, not as a
  finding about the data, and say so in the report.
- **Word claims at the right level.** Acceptable: "cells with the same
  annotated type are placed in the same neighborhood of the embedding."
  Not acceptable: "cell types form distinct, well-separated clusters," "group
  A is closer to group B than to group C," or "cluster A is larger."
- **Don't interpret between-cluster distances or cluster sizes** from t-SNE
  or UMAP at all, even when trustworthiness is high.
- **Labels are for evaluation only.** They must never be treated as
  confirmed by the plot; the plot can at most be consistent with them.

# Method-specific notes
- **PCA, classical MDS**: global distances and variance are meaningful, so
  between-group distances can be discussed, but only for the variance the
  shown components capture — report that fraction.
- **Isomap**: distances approximate geodesic distances along the manifold;
  a disconnected or short-circuited k-NN graph invalidates this.
- **LLE, Laplacian Eigenmaps**: preserve local relationships only; don't
  interpret global distances.
- **Diffusion Map**: Euclidean distance in Z approximates diffusion distance
  at time `t`, truncated to the shown components. Report `alpha` and `t`,
  since both change what "close" means. If the run warned that the k-NN
  graph is disconnected, the leading coordinates reflect graph components,
  not geometry — say so rather than interpreting them as structure.

# How to decide if a result is "good enough"
There's no dataset-independent pass/fail threshold — report:
1. The trustworthiness number and what it implies about local-structure fidelity.
2. Whether the labeled groups (if present) visually and numerically separate,
   with the embedding silhouette, input-space silhouette, and gap reported
   side by side, and the gap compared against PCA's.
3. A comparison across the methods actually run on this dataset — relative
   comparison is more informative than an absolute cutoff.
4. Any caveat specific to the method (see above).

# What to write in the report for each run
- Method + preprocessing + key hyperparameters (pull from `run_config.json`,
  don't restate from memory).
- The evaluation numbers.
- One or two sentences connecting the numbers to what they imply about the
  data's structure — this is the "quality of interpretation" the assignment
  grades, not just reporting numbers. Keep these claims within what the
  method can support (see the interpretation section above).
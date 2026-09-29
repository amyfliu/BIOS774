---
name: method-selection
description: Decide which dimension reduction method(s) to run, given a dataset profile from inspect_data.py.
---

# When to apply this skill
After running `tools/inspect_data.py` and reading the dataset's `DATA.md`, before calling `tools/run_dr.py`.

# Inputs to base the decision on
From `outputs/<dataset>/profile.json`:
- `n_samples`, `n_features` — sample size relative to dimensionality
- `sparsity_fraction` — near-zero fraction (e.g. count data)
- `feature_scale_ratio_max_over_min_std` — how heterogeneous feature scales are
- `rank_estimate_from_svd.dims_for_90pct_variance` — rough linear-structure signal:
  low value suggests strong linear structure; high value suggests linear
  methods alone may be insufficient

Also use `DATA.md` for what the *expected* structure is (discrete cell types?
continuous differentiation? tissue classes?). The profile tells you about the
numbers; `DATA.md` tells you what kind of structure is worth recovering. Both
should appear in your justification.

# Methods available in run_dr.py
`pca`, `kernel_pca`, `mds`, `isomap`, `lle`, `laplacian_eigenmaps`,
`diffusion_map`, `tsne`, `umap`.

PHATE, TriMap and PaCMAP are discussed below because they are sometimes the
theoretically better fit, but they are NOT implemented in `run_dr.py`. If you conclude one of them would be the best choice, say so
explicitly in the report (as a limitation / recommended extension) and run the
closest available method instead — do not silently substitute.

# First, decide what the embedding is for
Methods differ in *which relationships they preserve*. Pick the goal first,
then the method — this is the core of a defensible justification.

| Goal | What to preserve | Methods |
|---|---|---|
| Faithful low-dim representation, d may be > 2 | Global variance / distances | PCA, (classical) MDS |
| Nonlinear manifold, global geometry along it | Geodesic distances | Isomap |
| Nonlinear manifold, local geometry only | Local linear reconstructions | LLE |
| Graph / cluster structure | Local neighborhood connections | Laplacian Eigenmaps, Diffusion Map |
| Continuous trajectories and branches | Diffusion geometry | Diffusion Map, PHATE* |
| 2-D visualization of local neighborhoods and group separation | Local neighborhoods | t-SNE |
| 2-D visualization of local structure plus some larger-scale layout | Local neighborhoods + some larger-scale organization | UMAP |
| 2-D visualization balancing local, intermediate and global scale | Relations at multiple scales | TriMap*, PaCMAP* |

\* not in `run_dr.py` (see above).

The spectral methods (PCA, MDS, Isomap, LLE, Laplacian Eigenmaps, Diffusion Map) have
closed-form eigendecomposition solutions and can produce d > 2 coordinates.
The visualization methods (t-SNE, UMAP) are fitted by numerical optimization,
are meant for d = 2 or 3, and depend on random initialization — fix and report
the seed.

# Decision guidance (not a rigid lookup table — reason from the evidence)

- **Always run PCA first**, regardless of what else you run. It's cheap, it's
  a baseline, and `rank_estimate_from_svd` from inspect_data.py already gives
  you a preview of how much a linear method alone will capture.
- **Don't run classical MDS on Euclidean distances alongside PCA.** Classical
  MDS on Euclidean distances gives the same embedding as PCA scores (both
  come from the same centered Gram matrix), so it adds nothing. MDS is only
  worth running with a *different* dissimilarity. Note that Isomap is exactly
  classical MDS applied to graph-geodesic distances — that is where the
  "MDS idea" earns its place.
- **If `dims_for_90pct_variance` is small relative to `n_features`** (data is
  close to linear/low-rank): PCA (or a variant like Kernel PCA) alone may be
  sufficient. Don't reach for a nonlinear method just because it's available —
  justify the extra complexity.
- **If `dims_for_90pct_variance` stays high** (linear structure explains little):
  this suggests nonlinear structure. Then use `DATA.md` to decide which kind:
  - *Smooth low-dimensional manifold, and global position along it matters*
    → Isomap. It relies on k-NN graph shortest paths approximating geodesic
    distances, so it is fragile if the graph is disconnected or has
    "short-circuit" edges between distant parts of the manifold.
  - *Smooth manifold, only local geometry matters* → LLE. See the LLE
    constraints below.
  - *Graph or cluster structure more than a smooth manifold* → Laplacian
    Eigenmaps. It only uses connection strengths between neighbors, so it is
    simpler and less sensitive to neighbor configuration than LLE.
  - *Continuous processes (trajectories, branching — e.g. cell
    differentiation)* → Diffusion Map. PHATE is designed for exactly this
    too, but is not available; mention it as an extension if relevant.
  - *Diffusion Map vs Laplacian Eigenmaps*: both build a k-NN graph and take
    eigenvectors of a random walk on it. Diffusion Map adds density
    normalization, which matters when some regions of the data are sampled
    much more densely than others (common in scRNA-seq: abundant vs rare
    cell types). See the Diffusion Map parameters below.
  - *Visualizing discrete groups* → t-SNE or UMAP. t-SNE emphasizes local
    neighborhoods and separation of groups but says little about how groups
    are arranged relative to each other; UMAP usually keeps more large-scale
    organization but does not faithfully preserve global geometry either.
- **LLE-specific constraints.** Each point is reconstructed from its k
  neighbors, which assumes each neighborhood is well approximated by an
  affine (flat) patch — if the data are noisy or curved at the scale of k,
  the weights become unstable. The local k×k Gram matrix is only invertible
  if k ≤ p (neighbors ≤ input dimension); in particular, if you run LLE on a
  PCA-reduced input with few components, make sure `n_neighbors` is below the
  number of components or that regularization is applied, and state which.
- **Diffusion Map parameters** (pass via `--params`):
  - `alpha` (default 1.0). 0 gives the same operator as Laplacian
    Eigenmaps, so density differences shape the embedding. 1 removes the
    density effect and recovers the intrinsic geometry of the manifold.
    0.5 sits in between. State which you used and why: if you care about
    shape (trajectories), use 1; if density itself is informative, a lower
    value is defensible.
  - `t` (default 1, integer diffusion time). Larger t shrinks the
    fast-decaying coordinates, so the embedding reflects coarser,
    longer-range structure.
  - `n_neighbors` (default 15) and `epsilon` (default: median squared k-NN
    distance) set the local kernel scale.
  - If the run warns that the k-NN graph is disconnected, the leading
    coordinates mostly separate graph components rather than showing
    geometry. Increase `n_neighbors` and rerun, or report it.
- **`n_samples` vs `n_features`**: Isomap/LLE/Laplacian Eigenmaps/Diffusion Map depend on a
  k-NN graph — they need enough samples for that graph to be meaningful.
  Be cautious applying them when `n_samples` is small (rule of thumb: want
  `n_samples` at least in the low hundreds, and `n_neighbors` well below
  `n_samples`).
- **Neighborhood size is a scale choice, not a nuisance parameter.** For
  UMAP `n_neighbors` (and t-SNE perplexity), smaller values emphasize local
  structure and larger values let broader relationships shape the
  embedding. Report the value and why it suits the expected structure.
- **High sparsity** (count data like scRNA-seq): raw Euclidean-distance
  methods (MDS, Isomap, LLE) are sensitive to the preprocessing decision
  (log-transform, normalization) made beforehand — check the preprocessing
  skill and make sure that decision is made and logged before running these.
- **Large `n_samples`** (tens of thousands+): MDS, Kernel PCA, and Isomap
  each build a dense n x n matrix internally (dissimilarity / kernel /
  geodesic-distance matrix respectively) — at n ~ 90,000 that matrix alone
  is ~65GB, infeasible regardless of hardware. t-SNE (sklearn's default
  Barnes-Hut) avoids that dense-matrix wall but is still much slower than
  PCA/UMAP at this scale. Prefer PCA, UMAP, Diffusion Map, LLE, or Laplacian
  Eigenmaps (all use a sparse k-NN graph, so they scale to large n), or
  subsample for a method that doesn't scale.
  - **Sizing the subsample**: cap it at roughly 2,000-5,000 rows via
    `run_dr.py --subsample`. Justify this from algorithmic complexity, not
    a timing test on whatever machine happens to run the agent — SMACOF-style
    methods cost O(n^2) per iteration, so 2,000-5,000 keeps the per-iteration
    cost and memory footprint negligible on essentially any hardware, while a
    number derived from "how long it took on my laptop" may not hold on a
    different machine (e.g. a grader's). Aligning with this range also
    matches `evaluate.py`'s own default evaluation-metric subsample cap
    (`--silhouette_sample_size` / `--trustworthiness_sample_size`, both
    default 5000), so the embedding itself isn't smaller than what the
    evaluation step could have used anyway. State the exact size and seed
    used in the report — `run_dr.py --subsample` records both automatically
    in `run_config.json`'s `subsample` field.
- **Compare, don't just pick one.** Running 2-3 methods that preserve
  *different* relationships (e.g. PCA + one manifold method + one
  visualization method) and comparing their evaluation metrics
  (trustworthiness, silhouette-on-labels) produces a more defensible report
  than a single unjustified choice — the assignment explicitly allows "one or
  more" methods.

# Interpreting t-SNE / UMAP output (applies whenever you choose them)
These embeddings are visualizations of particular relationships, not faithful
reconstructions of the original geometry. Distances between groups, cluster
sizes, densities and the apparent separation of groups need not correspond to
the same quantities in the original data. Coloring an embedding by labels and
seeing separated groups is NOT evidence that the groups are separated in the
original space (see the All of Us UMAP controversy discussed in lecture).
Phrase conclusions accordingly, and back any separation claim with a
quantitative metric computed in a setting where it is meaningful. See the
evaluation skill for how to phrase this in the report.

# What NOT to do
- Don't run every method on every dataset "to be safe" — the assignment
  wants a *justified* selection, not exhaustive brute force.
- Don't pick a method based on what looks prettiest in the plot without
  checking the quantitative evaluation too.
- Don't skip PCA as a baseline even if you expect it to underperform —
  it's the reference point that makes claims about other methods meaningful.
- Don't run classical MDS on Euclidean distances as if it were a separate
  method from PCA.
- Don't justify a choice only in generic terms ("UMAP is good for
  clusters") — tie it to specific `profile.json` numbers and the expected
  structure from `DATA.md`.
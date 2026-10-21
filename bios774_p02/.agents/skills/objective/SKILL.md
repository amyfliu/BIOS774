---
name: objective
description: Turn the user's objective into a task type, a success criterion and an evaluation plan — and decide whether generative modeling is warranted at all.
---

# When to apply
Right after inspecting the data, before any modeling decision.

# Step 1 — classify the objective

| Task type | Typical wording | What the generative model is FOR | How success is measured (evaluate.py) | Families that fit |
|---|---|---|---|---|
| Synthetic data release / data sharing | "share instead of real data", "privacy", "synthetic dataset" | replace real data for others' analyses | fidelity + memorization + `utility --utility tstr` | conditional DDPM (images), cVAE (tabular), WGAN-GP if n is large |
| Class balancing | "rare classes", "under-represented", "rebalance" | add samples of chosen classes only | `utility --utility augment` (vs class-weighting / oversampling), per-class fidelity, label consistency | cVAE (small n), conditional DDPM (images, enough data) |
| Data augmentation (scarce data) | "small training set", "augment" | add realistic variety to a small real set | `utility --utility augment --real-max N` | as for balancing |
| Anomaly / OOD / QC screening | "flag unusual", "outliers", "quality control" | a density or typicality score for each real sample | `score.py` + `anomaly` (AUROC vs baselines) | flow (exact log p), VAE (ELBO) |
| Representation learning | "latent space", "embedding", "features" | an encoder z = f(x) | label consistency / downstream classifier on z (report qualitatively if no tool) | VAE (L14) |
| Imputation / denoising | "missing values", "reconstruct" | p(x_missing \| x_observed) | reconstruction error on held-out masked entries | VAE, DDPM (not supported by the current tools — say so) |
| Uncertainty quantification | "how confident", "how typical" | likelihoods / posterior spread | calibration of scores (qualitative here) | flow, VAE |

If an objective mixes types, name the primary one and evaluate that first.
If the requested task is not supported by the tools (e.g. imputation),
state it, do the closest supported analysis, and flag the gap in the report.

# Step 2 — decide whether to generate at all

Generative modeling is the WRONG first tool when:

- **The problem it would fix is not present.** Balancing a dataset whose
  class imbalance ratio (`profile.json → labels.imbalance_ratio_max_over_min`)
  is below ~3 rarely helps; say so and show the evidence (class counts, and
  optionally a `real_only` vs `real_class_weighted` comparison).
- **A cheap non-generative fix exists and has not been tried.** For class
  imbalance: class weights and random oversampling. These are ALWAYS the
  baselines; synthetic data must beat them to be recommended.
- **There is too little data to learn a distribution.** With ~10 training
  samples in a class, any "generated" sample is an interpolation of those
  few samples. That can still be useful for balancing, but must be described
  honestly (it adds smoothness, not new information).
- **The objective needs real-world evidence**, e.g. evaluating a
  classifier. Test sets must stay real; synthetic data never replaces them.

Declining is a complete, valid outcome: log the decision with the numbers
that justify it, run the cheap supporting evidence, and write the report.
If you decide to generate anyway in a borderline case, say why.

# Step 3 — write down the success criterion BEFORE training
Log it as the `objective` decision. Make it concrete and falsifiable, e.g.

- balancing: "recommend synthetic augmentation only if macro-F1 of
  real+synthetic beats the BEST non-generative baseline and the paired
  bootstrap CI of the difference excludes 0; otherwise recommend the baseline";
- release: "TSTR accuracy ≥ ~0.8× train-on-real at equal size, C2ST clearly
  below 1.0, and no sign of copying (`frac_nearest_real_is_train` close to
  its expected value)";
- anomaly: "AUROC well above 0.5 AND above the trivial baselines reported
  next to it (PCA reconstruction error, image complexity alone)".

Also decide what "the real reference" is (which classes, which split) and
whether labels are needed for conditioning.

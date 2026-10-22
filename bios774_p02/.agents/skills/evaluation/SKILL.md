---
name: evaluation
description: Choose which evaluate.py tasks measure the objective, and interpret every metric relative to its real-vs-real reference.
---

# When to apply
After sampling (or scoring), before deciding whether to iterate, and
again when writing the evaluation section.

# Pick tasks from the objective
| Objective | `--tasks` | extra flags |
|---|---|---|
| synthetic release | `fidelity,memorization,utility` | `--utility tstr` |
| class balancing | `fidelity,memorization,utility` | `--utility augment` (samples of the rare classes only) |
| augmentation for scarce data | `fidelity,utility` | `--utility augment --real-max N` |
| anomaly / QC | `anomaly` (after `score.py`) | `--anomaly-classes ...` |
Run fidelity on every sample set you plan to recommend.

# Golden rule: never read a metric alone
Every fidelity metric comes with `reference_real_train_vs_real_<split>`:
the same metric computed between two REAL samples. That is the best
achievable value at this sample size. Report both and judge the gap.
- The held-out reference is the `val` split when the dataset has one
  (`real_reference_split` in metrics.json), because it comes from the same
  source as the training data. The `test` split may come from a different
  source (e.g. another hospital); then real-train vs real-test differ and the
  classifier tests on `test` also measure "does it transfer to new data".
  Say which reference was used, and mention a source shift if `DATA.md`
  describes one.
- With class-specific samples, the real reference is automatically
  restricted to the generated classes (`real_reference_classes`).

# Fidelity (feature space: `feature_space` in metrics.json)
- **precision** = fraction of synthetic samples that land in the real
  data's region → *realism*. **recall** = fraction of real samples covered
  by the synthetic data's region → *diversity / coverage*. Typical
  patterns: VAE decoder means → high precision, low recall; mode collapse
  → low recall; noisy samples → low precision. k-NN recall is inflated when
  the synthetic set is much smaller than the real set — compare at similar
  sizes.
- **frechet** (FID-style): 0 = identical Gaussian summaries. Use the RATIO
  to the reference. Ignore it when `frechet_unreliable_small_n` is true.
- **mmd_rbf**: ≈ 0 when distributions match; compare to the reference.
- **c2st_accuracy** (classifier two-sample test): 0.5 = indistinguishable;
  ~0.5–0.6 hard to tell apart; > ~0.9 easily distinguishable. With a
  same-source reference it is ≈ 0.5; anything far above that is a real gap.
- **label_consistency** (conditional samples): a classifier trained on real
  data predicts the intended label of synthetic samples. Compare
  `accuracy_on_synthetic` with `reference_accuracy_on_real_<split>` and check
  `per_class` — a low value for one class means the model did not learn
  that class (common for very rare classes; see VAE skill).
- **embedding.png**: real vs synthetic in a shared PCA — look for missing
  classes or clusters.

# Memorization
- `frac_nearest_real_is_train` vs `..._expected_if_no_memorization`
  (≈ 0.5): for each synthetic sample, is its nearest real neighbour a
  training sample or a held-out sample (equal-sized pools)? Near the
  expected value → no sign of copying. ≥ ~0.7 → samples hug training data.
  Interpret against the sampling mode: posterior/interpolated VAE samples
  are anchored to training samples by design.
- `n_exact_copies` > 0 is a red flag for any data release.
- `ratio_synthetic_over_real` < 1 alone can just mean the samples are
  smoother (decoder means) — use it together with the membership fraction.

# Utility
Image classifiers are trained several times with different seeds (default
3; `n_seeds`); scores are averages and `macro_f1_sd_across_seeds` shows how
much a single run wobbles. A difference smaller than about two seed SDs is
not meaningful, whatever the bootstrap CI says (the CI only covers test-set
noise). For images, the augment comparison uses a 20,000-image stratified
real subset by default (`n_real_train`) to keep 4 arms × seeds affordable —
mention it. Classifier time ≈ (training images × epochs × seeds × arms);
check the `real_*_accuracy` of the judging classifier (`feature_space`) to
make sure it is a competent judge before trusting the comparisons.
- **tstr**: `train_on_synthetic` vs `train_on_real_same_size` on the real
  test set; `tstr_over_trtr_accuracy` ≥ ~0.8 is good for small models;
  `macro_f1_diff_95ci` shows if the gap is significant.
- **augment**: compare `real_plus_synthetic` against ALL three baselines
  (`real_only`, `real_class_weighted`, `real_random_oversample`), using
  `macro_f1`, `focus_classes_mean_f1` and `per_class_f1`. Use the paired
  bootstrap CIs in `macro_f1_diff_95ci`: if the CI vs the best baseline
  includes 0, there is NO demonstrated benefit. Check
  `n_test_per_focus_class`: with < ~10 test samples per class, per-class
  numbers are anecdotal — say so.

# Anomaly
- `auroc` (0.5 = chance) and `auprc` vs `auprc_chance`.
- Always compare with `baseline_pca_recon_error_auroc` and, for images,
  `complexity_check` (see flow skill). A model that does not beat trivial
  baselines is not useful as-is.
- `mean_logp_per_dim_by_class`: which classes the model finds typical.

# When to iterate (max three times, log each)
- label consistency low for the target classes → change sampling (VAE
  latent source, DDPM guidance);
- recall very low with high precision → more diversity: VAE
  `--temperature` > 1 (e.g. 1.5–2 with posterior sampling) and/or
  `--add-noise`; DDPM lower guidance; retrain with lower β. Check that
  label consistency stays high when diversity goes up;
- training health check failed → fix per the family skill;
- anomaly AUROC ≤ complexity baseline → report the corrected score or try
  the VAE ELBO.
Stop when the success criterion is met, the budget is spent, or the
remaining options are unlikely to change the recommendation.

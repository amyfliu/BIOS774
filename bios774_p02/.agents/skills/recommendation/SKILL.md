---
name: recommendation
description: Turn the evaluation into a concrete, honest recommendation for downstream use of the generated data (or of the model's scores).
---

# When to apply
After evaluation, before writing the report. Log the result as the
`recommendation` decision.

# Structure every recommendation as
1. **Verdict** — one of: *Use* · *Use with restrictions* · *Do not use
   (use <alternative> instead)*.
2. **For what, exactly** — the specific downstream task, and the files to
   use (e.g. `outputs/<run>/<label>/samples_<tag>.npz`, which classes,
   how many).
3. **Evidence** — the 2–3 metrics that decided it, each with its reference
   or baseline.
4. **Restrictions and caveats** — see the checklist below.
5. **What would change the verdict** — e.g. more training time, more real
   samples of a class, a different score.

# Decision guide by objective

**Class balancing / augmentation**
- *Use* only if real+synthetic beats the best non-generative baseline
  with a CI excluding 0 (or a clear, consistent gain on the focus classes
  when test counts are too small for a CI to be informative — say so).
- If not: recommend the best simple baseline (class weights or
  oversampling) and explain that the synthetic data showed no measurable
  benefit. That is a useful result, not a failure.
- If the samples were generated around real training samples (VAE
  posterior/interpolation), describe them as "smoothed variations of the
  N real cells", not as new biology.

**Synthetic data release**
- *Use for prototyping / method development* if TSTR is reasonably close
  to train-on-real, label consistency is high and there is no sign of
  copying.
- Never for final model evaluation or scientific claims about the real
  population; never replace the real test set.
- Any memorization flag → *Do not release* until fixed; this overrides
  good fidelity.

**Anomaly / QC screening**
- Recommend the score that performed best *including* trivial baselines
  (e.g. the complexity-corrected likelihood if it beats the raw one).
- Use as a triage tool that flags samples for human review, with a
  threshold chosen on validation data for the acceptable false-positive
  rate — not as automatic exclusion.
- Name which classes it catches and which it misses
  (`mean_logp_per_dim_by_class`).

# Caveat checklist (include all that apply)
- The space the samples live in (e.g. "log-normalized, standardized
  expression of 2,000 HVGs" — not raw counts, not other genes).
- Synthetic rows must be marked as synthetic in any downstream table.
- Evaluation and test data stay real.
- Small-sample warnings (rare classes with very few real test samples).
- Budget limits (training stopped by time limit; quality could improve).
- Label noise: synthetic labels are only as good as label consistency.
- Patient privacy: no formal privacy guarantee (no differential privacy);
  memorization checks are evidence, not proof.

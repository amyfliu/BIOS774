---
name: evaluation
description: Choose which evaluate.py tasks measure the objective, compare models on val, report on test, and interpret every metric against its baseline and chance level.
---

# When to apply
After training (on val, to choose), after the choice (on test, to report),
and when writing the results section.

# Pick tasks from the objective

| Objective task (objective skill) | `--tasks` | extra flags |
|---|---|---|
| classification / integration | `structure,collapse` | `--labels` all label columns; read the primary one first |
| clustering | `structure,collapse` | look at ARI/NMI and silhouette, not the probe |
| retrieval | `retrieval,collapse` | (+ `structure` if classification is also asked) |
| cross-modal prediction | `cross` | `--cross a->b:<target spec>,b->a:<target spec>` (targets from preprocessing) |

Run the SAME tasks on every model and baseline, so `compare_runs.py` can
put them side by side. `cross` needs the run to contain an embedding of the
source modality (`z_<src>`): the single-modality baseline of the source gives
the non-multimodal reference ("how well does RNA alone, compressed the same
way, predict ATAC?").

# Val to choose, test to report
1. `evaluate.py --split val` for every model → `compare_runs.py --split val`
   → choose. Log the choice with the val numbers.
2. `evaluate.py --split test` for the chosen model(s) AND the baselines →
   `compare_runs.py --split test` → these numbers go in the report.
If test disagrees with val (e.g. ranking flips within overlapping CIs), say
so — do not switch models because of it.

# Reading each metric

**Classification (`structure`)**
- *Probe balanced accuracy*: a linear classifier on the embedding (fit on
  train). Balanced = mean recall over classes, so a majority class cannot
  hide failures on rare ones. Compare with `chance_balanced_accuracy` and
  the single-modality baselines. Use `probe_per_class_recall` to name the
  classes a representation fails on.
- *kNN balanced accuracy*: does the neighbourhood structure (not just a
  linear direction) carry the label?
- Differences smaller than the overlap of the two 95% CIs are not
  meaningful; with ~1,500 test samples and tens of classes, expect CI widths
  of a few points.
- Labels made from one modality (DATA.md) cap what the other can add —
  read "joint ≈ that modality alone" as expected, not as failure.
- Text shortcut: if the label is predictable from words alone
  (`profile.json`), the text embedding's probe is not evidence of anything;
  use the image (non-text) embedding's probe.

**Clustering**: ARI (0 = random, 1 = identical to labels) and NMI; kmeans
uses k = number of classes, so very imbalanced labels depress ARI even for a
good embedding — interpret with the probe.

**Retrieval (`retrieval`)**: recall@k = fraction of queries whose true partner
is in the top k out of `n_candidates`; chance is k / n_candidates. Report
both directions; asymmetry is normal. FOSCTTM = average fraction of
candidates closer than the true partner (0 perfect, 0.5 random) — the most
stable single number for paired cells. Exact-duplicate texts (profile
`duplicate_text_fraction`) put a small ceiling on text-side recall@1.
For paired single cells, recall@1 is naturally low (thousands of cells of the
same type look alike); FOSCTTM and recall@10 are the better summaries.

**Cross-modal prediction (`cross`)**
- Continuous target: *per-sample Pearson r* (does the predicted profile of a
  cell correlate with its observed one?) compared with
  `per_sample_pearson_baseline_train_mean` — the baseline is high because all
  cells share an average profile, so the GAIN over it is the result; *per-
  feature r* (does a gene's predicted level track its observed level across
  cells? baseline 0) and `r2_vs_baseline`.
- Binary target: per-feature AUROC (baseline 0.5) and per-sample AUROC vs the
  train-mean baseline.

**Collapse / geometry (`collapse`)**: effective rank ≪ dim or `dead_dims` > 0
→ partial collapse (training skill). `alignment_paired_sq_dist` should be far
below `alignment_random_pairs_sq_dist`. `modality_gap_centroid_dist` > ~0.5
means two separate clouds (see fusion skill).

**Figures**: `embedding.png` (test) — left: joint embedding coloured by the
primary label; right: both modalities in the shared space with grey lines
joining a few pairs. Describe what is visible (separated classes? mixed
modalities or two clouds?). `comparison_<split>.png` for the report.

# When evaluation should trigger an iteration
- A learned model is below its own baselines on the primary metric → check
  training health first; then try the logged alternative.
- Fine-label probe far below coarse-label probe while coarse is near perfect
  → more input components (preprocessing skill) or larger `--dim`.
- Retrieval far above chance but classification below baselines → the joint
  embedding (average of modalities) may be hurt by a modality gap: compare
  per-modality probes, and consider using the best modality's embedding.
- Adapter never beats zero-shot on val → stop; zero-shot is the answer.

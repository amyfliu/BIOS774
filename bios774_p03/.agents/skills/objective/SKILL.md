---
name: objective
description: Turn the user's objective into downstream tasks, headline metrics, required baselines, a label-use policy and a falsifiable success criterion.
---

# When to apply
Right after inspecting the data, before any preprocessing or modeling decision.

# Step 1 — map the wording to downstream tasks

| Objective wording | Downstream task | Headline metric (evaluate.py) | What it needs from the representation |
|---|---|---|---|
| "identify / classify / annotate cell types", "classify modality", "predict the label" | classification | `structure` → linear-probe **balanced accuracy** (and macro-F1) on the named label | a joint embedding (or the relevant modality's embedding) |
| "cluster", "discover groups", "unsupervised", "without labels" | clustering | `structure` → k-means **ARI / NMI**, silhouette | a joint embedding trained WITHOUT labels |
| "retrieve", "find the matching caption/image/cell", "search", "match across modalities" | cross-modal retrieval | `retrieval` → **recall@k** both directions, FOSCTTM | one embedding per modality in a SHARED space of equal dimension |
| "predict one modality from the other", "impute", "translate RNA→ATAC" | cross-modal prediction | `cross` → per-sample / per-feature Pearson r (continuous) or AUROC (binary) vs the train-mean baseline | an embedding of the SOURCE modality alone |
| "integrate", "joint representation", "combine modalities" | integration | classification/clustering of the joint embedding **vs each modality alone** | joint + per-modality embeddings |
| "does modality X add information?" | ablation | difference joint − best single modality, with CIs | single-modality baselines on the same split |

An objective usually names two tasks; the first one named is primary.
If the objective asks for something no tool measures, say so, run the
closest supported analysis, and flag the gap in the report.

# Step 2 — decide the label policy

Labels in `dataset.json → label_columns` are first of all the *yardstick*.

- **Default: no labels in training.** Train self-/cross-modally supervised
  methods (`clip`, `vicreg`, `simclr`, `pca_concat`, `identity`); evaluate
  with labels.
- **Labels allowed in training** (`supcon_clip`) only when the objective
  says labels are available for training, or explicitly asks for the best
  possible classifier. Then (a) keep a label-free model as the comparison,
  (b) mark every label-trained row as such, (c) never present its
  classification score as evidence of what the modalities share.
- **Check how the labels were made** (`DATA.md`). Labels predicted from ONE
  modality (e.g. cell types called from RNA alone) favour that modality's
  representation: a joint model can at best tie on them. Say this when you
  interpret "does adding modality X help?".
- **Text shortcut.** If `profile.json → modalities.<text>.label_predictable_from_words_alone`
  is high (≳ 0.8 balanced accuracy), classifying that label from the text side
  tests word-spotting, not representation quality: judge classification on the
  NON-text side (e.g. the image embedding) and treat retrieval as the
  alignment test.

# Step 3 — fix the baselines before training

Always: each modality alone (`pca_concat` with one input, or `identity` of a
pretrained encoder) and early fusion (`pca_concat` of all inputs). Add the
pretrained zero-shot embedding when a pretrained shared-space encoder exists
for these modalities. Chance levels come with every metric — quote them.

# Step 4 — write the success criterion BEFORE training

Log it as the `objective` decision: concrete, falsifiable, and achievable
with this test set (check class sizes in `profile.json → labels`; a class
with < ~10 test samples cannot carry a conclusion on its own). Examples:

- integration: "the joint embedding's probe balanced accuracy on
  `label_fine` exceeds the best single modality's, with non-overlapping 95%
  CIs; otherwise report that the second modality adds no measurable label
  information";
- retrieval: "image→text recall@5 of the chosen model exceeds the zero-shot
  pretrained model's by more than the width of its 95% CI";
- prediction: "per-sample Pearson r of predicted vs observed target clearly
  above the train-mean baseline (CI excludes it)".

Also state which label column is primary (fine-grained labels are more
informative about what a second modality adds than coarse ones, if their
classes are large enough), and what metric would make you change course.

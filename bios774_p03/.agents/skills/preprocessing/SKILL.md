---
name: preprocessing
description: Choose a preprocess.py spec for each modality (encoder inputs and cross-modal prediction targets) from the profile, not from habit.
---

# When to apply
After the objective decision, before any training. One spec per modality for
the encoders; extra "target" specs only if the objective includes cross-modal
prediction. Log one `preprocessing` decision covering all specs, with the
`profile.json` numbers that justified each.

# Principles
- Everything fitted (feature selection, IDF, PCA/SVD, scaling) is fitted on
  train by the tool. You choose *what*, the tool guarantees *where*.
- Encoder inputs should be compact (tens to a few hundred dimensions) and
  denoised: contrastive encoders on 10⁴–10⁵ raw sparse features overfit and
  are slow.
- Read each `features/<mod>__<spec>.json` after running: dimension,
  explained variance, and for LSI `first_component_corr_with_depth`.

# Count modalities — decide from the profile

Read `profile.json → modalities.<m>`: `fraction_zero`, `library_size`,
`fraction_nonzero_values_equal_1`, `fraction_nonzero_values_le_2`,
`median_overdispersion_top2000`, `features_in_ge_1pct_samples`.

| Evidence | Data type it indicates | Encoder-input spec | Why |
|---|---|---|---|
| counts spread over many values, overdispersed, ~10⁴ features (gene expression) | RNA-like | `log1p_hvg2000_pca50` (n_train < ~1,000: `hvg1000_pca30`) | library-size normalisation + log tames depth and skew; HVGs keep the informative genes; PCA denoises |
| ≥ ~0.7 of non-zero values ≤ 2, ≥ ~0.9 zeros, ~10⁵ features (peaks, accessibility) | near-binary, ATAC-like (10x counts each fragment's two cut sites, so "2" usually just means "open") | `tfidf_lsi50_drop1` | TF-IDF weights rare informative peaks; LSI is the standard reduction; the first component tracks sequencing depth |
| `drop1`: check `first_component_corr_with_depth` | | keep `drop1` if \|r\| ≳ 0.7; if \|r\| is small, rerun without it and log that | dropping a biological component would lose signal |
| very few features (< ~200) or already-reduced values | dense | `scale` (optionally `_pcaK`) | nothing to select |

Variants worth trying only if evaluation points to the input: more
components (`pca100`, `lsi100`) when the fine-label probe is far below the
coarse one; `log1p_scale_pca50` when a few very high-variance genes dominate.

## Cross-modal prediction targets (only if the objective asks)
Predict an interpretable, preprocessed version of the target modality — not
raw counts, not the target's own LSI/PCA (those are not "the modality"):
- RNA-like target: `log1p_hvg2000` (continuous; scored by Pearson r).
- ATAC-like target: `binary_top10000` (most-detected peaks, on/off; scored
  by AUROC). Use `binary_top5000` if n_train < ~2,000.

# Image and text modalities

| Situation | Spec | Why |
|---|---|---|
| A pretrained encoder exists whose training data matches the domain (e.g. BiomedCLIP for biomedical figures/captions) and both modalities have it | `biomedclip` for BOTH image and text | in-domain pretraining on far more pairs than we have; the two outputs already share a space, which gives a zero-shot baseline for free |
| Baseline to show what pretraining buys | text `tfidf_svd128`, image `pixels32_pca50` | deliberately simple; any learned model should beat them |
| Images mostly grayscale (`fraction_grayscale`) | no change — the pretrained encoder expects RGB, the tool converts | |

Pretrained features are computed once and cached (`outputs/_cache/`); a
second run with the same spec reads the cache in seconds. The first
encoding of tens of thousands of images can take many minutes on a laptop:
the tool prints progress with elapsed time — use the first lines to estimate
the total, run it in the background if it is long, and wait for the
`features/<mod>__biomedclip.npz` file. It does not count toward the training
budget.

Do NOT clean or mask caption text (e.g. removing modality words) unless the
objective asks for it — but record the text-shortcut number from the profile
(objective skill) because it changes how classification results are read.

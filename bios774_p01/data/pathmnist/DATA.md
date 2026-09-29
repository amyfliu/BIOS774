# PathMNIST — Data Description

Source: MedMNIST v2, PathMNIST (colon pathology, 28x28 RGB patches), train split.

## Contents of prepared.npz
- `X`: (89996, 2352) flattened RGB pixel intensities per image
  (28*28*3 = 2352 "features"). Raw pixel values in [0, 255], NOT
  normalized/scaled.
- `feature_names`: pixel_0 ... pixel_2351. These are NOT
  biologically meaningful individual features the way genes are for PBMC3k —
  they're raw pixel positions. Don't interpret loadings on individual
  "features" here the same way you would for gene expression.
- `y`: tissue class label (9 classes) — ground truth, provided by the
  dataset. Usable for evaluation (silhouette etc.), NOT as a DR input feature.
- `sample_ids`: synthetic per-image IDs.
- `label_values` / `label_names`: the 9 numeric class codes and their real
  tissue-type names (adipose, background, debris, lymphocytes, mucus, smooth
  muscle, normal colon mucosa, cancer-associated stroma, colorectal
  adenocarcinoma epithelium), from `medmnist.INFO["pathmnist"]["label"]`.
  Optional/display-only — tools/run_dr.py uses these to label plot legends
  when present; nothing about the DR pipeline depends on them.

## Known characteristics worth confirming via inspect_data.py, not assuming
- Dense (not sparse) — real-valued pixel intensities, few exact zeros.
- Every "feature" is on the same raw scale (0-255) unlike PBMC3k's genes,
  so the scale-heterogeneity concern is different here.
- Much larger sample count than PBMC3k, worth checking before choosing
  compute-heavy methods (t-SNE/MDS) at full size.
- Structure here is spatial (adjacent pixels correlated) — flattening
  discards that; a method comparison could note whether that matters.

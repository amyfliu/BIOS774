# PathMNIST — Data Description

Source: MedMNIST v2, PathMNIST (colon pathology, 28x28 RGB patches).
Modality: **image**.

## Contents of prepared.npz
- `X`: (107180, 2352) uint8 pixel values in [0, 255], each row an image
  flattened in (H, W, C) order. NOT scaled.
- `image_shape`: (28, 28, 3) = (H, W, C).
- `y`: tissue class (int). `label_values` / `label_names` map codes to names.
  May be used as a conditioning variable when the objective needs it.
- `split`: official MedMNIST split — `train` (89996),
  `val` (10004) and `test` (7180).
  Fit on `train` ONLY.
  * `train` and `val` come from the same source collection (NCT-CRC-HE-100K):
    `val` is the like-for-like held-out reference (realism, copying checks).
  * `test` comes from a different collection (CRC-VAL-HE-7K, other patients /
    staining): a deliberate "new data from elsewhere" check, used for
    downstream classifier tests. Expect real train vs real test to differ.

## Classes
  - 0 = adipose: 9366 train / 1041 val / 1338 test
  - 1 = background: 9509 train / 1057 val / 847 test
  - 2 = debris: 10360 train / 1152 val / 339 test
  - 3 = lymphocytes: 10401 train / 1156 val / 634 test
  - 4 = mucus: 8006 train / 890 val / 1035 test
  - 5 = smooth muscle: 12182 train / 1354 val / 592 test
  - 6 = normal colon mucosa: 7886 train / 877 val / 741 test
  - 7 = cancer-associated stroma: 9401 train / 1045 val / 421 test
  - 8 = colorectal adenocarcinoma epithelium: 12885 train / 1432 val / 1233 test

## Things to confirm with inspect_data.py rather than assume
- Dense, homogeneous scale (all features are pixel intensities 0-255).
- Spatial structure: neighbouring pixels are strongly correlated, so
  convolutional architectures are natural; flattening discards that.
- Large n — training on a subsample may be a sensible compute decision.

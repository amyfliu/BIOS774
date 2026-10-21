# PathMNIST — Data Description

Source: MedMNIST v2, PathMNIST (colon pathology, 28x28 RGB patches).
Modality: **image**.

## Contents of prepared.npz
- `X`: (97176, 2352) uint8 pixel values in [0, 255], each row an image
  flattened in (H, W, C) order. NOT scaled.
- `image_shape`: (28, 28, 3) = (H, W, C).
- `y`: tissue class (int). `label_values` / `label_names` map codes to names.
  May be used as a conditioning variable when the objective needs it.
- `split`: official MedMNIST split — `train` (89996) and
  `test` (7180). Fit on `train` ONLY; `test` is the
  untouched real reference for evaluation.

## Classes
  - 0 = adipose: 9366 train / 1338 test
  - 1 = background: 9509 train / 847 test
  - 2 = debris: 10360 train / 339 test
  - 3 = lymphocytes: 10401 train / 634 test
  - 4 = mucus: 8006 train / 1035 test
  - 5 = smooth muscle: 12182 train / 592 test
  - 6 = normal colon mucosa: 7886 train / 741 test
  - 7 = cancer-associated stroma: 9401 train / 421 test
  - 8 = colorectal adenocarcinoma epithelium: 12885 train / 1233 test

## Things to confirm with inspect_data.py rather than assume
- Dense, homogeneous scale (all features are pixel intensities 0-255).
- Spatial structure: neighbouring pixels are strongly correlated, so
  convolutional architectures are natural; flattening discards that.
- Large n — training on a subsample may be a sensible compute decision.

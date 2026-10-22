"""
prepare.py — download and standardize PathMNIST (MedMNIST v2) for project 2.

Adapted from bios774_p01/data/pathmnist/prepare.py. Differences for project 2:
  * keeps the official MedMNIST `train`, `val` and `test` splits (as a `split`
    column). train/val come from the same source collection (NCT-CRC-HE-100K);
    test comes from a DIFFERENT collection (CRC-VAL-HE-7K). So `val` is the
    like-for-like real reference for realism/copying checks, and `test` is the
    "new data from elsewhere" check for downstream classifiers;
  * stores pixels as uint8 plus `image_shape` (H, W, C), so tools can rebuild
    images for convolutional models and sample grids;
  * labels may be used as a conditioning variable when the objective
    requires class-specific generation.

Run locally (needs internet to reach MedMNIST's Zenodo-hosted data):

    cd data/pathmnist && python prepare.py
"""
import medmnist
import numpy as np
from medmnist import PathMNIST


def main():
    parts = {s: PathMNIST(split=s, download=True) for s in ("train", "val", "test")}
    imgs = np.concatenate([parts[s].imgs for s in parts])            # (N, 28, 28, 3) uint8
    labels = np.concatenate([parts[s].labels.squeeze() for s in parts]).astype(np.int64)
    split = np.concatenate([np.full(len(parts[s].imgs), s, dtype="<U5") for s in parts])

    N = imgs.shape[0]
    image_shape = np.array(imgs.shape[1:])                           # (H, W, C)
    X = imgs.reshape(N, -1)                                          # uint8, HWC flattened

    label_map = medmnist.INFO["pathmnist"]["label"]
    label_values = np.array(sorted(label_map.keys(), key=int)).astype(np.int64)
    label_names = np.array([label_map[str(v)] for v in label_values])

    np.savez_compressed(
        "prepared.npz",
        X=X, y=labels, split=split, image_shape=image_shape,
        label_values=label_values, label_names=label_names,
        sample_ids=np.array([f"img_{i:06d}" for i in range(N)]),
    )

    lines = "\n".join(
        f"  - {v} = {n}: {int(((labels == v) & (split == 'train')).sum())} train / "
        f"{int(((labels == v) & (split == 'val')).sum())} val / "
        f"{int(((labels == v) & (split == 'test')).sum())} test"
        for v, n in zip(label_values, label_names)
    )
    with open("DATA.md", "w") as f:
        f.write(f"""# PathMNIST — Data Description

Source: MedMNIST v2, PathMNIST (colon pathology, 28x28 RGB patches).
Modality: **image**.

## Contents of prepared.npz
- `X`: ({N}, {X.shape[1]}) uint8 pixel values in [0, 255], each row an image
  flattened in (H, W, C) order. NOT scaled.
- `image_shape`: {tuple(int(v) for v in image_shape)} = (H, W, C).
- `y`: tissue class (int). `label_values` / `label_names` map codes to names.
  May be used as a conditioning variable when the objective needs it.
- `split`: official MedMNIST split — `train` ({int((split == 'train').sum())}),
  `val` ({int((split == 'val').sum())}) and `test` ({int((split == 'test').sum())}).
  Fit on `train` ONLY.
  * `train` and `val` come from the same source collection (NCT-CRC-HE-100K):
    `val` is the like-for-like held-out reference (realism, copying checks).
  * `test` comes from a different collection (CRC-VAL-HE-7K, other patients /
    staining): a deliberate "new data from elsewhere" check, used for
    downstream classifier tests. Expect real train vs real test to differ.

## Classes
{lines}

## Things to confirm with inspect_data.py rather than assume
- Dense, homogeneous scale (all features are pixel intensities 0-255).
- Spatial structure: neighbouring pixels are strongly correlated, so
  convolutional architectures are natural; flattening discards that.
- Large n — training on a subsample may be a sensible compute decision.
""")
    print(f"Saved prepared.npz: {N} images {tuple(image_shape)}, "
          f"{len(label_values)} classes")


if __name__ == "__main__":
    main()

"""
prepare.py — download and standardize PathMNIST (from MedMNIST).

Run this LOCALLY (not in a network-restricted sandbox) since it needs to
reach MedMNIST's Zenodo-hosted data.

    cd data/pathmnist
    python prepare.py

Produces:
    prepared.npz   — X (flattened pixel intensities, n_images x (28*28*3),
                         values in [0, 255], no preprocessing applied),
                      y (tissue-type class label, for evaluation only),
                      feature_names (pixel_0 ... pixel_N, not biologically
                         meaningful — flagged in DATA.md so the agent
                         doesn't over-interpret "features" here the way it
                         would for genes),
                      sample_ids
    DATA.md        — column/role description for the agent

Deliberately RAW: pixels are NOT normalized to [0,1], NOT standardized.
Whether/how to preprocess image-derived features is left to the agent.
"""
import medmnist
import numpy as np
from medmnist import PathMNIST


def main():
    # Use the train split; subsample if you want faster iteration —
    # left at default size here (~89,996 images) but this is a reasonable
    # place to subsample for compute reasons if needed; that's an
    # engineering choice worth stating explicitly in your report if you do it.
    train_ds = PathMNIST(split="train", download=True)

    images = train_ds.imgs  # (N, 28, 28, 3) uint8
    labels = train_ds.labels.squeeze()  # (N,) int class labels

    N = images.shape[0]
    X = images.reshape(N, -1).astype(np.float32)  # flatten to (N, 28*28*3)
    feature_names = np.array([f"pixel_{i}" for i in range(X.shape[1])])
    sample_ids = np.array([f"img_{i:06d}" for i in range(N)])

    # PathMNIST is one of the few datasets here where the numeric label has a
    # real, known display name (tissue type), bundled with the medmnist
    # package itself (no extra download). label_values/label_names are an
    # OPTIONAL pair of standardized fields — tools/run_dr.py checks for them
    # and falls back to raw numeric labels when absent (e.g. for pbmc3k,
    # whose Leiden cluster IDs have no such canonical name).
    label_map = medmnist.INFO["pathmnist"]["label"]  # {"0": "adipose", ...}
    label_values = np.array(sorted(label_map.keys(), key=int))
    label_names = np.array([label_map[v] for v in label_values])

    np.savez(
        "prepared.npz",
        X=X,
        y=labels,
        feature_names=feature_names,
        sample_ids=sample_ids,
        label_values=label_values,
        label_names=label_names,
    )

    with open("DATA.md", "w") as f:
        f.write(f"""# PathMNIST — Data Description

Source: MedMNIST v2, PathMNIST (colon pathology, 28x28 RGB patches), train split.

## Contents of prepared.npz
- `X`: ({X.shape[0]}, {X.shape[1]}) flattened RGB pixel intensities per image
  (28*28*3 = {X.shape[1]} "features"). Raw pixel values in [0, 255], NOT
  normalized/scaled.
- `feature_names`: pixel_0 ... pixel_{X.shape[1]-1}. These are NOT
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
""")
    print(f"Saved prepared.npz: X {X.shape}, {len(np.unique(labels))} label classes")


if __name__ == "__main__":
    main()

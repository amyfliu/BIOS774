# Prepared multimodal dataset format

Every dataset folder `data/<name>/` produced by its `prepare.py` has the same
layout, so the tools in `tools/` never need dataset-specific code:

```
data/<name>/
├── prepare.py        # dataset-specific: raw source -> the files below
├── DATA.md           # human/agent-readable description (written by prepare.py)
├── dataset.json      # manifest: modalities, label columns, splits, provenance
├── obs.csv           # one row per sample (sample_id, split, labels, ...)
└── <modality files>  # one entry per modality, listed in dataset.json
```

## `dataset.json`

```json
{
  "name": "pbmc_multiome",
  "description": "one line",
  "n_samples": 10970,
  "pairing": "same sample measured in every modality (row i of each modality = obs row i)",
  "obs": "obs.csv",
  "id_column": "sample_id",
  "split_column": "split",
  "splits": {"train": 7679, "val": 1645, "test": 1646},
  "label_columns": ["label", "label_fine"],
  "primary_label": "label",
  "modalities": [
    {"name": "rna",  "kind": "counts", "file": "rna.npz",  "features": "rna_features.tsv"},
    {"name": "atac", "kind": "counts", "file": "atac.npz", "features": "atac_features.tsv"}
  ],
  "source": {"url": "...", "license": "...", "citation": "..."}
}
```

Modality kinds:

| kind | stored as | row order |
|---|---|---|
| `counts` | `scipy.sparse` CSR matrix in `.npz` (`scipy.sparse.save_npz`), raw integer counts; `features` is a TSV with one row per column (first column = feature id) | same as `obs.csv` |
| `image` | a folder of image files; `column` names the `obs.csv` column holding each sample's path relative to the dataset folder | via `obs.csv` |
| `text` | a column of `obs.csv` (named by `column`) holding the raw text | `obs.csv` |

Rules:

- Raw values only. Normalisation, feature selection, TF-IDF, tokenisation,
  image transforms, etc. are the agent's decisions and happen in `tools/`.
- `split` is one of `train`, `val`, `test`; fixed by `prepare.py` with a
  fixed seed and stratified by the primary label where possible.
- Labels are for evaluation; the agent may use them for training only when
  its objective and skills allow it (e.g. supervised contrastive learning).

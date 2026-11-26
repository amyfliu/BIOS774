# Building an AI Agent for Multimodal Representation Learning (BIOS 774, Project 3)

Given **a multimodal dataset and a user objective**, an AI agent (Claude, run
headless through Claude Code) decides on its own how the representation
should be learned. It profiles each modality, interprets the objective,
chooses the preprocessing, the fusion strategy, the encoders and the
training objective, trains, checks training health, evaluates against
baselines, runs the downstream analyses the objective asks for, recommends a
representation and writes a report. Every decision is logged with the
evidence behind it.

```
./run_agent.sh <dataset> "<objective>" [report_name] [budget_minutes]
```

## Results at a glance

| Report | Dataset | Objective | What the agent chose | Verdict |
|---|---|---|---|---|
| [`generated_report_1`](reports/generated_report_1.md) | 10x PBMC Multiome (scRNA-seq + scATAC-seq) | joint cell representation for cell-type identification + predict each modality from the other | CLIP-style contrastive alignment (MLP encoders), regularised after it overfit | **Multimodal helps** on 17 fine cell types (0.884 vs 0.831 RNA alone, paired diff. +0.053 [+0.016, +0.086]); early fusion is a near-equal, simpler choice |
| [`generated_report_2`](reports/generated_report_2.md) | ROCOv2 (radiology images + captions) | caption ↔ image retrieval + imaging-modality classification from the image | light fine-tuning (adapter) of BiomedCLIP; it never beat its pretrained start | **Use frozen BiomedCLIP**: correct caption in top 5 of 2,000 for 81% of images; modality from image 0.972 balanced acc.; training added nothing |

Final write-up: [`reports/report.pdf`](reports/report.pdf) (≤ 4 pages + references).

Representation-learning methods available to the agent (lectures L20–L25):
single-modality and early-fusion **PCA** baselines, **CLIP-style
contrastive alignment** (symmetric InfoNCE), **VICReg** (non-contrastive),
**supervised contrastive + CLIP** (when labels may be used), **SimCLR**
(one modality), and **light fine-tuning of a pretrained multimodal
foundation model** (BiomedCLIP) with a residual adapter, or its frozen
**zero-shot** use. Encoders: linear, MLP or adapter.

## Deliverables (what the assignment asks for → where it is)

For each dataset (`<run>` = `generated_report_1_pbmc_multiome` or `generated_report_2_rocov2`):

| Required output | File(s) |
|---|---|
| learned multimodal representations | `outputs/<run>/recommended_representation.npz` (+ `.json`: source model, keys, how to rebuild the joint vector) |
| quantitative evaluation results | `outputs/<run>/comparison_test.md` (all models), `comparison_test_vs_<baseline>.md` (paired differences), `<model>/metrics.json` |
| downstream analysis results | `outputs/<run>/<model>/metrics.json` (classification, retrieval, cross-modal prediction) and the report's §6 |
| summary of methods and preprocessing | `outputs/<run>/decisions.md` (every decision with evidence and rejected alternatives), `features/*.json`, `<model>/run_config.json` |
| final analysis report | `reports/generated_report_1.md/.pdf`, `reports/generated_report_2.md/.pdf` |
| system report (≤ 4 pages) | `reports/report.pdf` |

## How it works

```
run_agent.sh ─► Claude reads AGENTS.md ─► per step, reads the matching skill ─► calls tools/ ─► reads JSON/PNG results
                                                                                      │
 ① inspect ─► ② objective ─► ③ preprocess ─► ④ fusion + method ─► ⑤ train + health check ─► ⑥ evaluate on val
                                                   ▲                                                │
                                                   └──────────────── iterate (≤ 3×) ◄───────────────┘
 ⑦ test (chosen models + baselines) ─► ⑧ downstream analyses ─► ⑨ recommend + export representation ─► ⑩ report
```

- **Decision layer** — `AGENTS.md` (workflow + rules) and `.agents/skills/*/SKILL.md`
  (decision criteria). Skills hold judgment; tools hold mechanics. No skill names a dataset.
- **Compute layer** — `tools/` (dataset-agnostic Python; no decision logic).
- **Evidence** — every tool writes JSON/PNG into `outputs/<run_id>/`; the agent
  logs each decision to `outputs/<run_id>/decisions.jsonl`; the report quotes
  only numbers from those files.
- **Rules that keep it honest** — fit on train, **choose on validation, report on
  test**; every learned model is compared with each modality alone, early fusion
  and (when available) the zero-shot pretrained model; "is A better than B?" is
  answered by a **paired bootstrap** on the same test samples; every training run
  also scores the untouched starting model (epoch 0) and keeps it if training
  never improves on it.

## Project structure

```
├── run_agent.sh              # THE single command: dataset + objective -> report + representation
├── run_all.sh                # the two runs used in reports/report.pdf
├── smoke_test.sh             # checks every tool/method on this machine (MPS); --real adds timings on the real data
├── AGENTS.md                 # workflow, rules, tool table, report format
├── .agents/skills/
│   ├── objective/            # downstream tasks, headline metrics, label policy, success criterion
│   ├── preprocessing/        # per-modality feature spec, decided from the profile
│   ├── fusion/               # early fusion vs CLIP vs VICReg vs SupCon vs pretrained/adapter; baselines
│   ├── training/             # hyperparameters, budget, health checks (overfitting, collapse, epoch 0)
│   ├── evaluation/           # which tasks; val vs test; paired differences; how to read each metric
│   └── recommendation/       # downstream analyses, verdict categories, caveats
├── data/
│   ├── FORMAT.md             # the common multimodal format every tool reads
│   ├── dataset_io.py         # shared download / split / validation helpers for prepare.py
│   ├── pbmc_multiome/prepare.py   # 10x download -> rna.npz, atac.npz, obs.csv, dataset.json, DATA.md
│   ├── rocov2/prepare.py          # Zenodo download -> images/, obs.csv (captions), dataset.json, DATA.md
│   └── _dev/make_dev.py           # tiny synthetic versions of both datasets (for tests)
├── tools/
│   ├── inspect_data.py       # profile.json (facts only, per modality)
│   ├── preprocess.py         # one modality -> feature matrix (e.g. log1p_hvg2000_pca50, tfidf_lsi50_drop1, biomedclip)
│   ├── train.py              # pca_concat | identity | clip | vicreg | supcon_clip | simclr; linear/mlp/adapter encoders
│   ├── evaluate.py           # structure (probe, kNN, k-means), retrieval, collapse, cross-modal prediction
│   ├── compare_runs.py       # one table across models + paired bootstrap differences vs a reference
│   ├── export_representation.py  # the recommended representation as a compact deliverable
│   ├── log_decision.py       # append to decisions.jsonl
│   ├── run_summary.py        # decision log + exact command history -> Markdown for the report
│   ├── export_report.py      # tidy report layout + export PDF (pandoc)
│   ├── show_progress.py      # pretty-prints the live agent transcript
│   └── common.py             # dataset loader, device selection, JSON helpers
├── outputs/<run_id>/         # everything a run produced (+ agent_transcript.jsonl)
└── reports/                  # generated_report_1/2 (.md/.pdf), report.md/.pdf,
                              #   architecture_tikz.tex (source of the architecture figure)
```

Local-only (gitignored, not on GitHub): the course instructions and lectures,
the raw and prepared data (`data/*/raw/`, `*.npz`, `images/`, `obs.csv`),
`outputs/_cache/` (BiomedCLIP features, computed once), `outputs/_previous/`
(earlier attempts, moved there automatically when a run is repeated),
`outputs/_smoke/` and `outputs/_check/`, per-model `features/`, `model.pt` and
full-precision `embeddings.npz` (up to 138 MB each; the recommended one is
committed as `recommended_representation.npz`).

## Setup

```bash
python3 -m pip install -r requirements.txt        # Python ≥ 3.9; tested on macOS (Apple silicon), Python 3.9.6, torch 2.8
cd data/pbmc_multiome && python3 prepare.py && cd ../..   # ~170 MB download, ~1 min + CellTypist labelling
cd data/rocov2        && python3 prepare.py && cd ../..   # ~6.4 GB download (Zenodo), 15-40 min; ~8 GB free disk
./smoke_test.sh --real    # 1-3 min; every step should PASS; prints the device (mps on a Mac) and real-data timings
```

The first BiomedCLIP use downloads its weights (~800 MB, from Hugging Face)
and caches the image and text features in `outputs/_cache/`; encoding the
24,000 images once takes several minutes on a laptop GPU.

The agent needs the Claude Code CLI (`claude`) on `PATH`, logged in (if it
reports an expired session, run `claude`, then `/login`). The submitted runs
used Claude Sonnet 5.5 (the CLI default at the time); set
`CLAUDE_MODEL=<model id>` before `run_agent.sh` to pin a specific model.

## Data

Each `prepare.py` only downloads, cleans and organises; every modelling choice
(normalisation, feature selection, TF-IDF, which encoder, ...) is the agent's.
Both write the common format in [`data/FORMAT.md`](data/FORMAT.md) and a
`DATA.md` that the agent reads first.

**PBMC Multiome** — [10k Human PBMCs, Multiome v1.0, Chromium X](https://www.10xgenomics.com/datasets/10-k-human-pbm-cs-multiome-v-1-0-chromium-x-1-standard-2-0-0)
(10x Genomics, Cell Ranger ARC 2.0.0, one healthy donor):

1. **Download** the filtered feature-barcode matrix
   ([`…_filtered_feature_bc_matrix.h5`](https://cf.10xgenomics.com/samples/cell-arc/2.0.0/10k_PBMC_Multiome_nextgem_Chromium_X/10k_PBMC_Multiome_nextgem_Chromium_X_filtered_feature_bc_matrix.h5),
   166 MB: 10,970 cells × 36,601 genes + 111,743 peaks) and the
   [ATAC peak annotation](https://cf.10xgenomics.com/samples/cell-arc/2.0.0/10k_PBMC_Multiome_nextgem_Chromium_X/10k_PBMC_Multiome_nextgem_Chromium_X_atac_peak_annotation.tsv)
   (nearest gene of each peak) into `data/pbmc_multiome/raw/`.
2. **Split** the matrix into an RNA matrix (cells × genes) and an ATAC matrix
   (cells × peaks); row *i* is the same cell in both.
3. **Cell QC**: ≥ 200 genes, ≤ 20% mitochondrial RNA counts, ≥ 1,000 ATAC counts
   in peaks → **10,405 cells**. Features: genes seen in ≥ 3 cells (25,598),
   peaks seen in ≥ 10 cells (111,698).
4. **Reference cell-type labels** with CellTypist on log-normalised RNA, with
   majority voting: coarse `label` (Immune_All_High, 8 types) and fine
   `label_fine` (Immune_All_Low, 17 types). They are predicted from RNA only,
   so they can favour RNA-based representations.
5. **Split** 70/15/15 train/val/test, stratified by `label`, seed 0
   (7,285 / 1,560 / 1,560 cells).
6. **Save** raw counts (`rna.npz`, `atac.npz`), feature tables, `obs.csv`,
   `dataset.json`, `DATA.md`.

**ROCOv2** — [Radiology Objects in COntext v2](https://zenodo.org/records/10821435)
(Rückert et al., 2024, *Scientific Data*; CC BY-NC 4.0):

1. **Download** captions, UMLS concepts, radiologist-checked manual concepts,
   the concept-name table and the image archives of the train / valid / test
   splits into `data/rocov2/raw/`.
2. **Imaging-modality label** from the manual concepts (CT, X-ray, MRI,
   ultrasound, angiography, PET); images with no or several modality concepts
   are dropped.
3. **Keep ROCOv2's own split** (test papers differ from training papers) and a
   fixed random subset, seed 0: **20,000 / 2,000 / 2,000** image–caption pairs.
4. **Resize** images so the shorter side is 256 px (aspect ratio kept); no
   other image or caption processing.
5. **Save** `images/<split>/`, `obs.csv` (image path, caption, label, concepts),
   `dataset.json`, `DATA.md`.

Adding a dataset needs only a `prepare.py` that writes this format; modalities
can be of kind `counts`, `image` or `text`.

## Run

The two reports (same commands as in `run_all.sh`):

```bash
./run_agent.sh pbmc_multiome \
  "Learn a joint RNA+ATAC cell representation that identifies cell types at least as well as either modality alone, and assess how well each modality can be predicted from the other." \
  generated_report_1 20

./run_agent.sh rocov2 \
  "Learn a shared image-text representation for radiology figures that supports retrieving the right caption for an image (and vice versa) and classifying the imaging modality from the image." \
  generated_report_2 20
```

Each run took about 5 minutes and US$0.50 of model usage on an Apple-silicon
MacBook Air. Progress is printed live; the full transcript is saved in
`outputs/<run_id>/agent_transcript.jsonl`, and if a run is repeated the
previous attempt is moved to `outputs/_previous/`.

To rebuild a report's PDF after editing its `.md` by hand:

```bash
python3 tools/export_report.py reports/<name>.md
```

To load a learned representation:

```python
import numpy as np, pandas as pd
rep = np.load("outputs/generated_report_1_pbmc_multiome/recommended_representation.npz")
obs = pd.read_csv("data/pbmc_multiome/obs.csv")      # same row order; join on sample_id
z_rna, z_atac = rep["z_rna"].astype("float32"), rep["z_atac"].astype("float32")
```

## Evaluation tools in brief

| Task | Metrics (each next to its chance level or baseline, with 95% bootstrap CIs) |
|---|---|
| structure | linear-probe balanced accuracy + macro-F1, per-class recall, 15-NN balanced accuracy, k-means ARI/NMI, silhouette — for the joint and each modality's embedding |
| retrieval | cross-modal recall@1/5/10 both directions, median rank, FOSCTTM |
| cross-modal prediction | ridge from one modality's embedding to the other's features: per-cell and per-feature Pearson r (continuous) or AUROC (binary), vs a train-mean baseline |
| collapse | effective rank, dead dimensions, alignment / uniformity, modality gap |
| comparisons | paired bootstrap of model − reference on the same samples (`compare_runs.py --reference`) |

Every report ends with one of four verdicts: *Multimodal helps*, *Multimodal
enables* (no gain on the shared tasks, but something one modality cannot do),
*No measurable gain*, or *Can't tell*.

## Reproducibility

Fixed seeds and splits (stored in `obs.csv`), every model's full configuration
in `run_config.json`, every agent decision in `decisions.jsonl`, and each
report ends with the exact commands that produced it. Three PBMC runs gave
identical test numbers. The tools are deterministic given their arguments up to
GPU nondeterminism; the agent's choices are LLM reasoning and can differ
between invocations (e.g. two ROCOv2 runs sized an alternative model
differently and labelled the same evidence "enables" vs "no measurable gain")
— the decision log records what was chosen and why.

## Limitations

Modalities must be counts, images or text; the learned methods align exactly
two paired modalities; BiomedCLIP is the only pretrained encoder; robustness
was shown on these two datasets only. See §5 of [`reports/report.pdf`](reports/report.pdf).

# Building an AI Agent for Generative Modeling (BIOS 774, Project 2)

Given **a dataset and a user objective**, an AI agent (Claude, run headless
through Claude Code) decides on its own whether and how to use generative
modeling. It inspects the data, interprets the objective, chooses the
model family and hyperparameters, trains, samples, evaluates, iterates,
recommends downstream use, and writes a report. Every decision is logged
with the evidence behind it.

```
./run_agent.sh <dataset> "<objective>" [report_name] [budget_minutes]
```

## Results at a glance

| Report | Dataset | Objective | Model the agent chose | Verdict |
|---|---|---|---|---|
| [`generated_report_1`](reports/generated_report_1.pdf) | PBMC3K | rebalance rare cell types | conditional VAE | Do not use (use class weights) |
| [`generated_report_2`](reports/generated_report_2.pdf) | PathMNIST | shareable synthetic labeled images | conditional DDPM | Use with restrictions (prototyping only) |
| [`appendix/generated_report_B`](reports/appendix/generated_report_B.pdf) | PathMNIST | balance tissue classes | smaller conditional DDPM | Use with restrictions (supplement only) |

Final write-up: [`reports/report.pdf`](reports/report.pdf) (≤ 4 pages).

Model families available to the agent (lectures L14–L17): **VAE / conditional
VAE**, **Wasserstein GAN-GP**, **RealNVP normalizing flow**, **DDPM** (with
DDIM sampling and classifier-free guidance). Each supports an MLP backbone
(tabular data) and a small CNN/UNet backbone (images).

## How it works

```
run_agent.sh ─► Claude reads AGENTS.md ─► per step, reads the matching skill ─► calls tools/ ─► reads JSON/PNG results
                                                                                     │
      ① inspect ─► ② objective ─► ③ prep ─► ④ model ─► ⑤ train ─► ⑥ sample ─► ⑦ evaluate ─► ⑧ recommend ─► ⑨ report
                                              ▲                                      │
                                              └──────── iterate (≤ 3×) ◄─────────────┘
```

- **Decision layer** — `AGENTS.md` (workflow + rules) and `.agents/skills/*/SKILL.md`
  (decision criteria). Skills hold judgment; tools hold mechanics.
- **Compute layer** — `tools/` (dataset-agnostic Python; no decision logic).
- **Evidence** — every tool writes JSON/PNG into `outputs/<run_id>/`; the agent
  logs each decision to `outputs/<run_id>/decisions.jsonl`; the report quotes
  only numbers from those files.

## Project structure

```
├── run_agent.sh              # THE single command: dataset + objective -> report
├── run_all.sh                # the four runs used in reports/report.pdf
├── smoke_test.sh             # checks every tool/model on this machine (MPS) + speed probe
├── AGENTS.md                 # workflow, rules, tool table, report format
├── .agents/skills/
│   ├── objective/            # task type, success criterion, generate at all?
│   ├── preprocessing/        # which space the model learns in
│   ├── model-selection/      # VAE vs GAN vs flow vs DDPM (the comparison)
│   ├── vae/ gan/ flow/ diffusion/   # configure, train, health-check, sample each family
│   ├── evaluation/           # which metrics for which objective; how to read them
│   └── recommendation/       # how (and whether) to use the outputs downstream
├── data/<name>/prepare.py    # download + standardize -> prepared.npz, DATA.md
├── tools/
│   ├── inspect_data.py       # profile.json (facts only)
│   ├── train.py              # train one model (time budget, smoke mode, MPS/CUDA/CPU)
│   ├── sample.py             # generate samples (per class, sampler knobs)
│   ├── score.py              # per-sample log-likelihood of real data (flow, VAE)
│   ├── evaluate.py           # fidelity / memorization / utility / anomaly metrics
│   ├── log_decision.py       # append to decisions.jsonl
│   ├── run_summary.py        # decision log + exact command history -> Markdown for the report
│   ├── export_report.py      # tidy report layout (figure sizes, table alignment) + export PDF
│   ├── show_progress.py      # pretty-prints the live agent transcript
│   ├── common.py             # data loading, preprocessing, device selection
│   └── models/               # vae.py gan.py flow.py ddpm.py nets.py (+ registry)
├── outputs/<run_id>/         # everything a run produced (+ agent_transcript.jsonl)
└── reports/                  # generated_report_1/2 (.md/.pdf), appendix/, report.md/.pdf,
                              #   architecture_tikz.tex (source of the architecture figure)
```

Local-only (gitignored, not on GitHub): `outputs/_previous/` (earlier
attempts, moved there automatically when a run is repeated), `outputs/_smoke/`
(smoke-test leftovers), `outputs/_cache/` (the saved judging classifier),
model weights (`model.pt`) and the prepared data (`prepared.npz`).

## Setup

```bash
python3 -m pip install -r requirements.txt
cd data/pbmc3k    && python3 prepare.py && cd ../..   # needs internet
cd data/pathmnist && python3 prepare.py && cd ../..
./smoke_test.sh        # a few minutes on a Mac GPU (longer on CPU); all steps should PASS; prints the device (mps on a Mac)
```

Splits: PBMC3K uses a fixed train/test split. PathMNIST keeps MedMNIST's
three splits: `train` and `val` come from the same source (NCT-CRC-HE-100K),
while `test` comes from a different collection (CRC-VAL-HE-7K: other patients and staining). So realism and
copying checks compare against `val`, and downstream classifier tests use
`test` (which also measures transfer to new data).

The agent needs the Claude Code CLI (`claude`) on `PATH`, logged in. The
submitted runs used Claude Sonnet 5.5 (the CLI default at the time); set
`CLAUDE_MODEL=<model id>` before `run_agent.sh` to pin a specific model.

## Run

The two required reports (same commands as in `run_all.sh`):

```bash
./run_agent.sh pbmc3k \
  "Some cell types (for example megakaryocytes and dendritic cells) have very few cells. Generate synthetic cells to rebalance the classes so that a downstream cell-type classifier does better on the rare types." \
  generated_report_1 10

./run_agent.sh pathmnist \
  "We are not allowed to share the real patient image patches. Generate a synthetic, labeled image dataset that could be shared instead, so that others can develop tissue-type classifiers on it. It should look realistic, cover all tissue types, and must not simply copy training images." \
  generated_report_2 40
```

`./run_all.sh` runs these plus two robustness runs on PathMNIST with
different objectives: B (class balancing, in `reports/appendix/`) and A
(anomaly / quality-control screening with a normalizing flow). **Run A was
not run for the submitted report** (time); its code path is covered by the
smoke test. Progress is printed live; the full transcript is saved in
`outputs/<run_id>/agent_transcript.jsonl`, and if a run is repeated the
previous attempt is moved to `outputs/_previous/`.

To rebuild a report's PDF after editing its `.md` by hand:

```bash
python3 tools/export_report.py reports/<name>.md
```

## Evaluation tools in brief

| Task | Metrics (each reported next to a real-vs-real reference) |
|---|---|
| fidelity | Fréchet distance, k-NN precision/recall, RBF-MMD, classifier two-sample test, label consistency, per-class precision/recall |
| memorization | nearest-neighbour membership test (train vs held-out), distance-to-closest-record, exact copies |
| utility | train-on-synthetic/test-on-real; augmentation vs class weighting & oversampling, with paired-bootstrap CIs |
| anomaly | AUROC/AUPRC of −log p, PCA-reconstruction baseline, image-complexity check and complexity-corrected score |

Image features come from a small CNN trained once on the real training data
(`outputs/_cache/`), a domain-specific stand-in for FID's Inception network.
Downstream image classifiers are trained with 3 seeds, and differences are
judged with paired-bootstrap CIs. Every report ends in one of four verdicts:
*Use*, *Use with restrictions*, *Do not use (use X instead)*, or *Can't tell
(not enough evidence)*.

## Reproducibility

Fixed seeds, fixed train/test splits (stored in `prepared.npz`), every run's
full configuration in `run_config.json`, every agent decision in
`decisions.jsonl`. The tools are deterministic given their arguments up to
GPU nondeterminism; the agent's choices are LLM reasoning and can differ
between invocations — the decision log records what was chosen and why.

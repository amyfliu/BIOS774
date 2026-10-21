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
                                              └──────── iterate (≤ 2×) ◄─────────────┘
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
│   ├── show_progress.py      # pretty-prints the live agent transcript
│   ├── common.py             # data loading, preprocessing, device selection
│   └── models/               # vae.py gan.py flow.py ddpm.py nets.py (+ registry)
├── outputs/<run_id>/         # everything a run produced (+ agent_transcript.jsonl)
└── reports/                  # generated_report_1/2 (.md/.pdf), appendix/, report.pdf
```

## Setup

```bash
python3 -m pip install -r requirements.txt
cd data/pbmc3k    && python3 prepare.py && cd ../..   # needs internet
cd data/pathmnist && python3 prepare.py && cd ../..
./smoke_test.sh        # ~2-5 min; all steps should PASS; prints the device (mps on a Mac)
```

The agent needs the Claude Code CLI (`claude`) on `PATH`, logged in.

## Run

```bash
./run_agent.sh pbmc3k "Some cell types have very few cells. Generate synthetic cells to rebalance the classes for a cell-type classifier." generated_report_1 20
./run_agent.sh pathmnist "Generate a synthetic labeled image dataset that could be shared instead of the real patient images." generated_report_2 40
```

or `./run_all.sh` for all four runs (two required reports + two robustness
runs in `reports/appendix/`). Progress is printed live; the full transcript
is saved in `outputs/<run_id>/agent_transcript.jsonl`.

## Evaluation tools in brief

| Task | Metrics (each reported next to a real-vs-real reference) |
|---|---|
| fidelity | Fréchet distance, k-NN precision/recall, RBF-MMD, classifier two-sample test, label consistency, per-class precision/recall |
| memorization | nearest-neighbour membership test (train vs held-out), distance-to-closest-record, exact copies |
| utility | train-on-synthetic/test-on-real; augmentation vs class weighting & oversampling, with paired-bootstrap CIs |
| anomaly | AUROC/AUPRC of −log p, PCA-reconstruction baseline, image-complexity check and complexity-corrected score |

Image features come from a small CNN trained once on the real training data
(`outputs/_cache/`), a domain-specific stand-in for FID's Inception network.

## Reproducibility

Fixed seeds, fixed train/test splits (stored in `prepared.npz`), every run's
full configuration in `run_config.json`, every agent decision in
`decisions.jsonl`. The tools are deterministic given their arguments up to
GPU nondeterminism; the agent's choices are LLM reasoning and can differ
between invocations — the decision log records what was chosen and why.

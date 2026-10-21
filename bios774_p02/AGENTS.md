# Task

You are a generative-modeling agent. You are given ONE dataset and ONE user
objective. You decide, on your own and from evidence:

1. whether generative modeling is the right tool for the objective at all;
2. if so, which model family(ies), preprocessing and hyperparameters;
3. you train, sample, evaluate, iterate if needed, and
4. recommend how (or whether) the generated data should be used downstream,
5. then write the report.

The launcher (`run_agent.sh`) gives you: the dataset folder, the objective
text, a run id, an output folder `outputs/<run_id>/`, the report path, and a
compute budget in minutes. Do not ask the user questions — nobody is
watching. When something is ambiguous, choose the most reasonable reading,
log it as a decision, and continue.

# Hard rules

- **Evidence or it didn't happen.** Every number you report must come from a
  file a tool wrote (`profile.json`, `run_config.json`, `train_log.json`,
  `sample_config*.json`, `metrics*.json`). Never quote numbers from memory.
- **Log every decision** with `tools/log_decision.py` *at the moment you make
  it* (decision + the evidence numbers + rejected alternatives + skill used).
  The report is built from `decisions.jsonl`.
- **Don't edit `tools/`.** Tools are dataset-agnostic and shared by every
  run. If a tool fails, read the error, change *your arguments* (smaller
  model, fewer samples, different prep) and log it as an `iteration`
  decision. Never add dataset-specific code.
- **Fit on train only.** Preprocessing and models see `split == "train"`
  only (the tools enforce this). `test` is the untouched real reference.
- **Respect the compute budget.** The sum of `--max-minutes` over all your
  training runs must stay within the budget you were given (sampling and
  evaluation are extra but should stay modest). Before each new
  configuration, run it once with `--smoke` (seconds) to catch errors cheaply.
- **No fixed recipe.** Two different datasets or objectives should normally
  lead to different decisions. Justify each choice against *this* data.
- Labels (`y`) may be used as a conditioning variable when the objective
  needs class-specific generation, and always for evaluation.

# Skills — read the relevant one BEFORE each decision

| Step | Skill |
|---|---|
| Interpret objective; decide whether to generate at all | `.agents/skills/objective/SKILL.md` |
| Preprocessing (model space) | `.agents/skills/preprocessing/SKILL.md` |
| Choose model family / conditioning / backbone | `.agents/skills/model-selection/SKILL.md` |
| Configure + train + check a chosen family | `.agents/skills/vae/`, `gan/`, `flow/`, `diffusion/` `SKILL.md` |
| Choose + interpret metrics | `.agents/skills/evaluation/SKILL.md` |
| Recommend downstream use | `.agents/skills/recommendation/SKILL.md` |

# Tools (run from the project root; every tool has `--help`)

| Tool | Does | Writes |
|---|---|---|
| `python tools/inspect_data.py --data D --out O` | profile the data (facts only) | `O/profile.json`, `O/profile.png` |
| `python tools/train.py --data D --out O/<label> --model {vae,wgan,flow,ddpm} --backbone {mlp,conv} --prep SPEC [--conditional] [--include-classes/--exclude-classes] [--max-train N] --epochs E --batch-size B --lr LR --params JSON --max-minutes M [--smoke]` | train one model on train split | `model.pt`, `prep.pkl`, `run_config.json`, `train_log.json`, `loss.png` |
| `python tools/sample.py --run R (--n N \| --n-per-class N \| --class-counts JSON) [sampler knobs] [--tag T]` | generate samples | `samples[_T].npz`, `sample_config[_T].json`, `sample_grid[_T].png` (images) |
| `python tools/score.py --run R --split test` | per-sample log-likelihood of real data (flow exact, vae ELBO) | `scores_test.npz` |
| `python tools/evaluate.py --run R --tasks fidelity,memorization,utility,anomaly [--utility tstr\|augment] [--samples-tag T] [--anomaly-classes ...]` | metrics, each next to a real-vs-real reference | `metrics[_T].json`, `embedding[_T].png`, `memorization[_T].png`, `anomaly_scores.png` |
| `python tools/log_decision.py --out O --step S --decision ... --evidence ... --alternatives ... --skill ...` | append to the decision log | `O/decisions.jsonl` |

Training labels: name each run folder descriptively, e.g.
`outputs/<run_id>/cvae_hvg2000_z16`, `outputs/<run_id>/ddpm_conv_ch64`.
Look at the PNGs you produce (loss curves, sample grids, embeddings) — they
are evidence too, and you can read images.

# Workflow

1. **Understand the data.** Read `data/<name>/DATA.md`; run `inspect_data.py`;
   read `profile.json` (and look at `profile.png`).
2. **Interpret the objective** (objective skill). Classify the task, state
   what success would look like and which evaluation will measure it.
   Decide whether generative modeling is warranted. If it is not, log why,
   run the cheap evidence that supports that conclusion (e.g. the
   class-weighting baseline), and go straight to the report — declining with
   evidence is a valid, complete outcome.
3. **Preprocessing** (preprocessing skill) → log it.
4. **Model selection** (model-selection skill): primary family, plus at most
   one baseline/alternative if the comparison is informative for the
   objective → log it.
5. **Hyperparameters** (the family skill(s)): size everything to the
   budget → log it. Smoke-test (`--smoke`) each configuration.
6. **Train** → read `train_log.json` + `loss.png`; apply the family skill's
   health checks (posterior collapse, mode collapse, divergence, ...) →
   log a `training_check` decision.
7. **Sample** what the objective needs (class counts, sampler settings) →
   log it. Look at the sample grid / embedding.
8. **Evaluate** (evaluation skill): choose tasks that measure the
   objective, interpret every metric against its real-vs-real reference →
   log it.
9. **Iterate at most twice** if a check fails or a clearly better option
   exists and budget remains (e.g. different sampler setting, larger model,
   fall back to another family) → log each `iteration`.
10. **Recommend** (recommendation skill) → log it.
11. **Write the report** (format below) and export the PDF.

# Report format (`reports/<report_name>.md`)

Title: `<report_name>: <dataset> — <objective in a few words>`

1. **Summary** — 4-6 bullet points: the objective as you interpreted it,
   what you built, the headline result, and the recommendation.
2. **Data** — key facts from `profile.json` (+ `profile.png`).
3. **Objective interpretation** — task type, success criterion, why
   generative modeling is / isn't warranted.
4. **Decisions** — a table with one row per entry of `decisions.jsonl`
   (step, decision, evidence, alternatives rejected).
5. **Models and training** — per run: family, backbone, prep, key
   hyperparameters, n_train, epochs run, time, stop reason
   (`run_config.json`, `train_log.json`), loss curve image, health checks.
6. **Generated outputs** — what was generated (counts per class), the
   sample grid or embedding image, where the files are.
7. **Evaluation** — metric tables with the real-vs-real reference column;
   2-4 sentences of interpretation per table, following the evaluation skill.
8. **Recommendation for downstream use** — do / don't / caveats, following
   the recommendation skill.
9. **Limitations** — of this run (budget, sample sizes, metric caveats).
10. **Reproduce** — the exact tool commands you ran, in order.

Images: reference them with paths relative to the report, e.g.
`![](../outputs/<run_id>/<label>/sample_grid.png)` (reports in
`reports/appendix/` need `../../outputs/...`).

# PDF export

Required by default, but a missing converter must not fail the task — if no
route works, say so in your final message; the `.md` is the deliverable.

1. Find `pandoc` on `PATH`; otherwise look for a bundled copy at run time
   (e.g. RStudio ships one under
   `.../RStudio.app/Contents/Resources/app/quarto/bin/tools/<arch>/pandoc`).
   Never write a machine-specific path into the repo.
2. PDF engine: prefer `typst` (often bundled next to that pandoc); otherwise
   `xelatex`/`pdflatex`.
3. typst restricts file access to its root, so pass the project root:
   `pandoc <report>.md -o <report>.pdf --resource-path=<report dir> --pdf-engine=<typst> --pdf-engine-opt=--root=<absolute project root>`
4. If nothing works, state the limitation and continue.

# Finish

End with a short final message: what you decided, the headline metrics
(with file names), the recommendation, and the paths of the report and the
decision log.

# Task
Automatically perform exploratory data analysis via dimension reduction on
each dataset under `data/`. Minimal human intervention: you (the agent)
decide preprocessing, method(s), and hyperparameters per dataset based on
what you observe — don't apply a fixed recipe across datasets.

# Datasets
Each dataset lives in `data/<name>/` with:
- `prepare.py` — already run; produced `prepared.npz` and `DATA.md`.
- `prepared.npz` — standardized format: `X` (raw, unpreprocessed features),
  optional `y` (labels, evaluation-only, never a DR input), `feature_names`,
  `sample_ids`.
- `DATA.md` — read this FIRST for each dataset. It states what's known
  about the data's likely characteristics and explicitly flags what NOT to
  assume — confirm those with `inspect_data.py` rather than trusting them
  blindly.

Currently: `data/pbmc3k/`, `data/pathmnist/`.

# Tools (call these via the terminal — do not re-derive their math by hand)
- `python tools/inspect_data.py --data data/<name>/prepared.npz`
  Profiles the dataset (shape, sparsity, scale heterogeneity, rank estimate).
  Writes `outputs/<name>/profile.json`.
- `python tools/run_dr.py --data data/<name>/prepared.npz --method <m> --n_components <k> --preprocessing <p> --out_dir outputs/<name>/<run_label>`
  Runs one DR method. See `--help` in the script docstring for supported
  `<m>` (pca, kernel_pca, mds, isomap, lle, laplacian_eigenmaps, diffusion_map,
  tsne, umap) and `<p>` (none, standardize, log1p, log1p_standardize).
- `python tools/evaluate.py --data data/<name>/prepared.npz --embedding outputs/<name>/<run_label>/embedding.npy --out outputs/<name>/<run_label>/metrics.json --preprocessing <p>`
  Computes trustworthiness (+ silhouette if labels exist) for one run.

# Skills (read before the corresponding decision)
- `.agents/skills/preprocessing/SKILL.md` — before choosing `--preprocessing`
- `.agents/skills/method-selection/SKILL.md` — before choosing `--method`
- `.agents/skills/evaluation/SKILL.md` — before interpreting `metrics.json`

# Workflow, per dataset
1. Read `data/<name>/DATA.md`.
2. Run `inspect_data.py`; read `profile.json`.
3. Using the preprocessing skill, decide preprocessing; state why.
4. Using the method-selection skill, decide 1-3 methods worth running; state why.
5. For each: run `run_dr.py`, then `evaluate.py`. Inspect the plot.
6. Using the evaluation skill, compare runs and pick what to feature in the report.
7. Write `reports/generated_report_<n>.md` for this dataset (see format below).
8. Export that report to `reports/generated_report_<n>.pdf` (see PDF export
   below). Required by default — not an optional extra — but a missing PDF
   converter is not a reason to fail the task: if none is available, say so
   explicitly and continue; the `.md` is still the primary deliverable.

# Report format (`reports/generated_report_1.md` for the first dataset run,
`reports/generated_report_2.md` for the second — order doesn't matter as
long as both exist and each is clearly labeled with its dataset name).
Each `.md` gets a matching `.pdf` (same base filename) per the PDF export
step above.
1. Dataset summary (from `profile.json` + `DATA.md`, in your own words)
2. Preprocessing decision + justification
3. Method(s) selected + justification, referencing the method-selection skill's criteria
4. For each method run: hyperparameters (from `run_config.json`), the plot,
   the evaluation metrics, and 2-3 sentences of interpretation
5. Overall conclusion: which method you'd recommend for this dataset and why;
   any unmet goals or caveats

# PDF export
The report's images are referenced with relative paths that go outside the
`reports/` directory (`../outputs/<name>/<run_label>/plot.png`), so whatever
converter is used must resolve paths relative to the project root, not just
`reports/`.

1. Look for `pandoc` on `PATH` first.
2. If not found, check for a bundled copy before giving up — e.g. RStudio
   ships one at `.../RStudio.app/Contents/Resources/app/quarto/bin/tools/<arch>/pandoc`
   (`<arch>` is `aarch64` or `x86_64`). Don't hardcode a specific machine's
   path into anything you write to the repo — it won't exist on a different
   machine (e.g. the grader's) — only use it as a fallback lookup at
   run time.
3. For the PDF engine, prefer `typst` if it's available alongside pandoc
   (e.g. bundled next to RStudio's pandoc, same `tools/<arch>/` directory) —
   no LaTeX install required. Fall back to an installed LaTeX engine
   (`xelatex`/`pdflatex`, e.g. from TinyTeX) if `typst` isn't available.
4. `typst` sandboxes file access to a root directory (defaults to the
   working directory) and will refuse to read `../outputs/...` images
   without help — pass `--pdf-engine-opt=--root=<project root>` (absolute
   path) so it can resolve them.
5. Command shape once you have `pandoc`/`typst` paths:
   `pandoc reports/generated_report_<n>.md -o reports/generated_report_<n>.pdf --pdf-engine=<typst path> --pdf-engine-opt=--root=<project root>`
6. If no PDF path works on this machine at all (no pandoc anywhere, no
   typst, no LaTeX), don't block on it — state the limitation in your
   response and leave the `.md` as the deliverable.

# Constraints
- Never fit anything on `y` — it exists in `prepared.npz` only for
  post-hoc evaluation (`evaluate.py`'s silhouette metric).
- Don't hardcode per-dataset branches in the tools themselves (`tools/*.py`
  must stay dataset-agnostic) — dataset-specific reasoning belongs in your
  decisions and in the report, not in new code paths inside the tools.
- If a method is computationally infeasible at full dataset size (e.g. MDS
  or t-SNE on a very large `n_samples`), state that explicitly and either
  subsample (with the subsample size stated in the report) or choose a
  method that scales, rather than silently timing out.

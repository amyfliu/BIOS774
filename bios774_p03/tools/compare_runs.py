"""compare_runs.py — one table comparing every evaluated model in a run folder,
plus paired differences against a reference model.

  python tools/compare_runs.py --out outputs/<run_id> --split val
  python tools/compare_runs.py --out outputs/<run_id> --split test --reference rna_only_pca

Reads <out>/*/metrics_val.json (--split val: for choosing models) or
<out>/*/metrics.json (--split test: final numbers) + run_config.json and writes
  <out>/comparison_<split>.md    markdown table (one row per model; best value per column in bold)
  <out>/comparison_<split>.json  the same numbers
  <out>/comparison_<split>.png   bar chart of the headline metrics

Columns: method + inputs, whether labels were used in training, linear-probe
balanced accuracy of the JOINT embedding and of EACH modality's own embedding
(column suffix "(image)", "(rna)", ...; only for multimodal models), k-means
ARI of the joint embedding, for EVERY label column,
cross-modal retrieval recall@5 both ways + FOSCTTM, cross-modal prediction
scores, effective rank. Missing values are shown as "–".

--reference <model folder name>  adds a second table: for every other model,
the difference (model − reference) in each headline metric — the probe for
every embedding both models share (joint, image, text, ...) — with a 95% PAIRED
bootstrap interval — both models scored on the SAME samples, resampled
together (from predictions_<split>.npz). This is the right test for "is A
better than B?": overlapping per-model intervals are NOT evidence of no
difference. Use --reference repeatedly (one call per reference) to compare
against several baselines; each call writes comparison_<split>_vs_<reference>.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_json, rounded, save_json  # noqa: E402

B = 2000


def row(run: Path, fname: str):
    m = load_json(run / fname)
    cfg = load_json(run / "run_config.json")
    r = {"model": run.name, "method": cfg.get("method"), "inputs": cfg.get("inputs"),
         "uses_labels": "yes" if cfg.get("uses_labels") else "no"}
    for lab, d in m.get("structure", {}).items():
        j = d.get("z_joint", {})
        r[f"probe_bal_acc[{lab}]"] = j.get("probe_balanced_accuracy")
        mod_keys = sorted(k for k in d if k != "z_joint")
        for key in (mod_keys if len(mod_keys) >= 2 else []):  # each modality's own side of a multimodal model
                r[f"probe_bal_acc[{lab}]({key[2:]})"] = d[key].get("probe_balanced_accuracy")
        r[f"kmeans_ari[{lab}]"] = j.get("kmeans_ari")
        r[f"chance[{lab}]"] = j.get("chance_balanced_accuracy")
    ret = m.get("retrieval", {})
    if "a_to_b" in ret:
        r[f"R@5 {ret['a']}->{ret['b']}"] = ret["a_to_b"]["recall_at_5"]
        r[f"R@5 {ret['b']}->{ret['a']}"] = ret["b_to_a"]["recall_at_5"]
        r["FOSCTTM"] = (ret["a_to_b"]["foscttm"] + ret["b_to_a"]["foscttm"]) / 2
        r["R@5 chance"] = ret["chance"]["recall_at_5"]
    for item, c in m.get("cross", {}).items():
        name = item.split(":")[0]
        if "per_sample_pearson_mean" in c:
            r[f"{name} per-cell r"] = c["per_sample_pearson_mean"]
            r[f"{name} per-cell r (baseline)"] = c["per_sample_pearson_baseline_train_mean"]
        elif "per_feature_auroc_mean" in c:
            r[f"{name} per-feature AUROC"] = c["per_feature_auroc_mean"]
            r[f"{name} per-cell AUROC"] = c["per_sample_auroc_mean"]
    col = m.get("collapse", {}).get("z_joint", {})
    if col:
        r["eff_rank(joint)"] = col.get("effective_rank")
    return rounded(r, 3)


# ----------------------------------------------------------------------------- paired bootstrap
def bal_acc(y, p, k):
    """balanced accuracy, vectorised: mean recall over classes present in y."""
    n_c = np.bincount(y, minlength=k)
    hit = np.bincount(y[p == y], minlength=k)
    present = n_c > 0
    return float((hit[present] / n_c[present]).mean())


def paired(stat_a, stat_b, n, rng):
    """point difference + 95% CI of stat_a(i) - stat_b(i) over B paired resamples."""
    full = np.arange(n)
    d0 = stat_a(full) - stat_b(full)
    ds = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        ds.append(stat_a(i) - stat_b(i))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return float(d0), float(lo), float(hi), float(stat_a(full)), float(stat_b(full))


def paired_table(out: Path, runs, ref: str, split: str):
    pf = f"predictions_{split}.npz"
    refdir = out / ref
    if not (refdir / pf).exists():
        raise SystemExit(f"{refdir / pf} missing — run evaluate.py --split {split} on {ref} "
                         f"(re-run it if it was evaluated before paired outputs existed)")
    R = dict(np.load(refdir / pf))
    rng = np.random.default_rng(0)
    rows = []
    for run in runs:
        if run.name == ref or not (run / pf).exists():
            continue
        M = dict(np.load(run / pf))
        if not np.array_equal(M["idx"], R["idx"]):
            rows.append({"model": run.name, "metric": "(skipped: evaluated on different samples)"})
            continue
        n = len(R["idx"])
        for key in sorted(M):
            if key.startswith("probe__") and key in R:
                lab, emb_key = key.split("__")[1], key.split("__")[2]
                y = R[f"y__{lab}"]
                k = int(max(y.max(), M[key].max(), R[key].max())) + 1
                pa, pb = M[key], R[key]
                d = paired(lambda i: bal_acc(y[i], pa[i], k), lambda i: bal_acc(y[i], pb[i], k), n, rng)
                rows.append(dict(model=run.name, metric=f"probe balanced accuracy [{lab}] ({emb_key[2:]})", diff=d))
            elif key.startswith("cross__") and key in R:
                a_, b_ = M[key], R[key]
                d = paired(lambda i: float(np.nanmean(a_[i])), lambda i: float(np.nanmean(b_[i])), n, rng)
                rows.append(dict(model=run.name, metric=f"cross {key[7:]} per-sample mean", diff=d))
        for dname in ("a_to_b", "b_to_a"):
            key = f"ranks_{dname}"
            if key in M and key in R and np.array_equal(M["retrieval_idx"], R["retrieval_idx"]):
                ra, rb = M[key], R[key]
                d = paired(lambda i: float((ra[i] < 5).mean()), lambda i: float((rb[i] < 5).mean()), len(ra), rng)
                rows.append(dict(model=run.name, metric=f"recall@5 {dname}", diff=d))
    lines = [f"Paired differences vs **{ref}** on the {split} split (model − reference; 95% paired bootstrap, "
             f"{B} resamples of the same samples)\n",
             "| model | metric | model | reference | difference [95% CI] | CI excludes 0 |",
             "|---|---|---|---|---|---|"]
    js = []
    for r in rows:
        if "diff" not in r:
            lines.append(f"| {r['model']} | {r['metric']} | – | – | – | – |")
            continue
        d0, lo, hi, va, vb = r["diff"]
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        lines.append(f"| {r['model']} | {r['metric']} | {va:.3f} | {vb:.3f} | {d0:+.3f} [{lo:+.3f}, {hi:+.3f}] | {sig} |")
        js.append(dict(model=r["model"], metric=r["metric"], model_value=va, reference_value=vb,
                       difference=d0, ci95=[lo, hi], ci_excludes_0=sig == "yes"))
    if not rows:
        lines.append("| – | no shared metrics (evaluate the reference and models with the same tasks) | – | – | – | – |")
    md = "\n".join(lines) + "\n"
    (out / f"comparison_{split}_vs_{ref}.md").write_text(md)
    save_json(rounded(js, 4), out / f"comparison_{split}_vs_{ref}.json")
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--reference", default=None, help="model folder name to compare every other model against (paired)")
    ap.add_argument("--label", default=None, help="(kept for compatibility; all label columns are shown)")
    a = ap.parse_args()
    out = Path(a.out)
    fname = "metrics_val.json" if a.split == "val" else "metrics.json"
    runs = sorted(p.parent for p in out.glob(f"*/{fname}"))
    if not runs:
        raise SystemExit(f"no */{fname} under {out} (run evaluate.py --split {a.split} first)")
    rows = [row(r, fname) for r in runs]
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    higher_better = lambda c: not (c.startswith("FOSCTTM") or "chance" in c or "baseline" in c)  # noqa: E731
    best = {}
    for c in cols[4:]:
        vals = [r.get(c) for r in rows if isinstance(r.get(c), (int, float))]
        if len(vals) > 1 and "chance" not in c and "baseline" not in c and not c.startswith("eff_rank"):
            best[c] = max(vals) if higher_better(c) else min(vals)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "–")
            s = str(v) if v is not None else "–"
            if c in best and v == best[c]:
                s = f"**{s}**"
            cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")
    md = (f"Split scored: **{a.split}**. Per-model numbers; for 'is A better than B' use the paired table "
          f"(--reference).\n\n" + "\n".join(lines) + "\n")
    (out / f"comparison_{a.split}.md").write_text(md)
    save_json(rows, out / f"comparison_{a.split}.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [c for c in cols if c.startswith("probe_bal_acc") or (c.startswith("R@5") and "chance" not in c)]
    if metrics:
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 0.45 * len(rows) + 1.4))
        for ax, c in zip(list(axes) if len(metrics) > 1 else [axes], metrics):
            vals = [r.get(c) if isinstance(r.get(c), (int, float)) else 0 for r in rows]
            ax.barh([r["model"] for r in rows][::-1], vals[::-1], color="#4C72B0")
            ref = rows[0].get("R@5 chance") if c.startswith("R@5") else rows[0].get(c.replace("probe_bal_acc", "chance"))
            if isinstance(ref, (int, float)):
                ax.axvline(ref, ls="--", c="grey", lw=1, label="chance")
                ax.legend(fontsize=7)
            ax.set_title(c, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.set_xlim(0, 1)
        fig.tight_layout(); fig.savefig(out / f"comparison_{a.split}.png", dpi=130); plt.close(fig)
    print(md)
    print(f"wrote {out / f'comparison_{a.split}.md'} (+ .json, .png)")
    if a.reference:
        pmd = paired_table(out, runs, a.reference, a.split)
        print(pmd)
        print(f"wrote {out / f'comparison_{a.split}_vs_{a.reference}.md'} (+ .json)")


if __name__ == "__main__":
    main()

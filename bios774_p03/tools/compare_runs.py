"""compare_runs.py — one table comparing every evaluated model in a run folder.

  python tools/compare_runs.py --out outputs/<run_id> [--label label_fine]

Reads <out>/*/metrics_val.json (--split val, default: for choosing models) or
<out>/*/metrics.json (--split test: final numbers) + run_config.json and writes
  <out>/comparison_<split>.md    markdown table (one row per model; best value per column in bold)
  <out>/comparison_<split>.json  the same numbers
  <out>/comparison_<split>.png   bar chart of the headline metrics
Columns: method + inputs, whether labels were used in training, linear-probe
balanced accuracy and k-means ARI of the joint embedding (for --label),
cross-modal retrieval recall@5 both ways + FOSCTTM, cross-modal prediction
scores, effective rank. Missing values are shown as "–".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_json, rounded, save_json  # noqa: E402


def row(run: Path, label: str | None, fname: str):
    m = load_json(run / fname)
    cfg = load_json(run / "run_config.json")
    r = {"model": run.name, "method": cfg.get("method"), "inputs": cfg.get("inputs"),
         "uses_labels": "yes" if cfg.get("uses_labels") else "no"}
    st = m.get("structure", {})
    lab = label if label in st else (next(iter(st)) if st else None)
    if lab:
        j = st[lab].get("z_joint", {})
        r[f"probe_bal_acc[{lab}]"] = j.get("probe_balanced_accuracy")
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
    col = m.get("collapse", {}).get("z_joint", {})
    if col:
        r["eff_rank(joint)"] = col.get("effective_rank")
    return rounded(r, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--label", default=None, help="label column for the structure columns (default: first)")
    a = ap.parse_args()
    out = Path(a.out)
    fname = "metrics_val.json" if a.split == "val" else "metrics.json"
    runs = sorted(p.parent for p in out.glob(f"*/{fname}"))
    if not runs:
        raise SystemExit(f"no */{fname} under {out} (run evaluate.py --split {a.split} first)")
    rows = [row(r, a.label, fname) for r in runs]
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
    md = "\n".join(lines) + "\n"
    md = f"Split scored: **{a.split}** (n per model in metrics files)\n\n" + md
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


if __name__ == "__main__":
    main()

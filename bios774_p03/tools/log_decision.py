"""log_decision.py — append one decision to the run's decision log.

  python tools/log_decision.py --out outputs/r1_pbmc_multiome --step fusion \
      --short "Align RNA and ATAC (CLIP-style)" \
      --why "Same cells in both; goal needs one modality to find the other" \
      --decision "clip: per-modality MLP encoders, symmetric InfoNCE, dim 32" \
      --evidence "paired n_train=7285 (profile.json); retrieval is an objective task" \
      --alternatives "pca_concat: no cross-modal map; vicreg: kept as comparison" \
      --skill fusion

Each call appends one JSON line to <out>/decisions.jsonl. The generated report
quotes this log, so every decision is traceable to evidence. --short / --why are
plain-language one-liners (no code names) used in the report's Decision path table;
--decision / --evidence / --alternatives hold the full technical detail (appendix).
Steps: objective | preprocessing | fusion | model_selection | hyperparameters |
       training_check | evaluation | downstream | iteration | recommendation | other
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

STEPS = ["objective", "preprocessing", "fusion", "model_selection", "hyperparameters", "training_check",
         "evaluation", "downstream", "iteration", "recommendation", "other"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--step", required=True, choices=STEPS)
    p.add_argument("--short", default="", help="what was decided, plain words, <= 8 words")
    p.add_argument("--why", default="", help="why, plain words, <= 15 words")
    p.add_argument("--decision", required=True)
    p.add_argument("--evidence", required=True, help="numbers + the file they came from")
    p.add_argument("--alternatives", default="", help="options considered and why rejected")
    p.add_argument("--skill", default="", help="skill whose criteria were applied")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"time": datetime.now().isoformat(timespec="seconds"), "step": a.step,
           "short": a.short, "why": a.why, "decision": a.decision,
           "evidence": a.evidence, "alternatives_rejected": a.alternatives, "skill": a.skill}
    with open(out / "decisions.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    n = sum(1 for _ in open(out / "decisions.jsonl"))
    print(f"[decision #{n}] {a.step}: {a.decision}")


if __name__ == "__main__":
    main()

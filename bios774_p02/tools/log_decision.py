"""log_decision.py — append one decision to the run's decision log.

  python tools/log_decision.py --out outputs/r1_pbmc3k --step model_selection \
      --decision "conditional VAE (mlp)" \
      --evidence "n_train=1978; smallest class has 11 train cells (profile.json)" \
      --alternatives "ddpm: too little data; wgan: unstable at this n" \
      --skill model-selection

Each call appends one JSON line to <out>/decisions.jsonl. The generated report
quotes this log, so every decision is traceable to evidence.
Steps: objective | preprocessing | model_selection | hyperparameters | training_check |
       sampling | evaluation | iteration | recommendation | other
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

STEPS = ["objective", "preprocessing", "model_selection", "hyperparameters", "training_check",
         "sampling", "evaluation", "iteration", "recommendation", "other"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--step", required=True, choices=STEPS)
    p.add_argument("--decision", required=True)
    p.add_argument("--evidence", required=True, help="numbers + the file they came from")
    p.add_argument("--alternatives", default="", help="options considered and why rejected")
    p.add_argument("--skill", default="", help="skill whose criteria were applied")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"time": datetime.now().isoformat(timespec="seconds"), "step": a.step, "decision": a.decision,
           "evidence": a.evidence, "alternatives_rejected": a.alternatives, "skill": a.skill}
    with open(out / "decisions.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    n = sum(1 for _ in open(out / "decisions.jsonl"))
    print(f"[decision #{n}] {a.step}: {a.decision}")


if __name__ == "__main__":
    main()

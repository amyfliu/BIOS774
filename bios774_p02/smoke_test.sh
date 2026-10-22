#!/usr/bin/env bash
# smoke_test.sh — check that every tool and model runs on THIS machine
# (Apple-GPU/MPS on a Mac), and measure training speed for budgeting.
#
#   ./smoke_test.sh            # ~2-5 minutes; needs data/*/prepared.npz
#
# It does NOT test model quality — only that the code path works end to end.
# Writes to outputs/_smoke/ (safe to delete).
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$(command -v python3 || command -v python)}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
OUT=outputs/_smoke
mkdir -p "$OUT"
PASS=0; FAIL=0; FAILED=()

step () {   # step "<label>" <command...>
  local label="$1"; shift
  if "$@" > "$OUT/log_$(echo "$label" | tr ' /' '__').txt" 2>&1; then
    printf "  PASS  %s\n" "$label"; PASS=$((PASS+1))
  else
    printf "  FAIL  %s   (see %s)\n" "$label" "$OUT/log_$(echo "$label" | tr ' /' '__').txt"; FAIL=$((FAIL+1)); FAILED+=("$label")
  fi
}

echo "== environment"
"$PY" - <<'EOF'
import sys, platform
print("  python", sys.version.split()[0], "|", platform.platform())
import torch, numpy, sklearn, scipy, matplotlib
print("  torch", torch.__version__, "| numpy", numpy.__version__, "| sklearn", sklearn.__version__)
print("  MPS built:", torch.backends.mps.is_built(), "| MPS available:", torch.backends.mps.is_available())
sys.path.insert(0, "tools")
from common import get_device
d = get_device(); print("  tools will use device:", d)
x = torch.randn(256, 256, device=d); y = (x @ x.T).sum().item()
print("  matmul on", d, "OK")
EOF
[ $? -eq 0 ] || { echo "Environment check failed — install requirements first: $PY -m pip install -r requirements.txt"; exit 1; }

for ds in pbmc3k pathmnist; do
  [ -f "data/$ds/prepared.npz" ] || { echo "data/$ds/prepared.npz missing — run: cd data/$ds && $PY prepare.py"; exit 1; }
done

echo "== tabular (pbmc3k)"
P=libnorm,log1p,hvg:500,standardize
step "inspect pbmc3k" "$PY" tools/inspect_data.py --data data/pbmc3k --out $OUT/pbmc3k
for m in vae wgan flow ddpm; do
  step "train $m mlp" "$PY" tools/train.py --data data/pbmc3k --out $OUT/pbmc3k/$m --model $m --prep $P --conditional --smoke
  step "sample $m mlp" "$PY" tools/sample.py --run $OUT/pbmc3k/$m --class-counts '{"Megakaryocytes": 20, "Dendritic cells": 20}' --steps 5
done
step "sample vae posterior" "$PY" tools/sample.py --run $OUT/pbmc3k/vae --class-counts '{"Megakaryocytes": 20}' --latent-source posterior --tag post
step "score flow mlp" "$PY" tools/score.py --run $OUT/pbmc3k/flow --split test
step "evaluate tabular" "$PY" tools/evaluate.py --run $OUT/pbmc3k/vae --tasks fidelity,memorization,utility --utility augment

echo "== images (pathmnist)"
step "inspect pathmnist" "$PY" tools/inspect_data.py --data data/pathmnist --out $OUT/pathmnist
for m in vae wgan ddpm; do
  step "train $m conv" "$PY" tools/train.py --data data/pathmnist --out $OUT/pathmnist/$m --model $m --backbone conv --prep scale_pm1 --conditional --smoke
  step "sample $m conv" "$PY" tools/sample.py --run $OUT/pathmnist/$m --n-per-class 4 --steps 5 --guidance 2.0
done
step "train flow conv" "$PY" tools/train.py --data data/pathmnist --out $OUT/pathmnist/flow --model flow --backbone conv --prep scale01 --exclude-classes background,debris --smoke
step "score flow conv" "$PY" tools/score.py --run $OUT/pathmnist/flow --split test --max-n 500
step "evaluate anomaly" "$PY" tools/evaluate.py --run $OUT/pathmnist/flow --tasks anomaly --anomaly-classes background,debris
step "evaluate images (trains + caches eval CNN once)" "$PY" tools/evaluate.py --run $OUT/pathmnist/ddpm --tasks fidelity,memorization,utility --utility tstr --clf-epochs 1
step "log_decision" "$PY" tools/log_decision.py --out $OUT --step other --decision "smoke test" --evidence "n/a"
printf '# t\n**Verdict:** ok\nIntro\n- a\n- b\n\n| x | n |\n|---|---|\n| + y | 1.0 |\n\n![c](pathmnist/ddpm/sample_grid.png)\n' > $OUT/smoke_report.md
step "export_report (tidy only)" "$PY" tools/export_report.py $OUT/smoke_report.md --no-pdf

echo "== speed probe (1 epoch, realistic sizes) — use these numbers to size budgets"
"$PY" tools/train.py --data data/pathmnist --out $OUT/probe_ddpm --model ddpm --backbone conv --prep scale_pm1 \
   --conditional --max-train 3000 --epochs 1 --batch-size 64 --lr 2e-4 --params '{"ch": 32}' 2>&1 | grep -E "\[train\] (ddpm|epoch)"
"$PY" tools/train.py --data data/pbmc3k --out $OUT/probe_vae --model vae --prep libnorm,log1p,hvg:2000,standardize \
   --conditional --epochs 5 --batch-size 128 2>&1 | grep -E "\[train\] (vae|epoch 5)"

echo
echo "== $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ] || { printf '   failed: %s\n' "${FAILED[@]}"; exit 1; }

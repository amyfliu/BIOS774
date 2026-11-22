#!/usr/bin/env bash
# smoke_test.sh — check that every tool and method runs on THIS machine
# (Apple GPU / MPS on a Mac). Does NOT test result quality.
#
#   ./smoke_test.sh          # ~1-3 min: whole tool chain on the synthetic dev datasets
#   ./smoke_test.sh --real   # + preprocessing/training speed on the real data and a
#                            #   BiomedCLIP check (downloads ~800 MB on first use)
#
# Writes to outputs/_smoke/ (safe to delete). The dev datasets are rebuilt by
# data/_dev/make_dev.py if missing.
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$(command -v python3 || command -v python)}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
OUT=outputs/_smoke
rm -rf "$OUT"; mkdir -p "$OUT"
PASS=0; FAIL=0; FAILED=()

step () {   # step "<label>" <command...>
  local label="$1"; shift
  local log="$OUT/log_$(echo "$label" | tr ' /:>,' '_____').txt"
  if "$@" > "$log" 2>&1; then
    printf "  PASS  %s\n" "$label"; PASS=$((PASS+1))
  else
    printf "  FAIL  %s   (see %s)\n" "$label" "$log"; FAIL=$((FAIL+1)); FAILED+=("$label")
  fi
}

echo "== environment"
"$PY" - <<'EOF'
import sys, platform
print("  python", sys.version.split()[0], "|", platform.platform())
import torch, numpy, sklearn, scipy, matplotlib, pandas
print("  torch", torch.__version__, "| numpy", numpy.__version__, "| sklearn", sklearn.__version__)
print("  MPS built:", torch.backends.mps.is_built(), "| MPS available:", torch.backends.mps.is_available())
sys.path.insert(0, "tools")
from common import get_device
d = get_device(); print("  tools will use device:", d)
x = torch.randn(256, 256, device=d); _ = (x @ x.T).sum().item()
print("  matmul on", d, "OK")
EOF
[ $? -eq 0 ] || { echo "Environment check failed — install requirements first: $PY -m pip install -r requirements.txt"; exit 1; }

if [ ! -f data/_dev/pbmc_multiome/dataset.json ] || [ ! -f data/_dev/rocov2/dataset.json ]; then
  step "build dev datasets" "$PY" data/_dev/make_dev.py
fi

echo "== paired counts (dev pbmc_multiome)"
D=data/_dev/pbmc_multiome; O=$OUT/pbmc
step "inspect" "$PY" tools/inspect_data.py --data $D --out $O
step "prep rna log1p_hvg200_pca20" "$PY" tools/preprocess.py --data $D --out $O --modality rna --spec log1p_hvg200_pca20
step "prep atac tfidf_lsi20_drop1" "$PY" tools/preprocess.py --data $D --out $O --modality atac --spec tfidf_lsi20_drop1
step "prep rna target log1p_hvg200" "$PY" tools/preprocess.py --data $D --out $O --modality rna --spec log1p_hvg200
step "prep atac target binary_top500" "$PY" tools/preprocess.py --data $D --out $O --modality atac --spec binary_top500
I=rna:log1p_hvg200_pca20,atac:tfidf_lsi20_drop1
step "train pca_concat rna" "$PY" tools/train.py --data $D --out $O/rna_only --inputs rna:log1p_hvg200_pca20 --method pca_concat --dim 16
step "train pca_concat both" "$PY" tools/train.py --data $D --out $O/early_fusion --inputs $I --method pca_concat --dim 16
step "train clip mlp" "$PY" tools/train.py --data $D --out $O/clip --inputs $I --method clip --dim 16 --smoke
step "train vicreg mlp" "$PY" tools/train.py --data $D --out $O/vicreg --inputs $I --method vicreg --dim 16 --smoke
step "train supcon_clip" "$PY" tools/train.py --data $D --out $O/supcon --inputs $I --method supcon_clip --dim 16 --smoke
step "train clip linear" "$PY" tools/train.py --data $D --out $O/clip_lin --inputs $I --method clip --encoder linear --dim 16 --smoke
step "train simclr" "$PY" tools/train.py --data $D --out $O/simclr --inputs rna:log1p_hvg200_pca20 --method simclr --dim 16 --smoke
X="rna->atac:binary_top500,atac->rna:log1p_hvg200"
step "evaluate val (rna_only)" "$PY" tools/evaluate.py --data $D --run $O/rna_only --split val --tasks structure,collapse,cross --cross "$X"
step "evaluate val (clip)" "$PY" tools/evaluate.py --data $D --run $O/clip --split val --tasks structure,retrieval,collapse,cross --cross "$X"
step "evaluate val (vicreg)" "$PY" tools/evaluate.py --data $D --run $O/vicreg --split val --tasks structure,retrieval,collapse,cross --cross "$X"
step "evaluate test (clip)" "$PY" tools/evaluate.py --data $D --run $O/clip --split test --tasks structure,retrieval,collapse,cross --cross "$X"
step "evaluate test (rna_only)" "$PY" tools/evaluate.py --data $D --run $O/rna_only --split test --tasks structure,collapse,cross --cross "$X" --no-plot
step "compare val" "$PY" tools/compare_runs.py --out $O --split val
step "compare test, paired vs rna_only" "$PY" tools/compare_runs.py --out $O --split test --reference rna_only

echo "== image + text (dev rocov2)"
D=data/_dev/rocov2; O=$OUT/rocov2
step "inspect" "$PY" tools/inspect_data.py --data $D --out $O
step "prep image pixels32_pca16" "$PY" tools/preprocess.py --data $D --out $O --modality image --spec pixels32_pca16
step "prep text tfidf_svd16" "$PY" tools/preprocess.py --data $D --out $O --modality text --spec tfidf_svd16
I=image:pixels32_pca16,text:tfidf_svd16
step "train identity" "$PY" tools/train.py --data $D --out $O/identity --inputs $I --method identity
step "train clip adapter" "$PY" tools/train.py --data $D --out $O/adapter --inputs $I --method clip --encoder adapter --dim 16 --hidden 32 --smoke
step "evaluate val (adapter)" "$PY" tools/evaluate.py --data $D --run $O/adapter --split val --tasks structure,retrieval,collapse
step "evaluate test (adapter)" "$PY" tools/evaluate.py --data $D --run $O/adapter --split test --tasks structure,retrieval,collapse
step "evaluate test (identity)" "$PY" tools/evaluate.py --data $D --run $O/identity --split test --tasks structure,retrieval,collapse
step "compare test, paired vs identity" "$PY" tools/compare_runs.py --out $O --split test --reference identity

echo "== logging + report"
step "log_decision" "$PY" tools/log_decision.py --out $OUT --step other --short "smoke" --why "test" --decision "smoke test" --evidence "n/a"
step "run_summary" "$PY" tools/run_summary.py --out $OUT
printf '# t\n**Verdict:** ok\nIntro\n- a\n- b\n\n| x | n |\n|---|---|\n| + y | 1.0 |\n\n![c](pbmc/clip/embedding.png)\n' > $OUT/smoke_report.md
step "export_report (tidy only)" "$PY" tools/export_report.py $OUT/smoke_report.md --no-pdf

if [ "${1:-}" = "--real" ]; then
  echo "== real data: timing (use these numbers to size budgets)"
  if [ -f data/pbmc_multiome/dataset.json ]; then
    D=data/pbmc_multiome; O=$OUT/real_pbmc
    /usr/bin/time -p "$PY" tools/preprocess.py --data $D --out $O --modality rna --spec log1p_hvg2000_pca50 2>&1 | grep -E "^rna|^real"
    /usr/bin/time -p "$PY" tools/preprocess.py --data $D --out $O --modality atac --spec tfidf_lsi50_drop1 2>&1 | grep -E "^atac|drop|^real"
    "$PY" tools/train.py --data $D --out $O/clip_probe --inputs rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 \
      --method clip --dim 32 --epochs 5 --eval-every 5 --batch-size 512 2>&1 | grep -E "epoch|done"
  fi
  if [ -f data/rocov2/dataset.json ]; then
    D=data/rocov2; O=$OUT/real_rocov2
    step "BiomedCLIP text (real rocov2; cached after first run)" "$PY" tools/preprocess.py --data $D --out $O --modality text --spec biomedclip
    echo "  BiomedCLIP image encoding of all images is NOT run here (can take a while);"
    echo "  the agent runs it, or start it yourself:"
    echo "    $PY tools/preprocess.py --data $D --out $O --modality image --spec biomedclip"
  fi
fi

echo
echo "== $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ] || { printf '   failed: %s\n' "${FAILED[@]}"; exit 1; }

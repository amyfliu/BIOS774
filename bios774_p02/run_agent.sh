#!/usr/bin/env bash
# run_agent.sh — the single command that runs the generative-modeling agent.
#
#   ./run_agent.sh <dataset> "<objective>" [report_name] [budget_minutes]
#
#   dataset        folder name under data/ (must contain prepared.npz + DATA.md)
#   objective      the user's goal, in plain language
#   report_name    default: generated_report_<dataset>   (e.g. generated_report_1)
#                  may include a subfolder, e.g. appendix/generated_report_A
#   budget_minutes total training-time budget for the agent (default 30)
#
# Example:
#   ./run_agent.sh pbmc3k "Rare cell types have very few cells; generate synthetic \
#       cells to rebalance the classes for a cell-type classifier" generated_report_1 20
#
# The agent (Claude Code, headless) reads AGENTS.md + .agents/skills/, then
# inspects, models, evaluates and writes reports/<report_name>.md (+ .pdf).
# Everything it produces goes to outputs/<run_id>/; the full transcript is
# saved there as agent_transcript.jsonl.
#
# Environment overrides: PYTHON (interpreter), CLAUDE_MODEL (e.g. sonnet, opus).
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 2 ]; then
  sed -n '2,20p' "$0"; exit 1
fi
DATASET="$1"; OBJECTIVE="$2"
REPORT="${3:-generated_report_${DATASET}}"
BUDGET="${4:-30}"
RUN_ID="$(basename "$REPORT")_${DATASET}"
OUT="outputs/${RUN_ID}"
PY="${PYTHON:-$(command -v python3 || command -v python)}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

[ -f "data/${DATASET}/prepared.npz" ] || { echo "data/${DATASET}/prepared.npz missing — run: cd data/${DATASET} && $PY prepare.py"; exit 1; }
command -v claude >/dev/null || { echo "Claude Code CLI ('claude') not found on PATH"; exit 1; }
# A repeated run never mixes with an earlier one: move the earlier attempt aside.
if [ -e "$OUT" ] || [ -e "reports/${REPORT}.md" ]; then
  PREV="outputs/_previous/${RUN_ID}_$(date '+%Y%m%d_%H%M%S')"
  mkdir -p "$PREV"
  [ -e "$OUT" ] && mv "$OUT" "$PREV/outputs"
  for ext in md pdf; do [ -e "reports/${REPORT}.${ext}" ] && mv "reports/${REPORT}.${ext}" "$PREV/"; done
  echo "Previous attempt moved to $PREV"
fi
mkdir -p "$OUT" "reports/$(dirname "$REPORT")"

DEVICE="$("$PY" -c 'import sys; sys.path.insert(0, "tools"); from common import get_device; print(get_device())')"
cat > "$OUT/objective.md" <<EOF
# Run ${RUN_ID}
- dataset: data/${DATASET}
- objective: ${OBJECTIVE}
- report: reports/${REPORT}.md
- training budget: ${BUDGET} minutes
- device: ${DEVICE}
- started: $(date '+%Y-%m-%d %H:%M:%S')
EOF

PROMPT="Follow AGENTS.md exactly. You are running unattended; do not ask questions.
Dataset folder: data/${DATASET}
User objective: \"${OBJECTIVE}\"
Run id: ${RUN_ID}
Output folder (all tool --out/--run paths go under it): ${OUT}
Report to write: reports/${REPORT}.md (and export reports/${REPORT}.pdf)
Total training budget: ${BUDGET} minutes (sum of --max-minutes over all training runs).
Compute device the tools will use: ${DEVICE}.
Python interpreter for every tool call: ${PY}   (e.g. ${PY} tools/inspect_data.py --data data/${DATASET} --out ${OUT})"

MODEL_ARGS=()
[ -n "${CLAUDE_MODEL:-}" ] && MODEL_ARGS=(--model "$CLAUDE_MODEL")

echo "=== agent run ${RUN_ID} | device ${DEVICE} | budget ${BUDGET} min ==="
echo "    transcript: ${OUT}/agent_transcript.jsonl"
# The agent may run shell commands and edit files inside this project without
# asking (it is unattended). It is instructed not to modify tools/.
claude -p "$PROMPT" ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} \
  --output-format stream-json --verbose \
  --permission-mode acceptEdits \
  --allowedTools "Bash,Read,Write,Edit,Glob,Grep" \
  | "$PY" tools/show_progress.py "${OUT}/agent_transcript.jsonl"

# Safety net: make sure the report's layout is tidied and the PDF exists even if
# the agent skipped its export step (running it twice is harmless).
if [ -f "reports/${REPORT}.md" ]; then
  "$PY" tools/export_report.py "reports/${REPORT}.md" || true
else
  echo "WARNING: the agent did not write reports/${REPORT}.md"
fi

echo "Report: reports/${REPORT}.md   Decisions: ${OUT}/decisions.jsonl"

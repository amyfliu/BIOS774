#!/usr/bin/env bash
# run_all.sh — the agent runs used in the project report (one command each).
#   1 : PBMC Multiome (paired scRNA-seq + scATAC-seq)  -> reports/generated_report_1
#   2 : ROCOv2 (radiology images + captions)           -> reports/generated_report_2
# Each line is an independent single-command run; comment out what you don't need.
set -euo pipefail
cd "$(dirname "$0")"

./run_agent.sh pbmc_multiome \
  "Learn a joint RNA+ATAC cell representation that identifies cell types at least as well as either modality alone, and assess how well each modality can be predicted from the other." \
  generated_report_1 20

./run_agent.sh rocov2 \
  "Learn a shared image-text representation for radiology figures that supports retrieving the right caption for an image (and vice versa) and classifying the imaging modality from the image." \
  generated_report_2 20

#!/usr/bin/env bash
# run_all.sh — the four agent runs used in the project report.
#   1, 2 : the two required reports (different datasets AND objectives)
#   A, B : robustness checks on PathMNIST (same data, different objectives)
# Each line is an independent single-command run; comment out what you don't need.
set -euo pipefail
cd "$(dirname "$0")"

./run_agent.sh pbmc3k \
  "Some cell types (for example megakaryocytes and dendritic cells) have very few cells. Generate synthetic cells to rebalance the classes so that a downstream cell-type classifier does better on the rare types." \
  generated_report_1 10

./run_agent.sh pathmnist \
  "We are not allowed to share the real patient image patches. Generate a synthetic, labeled image dataset that could be shared instead, so that others can develop tissue-type classifiers on it. It should look realistic, cover all tissue types, and must not simply copy training images." \
  generated_report_2 40

./run_agent.sh pathmnist \
  "Build a quality-control screen that flags image patches that are not informative tissue (background and debris) as unusual, by learning what typical tissue patches look like." \
  appendix/generated_report_A 25

./run_agent.sh pathmnist \
  "Some tissue types might be under-represented. Generate synthetic images to balance the classes for a downstream tissue classifier." \
  appendix/generated_report_B 15

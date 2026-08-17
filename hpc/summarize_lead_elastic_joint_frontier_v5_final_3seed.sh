#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/path/to/CSDNet}"
PREFIX="${PREFIX:-CSDNet/exp/lead/results_elastic_joint_frontier_v5r_final_base50k_seed}"
SUMMARY_DIR="${SUMMARY_DIR:-CSDNet/exp/lead/results_elastic_joint_frontier_v5r_final_base50k_3seed_summary}"
cd "$ROOT"
mkdir -p "$SUMMARY_DIR"

for SEED in 0 1 2; do
    DIR="${PREFIX}${SEED}"
    python -m CSDNet.exp.lead.aggregate \
        --input_dir "$DIR" \
        --output "$DIR/lead_summary.csv" \
        --planned_total 1000
done

python -m CSDNet.exp.lead.aggregate_multiseed \
    --seed_dir "0=${PREFIX}0" \
    --seed_dir "1=${PREFIX}1" \
    --seed_dir "2=${PREFIX}2" \
    --output_dir "$SUMMARY_DIR" \
    --planned_total 1000 \
    --expected_seeds 0,1,2

python -m CSDNet.exp.lead.invirtuogen_score \
    --input_dir "${PREFIX}0" \
    --input_dir "${PREFIX}1" \
    --input_dir "${PREFIX}2" \
    --planned_total 1000 \
    --expected_random_seeds 3 \
    --run_output "$SUMMARY_DIR/invirtuogen_score_runs.csv" \
    --cell_output "$SUMMARY_DIR/invirtuogen_score_cells.csv"

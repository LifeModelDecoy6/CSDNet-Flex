#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/path/to/CSDNet}"
PREFIX="${PREFIX:-CSDNet/exp/pmo/results/elastic_frontier_prescreen_truefsm_base50k_10k_seed}"
SUMMARY_DIR="${SUMMARY_DIR:-CSDNet/exp/pmo/results/elastic_frontier_prescreen_truefsm_base50k_10k_3seed_summary}"
MODE="elastic_frontier_prescreen"

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=3

ALL_COMPLETE=1
for SEED in 0 1 2; do
    OUT_DIR="${PREFIX}${SEED}"
    echo
    echo "================ PMO TRUEFSM SEED $SEED ================"
    if ! compgen -G "$OUT_DIR/*_${SEED}.csv" >/dev/null; then
        echo "No task histories found: $OUT_DIR"
        ALL_COMPLETE=0
        continue
    fi

    python -m CSDNet.exp.pmo.materialize_pmo_summary \
        --output_dir "$OUT_DIR" \
        --mode "$MODE" \
        --seed "$SEED" \
        --max_oracle_calls 10000
    python -m CSDNet.exp.pmo.summarize_pmo_progress \
        --output_dir "$OUT_DIR" \
        --mode "$MODE" \
        --seed "$SEED" \
        --expected_calls 10000

    SUMMARY="$OUT_DIR/summary_${MODE}.csv"
    COMPLETE=$(awk -F, -v seed="$SEED" \
        'NR > 1 && $1 == "elastic_frontier_prescreen" && $3 == seed && $4 + 0 >= 10000 {n += 1} END {print n + 0}' \
        "$SUMMARY")
    if [ "$COMPLETE" -ne 23 ]; then
        ALL_COMPLETE=0
    fi
done

if [ "$ALL_COMPLETE" -ne 1 ]; then
    echo
    echo "Three-seed aggregation deferred: at least one seed is incomplete."
    echo "Rerun this same command after the remaining jobs finish."
    exit 0
fi

python -m CSDNet.exp.pmo.aggregate_multiseed \
    --seed_dir "0=${PREFIX}0" \
    --seed_dir "1=${PREFIX}1" \
    --seed_dir "2=${PREFIX}2" \
    --output_dir "$SUMMARY_DIR" \
    --mode "$MODE" \
    --expected_calls 10000 \
    --expected_seeds 0,1,2


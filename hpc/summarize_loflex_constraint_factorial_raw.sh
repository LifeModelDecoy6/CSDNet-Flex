#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/path/to/CSDNet}"
BASE_DIR="${BASE_DIR:-results/csdnet_6m_loflex_base50k_constraint_factorial_raw}"
CKPT_PATH="${CKPT_PATH:-csdnet_checkpoints_6m_loflex_geometric_genmol50k/last.ckpt}"
OUTPUT="${OUTPUT:-$BASE_DIR/constraint_factorial_3seed.csv}"

cd "$ROOT"

for SPEC in \
    "none:elastic_loflex_constraint_none" \
    "fsm_only:elastic_loflex_constraint_fsm_only" \
    "rdkit_only:elastic_loflex_constraint_rdkit_only" \
    "full:elastic_loflex_constraint_full"; do
    MODE="${SPEC%%:*}"
    PROFILE="${SPEC#*:}"
    python -m CSDNet.exp.denovo.aggregate_promax \
        --input_dir "$BASE_DIR/$MODE" \
        --steps 1000 \
        --profiles "$PROFILE" \
        --seeds 0,1,2 \
        --output "$BASE_DIR/$MODE/${MODE}_raw_3seed.csv"
done

python -m CSDNet.exp.denovo.compare_constraint_factorial \
    --base_dir "$BASE_DIR" \
    --steps 1000 \
    --seeds 0,1,2 \
    --checkpoint "$CKPT_PATH" \
    --output "$OUTPUT"

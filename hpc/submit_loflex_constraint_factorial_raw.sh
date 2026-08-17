#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/path/to/CSDNet}"
PBS_FILE="hpc/run_loflex_base_denovo_constraint_factorial_3seed_1gpu.pbs"

cd "$ROOT"
if [ ! -s "$PBS_FILE" ]; then
    echo "Missing PBS file: $PBS_FILE"
    exit 2
fi
mkdir -p logs

for MODE in none fsm_only rdkit_only full; do
    JOB=$(qsub \
        -N "b6c2_${MODE}" \
        -o "$ROOT/logs/base6m_constraint_2x2_${MODE}_out.log" \
        -e "$ROOT/logs/base6m_constraint_2x2_${MODE}_err.log" \
        -v "CONSTRAINT_MODE=$MODE" \
        "$PBS_FILE")
    printf '%-12s %s\n' "$MODE" "$JOB"
done

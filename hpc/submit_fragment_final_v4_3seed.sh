#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/path/to/CSDNet}"
PBS_FILE="hpc/run_fragment_final_v4_task_1gpu.pbs"
CKPT_PATH="${CKPT_PATH:-csdnet_checkpoints_6m_loflex_geometric_genmol50k/last.ckpt}"
PRIOR="${PRIOR:-data/zinc250k_fragment_gap_prior_atom256.json}"
OUT_DIR="${OUT_DIR:-CSDNet/exp/frag/results/native_final_v4_base50k_3seeds}"
TASKS=(linker_design scaffold_morphing motif_extension scaffold_decoration superstructure_generation)
WALLTIMES=(06:00:00 06:00:00 05:00:00 05:00:00 05:00:00)

cd "$ROOT"
for REQUIRED in "$PBS_FILE" "$CKPT_PATH" "$PRIOR" csdnet_vocab.pkl data/fragments.csv; do
    if [ ! -s "$REQUIRED" ]; then
        echo "Missing or empty required file: $REQUIRED"
        exit 2
    fi
done
mkdir -p logs "$OUT_DIR"
MANIFEST="logs/fragment_final_v4_$(date +%Y%m%d_%H%M%S).tsv"
printf 'job_id\tseed\ttask_index\ttask\twalltime\toutput\n' > "$MANIFEST"

for SEED in 0 1 2; do
    for TASK_INDEX in 0 1 2 3 4; do
        TASK="${TASKS[$TASK_INDEX]}"
        JOB_NAME="ff4_s${SEED}t${TASK_INDEX}"
        JOB_ID=$(qsub \
            -N "$JOB_NAME" \
            -l "walltime=${WALLTIMES[$TASK_INDEX]}" \
            -o "$ROOT/logs/${JOB_NAME}_out.log" \
            -e "$ROOT/logs/${JOB_NAME}_err.log" \
            -v "ROOT=$ROOT,TASK_INDEX=$TASK_INDEX,SEED=$SEED,CKPT_PATH=$CKPT_PATH,FRAGMENT_LENGTH_PRIOR=$PRIOR,OUT_DIR=$OUT_DIR" \
            "$PBS_FILE")
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$JOB_ID" "$SEED" "$TASK_INDEX" "$TASK" \
            "${WALLTIMES[$TASK_INDEX]}" "$OUT_DIR" | tee -a "$MANIFEST"
    done
done

echo "Submitted 15 final-v4 fragment jobs."
echo "Output:   $OUT_DIR"
echo "Manifest: $MANIFEST"

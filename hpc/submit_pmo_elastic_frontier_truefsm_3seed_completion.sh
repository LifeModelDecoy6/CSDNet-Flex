#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/path/to/CSDNet}"
SEEDS="${SEEDS:-0,1,2}"
PBS_FILE="hpc/run_elastic_frontier_prescreen_true_fsm_task_1gpu.pbs"
CKPT_PATH="${CKPT_PATH:-csdnet_checkpoints_6m_loflex_geometric_genmol50k/last.ckpt}"
ATOMIC_LENGTH_PRIOR="${ATOMIC_LENGTH_PRIOR:-data/zinc250k_csdnet_atomic_lengths_max256.json}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-CSDNet/exp/pmo/results/elastic_frontier_prescreen_truefsm_base50k_10k_seed}"
DRY_RUN="${DRY_RUN:-0}"

TASKS=(
    albuterol_similarity amlodipine_mpo celecoxib_rediscovery deco_hop drd2
    fexofenadine_mpo gsk3b isomers_c7h8n2o2 isomers_c9h10n2o2pf2cl jnk3
    median1 median2 mestranol_similarity osimertinib_mpo perindopril_mpo qed
    ranolazine_mpo scaffold_hop sitagliptin_mpo thiothixene_rediscovery
    troglitazone_rediscovery valsartan_smarts zaleplon_mpo
)

# Timings include headroom over the observed seed-0 run. The three tasks that
# previously hit two hours are deliberately allocated four hours.
WALLTIMES=(
    03:00:00 03:00:00 04:00:00 03:00:00 03:00:00 03:00:00
    05:00:00 04:00:00 03:00:00 04:00:00 03:00:00 04:00:00
    05:00:00 03:00:00 03:00:00 03:00:00 03:00:00 03:00:00
    03:00:00 03:00:00 04:00:00 03:00:00 03:00:00
)

cd "$ROOT"
for REQUIRED in "$PBS_FILE" "$CKPT_PATH" "$ATOMIC_LENGTH_PRIOR" csdnet_vocab.pkl; do
    if [ ! -s "$REQUIRED" ]; then
        echo "Missing or empty required file: $REQUIRED"
        exit 2
    fi
done
for ORACLE in "${TASKS[@]}"; do
    PRIOR_FILE="CSDNet/exp/pmo/vocab/${ORACLE}.csv"
    if [ ! -s "$PRIOR_FILE" ]; then
        echo "Missing oracle-specific ZINC prior: $PRIOR_FILE"
        exit 2
    fi
done

mkdir -p logs
MANIFEST="logs/pmo_truefsm_3seed_completion_$(date +%Y%m%d_%H%M%S).tsv"
printf 'job_id\tseed\ttask_index\toracle\tstarting_calls\twalltime\toutput\n' > "$MANIFEST"

SUBMITTED=0
SKIPPED=0
for SEED in ${SEEDS//,/ }; do
    if ! [[ "$SEED" =~ ^[0-2]$ ]]; then
        echo "Invalid seed in SEEDS=$SEEDS: $SEED"
        exit 2
    fi
    OUT_DIR="${OUTPUT_PREFIX}${SEED}"
    mkdir -p "$OUT_DIR"

    for TASK_INDEX in $(seq 0 22); do
        ORACLE="${TASKS[$TASK_INDEX]}"
        SCORE_FILE="$OUT_DIR/${ORACLE}_${SEED}.csv"
        STARTING_CALLS=0
        if [ -f "$SCORE_FILE" ]; then
            STARTING_CALLS=$(awk 'NF {n += 1} END {print n + 0}' "$SCORE_FILE")
        fi
        if [ "$STARTING_CALLS" -ge 10000 ]; then
            echo "skip complete seed=$SEED task=$TASK_INDEX oracle=$ORACLE calls=$STARTING_CALLS"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        JOB_NAME="pet_s${SEED}t${TASK_INDEX}"
        QSUB_ARGS=(
            -N "$JOB_NAME"
            -l "walltime=${WALLTIMES[$TASK_INDEX]}"
            -o "$ROOT/logs/${JOB_NAME}_out.log"
            -e "$ROOT/logs/${JOB_NAME}_err.log"
            -v "ROOT=$ROOT,SEED=$SEED,TASK_INDEX=$TASK_INDEX,CKPT_PATH=$CKPT_PATH,ATOMIC_LENGTH_PRIOR=$ATOMIC_LENGTH_PRIOR,OUT_DIR=$OUT_DIR,RUN_TAG=truefsm_complete_s${SEED},SKIP_DONE=1,MAX_CALLS=10000,MAX_LEN=256,V5_MOTIF_POOL_SIZE=240,V9_PRIOR_POPULATION_SIZE=100,V9_PRESCREEN_ACTIVE_MOTIF_SIZE=240,V9_ROOT_OVERGENERATE_FACTOR=2.0,LEARNED_INSERTION_FRACTION_SCALE=1.0"
            "$PBS_FILE"
        )
        if [ "$DRY_RUN" = "1" ]; then
            JOB_ID="DRY_RUN"
            printf 'qsub'
            printf ' %q' "${QSUB_ARGS[@]}"
            printf '\n'
        else
            JOB_ID=$(qsub "${QSUB_ARGS[@]}")
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$JOB_ID" "$SEED" "$TASK_INDEX" "$ORACLE" "$STARTING_CALLS" \
            "${WALLTIMES[$TASK_INDEX]}" "$OUT_DIR" | tee -a "$MANIFEST"
        SUBMITTED=$((SUBMITTED + 1))
    done
done

echo "Submitted: $SUBMITTED; skipped complete: $SKIPPED"
echo "Protocol: elastic_frontier_prescreen, top-100 oracle-specific prior"
echo "Checkpoint: $CKPT_PATH"
echo "Manifest: $MANIFEST"


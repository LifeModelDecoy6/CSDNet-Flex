#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/path/to/CSDNet}"
SEEDS="${SEEDS:-0,1,2}"
SAMPLER_PROFILE="elastic_joint_frontier_v5"
CKPT_PATH="${CKPT_PATH:-csdnet_checkpoints_6m_loflex_geometric_genmol50k/last.ckpt}"
ATOMIC_LENGTH_PRIOR="${ATOMIC_LENGTH_PRIOR:-data/zinc250k_csdnet_atomic_lengths_max256.json}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-CSDNet/exp/lead/results_elastic_joint_frontier_v5r_final_base50k_seed}"
PBS_FILE="hpc/run_lead_elastic_joint_frontier_v5_feasible_chunk_1gpu.pbs"

cd "$ROOT"
for REQUIRED in "$PBS_FILE" "$CKPT_PATH" "$ATOMIC_LENGTH_PRIOR" csdnet_vocab.pkl; do
    if [ ! -s "$REQUIRED" ]; then
        echo "Missing or empty required file: $REQUIRED"
        exit 2
    fi
done
grep -Fq '"elastic_joint_frontier_v5"' CSDNet/exp/lead/run.py || {
    echo "Final Lead v5 profile is missing"
    exit 2
}
grep -Fq '"--oracle_feasible_only"' CSDNet/exp/lead/run.py || {
    echo "Final Lead oracle feasibility gate is unavailable"
    exit 2
}
TASK_COUNT=$(grep -Eo \
    '"(parp1|fa7|5ht1b|braf|jak2):[012]:0\.[46]"' \
    "$PBS_FILE" | sort -u | wc -l | tr -d ' ')
if [ "$TASK_COUNT" -ne 30 ]; then
    echo "Lead v5 chunk map must contain all 30 unique cells; found $TASK_COUNT"
    exit 2
fi

mkdir -p logs
MANIFEST="logs/lead_v5r_final_$(date +%Y%m%d_%H%M%S).tsv"
printf 'job_id\tseed\tchunk\ttasks\twalltime\toutput\n' > "$MANIFEST"

for SEED in ${SEEDS//,/ }; do
    if ! [[ "$SEED" =~ ^[0-2]$ ]]; then
        echo "Invalid seed in SEEDS=$SEEDS: $SEED"
        exit 2
    fi
    OUT_DIR="${OUTPUT_PREFIX}${SEED}"
    RUN_TAG="elastic_joint_v5r_final_base50k_s${SEED}"
    mkdir -p "$OUT_DIR"
    for CHUNK in 0 1 2 3 4 5; do
        case "$CHUNK" in
            0)
                TASK_LABEL="delta0.4_all15"
                WALLTIME="12:00:00"
                MAX_ROUNDS=4
                OVERGENERATE=2.0
                MAX_PROPOSALS=400
                ;;
            1)
                TASK_LABEL="delta0.6_fa7_id1_slow"
                WALLTIME="06:00:00"
                MAX_ROUNDS=4
                OVERGENERATE=3.0
                MAX_PROPOSALS=600
                ;;
            2)
                TASK_LABEL="delta0.6_5ht1b_id0_slow"
                WALLTIME="06:00:00"
                MAX_ROUNDS=4
                OVERGENERATE=3.0
                MAX_PROPOSALS=600
                ;;
            3)
                TASK_LABEL="delta0.6_fa7_id0_id2"
                WALLTIME="08:00:00"
                MAX_ROUNDS=4
                OVERGENERATE=3.0
                MAX_PROPOSALS=600
                ;;
            4)
                TASK_LABEL="delta0.6_parp1_5ht1b"
                WALLTIME="10:00:00"
                MAX_ROUNDS=4
                OVERGENERATE=3.0
                MAX_PROPOSALS=600
                ;;
            5)
                TASK_LABEL="delta0.6_braf_jak2"
                WALLTIME="10:00:00"
                MAX_ROUNDS=4
                OVERGENERATE=3.0
                MAX_PROPOSALS=600
                ;;
        esac
        JOB_NAME="ej5r_s${SEED}c${CHUNK}"
        JOB_ID=$(qsub \
            -N "$JOB_NAME" \
            -l "walltime=$WALLTIME" \
            -o "$ROOT/logs/${JOB_NAME}_out.log" \
            -e "$ROOT/logs/${JOB_NAME}_err.log" \
            -v "ROOT=$ROOT,SEED=$SEED,CHUNK=$CHUNK,SAMPLER_PROFILE=$SAMPLER_PROFILE,CKPT_PATH=$CKPT_PATH,ATOMIC_LENGTH_PRIOR=$ATOMIC_LENGTH_PRIOR,OUT_DIR=$OUT_DIR,RUN_TAG=$RUN_TAG,SKIP_DONE=1,RESUME_PARTIAL=1,REQUIRE_RESULT=1,NUM_ITER=10,ORACLE_BUDGET=1000,MIN_ORACLE_CALLS=0,MAX_GENERATION_ROUNDS=$MAX_ROUNDS,MAX_PROPOSAL_INPUTS_PER_ITERATION=$MAX_PROPOSALS,DIRECT_OVERGENERATE_FACTOR=$OVERGENERATE" \
            "$PBS_FILE")
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$JOB_ID" "$SEED" "$CHUNK" "$TASK_LABEL" "$WALLTIME" "$OUT_DIR" \
            | tee -a "$MANIFEST"
    done
done

echo "Submitted 18 repaired Lead v5 jobs for seeds: $SEEDS"
echo "Protocol: 10 bounded proposal iterations; feasible-only docking; 1000 calls is a hard upper bound, not a quota."
echo "Manifest: $MANIFEST"

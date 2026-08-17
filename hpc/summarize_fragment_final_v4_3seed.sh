#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/path/to/CSDNet}"
OUT_DIR="${OUT_DIR:-CSDNet/exp/frag/results/native_final_v4_base50k_3seeds}"
cd "$ROOT"

python -m CSDNet.exp.frag.aggregate_frontier \
    --input_dir "$OUT_DIR" \
    --seeds 0,1,2 \
    --output_prefix "$OUT_DIR/native_final_v4_base50k_3seed"
python -m CSDNet.exp.frag.report_insertion_trajectories \
    --input_dir "$OUT_DIR"

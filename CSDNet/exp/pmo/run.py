#!/usr/bin/env python
import argparse
import os
from time import time

import yaml

from CSDNet.exp.pmo.optimizer import CSDNetOptimizer, PMO_TASKS, load_csdnet_model


def result_path(output_dir, mode, oracle_name, seed):
    return os.path.join(
        output_dir,
        f"results_CSDNet_{mode}_{oracle_name}_{seed}.yaml",
    )


def completed(output_dir, mode, oracle_name, seed, max_oracle_calls):
    path = result_path(output_dir, mode, oracle_name, seed)
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return False
    return len(data) >= max_oracle_calls


def load_resume_buffer(output_dir, mode, oracle_name, seed, max_oracle_calls):
    """Load and validate an incomplete oracle buffer saved by BaseOptimizer."""
    path = result_path(output_dir, mode, oracle_name, seed)
    score_path = os.path.join(output_dir, f"{oracle_name}_{seed}.csv")
    yaml_rows = []
    if os.path.exists(path):
        with open(path) as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise RuntimeError(f"Resume checkpoint is not a mapping: {path}")
        seen_calls = set()
        for smiles, value in data.items():
            if not isinstance(smiles, str) or not isinstance(value, (list, tuple)) or len(value) < 2:
                raise RuntimeError(f"Malformed resume entry in {path}: {smiles!r}: {value!r}")
            score = float(value[0])
            call_index = int(value[1])
            if call_index < 1 or call_index > max_oracle_calls:
                raise RuntimeError(f"Invalid call index {call_index} in {path}")
            if call_index in seen_calls:
                raise RuntimeError(f"Duplicate call index {call_index} in {path}")
            seen_calls.add(call_index)
            yaml_rows.append((call_index, smiles, score))
        yaml_rows.sort()

    csv_rows = []
    if os.path.exists(score_path):
        seen_smiles = set()
        with open(score_path) as handle:
            for line_number, raw in enumerate(handle, start=1):
                raw = raw.rstrip("\n")
                if not raw:
                    continue
                try:
                    smiles, score_text = raw.rsplit(",", 1)
                    score = float(score_text)
                except Exception as exc:
                    raise RuntimeError(
                        f"Malformed score row {line_number} in {score_path}"
                    ) from exc
                if smiles in seen_smiles:
                    raise RuntimeError(
                        f"Duplicate molecule at row {line_number} in {score_path}"
                    )
                seen_smiles.add(smiles)
                csv_rows.append((len(csv_rows) + 1, smiles, score))
                if len(csv_rows) >= max_oracle_calls:
                    break

    rows = csv_rows if len(csv_rows) > len(yaml_rows) else yaml_rows
    if yaml_rows and csv_rows:
        shared = min(len(yaml_rows), len(csv_rows))
        for yaml_row, csv_row in zip(yaml_rows[:shared], csv_rows[:shared]):
            if yaml_row[1] != csv_row[1] or abs(yaml_row[2] - csv_row[2]) > 1e-8:
                raise RuntimeError(
                    f"YAML and score CSV histories diverge at call {yaml_row[0]} "
                    f"for {oracle_name}; refusing an unsafe resume."
                )

    expected = list(range(1, len(rows) + 1))
    observed = [call_index for call_index, _, _ in rows]
    if observed != expected:
        raise RuntimeError(
            f"Non-contiguous oracle call history for {oracle_name}: "
            f"expected 1..{len(rows)}, got {observed[:3]}...{observed[-3:]}"
        )
    return {
        smiles: [score, call_index]
        for call_index, smiles, score in rows
    }


def clear_incomplete_outputs(output_dir, mode, oracle_name, seed):
    paths = [
        os.path.join(output_dir, f"{oracle_name}_{seed}.csv"),
        os.path.join(output_dir, f"results_CSDNet_{mode}_{oracle_name}_{seed}.yaml"),
        os.path.join(output_dir, f"diagnostics_{mode}_{oracle_name}_{seed}.csv"),
        os.path.join(output_dir, f"transitions_{mode}_{oracle_name}_{seed}.csv"),
        os.path.join(output_dir, f"frontier_state_{mode}_{oracle_name}_{seed}.json"),
    ]
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "motif_seeded",
            "iterative_remask",
            "iterative_remask_v2",
            "iterative_remask_v3",
            "iterative_remask_v4",
            "iterative_remask_v5",
            "iterative_remask_v6",
            "iterative_remask_v7",
            "iterative_remask_v8",
            "iterative_remask_v9",
            "iterative_remask_v9_no_prescreen",
            "iterative_remask_v10",
            "elastic_direct",
            "elastic_frontier",
            "elastic_frontier_prescreen",
            "elastic_frontier_prescreen_v2",
            "safe_frontier_final",
            "iterative_remask_v9_gated",
            "iterative_remask_v9_reversible",
            "unified_frontier",
            "unified_frontier_v2",
            "unified_frontier_restored",
        ],
        required=True,
    )
    parser.add_argument("-o", "--oracle", default="all",
                        choices=["all", *PMO_TASKS])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip oracle tasks whose YAML result already has max_oracle_calls entries.")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Log failed oracle tasks and continue with the remaining tasks.")
    parser.add_argument("--max_oracle_calls", type=int, default=10000)
    parser.add_argument("--freq_log", type=int, default=100)
    parser.add_argument("-s", "--seed", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=-1)

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--vocab", type=str, default="csdnet_vocab.pkl")
    parser.add_argument("--data_dir", type=str, default="csdnet_data/pubchem_10m_with_props_v2")
    parser.add_argument("--ref_sample_n", type=int, default=50000)
    parser.add_argument(
        "--atomic_length_prior",
        type=str,
        default=None,
        help="Validated CSDNet atomic-token length prior for global PMO restarts.",
    )
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--candidate_batch_size", type=int, default=128)
    parser.add_argument("--n_steps", type=int, default=250)
    parser.add_argument("--population_size", type=int, default=100)
    parser.add_argument("--elite_size", type=int, default=100)
    parser.add_argument("--elite_seed_prob", type=float, default=0.5)

    parser.add_argument("--motif_min_atoms", type=int, default=4)
    parser.add_argument("--motif_max_atoms", type=int, default=36)
    parser.add_argument("--remask_fraction", type=float, default=0.35)
    parser.add_argument("--min_remask_tokens", type=int, default=2)
    parser.add_argument("--span_prob", type=float, default=0.7)
    parser.add_argument("--length_delta_choices", type=str, default="0")
    parser.add_argument("--length_edit_prob", type=float, default=0.0)
    parser.add_argument("--length_edit_min_span", type=int, default=1)
    parser.add_argument("--length_edit_max_span", type=int, default=8)
    parser.add_argument("--disable_learned_insertion", action="store_true")
    parser.add_argument("--learned_insertion_fraction_scale", type=float, default=1.0)
    parser.add_argument("--learned_insertion_max_growth", type=int, default=4)
    parser.add_argument("--learned_insertion_max_shrink", type=int, default=4)
    parser.add_argument("--learned_insertion_max_per_step", type=int, default=4)

    parser.add_argument("--disable_fsm_check", action="store_true")
    parser.add_argument("--disable_rdkit_kekulize_check", action="store_true")
    parser.add_argument("--rdkit_check_interval", type=int, default=25)
    parser.add_argument("--max_sample_retries", type=int, default=2)
    parser.add_argument("--violation_neighborhood", type=int, default=2)
    parser.add_argument("--temperature_start", type=float, default=1.2)
    parser.add_argument("--temperature_end", type=float, default=0.2)
    parser.add_argument("--temperature_power", type=float, default=1.5)
    parser.add_argument("--v2_remask_fractions", type=str, default="0.15,0.30,0.50")
    parser.add_argument("--v2_rescue_patience", type=int, default=1200)
    parser.add_argument("--v2_diversity_weight", type=float, default=0.20)
    parser.add_argument("--v2_restart_prob", type=float, default=0.20)
    parser.add_argument("--v2_near_duplicate_sim", type=float, default=0.985)
    parser.add_argument("--v2_score_elite_fraction", type=float, default=0.35)
    parser.add_argument("--v3_remask_fractions", type=str, default="0.10,0.25,0.45,0.65")
    parser.add_argument("--v3_zero_rescue_patience", type=int, default=1500)
    parser.add_argument("--v3_stagnation_rescue_patience", type=int, default=1800)
    parser.add_argument("--v3_nonzero_threshold", type=float, default=1e-8)
    parser.add_argument("--v3_restart_prob", type=float, default=0.35)
    parser.add_argument("--v3_rescue_temperature", type=float, default=1.6)
    parser.add_argument("--v3_near_duplicate_sim", type=float, default=0.990)
    parser.add_argument("--v3_recent_memory", type=int, default=3000)
    parser.add_argument("--v4_remask_fractions", type=str, default="0.08,0.18,0.32,0.55")
    parser.add_argument("--v4_zero_rescue_patience", type=int, default=250)
    parser.add_argument("--v4_stagnation_rescue_patience", type=int, default=350)
    parser.add_argument("--v4_nonzero_threshold", type=float, default=1e-8)
    parser.add_argument("--v4_rescue_temperature", type=float, default=1.65)
    parser.add_argument("--v4_near_duplicate_sim", type=float, default=0.995)
    parser.add_argument("--v4_recent_memory", type=int, default=1500)
    parser.add_argument("--v4_overgenerate_factor", type=float, default=1.5)
    parser.add_argument("--v4_motif_pool_size", type=int, default=220)
    parser.add_argument("--v4_diverse_size", type=int, default=120)
    parser.add_argument("--v4_score_elite_fraction", type=float, default=0.35)
    parser.add_argument("--v4_diversity_weight", type=float, default=0.25)
    parser.add_argument("--v5_remask_fractions", type=str, default="0.06,0.14,0.28,0.50")
    parser.add_argument("--v5_warmup_calls", type=int, default=160)
    parser.add_argument("--v5_sparse_nonzero_rate", type=float, default=0.35)
    parser.add_argument("--v5_good_top1_threshold", type=float, default=0.65)
    parser.add_argument("--v5_high_top10_threshold", type=float, default=0.78)
    parser.add_argument("--v5_low_top1_threshold", type=float, default=0.45)
    parser.add_argument("--v5_late_gap_threshold", type=float, default=0.12)
    parser.add_argument("--v5_stagnation_rescue_patience", type=int, default=260)
    parser.add_argument("--v5_nonzero_threshold", type=float, default=1e-8)
    parser.add_argument("--v5_rescue_temperature", type=float, default=1.65)
    parser.add_argument("--v5_near_duplicate_sim", type=float, default=0.995)
    parser.add_argument("--v5_recent_memory", type=int, default=1500)
    parser.add_argument("--v5_overgenerate_factor", type=float, default=1.5)
    parser.add_argument("--v5_motif_pool_size", type=int, default=240)
    parser.add_argument("--v5_diverse_size", type=int, default=120)
    parser.add_argument("--v5_score_elite_fraction", type=float, default=0.35)
    parser.add_argument("--v5_diversity_weight", type=float, default=0.25)
    parser.add_argument("--v6_bandit_alpha", type=float, default=0.25)
    parser.add_argument("--v6_bandit_temperature", type=float, default=2.0)
    parser.add_argument("--v6_ucb_weight", type=float, default=0.35)
    parser.add_argument("--v6_min_operator_weight", type=float, default=0.03)
    parser.add_argument("--v6_reward_topk_weight", type=float, default=0.45)
    parser.add_argument("--v6_reward_best_weight", type=float, default=0.25)
    parser.add_argument("--v6_reward_nonzero_weight", type=float, default=0.20)
    parser.add_argument("--v6_reward_gain_weight", type=float, default=0.10)
    parser.add_argument("--v7_bandit_alpha", type=float, default=0.25)
    parser.add_argument("--v7_bandit_temperature", type=float, default=2.2)
    parser.add_argument("--v7_ucb_weight", type=float, default=0.35)
    parser.add_argument("--v7_min_operator_weight", type=float, default=0.03)
    parser.add_argument("--v7_reward_delta_top10_weight", type=float, default=0.38)
    parser.add_argument("--v7_reward_delta_auc_weight", type=float, default=0.22)
    parser.add_argument("--v7_reward_top10_entry_weight", type=float, default=0.22)
    parser.add_argument("--v7_reward_best_gain_weight", type=float, default=0.10)
    parser.add_argument("--v7_reward_nonzero_weight", type=float, default=0.08)
    parser.add_argument("--v7_length_rescue_after_stagnant", type=int, default=260)
    parser.add_argument("--v7_length_rescue_weight", type=float, default=0.08)
    parser.add_argument("--v7_length_edit_prob", type=float, default=0.75)
    parser.add_argument("--v7_length_shrink_deltas", type=str, default="-6:-4:-2:0")
    parser.add_argument("--v7_length_expand_deltas", type=str, default="0:1:2:4")
    parser.add_argument("--v7_length_edit_min_span", type=int, default=2)
    parser.add_argument("--v7_length_edit_max_span", type=int, default=10)
    parser.add_argument("--v8_bandit_alpha", type=float, default=0.08)
    parser.add_argument("--v8_bandit_temperature", type=float, default=2.0)
    parser.add_argument("--v8_ucb_weight", type=float, default=0.40)
    parser.add_argument("--v8_min_operator_weight", type=float, default=0.04)
    parser.add_argument("--v8_positive_reward_threshold", type=float, default=0.60)
    parser.add_argument("--v8_reward_delta_weight", type=float, default=0.45)
    parser.add_argument("--v8_reward_frontier_weight", type=float, default=0.25)
    parser.add_argument("--v8_reward_top10_entry_weight", type=float, default=0.20)
    parser.add_argument("--v8_reward_percentile_weight", type=float, default=0.10)
    parser.add_argument("--v8_delta_scale", type=float, default=0.08)
    parser.add_argument("--v8_frontier_gain_scale", type=float, default=0.02)
    parser.add_argument("--v8_overgenerate_factor", type=float, default=1.5)
    parser.add_argument("--v8_near_duplicate_sim", type=float, default=0.997)
    parser.add_argument("--v8_absolute_min_atoms", type=int, default=2)
    parser.add_argument("--v8_size_probe_samples", type=int, default=600)
    parser.add_argument("--v8_size_low_quantile", type=float, default=0.02)
    parser.add_argument("--v8_size_high_quantile", type=float, default=0.98)
    parser.add_argument("--v8_size_margin", type=int, default=6)
    parser.add_argument("--v8_graph_cut_rounds", type=int, default=6)
    parser.add_argument("--v8_graph_edit_attempts", type=int, default=120)
    parser.add_argument("--v8_credit_alpha", type=float, default=0.15)
    parser.add_argument("--v9_prior_population_size", type=int, default=100)
    parser.add_argument(
        "--v9_prescreen_active_motif_size",
        type=int,
        default=240,
        help=(
            "Per-round motif subset used by elastic_frontier_prescreen_v2. "
            "The full ranked prior can be much deeper without diluting the "
            "active proposal distribution."
        ),
    )
    parser.add_argument(
        "--v9_online_bootstrap_calls",
        type=int,
        default=256,
        help=(
            "Budgeted unconditional oracle calls used to learn the initial fragment "
            "prior in iterative_remask_v9_no_prescreen."
        ),
    )
    parser.add_argument("--v9_warmup_calls", type=int, default=1000)
    parser.add_argument("--v9_score_archive_size", type=int, default=84)
    parser.add_argument("--v9_lineage_archive_size", type=int, default=36)
    parser.add_argument("--v9_lineage_ucb_weight", type=float, default=0.04)
    parser.add_argument("--v9_bandit_alpha", type=float, default=0.20)
    parser.add_argument("--v9_bandit_temperature", type=float, default=2.0)
    parser.add_argument("--v9_ucb_weight", type=float, default=0.30)
    parser.add_argument("--v9_min_operator_multiplier", type=float, default=0.05)
    parser.add_argument("--v9_root_bandit_alpha", type=float, default=0.15)
    parser.add_argument("--v9_root_bandit_temperature", type=float, default=1.5)
    parser.add_argument("--v9_root_ucb_weight", type=float, default=0.30)
    parser.add_argument("--v9_root_overgenerate_factor", type=float, default=2.0)
    parser.add_argument("--v9_anchor_remask_fraction", type=float, default=0.10)
    parser.add_argument("--v9_saturation_top10", type=float, default=0.90)
    parser.add_argument("--v9_sparse_nonzero_rate", type=float, default=0.25)
    parser.add_argument("--v9_stagnation_patience", type=int, default=800)
    parser.add_argument("--v9_lineage_collapse_threshold", type=float, default=0.60)
    parser.add_argument("--v9_empty_round_fallback", type=int, default=2)
    parser.add_argument("--v9_max_empty_rounds", type=int, default=8)
    parser.add_argument("--v9_max_global_rescues", type=int, default=6)
    parser.add_argument("--v9_frontier_min_scale", type=float, default=0.01)
    parser.add_argument("--v9_delta_min_scale", type=float, default=0.03)
    parser.add_argument("--v9_delayed_credit_alpha", type=float, default=0.05)
    parser.add_argument("--v9_archive_percentile", type=float, default=0.90)
    parser.add_argument("--v9_gate_probe_fraction", type=float, default=0.04)
    parser.add_argument("--v9_gate_max_fraction", type=float, default=0.12)
    parser.add_argument("--v9_gate_window_calls", type=int, default=500)
    parser.add_argument("--v9_gate_positive_margin", type=float, default=0.02)
    parser.add_argument("--v9_gate_negative_margin", type=float, default=0.01)
    parser.add_argument("--v9_gate_reprobe_calls", type=int, default=1000)
    parser.add_argument("--v9_gate_promotion_windows", type=int, default=2)
    parser.add_argument("--v9_gate_neutral_patience", type=int, default=2)
    parser.add_argument("--v9_gate_confidence_z", type=float, default=1.2816)
    parser.add_argument("--v9_gate_frontier_margin", type=float, default=0.05)
    parser.add_argument("--v9_gate_entry_tolerance", type=float, default=0.002)
    parser.add_argument("--v9_gate_min_target_calls", type=int, default=40)
    parser.add_argument("--v9_gate_min_reference_calls", type=int, default=160)
    parser.add_argument("--v9_gate_state_reprobe_calls", type=int, default=250)
    parser.add_argument("--v10_warmup_calls", type=int, default=1000)
    parser.add_argument("--v10_overgenerate_factor", type=float, default=4.0)
    parser.add_argument("--v10_surrogate_start_calls", type=int, default=1000)
    parser.add_argument("--v10_surrogate_history", type=int, default=2000)
    parser.add_argument("--v10_knn_k", type=int, default=16)
    parser.add_argument("--v10_ucb_beta_start", type=float, default=0.85)
    parser.add_argument("--v10_ucb_beta_end", type=float, default=0.20)
    parser.add_argument("--v10_exploration_floor", type=float, default=0.15)
    parser.add_argument("--v10_group_ucb_weight", type=float, default=0.35)
    parser.add_argument("--v10_operator_ucb_weight", type=float, default=0.25)
    parser.add_argument("--v10_length_deltas", type=str, default="-2,0,2")
    parser.add_argument("--v10_peripheral_fraction", type=float, default=0.20)
    parser.add_argument("--v10_micro_fraction", type=float, default=0.08)
    parser.add_argument("--v10_max_empty_rounds", type=int, default=6)
    parser.add_argument("--direct_bootstrap_calls", type=int, default=256)
    parser.add_argument("--direct_global_fraction", type=float, default=0.10)
    parser.add_argument("--direct_global_steps", type=int, default=256)
    parser.add_argument("--direct_parent_pool_size", type=int, default=160)
    parser.add_argument("--direct_overgenerate_factor", type=float, default=1.40)
    parser.add_argument("--direct_stagnation_rounds", type=int, default=5)
    parser.add_argument("--direct_max_empty_rounds", type=int, default=8)
    parser.add_argument("--direct_length_quantile_low", type=float, default=0.01)
    parser.add_argument("--direct_length_quantile_high", type=float, default=0.995)
    parser.add_argument("--direct_local_rounds", type=int, default=3)
    parser.add_argument("--direct_frontier_stagnation_rounds", type=int, default=8)
    parser.add_argument("--direct_rescue_global_fraction", type=float, default=0.05)
    parser.add_argument("--direct_absolute_length_slack", type=int, default=30)
    parser.add_argument("--direct_quality_parent_fraction", type=float, default=0.25)
    return parser.parse_args()


def main():
    start = time()
    args = parse_args()
    try:
        from tdc import Oracle
    except ImportError as exc:
        raise SystemExit("PMO requires pytdc. Install the project dependencies before running PMO.") from exc

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "CSDNet",
            "exp",
            "pmo",
            "results",
            args.mode,
        )
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = PMO_TASKS if args.oracle == "all" else [args.oracle]
    print(f"CSDNet PMO mode: {args.mode}")
    print(f"Tasks: {len(tasks)} ({', '.join(tasks)})")
    print(f"Output dir: {args.output_dir}")
    print(f"Max oracle calls per task: {args.max_oracle_calls}")

    model_bundle = load_csdnet_model(args)
    for oracle_name in tasks:
        if args.resume and completed(
            args.output_dir,
            args.mode,
            oracle_name,
            args.seed,
            args.max_oracle_calls,
        ):
            print(f"[skip] {oracle_name}: completed result found.")
            continue
        if args.resume:
            resume_buffer = load_resume_buffer(
                args.output_dir,
                args.mode,
                oracle_name,
                args.seed,
                args.max_oracle_calls,
            )
        else:
            resume_buffer = {}
            clear_incomplete_outputs(args.output_dir, args.mode, oracle_name, args.seed)

        print("=" * 80)
        print(f"Optimizing PMO oracle: {oracle_name}")
        if resume_buffer:
            print(
                f"[resume] {oracle_name}: restoring "
                f"{len(resume_buffer)}/{args.max_oracle_calls} oracle calls."
            )
        args.oracle = oracle_name
        try:
            oracle = Oracle(name=oracle_name)
            optimizer = CSDNetOptimizer(args=args, model_bundle=model_bundle)
            if resume_buffer:
                optimizer.oracle.mol_buffer = resume_buffer
                optimizer.oracle.last_log = len(resume_buffer)
            optimizer.optimize(oracle=oracle, config={}, seed=args.seed)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            fail_path = os.path.join(args.output_dir, f"failed_{args.mode}.csv")
            exists = os.path.exists(fail_path)
            with open(fail_path, "a") as f:
                if not exists:
                    f.write("oracle,seed,error\n")
                msg = str(exc).replace("\n", " ")[:1000]
                f.write(f"{oracle_name},{args.seed},{type(exc).__name__}: {msg}\n")
            print(f"[failed] {oracle_name}: {type(exc).__name__}: {exc}")
            continue

    hours = (time() - start) / 3600.0
    print(f"CSDNet PMO {args.mode} finished in {hours:.2f} hours")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run the five fragment-constrained tasks with the V1 frontier core."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger

from CSDNet.exp.frag.run_linker_design import evaluate_samples
from CSDNet.exp.frag.task_head import (
    CANONICAL_TASKS,
    OPERATOR_PROFILES,
    OPERATOR_PROFILES_V2,
    FragmentConstraintAdapter,
    FragmentConstraintAdapterV2,
    assess_candidate,
    available_operators,
    build_constraint_spec,
    build_seed_pool,
    build_seed_pool_v2,
    largest_root_fraction,
    make_edit_plan,
    normalize_task,
    prepare_model_seed,
)
from CSDNet.exp.pmo.optimizer import (
    load_csdnet_model,
    sample_csdnet_local_remask,
)
from CSDNet.optim.frontier import (
    UnifiedFrontierEngine,
    allocate_insertion_flags,
)


RDLogger.DisableLog("rdApp.*")


def stable_case_seed(seed: int, task: str, name: str) -> int:
    # The published benchmark defines scaffold morphing and linker design with
    # the same prompt.  Sharing their random stream preserves that equivalence.
    random_task = "linker_design" if task == "scaffold_morphing" else task
    payload = f"{int(seed)}:{random_task}:{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def make_engine(spec, args):
    operators = available_operators(spec)
    adapter_kwargs = {
        "warmup_attempts": min(args.round_size, args.num_samples),
        "feasible_rate": args.feasible_rate,
        "plateau_patience": args.plateau_patience,
        "collapse_threshold": args.collapse_threshold,
    }
    if args.adapter_version == "v2":
        adapter = FragmentConstraintAdapterV2(
            unique_target=args.unique_target,
            **adapter_kwargs,
        )
    else:
        adapter = FragmentConstraintAdapter(**adapter_kwargs)
    engine = UnifiedFrontierEngine(
        adapter=adapter,
        operator_groups={"proposal": operators},
        bandit_configs={
            "proposal": {
                "alpha": args.bandit_alpha,
                "temperature": args.bandit_temperature,
                "ucb_weight": args.bandit_ucb_weight,
                "min_multiplier": args.bandit_min_multiplier,
                "base_floor": args.bandit_base_floor,
            }
        },
    )
    return engine, operators


def _seed_batch_with_plans(
    seeds,
    seed_roots,
    queries,
    operator,
    amount,
    rng,
    profiles,
):
    selected = []
    plans = []
    roots = []
    for _ in range(max(0, int(amount))):
        seed_index = rng.randrange(len(seeds))
        seed, plan = make_edit_plan(
            seeds[seed_index],
            queries,
            operator,
            rng,
            profiles=profiles,
        )
        if seed is None or plan is None:
            selected.append(None)
            plans.append(None)
            roots.append(seed_roots[seed_index])
            continue
        selected.append(seed)
        plans.append(plan)
        roots.append(seed_roots[seed_index])
    return selected, plans, roots


def _generate_operator_batch(
    *,
    args,
    model_bundle,
    spec,
    seeds,
    seed_roots,
    operator,
    amount,
    rng,
    profiles,
    insertion_fraction,
):
    model, tk, device = model_bundle
    selected, plans, roots = _seed_batch_with_plans(
        seeds,
        seed_roots,
        spec.queries,
        operator,
        amount,
        rng,
        profiles,
    )
    runnable_indices = [
        index
        for index, (seed, plan) in enumerate(zip(selected, plans))
        if seed is not None and plan is not None
    ]
    outputs_by_slot = {}
    diagnostics_by_slot = {}
    if runnable_indices:
        runnable_seeds = [selected[index] for index in runnable_indices]
        runnable_plans = [plans[index] for index in runnable_indices]
        profile = profiles[operator]
        insertion_flags = allocate_insertion_flags(
            len(runnable_plans),
            insertion_fraction if profile.learned_insertion else 0.0,
            rng,
        )
        for plan, learned in zip(runnable_plans, insertion_flags):
            if not learned:
                plan["delta"] = 0
                continue
            plan["length_mode"] = "learned_insertion"
            if not getattr(model, "is_elastic", False):
                removed = max(1, int(plan["stop"]) - int(plan["start"]))
                plan.update(
                    {
                        "min_replacement_len": max(
                            0,
                            removed - int(profile.max_shrink_tokens),
                        ),
                        "max_replacement_len": (
                            removed + int(profile.max_growth_tokens)
                        ),
                    }
                )
        outputs = sample_csdnet_local_remask(
            model=model,
            tk=tk,
            seed_smiles=runnable_seeds,
            max_len=args.max_len,
            device=device,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            remask_fraction=0.20,
            min_remask_tokens=1,
            span_prob=profile.span_prob,
            use_fsm_check=not args.disable_fsm_check,
            use_rdkit_kekulize_check=not args.disable_rdkit_kekulize_check,
            rdkit_check_interval=args.rdkit_check_interval,
            max_sample_retries=args.max_sample_retries,
            violation_neighborhood=args.violation_neighborhood,
            temperature_start=profile.temperature_start,
            temperature_end=args.temperature_end,
            temperature_power=args.temperature_power,
            top_k=getattr(args, "top_k", 0),
            top_p=getattr(args, "top_p", 1.0),
            edit_plans=runnable_plans,
            return_seed_indices=True,
            return_diagnostics=True,
        )
        for smiles, runnable_index, diagnostics in outputs:
            slot = runnable_indices[int(runnable_index)]
            outputs_by_slot[slot] = smiles
            diagnostics_by_slot[slot] = diagnostics

    assessments = []
    attempt_rows = []
    for slot in range(int(amount)):
        assessment = assess_candidate(outputs_by_slot.get(slot), spec)
        assessments.append((slot, roots[slot], assessment))
        plan = plans[slot] or {}
        diagnostics = diagnostics_by_slot.get(slot, {})
        attempt_rows.append(
            {
                "operator": operator,
                "operator_slot": slot,
                "root_index": roots[slot],
                "seed_smiles": selected[slot],
                "edit_start": plan.get("start"),
                "edit_stop": plan.get("stop"),
                "edit_delta": plan.get("delta"),
                "length_mode": diagnostics.get(
                    "length_mode",
                    plan.get("length_mode", "fixed"),
                ),
                "removed_tokens": diagnostics.get("removed_tokens"),
                "inserted_tokens": diagnostics.get("inserted_tokens"),
                "actual_delta": diagnostics.get("actual_delta"),
                "generated": assessment.valid,
                "smiles": assessment.smiles,
                "connected": assessment.connected,
                "no_dummies": assessment.no_dummies,
                "preserved": assessment.preserved,
                "required": assessment.required,
                "preserved_fraction": assessment.preserved_fraction,
                "structural_success": assessment.structural_success,
                "structural_score": assessment.structural_score,
            }
        )
    return assessments, attempt_rows


def run_case(args, model_bundle, row):
    spec = build_constraint_spec(args.task, row)
    case_seed = stable_case_seed(args.seed, spec.task, spec.name)
    random.seed(case_seed)
    np.random.seed(case_seed)
    torch.manual_seed(case_seed)
    rng = random.Random(case_seed)

    seed_builder = (
        build_seed_pool_v2 if args.adapter_version == "v2" else build_seed_pool
    )
    profiles = (
        OPERATOR_PROFILES_V2 if args.adapter_version == "v2" else OPERATOR_PROFILES
    )
    seeds = seed_builder(spec, limit=args.seed_pool_size, rng=rng)
    model, tk, _ = model_bundle
    del model
    seeds = [prepare_model_seed(seed, tk, args.max_len) for seed in seeds]
    seeds = list(dict.fromkeys(seed for seed in seeds if seed is not None))
    if not seeds:
        raise RuntimeError(
            f"No tokenizable seed structures for {spec.task}:{spec.name}"
        )
    seed_roots = list(range(len(seeds)))
    dynamic_seed_seen = set(seeds)
    base_seed_count = len(seeds)

    engine, operators = make_engine(spec, args)
    attempts = 0
    generated = 0
    accepted = []
    accepted_roots = []
    accepted_unique = set()
    root_success_counts = Counter()
    all_attempt_rows = []
    policy_rows = []
    stagnant_rounds = 0
    previous_successes = 0
    round_index = 0

    while attempts < args.num_samples:
        success_rate = len(accepted) / max(1, attempts)
        unique_success_rate = len(accepted_unique) / max(1, len(accepted))
        valid_rate = generated / max(1, attempts)
        context = {
            "attempts": attempts,
            "structural_success_rate": success_rate,
            "valid_rate": valid_rate,
            "stagnant_rounds": stagnant_rounds,
            "largest_root_fraction": largest_root_fraction(accepted_roots),
            "available_operators": operators,
            "geometry": spec.geometry,
        }
        if args.adapter_version == "v2":
            context.update(
                {
                    "unique_success_rate": unique_success_rate,
                    "dynamic_seed_pool_size": len(seeds),
                }
            )
        state = engine.classify(**context)
        round_budget = min(args.round_size, args.num_samples - attempts)
        allocation = engine.allocate(round_budget, state=state, context=context)
        allocation = allocation.get("proposal", {})
        if sum(allocation.values()) != round_budget:
            raise RuntimeError(
                f"V1 allocation lost budget: {allocation} vs {round_budget}"
            )

        operator_items = list(allocation.items())
        if args.adapter_version == "v2":
            rng.shuffle(operator_items)
        round_new_seeds = []
        for operator, amount in operator_items:
            assessments, rows = _generate_operator_batch(
                args=args,
                model_bundle=model_bundle,
                spec=spec,
                seeds=seeds,
                seed_roots=seed_roots,
                operator=operator,
                amount=amount,
                rng=rng,
                profiles=profiles,
                insertion_fraction=engine.adapter.insertion_fraction(
                    state,
                    context,
                ),
            )
            transitions = []
            operator_generated = 0
            operator_accepted = 0
            operator_novel = 0
            for local_slot, root_index, assessment in assessments:
                is_novel = False
                lineage_credit = 0.0
                if assessment.valid:
                    generated += 1
                    operator_generated += 1
                    transition = assessment.transition()
                    is_novel = bool(
                        assessment.structural_success
                        and assessment.smiles not in accepted_unique
                    )
                    lineage_credit = (
                        1.0 / math.sqrt(root_success_counts[root_index] + 1.0)
                        if assessment.structural_success
                        else 0.0
                    )
                    transition.update(
                        {
                            "novel": is_novel,
                            "lineage_credit": lineage_credit,
                        }
                    )
                    transitions.append(transition)
                if assessment.structural_success:
                    accepted.append(assessment.smiles)
                    accepted_roots.append(root_index)
                    operator_accepted += 1
                    if is_novel:
                        accepted_unique.add(assessment.smiles)
                        round_new_seeds.append((assessment.smiles, root_index))
                        operator_novel += 1
                    root_success_counts[root_index] += 1
                rows[local_slot].update(
                    {
                        "task": spec.task,
                        "name": spec.name,
                        "seed": args.seed,
                        "case_seed": case_seed,
                        "round": round_index,
                        "state": state,
                        "global_attempt": attempts + local_slot + 1,
                        "novel": is_novel,
                        "lineage_credit": lineage_credit,
                    }
                )
            reward, reward_parts = engine.update_constrained_batch(
                group="proposal",
                operator=operator,
                transitions=transitions,
                requested=amount,
            )
            policy_rows.append(
                {
                    "task": spec.task,
                    "name": spec.name,
                    "seed": args.seed,
                    "round": round_index,
                    "state": state,
                    "geometry": spec.geometry,
                    "operator": operator,
                    "requested": amount,
                    "generated": operator_generated,
                    "structural_successes": operator_accepted,
                    "novel_structural_successes": operator_novel,
                    "dynamic_seed_pool_size": len(seeds),
                    "reward": reward,
                    **reward_parts,
                }
            )
            all_attempt_rows.extend(rows)
            attempts += amount

        if args.adapter_version == "v2":
            for candidate, root_index in round_new_seeds:
                if len(seeds) >= int(args.adaptive_seed_pool_size):
                    break
                prepared = prepare_model_seed(candidate, tk, args.max_len)
                if prepared is None or prepared in dynamic_seed_seen:
                    continue
                dynamic_seed_seen.add(prepared)
                seeds.append(prepared)
                seed_roots.append(root_index)

        progress = (
            len(accepted_unique) if args.adapter_version == "v2" else len(accepted)
        )
        if progress > previous_successes:
            stagnant_rounds = 0
        else:
            stagnant_rounds += 1
        previous_successes = progress
        round_index += 1

    snapshot = engine.snapshot()
    snapshot["adapter_version"] = args.adapter_version
    snapshot["base_seed_pool_size"] = base_seed_count
    snapshot["final_seed_pool_size"] = len(seeds)
    snapshot["unique_structural_successes"] = len(accepted_unique)
    return spec, seeds, accepted, all_attempt_rows, policy_rows, snapshot


def summarize_metrics(rows):
    columns = [
        "validity",
        "uniqueness",
        "diversity",
        "distance",
        "quality",
        "mean_qed",
        "mean_sa",
    ]
    df = pd.DataFrame(rows)
    summary = {"n_cases": len(df)}
    for column in columns:
        summary[f"{column}_mean"] = float(df[column].mean()) if len(df) else 0.0
    return summary


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def read_rows(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_task(args):
    from tdc import Evaluator, Oracle

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.task = normalize_task(args.task)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.fragments_csv)
    model_bundle = load_csdnet_model(args)
    oracle_qed = Oracle("qed")
    oracle_sa = Oracle("sa")
    diversity_evaluator = Evaluator("diversity")

    stem = f"{args.task}_seed{args.seed}"
    metrics_path = output_dir / f"metrics_{stem}.csv"
    samples_path = output_dir / f"samples_{stem}.csv"
    attempts_path = output_dir / f"attempts_{stem}.csv"
    policy_path = output_dir / f"policy_{stem}.csv"
    snapshot_path = output_dir / f"policy_state_{stem}.json"
    summary_path = output_dir / f"summary_{stem}.csv"

    metric_rows = read_rows(metrics_path) if args.resume else []
    sample_rows = read_rows(samples_path) if args.resume else []
    attempt_rows = read_rows(attempts_path) if args.resume else []
    policy_rows = read_rows(policy_path) if args.resume else []
    if args.resume and snapshot_path.exists():
        policy_snapshots = json.loads(snapshot_path.read_text(encoding="utf-8"))
    else:
        policy_snapshots = {}
    completed_names = {str(record["name"]) for record in metric_rows}

    for _, row in data.iterrows():
        name = str(row["name"])
        if name in completed_names:
            print(f"[{args.task}] {name}: already complete, skipping")
            continue
        print("=" * 72)
        print(f"[{args.task}] {name}: fixed raw budget={args.num_samples}")
        spec, seeds, accepted, attempts, policy, snapshot = run_case(
            args, model_bundle, row
        )

        metrics, unique_records = evaluate_samples(
            spec.original,
            accepted,
            args.num_samples,
            oracle_qed,
            oracle_sa,
            diversity_evaluator,
        )
        metrics.update(
            {
                "task": spec.task,
                "name": spec.name,
                "seed": args.seed,
                "adapter_version": args.adapter_version,
                "original": spec.original,
                "fragment": spec.fragment,
                "geometry": spec.geometry,
                "raw_attempts": args.num_samples,
                "model_outputs": sum(bool(item.get("generated")) for item in attempts),
                "structural_successes": len(accepted),
                "seed_pool_size": len(seeds),
                "base_seed_pool_size": int(
                    snapshot.get("base_seed_pool_size", len(seeds))
                ),
            }
        )
        metric_rows.append(metrics)
        attempt_rows.extend(attempts)
        policy_rows.extend(policy)
        policy_snapshots[spec.name] = snapshot
        for record in unique_records:
            sample_rows.append(
                {
                    "task": spec.task,
                    "name": spec.name,
                    "seed": args.seed,
                    "smiles": record["smiles"],
                    "qed": record["qed"],
                    "sa": record["sa"],
                }
            )
        print(
            f"[{args.task}] {name}: validity={metrics['validity']:.3f} "
            f"uniqueness={metrics['uniqueness']:.3f} "
            f"quality={metrics['quality']:.3f} "
            f"outputs={metrics['model_outputs']}/{args.num_samples}"
        )
        write_rows(metrics_path, metric_rows)
        write_rows(samples_path, sample_rows)
        write_rows(attempts_path, attempt_rows)
        write_rows(policy_path, policy_rows)
        write_json(snapshot_path, policy_snapshots)

    summary = {
        "task": args.task,
        "seed": args.seed,
        "adapter_version": args.adapter_version,
        **summarize_metrics(metric_rows),
    }
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print("=" * 72)
    print(f"Task: {args.task}, seed: {args.seed}")
    for metric in ("validity", "uniqueness", "quality", "diversity", "distance"):
        print(f"{metric}: {summary[f'{metric}_mean']:.4f}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {samples_path}")
    print(f"Saved: {attempts_path}")
    print(f"Saved: {policy_path}")
    print(f"Saved: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True, choices=(*CANONICAL_TASKS, "superstructure_design")
    )
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--fragments_csv", default="data/fragments.csv")
    parser.add_argument(
        "--output_dir",
        default=os.path.join("CSDNet", "exp", "frag", "results", "frontier_v1"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter_version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--round_size", type=int, default=25)
    parser.add_argument("--seed_pool_size", type=int, default=48)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=160)
    parser.add_argument("--temperature_end", type=float, default=0.18)
    parser.add_argument("--temperature_power", type=float, default=1.6)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--feasible_rate", type=float, default=0.35)
    parser.add_argument("--unique_target", type=float, default=0.70)
    parser.add_argument("--plateau_patience", type=int, default=2)
    parser.add_argument("--collapse_threshold", type=float, default=0.75)
    parser.add_argument("--bandit_alpha", type=float, default=0.25)
    parser.add_argument("--bandit_temperature", type=float, default=1.5)
    parser.add_argument("--bandit_ucb_weight", type=float, default=0.25)
    parser.add_argument("--bandit_min_multiplier", type=float, default=0.05)
    parser.add_argument("--bandit_base_floor", type=float, default=0.20)
    parser.add_argument("--disable_fsm_check", action="store_true")
    parser.add_argument("--disable_rdkit_kekulize_check", action="store_true")
    parser.add_argument("--rdkit_check_interval", type=int, default=25)
    parser.add_argument("--max_sample_retries", type=int, default=2)
    parser.add_argument("--violation_neighborhood", type=int, default=2)
    parser.add_argument("--adaptive_seed_pool_size", type=int, default=160)
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.fragments_csv).exists():
        raise SystemExit(f"Cannot find fragments CSV: {args.fragments_csv}")
    if args.num_samples <= 0 or args.round_size <= 0:
        raise SystemExit("num_samples and round_size must be positive")
    if args.top_k < 0 or not 0.0 < args.top_p <= 1.0:
        raise SystemExit("top_k must be non-negative and top_p must be in (0, 1]")
    if args.adaptive_seed_pool_size < args.seed_pool_size:
        raise SystemExit("adaptive_seed_pool_size must be at least seed_pool_size")
    run_task(args)


if __name__ == "__main__":
    main()

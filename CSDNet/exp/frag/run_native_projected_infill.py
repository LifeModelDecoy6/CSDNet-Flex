#!/usr/bin/env python
"""Run fair fragment infilling in the elastic model's native state space."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger

from CSDNet.exp.frag.direct_infill import (
    apply_native_gap_constraint_policy,
    build_native_projected_template,
    native_gap_insertion_rate_scale,
    native_nucleus_support,
    native_sampler_arm,
)
from CSDNet.exp.frag.fragment_length_prior import (
    FragmentGapLengthPrior,
    PrefillProposal,
    apply_prefill_lengths,
)
from CSDNet.exp.frag.run_direct_infill import (
    _read_rows,
    _validate_fixed_tokens,
    _write_rows,
)
from CSDNet.exp.frag.run_linker_design import evaluate_samples
from CSDNet.exp.frag.task_head import (
    CANONICAL_TASKS,
    assess_candidate,
    build_constraint_spec,
    normalize_task,
    task_column,
)
from CSDNet.exp.pmo.optimizer import load_csdnet_model
from CSDNet.util.elastic_sampling import sample_elastic_local_infill
from CSDNet.util.tokenizer import tokenize_smiles


RDLogger.DisableLog("rdApp.*")


def stable_case_seed(seed: int, task: str, name: str) -> int:
    random_task = "linker_design" if task == "scaffold_morphing" else task
    payload = f"native-projected-infill-v1:{seed}:{random_task}:{name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fixed_template_tokens(template) -> int:
    tokens = tokenize_smiles(template.seed_smiles)
    editable = set()
    for plan in template.edit_plans:
        editable.update(range(int(plan["start"]), int(plan["stop"])))
    return max(0, len(tokens) - len(editable))


def run_case(args, model_bundle, row, length_prior=None):
    spec = build_constraint_spec(args.task, row)
    case_seed = stable_case_seed(args.seed, spec.task, spec.name)
    _seed_everything(case_seed)
    rng = random.Random(case_seed)
    model, tokenizer, device = model_bundle

    templates = [
        build_native_projected_template(
            spec,
            max_len=args.max_len,
            rng=rng,
            initial_gap_tokens=args.initial_gap_tokens,
        )
        for _ in range(args.num_samples)
    ]
    for template in templates:
        _validate_fixed_tokens(template, tokenizer)

    planned_edits = []
    constrained_trajectories = []
    prefill_proposals = []
    for attempt_index, template in enumerate(templates):
        plans, constrained = apply_native_gap_constraint_policy(
            template,
            geometry=spec.geometry,
            attempt_index=attempt_index,
            case_seed=case_seed,
            policy=args.gap_constraint_policy,
        )
        fixed_tokens = _fixed_template_tokens(template)
        if args.length_prefill_policy in {
            "zinc_geometry_lower",
            "zinc_geometry_calibrated",
            "zinc_geometry_diversity_guarded",
        }:
            if length_prior is None:
                raise RuntimeError("ZINC geometry prefill requires a loaded prior")
            proposal = length_prior.propose(
                geometry=spec.geometry,
                fixed_tokens=fixed_tokens,
                gap_count=len(plans),
                attempt_index=attempt_index,
                constrained_atoms=constrained,
                maximum_total=max(
                    len(plans),
                    int(args.max_len) - 2 - fixed_tokens,
                ),
                maximum_per_gap=args.max_prefill_tokens_per_gap,
                rng=rng,
                allocation_profile=(
                    "diversity_guarded"
                    if args.length_prefill_policy
                    == "zinc_geometry_diversity_guarded"
                    else (
                        "geometry_calibrated"
                        if args.length_prefill_policy
                        == "zinc_geometry_calibrated"
                        else "balanced"
                    )
                ),
            )
            plans = apply_prefill_lengths(plans, proposal.lengths)
        else:
            proposal = PrefillProposal(
                lengths=tuple(
                    int(plan["initial_replacement_len"]) for plan in plans
                ),
                source="native_one_mask",
                measure="atoms" if constrained else "tokens",
                quantile=None,
                prior_total=len(plans),
            )
        planned_edits.append(list(plans))
        constrained_trajectories.append(constrained)
        prefill_proposals.append(proposal)

    effective_insertion_rate_scale = native_gap_insertion_rate_scale(
        geometry=spec.geometry,
        base_scale=args.insertion_rate_scale,
        policy=args.insertion_rate_policy,
    )
    effective_nucleus_start, effective_nucleus_end = native_nucleus_support(
        geometry=spec.geometry,
        start=args.nucleus_min_tokens_start,
        end=args.nucleus_min_tokens_end,
        policy=args.nucleus_support_policy,
    )

    sampler_arms = [
        native_sampler_arm(
            attempt_index=index,
            case_seed=case_seed,
            exploration_fraction=args.diversity_fraction,
            policy=args.sampler_portfolio_policy,
            prefill_source=prefill_proposals[index].source,
        )
        for index in range(len(templates))
    ]
    outputs = {}
    for arm in ("core", "exploration"):
        group_indices = [
            index
            for index, assigned_arm in enumerate(sampler_arms)
            if assigned_arm == arm
        ]
        if not group_indices:
            continue
        if arm == "exploration":
            temperature = args.diversity_temperature
            top_p = args.diversity_top_p
            nucleus_start = max(
                effective_nucleus_start,
                args.diversity_nucleus_min_tokens_start,
            )
            nucleus_end = max(
                effective_nucleus_end,
                args.diversity_nucleus_min_tokens_end,
            )
            unmask_selection = args.diversity_unmask_selection
        else:
            temperature = args.temperature
            top_p = args.top_p
            nucleus_start = effective_nucleus_start
            nucleus_end = effective_nucleus_end
            unmask_selection = "top_prob"

        generated = sample_elastic_local_infill(
            model=model,
            tk=tokenizer,
            seed_smiles=[templates[index].seed_smiles for index in group_indices],
            edit_plans=[planned_edits[index] for index in group_indices],
            max_len=args.max_len,
            device=device,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            use_fsm_check=not args.disable_fsm_check,
            use_rdkit_kekulize_check=not args.disable_rdkit_kekulize_check,
            max_sample_retries=args.max_sample_retries,
            violation_neighborhood=args.violation_neighborhood,
            fsm_repair_progressive_steps=args.fsm_repair_progressive_steps,
            fsm_repair_prefer_localization=(
                not args.disable_fsm_repair_localization
            ),
            temperature_start=temperature,
            temperature_end=temperature,
            temperature_power=1.0,
            top_k=args.top_k,
            top_p=top_p,
            nucleus_min_tokens_start=nucleus_start,
            nucleus_min_tokens_end=nucleus_end,
            max_insertions_per_step=args.max_insertions_per_step,
            insertion_rate_scale=effective_insertion_rate_scale,
            unmask_selection=unmask_selection,
            deterministic_final_unmask=True,
            recursive_gap_insertions=(args.gap_insertion_mode == "recursive"),
            trajectory_mode=args.trajectory_mode,
            planning_fraction=args.planning_fraction,
            fill_mode=args.fill_mode,
            fill_remask_power=args.fill_remask_power,
            fill_gumbel_scale=args.fill_gumbel_scale,
            sequence_validators=[
                lambda smiles, current_spec=spec: assess_candidate(
                    smiles,
                    current_spec,
                ).structural_success
                for _ in group_indices
            ]
            if args.condition_repair
            else None,
            return_seed_indices=True,
            return_diagnostics=True,
        )
        for smiles, local_index, diagnostics in generated:
            source_index = group_indices[int(local_index)]
            diagnostics = dict(diagnostics)
            diagnostics.update(
                {
                    "sampler_arm": arm,
                    "effective_temperature": float(temperature),
                    "effective_top_p": float(top_p),
                    "effective_nucleus_min_tokens_start": int(nucleus_start),
                    "effective_nucleus_min_tokens_end": int(nucleus_end),
                    "effective_unmask_selection": unmask_selection,
                }
            )
            outputs[source_index] = (smiles, diagnostics)

    attempts = []
    accepted = []
    for index, template in enumerate(templates):
        smiles, diagnostics = outputs.get(index, (None, {}))
        assessment = assess_candidate(smiles, spec)
        if assessment.structural_success:
            accepted.append(assessment.smiles)
        attempts.append(
            {
                "task": spec.task,
                "name": spec.name,
                "seed": args.seed,
                "case_seed": case_seed,
                "attempt": index,
                "method": (
                    f"native_projected_{args.gap_insertion_mode}_"
                    f"{args.trajectory_mode}_{args.fill_mode}_"
                    f"{args.gap_constraint_policy}_"
                    f"{args.insertion_rate_policy}_"
                    f"{args.length_prefill_policy}_"
                    f"{args.sampler_portfolio_policy}"
                ),
                "gap_insertion_mode": args.gap_insertion_mode,
                "trajectory_mode": args.trajectory_mode,
                "planning_fraction": args.planning_fraction,
                "fill_mode": args.fill_mode,
                "fill_remask_power": args.fill_remask_power,
                "fill_gumbel_scale": args.fill_gumbel_scale,
                "geometry": spec.geometry,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "nucleus_min_tokens_start": args.nucleus_min_tokens_start,
                "nucleus_min_tokens_end": args.nucleus_min_tokens_end,
                "nucleus_support_policy": args.nucleus_support_policy,
                "sampler_portfolio_policy": args.sampler_portfolio_policy,
                "diversity_fraction": args.diversity_fraction,
                "sampler_arm": diagnostics.get("sampler_arm", "missing"),
                "effective_temperature": diagnostics.get(
                    "effective_temperature", args.temperature
                ),
                "effective_top_p": diagnostics.get(
                    "effective_top_p", args.top_p
                ),
                "effective_nucleus_min_tokens_start": (
                    diagnostics.get(
                        "effective_nucleus_min_tokens_start",
                        effective_nucleus_start,
                    )
                ),
                "effective_nucleus_min_tokens_end": diagnostics.get(
                    "effective_nucleus_min_tokens_end",
                    effective_nucleus_end,
                ),
                "effective_unmask_selection": diagnostics.get(
                    "effective_unmask_selection", "top_prob"
                ),
                "n_steps": args.n_steps,
                "initial_gap_tokens": args.initial_gap_tokens,
                "insertion_rate_scale": args.insertion_rate_scale,
                "insertion_rate_policy": args.insertion_rate_policy,
                "effective_insertion_rate_scale": (
                    effective_insertion_rate_scale
                ),
                "gap_constraint_policy": args.gap_constraint_policy,
                "gap_constraint_applied": constrained_trajectories[index],
                "length_prefill_policy": args.length_prefill_policy,
                "prefill_source": prefill_proposals[index].source,
                "prefill_measure": prefill_proposals[index].measure,
                "prefill_quantile": prefill_proposals[index].quantile,
                "prefill_prior_total": prefill_proposals[index].prior_total,
                "attachment_count": template.attachment_count,
                "open_gaps": len(planned_edits[index]),
                "initial_mask_tokens": sum(
                    int(plan["initial_replacement_len"])
                    for plan in planned_edits[index]
                ),
                "removed_tokens": diagnostics.get("removed_tokens", 0),
                "inserted_tokens": diagnostics.get("inserted_tokens", 0),
                "initial_inserted_tokens": diagnostics.get(
                    "initial_inserted_tokens", 0
                ),
                "learned_inserted_tokens": diagnostics.get(
                    "learned_inserted_tokens", 0
                ),
                "actual_delta": diagnostics.get("actual_delta", 0),
                "insertion_event_sites": diagnostics.get(
                    "insertion_event_sites", 0
                ),
                "insertion_steps": diagnostics.get("insertion_steps", 0),
                "unmask_events": diagnostics.get("unmask_events", 0),
                "forced_final_unmasks": diagnostics.get(
                    "forced_final_unmasks", 0
                ),
                "fsm_constraint_mode": diagnostics.get(
                    "fsm_constraint_mode", "unknown"
                ),
                "fsm_check_enabled": diagnostics.get(
                    "fsm_check_enabled", False
                ),
                "rdkit_kekulize_check_enabled": diagnostics.get(
                    "rdkit_kekulize_check_enabled", False
                ),
                "condition_validator_enabled": diagnostics.get(
                    "condition_validator_enabled", False
                ),
                "condition_repair_initial_constraint_invalid_rows": diagnostics.get(
                    "condition_repair_initial_constraint_invalid_rows", 0
                ),
                "condition_repair_final_constraint_invalid_rows": diagnostics.get(
                    "condition_repair_final_constraint_invalid_rows", 0
                ),
                "fsm_repair_progressive_steps": diagnostics.get(
                    "fsm_repair_progressive_steps", 0
                ),
                "fsm_repair_prefer_localization": diagnostics.get(
                    "fsm_repair_prefer_localization", False
                ),
                "online_fsm_repair_events": diagnostics.get(
                    "online_fsm_repair_events", 0
                ),
                "online_fsm_remasked_tokens": diagnostics.get(
                    "online_fsm_remasked_tokens", 0
                ),
                "chain_atom_constrained_tokens": diagnostics.get(
                    "chain_atom_constrained_tokens", 0
                ),
                "planning_steps": diagnostics.get("planning_steps", 0),
                "max_open_sites": diagnostics.get("max_open_sites", 0),
                "max_sequence_tokens": diagnostics.get(
                    "max_sequence_tokens", 0
                ),
                "mean_open_site_rate": diagnostics.get(
                    "mean_open_site_rate", 0.0
                ),
                "model_output": assessment.valid,
                # Keep the undecoded model string: invalid outputs otherwise
                # collapse to an empty canonical SMILES and cannot be audited.
                "raw_smiles": smiles,
                "smiles": assessment.smiles,
                "connected": assessment.connected,
                "no_dummies": assessment.no_dummies,
                "preserved": assessment.preserved,
                "required": assessment.required,
                "structural_success": assessment.structural_success,
            }
        )
    return spec, accepted, attempts, effective_insertion_rate_scale


def run_task(args):
    from tdc import Evaluator, Oracle

    args.task = normalize_task(args.task)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.fragments_csv)
    required_columns = {"name", "smiles", task_column(args.task)}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(
            f"{args.task} benchmark is missing explicit columns "
            f"{missing_columns} in {args.fragments_csv}"
        )
    model_bundle = load_csdnet_model(args)
    model = model_bundle[0]
    if not bool(getattr(model, "is_elastic", False)):
        raise RuntimeError(
            "native projected infill requires an elastic checkpoint with "
            "learned insertion dynamics"
        )

    length_prior = None
    length_prior_source = ""
    length_prior_source_md5 = ""
    if args.length_prefill_policy in {
        "zinc_geometry_lower",
        "zinc_geometry_calibrated",
        "zinc_geometry_diversity_guarded",
    }:
        length_prior = FragmentGapLengthPrior.load(args.fragment_length_prior)
        length_prior_source = str(length_prior.payload.get("source", ""))
        length_prior_source_md5 = str(
            length_prior.payload.get("source_md5", "")
        )
        print(
            "Fragment length prior: "
            f"path={length_prior.path} "
            f"source={length_prior.payload.get('source')} "
            f"rows={length_prior.payload.get('source_rows')} "
            f"tokenizable={length_prior.payload.get('tokenizable_rows')}"
        )

    print(
        "Elastic fragment audit: "
        f"is_elastic={getattr(model, 'is_elastic', False)} "
        f"fragment_corruption_prob="
        f"{getattr(model, 'fragment_corruption_prob', 'checkpoint-only')} "
        f"kuma_shape_a={getattr(model, 'kuma_shape_a', 'unknown')} "
        f"gap_insertion_mode={args.gap_insertion_mode}"
    )
    oracle_qed = Oracle("qed")
    oracle_sa = Oracle("sa")
    diversity_evaluator = Evaluator("diversity")

    stem = f"{args.task}_seed{args.seed}"
    metrics_path = output_dir / f"metrics_{stem}.csv"
    samples_path = output_dir / f"samples_{stem}.csv"
    attempts_path = output_dir / f"attempts_{stem}.csv"
    summary_path = output_dir / f"summary_{stem}.csv"
    metric_rows = _read_rows(metrics_path) if args.resume else []
    sample_rows = _read_rows(samples_path) if args.resume else []
    attempt_rows = _read_rows(attempts_path) if args.resume else []
    completed = {str(record["name"]) for record in metric_rows}

    for _, row in data.iterrows():
        name = str(row["name"])
        if name in completed:
            print(f"[{args.task}] {name}: already complete, skipping")
            continue
        print("=" * 72)
        print(
            f"[{args.task}] {name}: native projected attempts="
            f"{args.num_samples} steps={args.n_steps} top_p={args.top_p}"
        )
        (
            spec,
            accepted,
            attempts,
            effective_insertion_rate_scale,
        ) = run_case(args, model_bundle, row, length_prior=length_prior)
        core_attempt = next(
            (
                item
                for item in attempts
                if item.get("sampler_arm", "core") == "core"
            ),
            attempts[0],
        )
        effective_nucleus_start = int(
            core_attempt.get(
                "effective_nucleus_min_tokens_start",
                args.nucleus_min_tokens_start,
            )
        )
        effective_nucleus_end = int(
            core_attempt.get(
                "effective_nucleus_min_tokens_end",
                args.nucleus_min_tokens_end,
            )
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
                "method": (
                    f"native_projected_{args.gap_insertion_mode}_"
                    f"{args.trajectory_mode}_{args.fill_mode}_"
                    f"{args.gap_constraint_policy}_"
                    f"{args.insertion_rate_policy}_"
                    f"{args.length_prefill_policy}_"
                    f"{args.sampler_portfolio_policy}"
                ),
                "gap_insertion_mode": args.gap_insertion_mode,
                "trajectory_mode": args.trajectory_mode,
                "planning_fraction": args.planning_fraction,
                "fill_mode": args.fill_mode,
                "fill_remask_power": args.fill_remask_power,
                "fill_gumbel_scale": args.fill_gumbel_scale,
                "nucleus_min_tokens_start": args.nucleus_min_tokens_start,
                "nucleus_min_tokens_end": args.nucleus_min_tokens_end,
                "nucleus_support_policy": args.nucleus_support_policy,
                "sampler_portfolio_policy": args.sampler_portfolio_policy,
                "diversity_fraction": args.diversity_fraction,
                "diversity_temperature": args.diversity_temperature,
                "diversity_top_p": args.diversity_top_p,
                "diversity_nucleus_min_tokens_start": (
                    args.diversity_nucleus_min_tokens_start
                ),
                "diversity_nucleus_min_tokens_end": (
                    args.diversity_nucleus_min_tokens_end
                ),
                "diversity_unmask_selection": (
                    args.diversity_unmask_selection
                ),
                "exploration_fraction_observed": float(
                    np.mean(
                        [
                            item.get("sampler_arm", "core") == "exploration"
                            for item in attempts
                        ]
                    )
                ),
                "effective_nucleus_min_tokens_start": (
                    effective_nucleus_start
                ),
                "effective_nucleus_min_tokens_end": effective_nucleus_end,
                "initial_gap_tokens": args.initial_gap_tokens,
                "insertion_rate_scale": args.insertion_rate_scale,
                "insertion_rate_policy": args.insertion_rate_policy,
                "effective_insertion_rate_scale": (
                    effective_insertion_rate_scale
                ),
                "gap_constraint_policy": args.gap_constraint_policy,
                "length_prefill_policy": args.length_prefill_policy,
                "fragment_length_prior": (
                    str(args.fragment_length_prior) if length_prior else ""
                ),
                "length_prior_source": length_prior_source,
                "length_prior_source_md5": length_prior_source_md5,
                "gap_constraint_fraction": float(
                    np.mean(
                        [
                            item["gap_constraint_applied"]
                            for item in attempts
                        ]
                    )
                ),
                "original": spec.original,
                "fragment": spec.fragment,
                "geometry": spec.geometry,
                "raw_attempts": args.num_samples,
                "model_draws": args.num_samples,
                "model_outputs": sum(item["model_output"] for item in attempts),
                "structural_successes": len(accepted),
                "mean_inserted_tokens": float(
                    np.mean([item["inserted_tokens"] for item in attempts])
                ),
                "mean_actual_delta": float(
                    np.mean([item["actual_delta"] for item in attempts])
                ),
                "mean_learned_inserted_tokens": float(
                    np.mean(
                        [item["learned_inserted_tokens"] for item in attempts]
                    )
                ),
                "mean_initial_mask_tokens": float(
                    np.mean([item["initial_mask_tokens"] for item in attempts])
                ),
                "mean_prefill_prior_total": float(
                    np.mean([item["prefill_prior_total"] for item in attempts])
                ),
                "native_prefill_fraction": float(
                    np.mean(
                        [
                            item["prefill_source"] == "native_one_mask"
                            for item in attempts
                        ]
                    )
                ),
                "mean_max_open_sites": float(
                    np.mean([item["max_open_sites"] for item in attempts])
                ),
                "mean_open_site_rate": float(
                    np.mean([item["mean_open_site_rate"] for item in attempts])
                ),
                "mean_forced_final_unmasks": float(
                    np.mean(
                        [item["forced_final_unmasks"] for item in attempts]
                    )
                ),
                "mean_chain_atom_constrained_tokens": float(
                    np.mean(
                        [
                            item["chain_atom_constrained_tokens"]
                            for item in attempts
                        ]
                    )
                ),
            }
        )
        metric_rows.append(metrics)
        attempt_rows.extend(attempts)
        sample_rows.extend(
            {
                "task": spec.task,
                "name": spec.name,
                "seed": args.seed,
                **record,
            }
            for record in unique_records
        )
        print(
            f"[{args.task}] {name}: validity={metrics['validity']:.3f} "
            f"uniqueness={metrics['uniqueness']:.3f} "
            f"quality={metrics['quality']:.3f} "
            f"inserted={metrics['mean_inserted_tokens']:.2f}"
        )
        _write_rows(metrics_path, metric_rows)
        _write_rows(samples_path, sample_rows)
        _write_rows(attempts_path, attempt_rows)

    frame = pd.DataFrame(metric_rows)
    summary = {
        "task": args.task,
        "seed": args.seed,
        "method": (
            f"native_projected_{args.gap_insertion_mode}_"
            f"{args.trajectory_mode}_{args.fill_mode}_"
            f"{args.gap_constraint_policy}_"
            f"{args.insertion_rate_policy}_"
            f"{args.length_prefill_policy}_"
            f"{args.sampler_portfolio_policy}"
        ),
        "gap_insertion_mode": args.gap_insertion_mode,
        "trajectory_mode": args.trajectory_mode,
        "planning_fraction": args.planning_fraction,
        "fill_mode": args.fill_mode,
        "fill_remask_power": args.fill_remask_power,
        "fill_gumbel_scale": args.fill_gumbel_scale,
        "initial_gap_tokens": args.initial_gap_tokens,
        "insertion_rate_scale": args.insertion_rate_scale,
        "insertion_rate_policy": args.insertion_rate_policy,
        "gap_constraint_policy": args.gap_constraint_policy,
        "length_prefill_policy": args.length_prefill_policy,
        "fragment_length_prior": (
            str(args.fragment_length_prior) if length_prior else ""
        ),
        "length_prior_source": length_prior_source,
        "length_prior_source_md5": length_prior_source_md5,
        "n_cases": len(frame),
        "n_steps": args.n_steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "nucleus_min_tokens_start": args.nucleus_min_tokens_start,
        "nucleus_min_tokens_end": args.nucleus_min_tokens_end,
        "nucleus_support_policy": args.nucleus_support_policy,
        "sampler_portfolio_policy": args.sampler_portfolio_policy,
        "diversity_fraction": args.diversity_fraction,
        "diversity_temperature": args.diversity_temperature,
        "diversity_top_p": args.diversity_top_p,
        "diversity_nucleus_min_tokens_start": (
            args.diversity_nucleus_min_tokens_start
        ),
        "diversity_nucleus_min_tokens_end": (
            args.diversity_nucleus_min_tokens_end
        ),
        "diversity_unmask_selection": args.diversity_unmask_selection,
        "max_len": args.max_len,
    }
    for metric in (
        "validity",
        "uniqueness",
        "quality",
        "diversity",
        "distance",
        "mean_qed",
        "mean_sa",
        "mean_inserted_tokens",
        "mean_actual_delta",
        "mean_learned_inserted_tokens",
        "mean_initial_mask_tokens",
        "mean_prefill_prior_total",
        "native_prefill_fraction",
        "mean_max_open_sites",
        "mean_open_site_rate",
        "mean_forced_final_unmasks",
        "gap_constraint_fraction",
        "mean_chain_atom_constrained_tokens",
        "effective_insertion_rate_scale",
        "effective_nucleus_min_tokens_start",
        "effective_nucleus_min_tokens_end",
        "exploration_fraction_observed",
    ):
        summary[f"{metric}_mean"] = (
            float(frame[metric].mean()) if len(frame) else 0.0
        )
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Saved: {metrics_path}")
    print(f"Saved: {samples_path}")
    print(f"Saved: {attempts_path}")
    print(f"Saved: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        choices=(*CANONICAL_TASKS, "superstructure_design"),
    )
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--vocab", default="csdnet_vocab.pkl")
    parser.add_argument("--fragments_csv", default="data/fragments.csv")
    parser.add_argument(
        "--output_dir",
        default="CSDNet/exp/frag/results/native_recursive_base50k_3seeds",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--nucleus_min_tokens_start", type=int, default=1)
    parser.add_argument("--nucleus_min_tokens_end", type=int, default=1)
    parser.add_argument(
        "--nucleus_support_policy",
        choices=("uniform", "multi_anchor_annealed"),
        default="uniform",
        help=(
            "Use one support schedule everywhere or anneal wider early "
            "support only for multi-anchor bridge geometries."
        ),
    )
    parser.add_argument(
        "--sampler_portfolio_policy",
        choices=("uniform", "fixed_diversity", "prefill_guarded"),
        default="uniform",
        help=(
            "Use one sampler for every proposal or a pre-registered fixed "
            "mixture in which every proposal remains in the benchmark. "
            "prefill_guarded widens support only for empirical upper-tail "
            "length proposals."
        ),
    )
    parser.add_argument("--diversity_fraction", type=float, default=0.0)
    parser.add_argument("--diversity_temperature", type=float, default=1.08)
    parser.add_argument("--diversity_top_p", type=float, default=0.8)
    parser.add_argument(
        "--diversity_nucleus_min_tokens_start", type=int, default=8
    )
    parser.add_argument(
        "--diversity_nucleus_min_tokens_end", type=int, default=2
    )
    parser.add_argument(
        "--diversity_unmask_selection",
        choices=("top_prob", "random"),
        default="top_prob",
    )
    parser.add_argument("--max_insertions_per_step", type=int, default=4)
    parser.add_argument(
        "--initial_gap_tokens",
        type=int,
        choices=(0, 1),
        default=1,
        help="Start each projected editable gap empty or with one mask.",
    )
    parser.add_argument(
        "--insertion_rate_scale",
        type=float,
        default=1.0,
        help=(
            "Global posterior-rate calibration for local conditional gaps. "
            "One exactly preserves the learned insertion process."
        ),
    )
    parser.add_argument(
        "--insertion_rate_policy",
        choices=("uniform", "geometry_adaptive"),
        default="uniform",
        help=(
            "Keep the checkpoint rate unchanged or apply a conservative "
            "attachment-geometry calibration without task labels."
        ),
    )
    parser.add_argument(
        "--gap_insertion_mode",
        choices=("single_anchor", "recursive"),
        default="recursive",
        help=(
            "Use recursive to keep newly inserted positions open, matching "
            "the elastic reverse process; single_anchor preserves legacy runs."
        ),
    )
    parser.add_argument(
        "--gap_constraint_policy",
        choices=(
            "none",
            "geometry_adaptive",
            "geometry_calibrated",
            "structural_feasible",
            "all_chain",
        ),
        default="geometry_adaptive",
        help=(
            "Apply atom-chain token roles to all gaps, no gaps, or one of "
            "the geometry-only mixtures calibrated without target molecules."
        ),
    )
    parser.add_argument(
        "--length_prefill_policy",
        choices=(
            "native",
            "zinc_geometry_lower",
            "zinc_geometry_calibrated",
            "zinc_geometry_diversity_guarded",
        ),
        default="native",
        help=(
            "Initialize every gap with one mask or use a stratified, "
            "geometry-conditioned ZINC prior before learned insertion."
        ),
    )
    parser.add_argument(
        "--fragment_length_prior",
        default="data/zinc250k_fragment_gap_prior_atom256.json",
    )
    parser.add_argument("--max_prefill_tokens_per_gap", type=int, default=16)
    parser.add_argument(
        "--trajectory_mode",
        choices=("coupled", "plan_then_fill"),
        default="coupled",
        help=(
            "Use plan_then_fill to infer gap lengths before jointly decoding "
            "their token content; coupled preserves the native trajectory."
        ),
    )
    parser.add_argument("--planning_fraction", type=float, default=0.5)
    parser.add_argument(
        "--fill_mode",
        choices=("absorbing", "progressive_remask"),
        default="absorbing",
    )
    parser.add_argument("--fill_remask_power", type=float, default=0.8)
    parser.add_argument("--fill_gumbel_scale", type=float, default=0.65)
    parser.add_argument("--disable_fsm_check", action="store_true")
    parser.add_argument("--disable_rdkit_kekulize_check", action="store_true")
    parser.add_argument(
        "--condition_repair",
        action="store_true",
        help=(
            "Repair chemistry-valid outputs that lost a protected benchmark "
            "fragment by reopening only the declared editable gap."
        ),
    )
    parser.add_argument(
        "--fsm_repair_progressive_steps",
        type=int,
        default=8,
        help=(
            "Number of confidence-ordered completion steps used by the final "
            "neural repair before deterministic syntax projection."
        ),
    )
    parser.add_argument(
        "--disable_fsm_repair_localization",
        action="store_true",
        help=(
            "Use whole-row RDKit localization immediately instead of giving "
            "the syntax FSM's local diagnosis priority."
        ),
    )
    parser.add_argument("--max_sample_retries", type=int, default=2)
    parser.add_argument("--violation_neighborhood", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_samples <= 0 or args.n_steps <= 0:
        raise SystemExit("num_samples and n_steps must be positive")
    if not 0.0 < args.top_p <= 1.0:
        raise SystemExit("top_p must be in (0, 1]")
    if args.nucleus_min_tokens_start < 1 or args.nucleus_min_tokens_end < 1:
        raise SystemExit("nucleus minimum token counts must be positive")
    if not 0.0 <= args.diversity_fraction <= 1.0:
        raise SystemExit("diversity_fraction must be in [0, 1]")
    if (
        args.sampler_portfolio_policy == "fixed_diversity"
        and args.diversity_fraction <= 0.0
    ):
        raise SystemExit(
            "fixed_diversity requires a positive diversity_fraction"
        )
    if args.diversity_temperature <= 0.0:
        raise SystemExit("diversity_temperature must be positive")
    if not 0.0 < args.diversity_top_p <= 1.0:
        raise SystemExit("diversity_top_p must be in (0, 1]")
    if (
        args.diversity_nucleus_min_tokens_start < 1
        or args.diversity_nucleus_min_tokens_end < 1
    ):
        raise SystemExit("diversity nucleus support must be positive")
    if args.insertion_rate_scale <= 0.0:
        raise SystemExit("insertion_rate_scale must be positive")
    if args.max_prefill_tokens_per_gap < 1:
        raise SystemExit("max_prefill_tokens_per_gap must be positive")
    if args.fsm_repair_progressive_steps < 1:
        raise SystemExit("fsm_repair_progressive_steps must be positive")
    if args.length_prefill_policy in {
        "zinc_geometry_lower",
        "zinc_geometry_calibrated",
        "zinc_geometry_diversity_guarded",
    }:
        prior_path = Path(args.fragment_length_prior)
        if not prior_path.is_file() or prior_path.stat().st_size == 0:
            raise SystemExit(
                f"Cannot find non-empty fragment length prior: {prior_path}"
            )
    if args.trajectory_mode == "plan_then_fill" and not (
        0.0 < args.planning_fraction < 1.0
    ):
        raise SystemExit("planning_fraction must be in (0, 1)")
    if args.fill_mode == "progressive_remask" and (
        args.trajectory_mode != "plan_then_fill"
    ):
        raise SystemExit(
            "progressive_remask requires --trajectory_mode plan_then_fill"
        )
    for path in (args.fragments_csv, args.vocab, args.ckpt_path):
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise SystemExit(f"Cannot find non-empty required file: {path}")
    run_task(args)


if __name__ == "__main__":
    main()

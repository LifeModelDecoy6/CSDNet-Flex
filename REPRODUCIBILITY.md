# Reproducibility guide

## Evidence hierarchy

The [Zenodo archive](https://doi.org/10.5281/zenodo.21980242) is the
authoritative frozen record. It contains the exact
checkpoint, raw generated molecules, all 90 Lead terminal cells, all 69 PMO
histories, aggregate tables, and SHA-256 checksums. This GitHub repository is a
lighter code distribution with compact summaries.

`manifests/final_result_lock.json` fixes seeds, budgets, sampler profiles,
expected output counts, and expected aggregate values. The claim-to-file map is
`manifests/paper_result_manifest.csv`.

## De novo generation

Final profile: `elastic_loflex`; 1,000 denoising steps; seeds 0, 1, and 2;
1,000 delivered molecules per seed. The delivered-pool protocol permits at
most a 1.5x refill factor after strict final sanitization. The raw 2x2
constraint factorial instead fixes exactly 1,000 proposals per seed and uses
no refill or strict final sanitization.

Reference runner:

```text
hpc/run_loflex_base_denovo_seed_1gpu.pbs
hpc/run_loflex_base_denovo_constraint_factorial_3seed_1gpu.pbs
```

## Fragment-constrained generation

Five reported task rows, three seeds, ten cases per task and seed, and 100
proposals per case. The final sampler uses recursive learned insertion,
geometry-adaptive rates, atom-token ZINC gap priors, condition repair, and the
validated exploration arm.

Reference runner: `hpc/run_fragment_final_v4_task_1gpu.pbs`.

## Lead optimization

The matrix contains three seeds x five targets x three starting molecules x two
Morgan-similarity thresholds. The frozen v5r protocol performs at most ten
proposal iterations, docks only candidates already satisfying QED, SA, and
similarity constraints, and imposes a hard upper oracle budget of 1,000 per
cell. A terminal `DONE_ZERO` cell is a completed scientific outcome and
contributes zero to the signed score sum.

Reference runner:
`hpc/run_lead_elastic_joint_frontier_v5_feasible_chunk_1gpu.pbs`.

## PMO

The reported mode is `elastic_frontier_prescreen`, evaluated for three seeds
and 23 tasks with 10,000 oracle calls per task. It uses the task-specific ranked
ZINC priors documented in `DATA.md`.

Reference runner:
`hpc/run_elastic_frontier_prescreen_true_fsm_task_1gpu.pbs`.

## Verify the archived evidence

From the extracted Zenodo release root:

```bash
bash verify_release.sh
```

This verifies every archived file against `manifests/SHA256SUMS.txt`, then runs
the benchmark-aware auditor without importing the GPU model stack.

## Portability notes

The PBS files preserve the reported CX3 resource requests and command-line
arguments. Their root paths and notification addresses are placeholders. Set
`ROOT`, checkpoint paths, and scheduler resources for the target cluster.

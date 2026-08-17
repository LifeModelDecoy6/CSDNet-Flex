# CSDNet Python package

This directory contains the model, training objectives, chemical constraints,
samplers, and benchmark adapters used in the final CSDNet evaluation.

## Layout

- `model/`: Transformer backbones, learned insertion modules, and Lightning
  training objectives.
- `util/`: atom/syntax SMILES tokenization, elastic sampling, finite-state
  chemical tracking, checkpoint loading, and shared utilities.
- `config/`: model-size presets and the explicit valence-state table.
- `optim/`: frontier, length, structure, and protected-search policies.
- `exp/denovo/`: de novo generation, sampler profiles, and metric aggregation.
- `exp/frag/`: fragment benchmark construction, learned local insertion, and
  final result aggregation.
- `exp/lead/`: feasibility-first Lead optimization and docking-score audits.
- `exp/pmo/`: Practical Molecular Optimization runners and reporting.
- `tests/`: unit and integration tests for the released implementation.

The authoritative commands, checkpoint placement, fixed seeds, budgets, and
result paths are documented in the repository-level `REPRODUCIBILITY.md` and
the scripts under `hpc/`. Earlier exploratory entry points remain in the
package for provenance, but they are not part of the frozen dissertation
claims in `manifests/final_result_lock.json`.

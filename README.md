# CSDNet

**Chemically Guided Variable-Length Masked Discrete Diffusion for Molecular Generation and Optimization**

CSDNet is a compact atom-tokenized molecular generator developed for Shuo
Cheng's MSc Digital Chemistry dissertation at Imperial College London. A
single 6.74M-parameter checkpoint learns an unconditional molecular prior from
canonical SMILES decoded from SAFE-GPT. At inference, learned insertion,
finite-state chemical tracking, RDKit feedback, and task-specific search adapt
the frozen model to de novo generation, fragment-constrained generation, Lead
optimization, and Practical Molecular Optimization (PMO).

The training checkpoint does not use benchmark property labels, and the
backbone is not updated for downstream tasks.

## Headline results

All values are means over three random seeds.

| Evaluation | Result |
| --- | --- |
| De novo completed output pool | validity 100.00%, total uniqueness 92.90%, quality 82.43%, diversity 0.8132 |
| Raw FSM + RDKit factorial | validity 98.73%, quality 79.43% with both constraints; 89.20% and 75.33% with neither |
| Fragment-constrained generation | validity 93.83%, uniqueness 75.96%, quality 32.85%, diversity 0.5057 |
| Lead optimization | signed docking-score sum -176.2 +/- 3.1 at delta=0.4 and -127.4 +/- 7.8 at delta=0.6 |
| Oracle-prescreened PMO | sum AUC top-10 17.664 +/- 0.201 across 23 tasks |

Compact aggregate tables are in `results_summary/`. The checkpoint, raw
generated molecules, full optimization histories, and integrity manifests are
distributed in the [companion Zenodo release](https://doi.org/10.5281/zenodo.21980242).

## Repository layout

- `CSDNet/`: model, training losses, samplers, chemical constraints, and task runners.
- `data/`: atom-token length and fragment-gap priors derived from ZINC250K.
- `scripts/`: Python utilities and upstream-compatible benchmark interfaces.
- `hpc/`: the exact CX3/PBS protocol scripts, with local paths anonymized.
- `results_summary/`: compact tables used in the dissertation.
- `manifests/`: frozen benchmark definitions and claim-to-result mapping.
- `MODEL_CARD.md`, `DATA.md`, and `REPRODUCIBILITY.md`: model and evidence documentation.

## Installation

Python 3.10.8 was used for the reported runs. Create an isolated environment,
then install this checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Lead docking additionally requires the `obabel` executable. The archived
QuickVina binary targets Linux x86-64; users on other platforms should provide
a compatible QuickVina/AutoDock Vina executable and preserve the reported
docking configuration.

## Checkpoint placement

Download `model/last.ckpt` from the
[companion Zenodo release](https://doi.org/10.5281/zenodo.21980242) and place it at:

```text
checkpoints/csdnet_6m_loflex_geometric_genmol50k/last.ckpt
```

The exact file must have global step 50,000 and SHA-256 recorded in the Zenodo
manifest. The small vocabulary files required to load it are included here.

## Minimal checks

The code-only checks do not require the model checkpoint:

```bash
python -m compileall -q CSDNet
pytest -q CSDNet/tests
```

After downloading the Zenodo archive, verify its checksums and scientific
claims with its root-level `verify_release.sh`.

## Reproducing the evaluations

The commands and exact protocol fingerprints are documented in
`REPRODUCIBILITY.md`. The scripts in `hpc/` retain the final batch sizes,
sampler profiles, budgets, seeds, and stopping rules used on Imperial's CX3
cluster. Scheduler-specific paths and email addresses have been replaced with
portable placeholders only.

## Data and scope

SAFE-GPT and the raw ZINC250K table are external datasets and are not
redistributed. See `DATA.md` for revisions, checksums, and the distinction
between training data, derived structural priors, and oracle-specific PMO
prescreening. This repository is research software and is not a clinical or
experimental activity predictor.

## Licence and attribution

Code and original CSDNet artifacts are released under Apache-2.0. This work
builds on the Apache-2.0 GenMol codebase and includes separately licensed
third-party components. Copyright and modification notices are summarized in
`NOTICE`; complete third-party licence texts are under `LICENSES/third_party/`.

## Citation

Use the metadata in `CITATION.cff` and cite the archived release using
[DOI 10.5281/zenodo.21980242](https://doi.org/10.5281/zenodo.21980242).

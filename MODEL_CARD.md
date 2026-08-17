# Model card

## Model

- Name: CSDNet-Flex 6.74M
- Architecture: bidirectional Transformer masked language model with learned
  insertion timing and atom/syntax-level SMILES tokens
- Parameters: 6.74 million trainable parameters
- Vocabulary: 344 tokens
- Maximum sequence capacity: 256 positions including boundary tokens
- Position encoding: rotary
- Checkpoint: EMA weights at optimizer step 50,000

The exact hyperparameters are recorded in the companion Zenodo file
`model/checkpoint_metadata.json`.

## Training

The model was trained from random initialization for 50,000 optimizer steps on
canonical SMILES decoded from the SAFE-GPT training split. With a global batch
of 2,048, this corresponds to 102,400,000 molecular examples presented to the
optimizer. The stream may revisit molecules; this number is exposure, not a
claim of 102.4 million unique structures.

The objective combines masked token reconstruction, learned insertion timing,
four equally weighted fragment-corruption geometries within a 15% fragment
branch, 10% standard masked-diffusion corruption, 5% visible-token refinement,
PAPL, and normalized aromatic loss allocation. No molecular-property labels or
benchmark scores are used in backbone training.

## Intended use

Research on molecular generation, constrained molecular editing, and
budget-aware optimization. Generated molecules require independent chemical,
synthetic, safety, and experimental assessment.

## Limitations

- SMILES validity is improved but not guaranteed by the raw model alone.
- Fragment-constrained diversity is lower than the strongest reported methods.
- High-similarity Lead editing remains more difficult than delta=0.4 editing.
- PMO results use an explicitly oracle-prescreened ZINC prior and should only be
  compared with methods using the same benchmark setting.
- Docking and PMO oracles are computational proxies, not experimental evidence.

## Evaluation protocol

All headline results use the same frozen pre-ZINC-fine-tuning checkpoint. See
`REPRODUCIBILITY.md` and `manifests/final_result_lock.json`.


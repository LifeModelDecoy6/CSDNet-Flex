# Data provenance

## SAFE-GPT training data

- Dataset: `datamol-io/safe-gpt`
- Split: `train`
- Access: Hugging Face streaming
- Revision observed in the final run:
  `16d0be9ad6177ae683a32a86204530e8ee624a0f`
- Processing: SAFE records were decoded, canonicalized as SMILES, and tokenized
  with the archived atom/syntax vocabulary.

The external parquet shards are not redistributed. Obtain the dataset from its
provider and follow its licence. The frozen checkpoint, rather than a mutable
remote stream, is authoritative for the reported evaluation.

## ZINC250K-derived priors

The raw ZINC250K CSV is not redistributed. The accepted local source contained
249,455 rows. The following source fingerprints were recorded independently:

- SHA-256: `35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae`
- MD5: `b59078b2b04c6e9431280e3dc42048d5`
- Atom-token vocabulary SHA-256:
  `6ba2ea505c17bf8662c34bd145cd5de3e85ae2be9559ad29b2095f3ff468cd30`

The derived files in `data/` use the same CSDNet atom/syntax-token units as the
model. They guide inference only and do not update backbone parameters.

## PMO oracle-specific priors

`CSDNet/exp/pmo/vocab/` contains one ZINC-derived ranked prior for each of the
23 PMO tasks. These files are used only by the explicitly prescreened PMO
protocol. They make that protocol different from an optimizer that has no
oracle-specific prior, which must be stated in every comparison.

## Fragment and Lead benchmark assets

- `data/fragments.csv` contains the fragment-constrained benchmark inputs.
- `scripts/exps/lead/docking/` contains the five receptor files, starting
  molecules, docking wrapper, and Linux QuickVina executable inherited from the
  GenMol benchmark interface.

Runtime docking intermediates have been removed.


import json

import pytest

from CSDNet.util.length_prior import (
    ATOMIC_LENGTH_PRIOR_SCHEMA,
    load_atomic_length_prior,
)


def test_load_atomic_length_prior_validates_tokenizer_metadata(tmp_path):
    path = tmp_path / "prior.json"
    path.write_text(
        json.dumps(
            {
                "schema": ATOMIC_LENGTH_PRIOR_SCHEMA,
                "tokenizer": "csdnet_atomic_smiles",
                "include_special_tokens": True,
                "lengths": [8, 9, 9, 12],
            }
        )
    )
    lengths, metadata = load_atomic_length_prior(path, max_len=12)
    assert lengths == [8, 9, 9, 12]
    assert metadata["histogram"] == {"8": 1, "9": 2, "12": 1}


def test_load_atomic_length_prior_rejects_fragment_token_lengths(tmp_path):
    path = tmp_path / "prior.json"
    path.write_text(json.dumps({"lengths": [10, 11]}))
    with pytest.raises(ValueError, match="GenMol data/len.pk"):
        load_atomic_length_prior(path)

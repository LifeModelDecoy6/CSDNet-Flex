import torch

from CSDNet.util.data import make_collate_fn


class DummyTokenizer:
    pad_id = 0


def _batch():
    return [
        {
            "input_ids": torch.tensor([2, 10, 11, 3, 0, 0, 0, 0]),
            "cond": torch.empty(0),
            "aromatic_mask": torch.tensor(
                [False, True, True, False, False, False, False, False]
            ),
        },
        {
            "input_ids": torch.tensor([2, 12, 13, 14, 15, 3, 0, 0]),
            "cond": torch.empty(0),
            "aromatic_mask": torch.tensor(
                [False, False, True, True, False, False, False, False]
            ),
        },
    ]


def test_dynamic_padding_uses_exact_batch_maximum():
    collate = make_collate_fn(
        DummyTokenizer(),
        include_cond=True,
        use_aromatic_cbi=True,
        dynamic_padding=True,
        pad_to_multiple_of=1,
    )

    result = collate(_batch())

    assert result["input_ids"].shape == (2, 6)
    assert result["attention_mask"].shape == (2, 6)
    assert result["aromatic_mask"].shape == (2, 6)
    assert result["attention_mask"].sum(dim=1).tolist() == [4, 6]


def test_static_padding_preserves_global_width():
    collate = make_collate_fn(
        DummyTokenizer(),
        include_cond=True,
        use_aromatic_cbi=True,
    )

    result = collate(_batch())

    assert result["input_ids"].shape == (2, 8)
    assert result["aromatic_mask"].shape == (2, 8)


def test_dynamic_padding_can_round_up_when_requested():
    collate = make_collate_fn(
        DummyTokenizer(),
        include_cond=False,
        dynamic_padding=True,
        pad_to_multiple_of=4,
    )

    result = collate(_batch())

    assert result["input_ids"].shape == (2, 8)

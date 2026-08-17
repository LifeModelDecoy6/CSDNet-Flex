import torch

from CSDNet.model.loss_weighting import redistribute_priority_with_fixed_mass


def _valid_mass(weights, valid):
    return (weights * valid.to(weights.dtype)).sum(dim=-1)


def test_normalized_priority_preserves_each_sample_weight_mass():
    base = torch.tensor(
        [
            [1.0, 2.0, 4.0, 8.0],
            [3.0, 1.0, 5.0, 7.0],
        ]
    )
    relative = torch.tensor(
        [
            [1.0, 1.5, 1.5, 1.0],
            [3.0, 1.0, 1.5, 1.0],
        ]
    )
    valid = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
        ]
    )

    output = redistribute_priority_with_fixed_mass(base, relative, valid)

    torch.testing.assert_close(
        _valid_mass(output, valid),
        _valid_mass(base, valid),
    )


def test_normalized_priority_increases_priority_share_for_mixed_sample():
    base = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    priority = torch.tensor([[False, True, True, False]])
    relative = torch.where(priority, torch.tensor(1.5), torch.tensor(1.0))
    valid = torch.ones_like(priority)

    output = redistribute_priority_with_fixed_mass(base, relative, valid)
    base_share = base[priority].sum() / base.sum()
    output_share = output[priority].sum() / output.sum()

    assert output_share > base_share


def test_combined_cbi_and_arocbi_preserve_mass_and_relative_preferences():
    base = torch.ones(1, 4)
    # aromatic scaffold, non-aromatic scaffold, ordinary token, ordinary token
    combined_multiplier = torch.tensor([[3.0, 2.0, 1.0, 1.0]])
    valid = torch.ones_like(base, dtype=torch.bool)

    output = redistribute_priority_with_fixed_mass(
        base,
        combined_multiplier,
        valid,
    )

    torch.testing.assert_close(output.sum(), base.sum())
    torch.testing.assert_close(output[0, 0] / output[0, 1], torch.tensor(1.5))
    torch.testing.assert_close(output[0, 1] / output[0, 2], torch.tensor(2.0))
    torch.testing.assert_close(output[0, 0] / output[0, 2], torch.tensor(3.0))


def test_normalized_priority_leaves_uniform_samples_unchanged():
    base = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    relative = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.5, 1.5, 1.5],
        ]
    )
    valid = torch.ones_like(base, dtype=torch.bool)

    output = redistribute_priority_with_fixed_mass(base, relative, valid)

    torch.testing.assert_close(output, base)


def test_normalized_priority_ignores_invalid_positions_when_scaling():
    base = torch.tensor([[1.0, 2.0, 100.0]])
    relative = torch.tensor([[1.0, 1.5, 10.0]])
    valid = torch.tensor([[True, True, False]])

    output = redistribute_priority_with_fixed_mass(base, relative, valid)

    torch.testing.assert_close(
        _valid_mass(output, valid),
        _valid_mass(base, valid),
    )
    expected_scale = 3.0 / 4.0
    torch.testing.assert_close(output[0, :2], torch.tensor([0.75, 2.25]))
    assert output[0, 2].item() == 100.0 * 10.0 * expected_scale

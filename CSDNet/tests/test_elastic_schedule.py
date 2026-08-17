import torch

from CSDNet.model.elastic_schedule import (
    ElasticKumaSchedule,
    bregman_poisson,
    sample_variable_length_state,
)


def test_kuma_cdf_and_inverse_are_consistent():
    schedule = ElasticKumaSchedule(shape_a=2.0)
    time = torch.tensor([[0.05, 0.25, 0.75, 0.80]])
    rate = torch.tensor([[0.1, 0.5, 2.0, 10.0]])

    recovered = schedule.inverse_cdf(schedule.cdf(time, rate), rate)

    assert torch.allclose(recovered, time, atol=2e-5, rtol=2e-5)


def test_kuma_state_probabilities_form_a_distribution():
    schedule = ElasticKumaSchedule(shape_a=2.0)
    time = torch.tensor([0.01, 0.2, 0.5, 0.9, 0.99]).unsqueeze(-1)
    insertion_rate = torch.tensor([[0.05, 0.5, 1.0, 5.0]])
    unmask_rate = torch.tensor([[0.1, 0.75, 2.0, 10.0]])

    log_probabilities = schedule.state_log_probabilities(
        time,
        insertion_rate,
        unmask_rate,
    )
    probabilities = torch.stack(log_probabilities, dim=-1).exp()

    assert torch.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones_like(probabilities[..., 0]),
        atol=5e-5,
        rtol=5e-5,
    )


def test_truncated_unmask_event_never_precedes_insertion():
    schedule = ElasticKumaSchedule(shape_a=2.0)
    torch.manual_seed(11)
    rate = torch.linspace(0.001, 20.0, 4096)

    insertion_time = schedule.sample_event_time(rate)
    insertion_time[-512:] = torch.linspace(0.90, 0.999, 512)
    unmask_time = schedule.sample_truncated_event_time(
        insertion_time,
        rate,
    )

    assert torch.all(unmask_time >= insertion_time)


def test_deleted_tokens_are_assigned_to_the_retained_right_anchor():
    clean_ids = torch.tensor([[1, 10, 11, 2, 0]])
    fixed = torch.tensor([[True, False, False, True, True]])
    insertion_times = torch.tensor([[0.0, 0.8, 0.2, 0.0, 0.0]])
    unmask_times = torch.tensor([[0.0, 0.9, 0.3, 0.0, 0.0]])
    insertion_hazard = torch.tensor([[0.0, 2.5, 3.5, 0.0, 0.0]])

    state = sample_variable_length_state(
        clean_ids=clean_ids,
        t=torch.tensor([0.5]),
        insertion_times=insertion_times,
        unmask_times=unmask_times,
        insertion_hazard=insertion_hazard,
        fixed=fixed,
        mask_id=3,
        pad_id=0,
    )

    assert state["input_ids"].tolist() == [[1, 11, 2, 0, 0]]
    assert state["source_positions"][0, :3].tolist() == [0, 2, 3]
    assert state["gap_sizes"][0, :3].tolist() == [0, 1, 0]
    assert state["gap_rate_target"][0, :3].tolist() == [0.0, 2.5, 0.0]


def test_poisson_bregman_is_nonnegative_and_zero_at_match():
    target = torch.tensor([0.1, 1.0, 10.0])
    matched = bregman_poisson(target, target)
    mismatched = bregman_poisson(target, target * 1.7)

    assert torch.allclose(matched, torch.zeros_like(matched), atol=1e-7)
    assert (mismatched > 0).all()


def test_poisson_bregman_handles_zero_target_exactly():
    predicted = torch.tensor([0.1, 1.0, 10.0])
    divergence = bregman_poisson(torch.zeros_like(predicted), predicted)

    assert torch.equal(divergence, predicted)


def test_loflex_a1_fixed_unmask_is_uniform_and_regularizer_free():
    schedule = ElasticKumaSchedule(
        shape_a=1.0,
        regularizer_mode="loflex",
    )
    time = torch.tensor([0.1, 0.4, 0.9]).unsqueeze(-1)
    unit_rate = torch.ones(3, 4)

    assert torch.allclose(schedule.cdf(time, unit_rate), time.expand_as(unit_rate))
    assert torch.allclose(
        schedule.hazard(time, unit_rate),
        1.0 / (1.0 - time).expand_as(unit_rate),
    )

    active = torch.ones_like(unit_rate, dtype=torch.bool)
    regularizer = schedule.regularizer(unit_rate, unit_rate, active)
    assert torch.allclose(regularizer, torch.zeros_like(regularizer), atol=1e-6)

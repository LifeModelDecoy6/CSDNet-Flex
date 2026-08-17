import torch
import torch.nn.functional as F


def bregman_poisson(target_rate, predicted_rate, eps=1e-8):
    target = target_rate.float().clamp_min(0.0)
    predicted = predicted_rate.float().clamp_min(eps)
    relative_entropy = torch.where(
        target > 0,
        target * torch.log(target.clamp_min(eps) / predicted),
        torch.zeros_like(target),
    )
    return predicted - target + relative_entropy


class ElasticKumaSchedule:
    """Simplified Kumaraswamy insertion and unmasking schedule."""

    def __init__(
        self,
        shape_a=2.0,
        eps=1e-5,
        regularizer_mode="legacy",
        regularizer_grid_points=100,
        boundary_epsilon=0.01,
        boundary_delta=0.01,
    ):
        self.shape_a = float(shape_a)
        self.eps = float(eps)
        self.regularizer_mode = str(regularizer_mode)
        if self.regularizer_mode not in {"legacy", "loflex"}:
            raise ValueError(
                "regularizer_mode must be 'legacy' or 'loflex'."
            )
        self.regularizer_grid_points = int(regularizer_grid_points)
        self.boundary_epsilon = float(boundary_epsilon)
        self.boundary_delta = float(boundary_delta)
        if self.regularizer_grid_points < 2:
            raise ValueError("regularizer_grid_points must be at least two.")

    def sample_time(self, batch_size, device, antithetic=True):
        if antithetic:
            offset = torch.rand((), device=device)
            t = (
                torch.arange(batch_size, device=device, dtype=torch.float32)
                + offset
            ) / max(batch_size, 1)
            t = torch.remainder(t, 1.0)
        else:
            t = torch.rand(batch_size, device=device)
        return t.clamp(self.eps, 1.0 - self.eps)

    def cdf(self, t, rate):
        t = t.float().clamp(self.eps, 1.0 - self.eps)
        rate = rate.float().clamp_min(self.eps)
        log_survival = rate * torch.log1p(
            -t.pow(self.shape_a).clamp(max=1.0 - self.eps)
        )
        return -torch.expm1(log_survival)

    def inverse_cdf(self, probability, rate):
        probability = probability.float().clamp(self.eps, 1.0 - self.eps)
        rate = rate.float().clamp_min(self.eps)
        base = -torch.expm1(torch.log1p(-probability) / rate)
        return base.clamp_min(self.eps).pow(1.0 / self.shape_a).clamp(
            self.eps,
            1.0 - self.eps,
        )

    def sample_event_time(self, rate):
        uniform = torch.rand_like(rate.float()).clamp(self.eps, 1.0 - self.eps)
        return self.inverse_cdf(uniform, rate)

    def sample_truncated_event_time(self, lower, rate):
        lower = lower.float().clamp(self.eps, 1.0 - self.eps)
        rate = rate.float().clamp_min(self.eps)
        log_survival_at_lower = rate * torch.log1p(
            -lower.pow(self.shape_a).clamp(max=1.0 - self.eps)
        )
        uniform = torch.rand_like(rate).clamp(self.eps, 1.0 - self.eps)
        conditional_log_survival = log_survival_at_lower + torch.log(uniform)
        base = -torch.expm1(conditional_log_survival / rate)
        event_time = base.clamp_min(self.eps).pow(1.0 / self.shape_a).clamp(
            self.eps,
            1.0 - self.eps,
        )
        return torch.maximum(event_time, lower)

    def hazard(self, t, rate):
        t = t.float().clamp(self.eps, 1.0 - self.eps)
        rate = rate.float().clamp_min(self.eps)
        denominator = (1.0 - t.pow(self.shape_a)).clamp_min(self.eps)
        return (
            self.shape_a
            * rate
            * t.pow(self.shape_a - 1.0)
            / denominator
        )

    def state_log_probabilities(self, t, insertion_rate, unmask_rate):
        t = t.float()
        while t.ndim < insertion_rate.ndim:
            t = t.unsqueeze(-1)
        t = t.clamp(self.eps, 1.0 - self.eps)
        insertion_rate = insertion_rate.float().clamp_min(self.eps)
        unmask_rate = unmask_rate.float().clamp_min(self.eps)

        log_survival = torch.log1p(
            -t.pow(self.shape_a).clamp(max=1.0 - self.eps)
        )
        log_dropped = insertion_rate * log_survival

        difference = insertion_rate - unmask_rate
        x = difference * log_survival
        abs_difference = difference.abs().clamp_min(self.eps)
        positive_inner = torch.log(
            (-torch.expm1(x.clamp(max=-self.eps))).clamp_min(self.eps)
        )
        negative_inner = (
            x
            + torch.log(
                (-torch.expm1((-x).clamp(max=-self.eps))).clamp_min(self.eps)
            )
        )
        log_inner = torch.where(difference > 0, positive_inner, negative_inner)
        general = (
            torch.log(insertion_rate)
            - torch.log(abs_difference)
            + log_inner
        )
        equal_limit = torch.log(insertion_rate) + torch.log(
            (-log_survival).clamp_min(self.eps)
        )
        integral = torch.where(difference.abs() < 1e-5, equal_limit, general)
        log_masked = (unmask_rate * log_survival + integral).clamp(max=0.0)

        occupied = torch.logaddexp(log_dropped, log_masked).clamp(
            max=-self.eps
        )
        log_unmasked = torch.log((-torch.expm1(occupied)).clamp_min(self.eps))
        return log_dropped, log_masked, log_unmasked

    def regularizer(self, insertion_rate, unmask_rate, active_mask):
        if self.regularizer_mode == "loflex":
            return self._loflex_regularizer(
                insertion_rate,
                unmask_rate,
                active_mask,
            )

        active = active_mask.bool()
        if not active.any():
            return insertion_rate.new_zeros(insertion_rate.size(0))

        grid = torch.linspace(
            0.02,
            0.98,
            32,
            device=insertion_rate.device,
            dtype=torch.float32,
        ).view(32, 1, 1)
        active_f = active.float().unsqueeze(0)
        denominator = active_f.sum(dim=-1).clamp(min=1.0)
        losses = []
        rates = [insertion_rate]
        if unmask_rate is not None:
            rates.append(unmask_rate)
        for rate in rates:
            mixture = (
                self.cdf(grid, rate.float().unsqueeze(0)) * active_f
            ).sum(dim=-1) / denominator
            losses.append((mixture - grid.squeeze(-1)).pow(2).mean(dim=0))
        return torch.stack(losses, dim=0).sum(dim=0)

    def _loflex_regularizer(
        self,
        insertion_rate,
        unmask_rate,
        active_mask,
    ):
        """Mixture-CDF and endpoint regularizer used by LoFlexMDM."""
        active = active_mask.bool()
        if not active.any():
            return insertion_rate.new_zeros(insertion_rate.size(0))

        grid = torch.linspace(
            self.eps,
            1.0 - self.eps,
            self.regularizer_grid_points,
            device=insertion_rate.device,
            dtype=torch.float32,
        ).view(self.regularizer_grid_points, 1, 1)
        active_f = active.float().unsqueeze(0)
        denominator = active_f.sum(dim=-1).clamp(min=1.0)
        epsilon = torch.tensor(
            self.boundary_epsilon,
            device=insertion_rate.device,
            dtype=torch.float32,
        )
        delta = torch.tensor(
            self.boundary_delta,
            device=insertion_rate.device,
            dtype=torch.float32,
        )

        losses = []
        rates = [insertion_rate]
        if unmask_rate is not None:
            rates.append(unmask_rate)
        for rate in rates:
            expanded_rate = rate.float().unsqueeze(0)
            mixture = (
                self.cdf(grid, expanded_rate) * active_f
            ).sum(dim=-1) / denominator
            mixture_loss = (mixture - grid.squeeze(-1)).pow(2).sum(dim=0)

            lower = torch.relu(self.cdf(epsilon, rate.float()) - delta).pow(2)
            upper = torch.relu(
                1.0 - self.cdf(1.0 - epsilon, rate.float()) - delta
            ).pow(2)
            boundary_loss = ((lower + upper) * active.float()).sum(dim=-1)
            losses.append(mixture_loss + boundary_loss)
        return torch.stack(losses, dim=0).sum(dim=0)


def sample_variable_length_state(
    clean_ids,
    t,
    insertion_times,
    unmask_times,
    insertion_hazard,
    fixed,
    mask_id,
    pad_id,
):
    """Delete, mask, and compact a clean sequence at diffusion time t."""
    batch_size, max_length = clean_ids.shape
    t_expanded = t.unsqueeze(-1)
    insertion_times = torch.where(fixed, torch.zeros_like(insertion_times), insertion_times)
    unmask_times = torch.where(fixed, torch.zeros_like(unmask_times), unmask_times)

    deleted = t_expanded < insertion_times
    masked = (t_expanded >= insertion_times) & (t_expanded < unmask_times)
    deleted = deleted & ~fixed
    masked = masked & ~fixed

    noisy = clean_ids.clone()
    noisy[deleted] = pad_id
    noisy[masked] = mask_id

    source_positions = noisy.ne(pad_id).argsort(
        dim=1,
        descending=True,
        stable=True,
    )
    noisy = torch.gather(noisy, 1, source_positions)
    source_positions = source_positions.clone()
    source_positions[noisy == pad_id] = 0
    noisy_length = noisy.ne(pad_id).sum(dim=1)

    previous = F.pad(source_positions[:, :-1], (1, 0), value=-1)
    gap_sizes = (source_positions - previous - 1).clamp(min=0)
    gap_index = torch.arange(max_length, device=clean_ids.device).unsqueeze(0)
    gap_mask = gap_index < noisy_length.unsqueeze(1)
    gap_sizes = gap_sizes.masked_fill(~gap_mask, 0)

    deleted_hazard = insertion_hazard * deleted.float() * (~fixed).float()
    cumulative = F.pad(torch.cumsum(deleted_hazard, dim=1), (1, 0), value=0.0)
    end_index = source_positions.clamp(min=0, max=max_length)
    start_index = (previous + 1).clamp(min=0, max=max_length)
    gap_rate_target = (
        torch.gather(cumulative, 1, end_index)
        - torch.gather(cumulative, 1, start_index)
    )
    gap_rate_target = gap_rate_target * gap_mask.float()

    return {
        "input_ids": noisy,
        "source_positions": source_positions,
        "gap_sizes": gap_sizes,
        "gap_mask": gap_mask,
        "gap_rate_target": gap_rate_target,
        "deleted": deleted,
        "masked": masked,
    }


def apply_structured_span_mask(
    state,
    fixed,
    mask_id,
    pad_id,
    selected_samples,
    min_span=2,
    max_span=8,
):
    """Add a small task-agnostic contiguous-mask curriculum to a state."""
    if min_span < 1 or max_span < min_span:
        raise ValueError("Structured span bounds must satisfy 1 <= min <= max.")

    output = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in state.items()
    }
    noisy = output["input_ids"]
    source_positions = output["source_positions"]
    selected_samples = selected_samples.to(
        device=noisy.device,
        dtype=torch.bool,
    )
    applied = torch.zeros_like(selected_samples)

    source_fixed = torch.gather(fixed, 1, source_positions)
    eligible = noisy.ne(pad_id) & noisy.ne(mask_id) & ~source_fixed
    for batch_index_tensor in selected_samples.nonzero(
        as_tuple=False
    ).flatten():
        batch_index = int(batch_index_tensor.item())
        positions = eligible[batch_index].nonzero(as_tuple=False).flatten()
        if positions.numel() < min_span:
            continue

        runs = []
        run_start = 0
        for offset in range(1, positions.numel() + 1):
            at_end = offset == positions.numel()
            if (
                not at_end
                and positions[offset] == positions[offset - 1] + 1
            ):
                continue
            run = positions[run_start:offset]
            if run.numel() >= min_span:
                runs.append(run)
            run_start = offset
        if not runs:
            continue

        run_index = int(
            torch.randint(len(runs), (1,), device=noisy.device).item()
        )
        run = runs[run_index]
        upper = min(max_span, int(run.numel()))
        span_length = int(
            torch.randint(
                min_span,
                upper + 1,
                (1,),
                device=noisy.device,
            ).item()
        )
        start_offset = int(
            torch.randint(
                int(run.numel()) - span_length + 1,
                (1,),
                device=noisy.device,
            ).item()
        )
        compact_positions = run[start_offset:start_offset + span_length]
        original_positions = source_positions[batch_index, compact_positions]

        noisy[batch_index, compact_positions] = mask_id
        output["masked"][batch_index, original_positions] = True
        output["deleted"][batch_index, original_positions] = False
        applied[batch_index] = True

    return output, applied

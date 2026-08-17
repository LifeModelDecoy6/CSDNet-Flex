import numpy as np
import torch

from CSDNet.util.sampling import _block_refine_tokens, sample_csdnet
from CSDNet.util.tokenizer import SMILESTokenizer


class CarbonModel(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask):
        del attention_mask
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.tokenizer.vocab_size),
            -20.0,
            device=input_ids.device,
        )
        logits[:, :, self.tokenizer.vocab["C"]] = 20.0
        return logits


class CarbonRefinementModel(CarbonModel):
    corruption_level_conditioning = True


class AlternatingModel(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        del attention_mask
        self.calls += 1
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.tokenizer.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        token = "C" if self.calls == 1 else "O"
        logits[:, :, self.tokenizer.vocab[token]] = 100.0
        return logits


def test_length_adaptive_sampler_runs_per_sequence_tensor_schedules():
    tokenizer = SMILESTokenizer(
        ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>", "C", "O"]
    )
    model = CarbonModel(tokenizer)
    np.random.seed(3)
    torch.manual_seed(3)

    generated, diagnostics = sample_csdnet(
        model=model,
        tk=tokenizer,
        ref_lengths=[5, 9],
        n_mol=8,
        device="cpu",
        batch_size=8,
        n_steps=3,
        use_fsm_check=False,
        use_rdkit_kekulize_check=False,
        length_batching="sorted",
        length_adaptive=True,
        adaptive_length_low=5.0,
        adaptive_length_high=9.0,
        temperature_start=1.2,
        temperature_end=0.15,
        temperature_power=1.5,
        gumbel_scale=0.65,
        remask_power=1.35,
        adaptive_temperature_start_short=1.8,
        adaptive_temperature_end_short=0.35,
        adaptive_temperature_power_short=1.25,
        adaptive_gumbel_scale_short=1.35,
        adaptive_remask_power_short=0.8,
        return_diagnostics=True,
    )

    assert len(generated) == 8
    assert set(generated).issubset({"CCC", "CCCCCCC"})
    assert diagnostics["length_adaptive"] is True
    assert diagnostics["adaptive_length_low"] == 5.0
    assert diagnostics["adaptive_length_high"] == 9.0
    assert sum(diagnostics["sampled_length_histogram"].values()) == 8


def test_progressive_commit_preserves_a_confirmed_token():
    tokenizer = SMILESTokenizer(
        ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>", "C", "O"]
    )

    def generate(progressive_commit):
        model = AlternatingModel(tokenizer)
        output, diagnostics = sample_csdnet(
            model=model,
            tk=tokenizer,
            ref_lengths=[3],
            n_mol=1,
            device="cpu",
            batch_size=1,
            n_steps=2,
            use_fsm_check=False,
            use_rdkit_kekulize_check=False,
            gumbel_scale=0.0,
            confidence_temperature=1.0,
            progressive_commit=progressive_commit,
            return_diagnostics=True,
        )
        return output[0], diagnostics

    resampled, old_diagnostics = generate(False)
    committed, progressive_diagnostics = generate(True)

    assert resampled == "O"
    assert committed == "C"
    assert old_diagnostics["progressive_commit"] is False
    assert progressive_diagnostics["progressive_commit"] is True


def test_progressive_refresh_activates_only_on_committed_positions():
    tokenizer = SMILESTokenizer(
        ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>", "C", "O"]
    )
    model = CarbonRefinementModel(tokenizer)
    output, diagnostics = sample_csdnet(
        model=model,
        tk=tokenizer,
        ref_lengths=[5],
        n_mol=1,
        device="cpu",
        batch_size=1,
        n_steps=4,
        use_fsm_check=False,
        use_rdkit_kekulize_check=False,
        gumbel_scale=0.0,
        progressive_commit=True,
        progressive_refresh_confidence=True,
        progressive_refresh_start=0.5,
        return_diagnostics=True,
    )

    assert output == ["CCC"]
    assert diagnostics["progressive_refresh"]["enabled"] is True
    assert diagnostics["progressive_refresh"]["steps"] > 0
    assert diagnostics["progressive_refresh"]["positions"] > 0


def test_length_adaptive_confidence_runs_without_changing_length_protocol():
    tokenizer = SMILESTokenizer(
        ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>", "C", "O"]
    )
    model = CarbonModel(tokenizer)
    output, diagnostics = sample_csdnet(
        model=model,
        tk=tokenizer,
        ref_lengths=[5, 9],
        n_mol=4,
        device="cpu",
        batch_size=4,
        n_steps=3,
        use_fsm_check=False,
        use_rdkit_kekulize_check=False,
        progressive_commit=True,
        confidence_length_adaptive=True,
        adaptive_confidence_length_low=5.0,
        adaptive_confidence_length_high=9.0,
        adaptive_confidence_temperature_short=1.0,
        return_diagnostics=True,
    )

    assert len(output) == 4
    assert diagnostics["confidence_length_adaptive"] is True
    assert diagnostics["adaptive_confidence_length_low"] == 5.0
    assert diagnostics["adaptive_confidence_length_high"] == 9.0


def test_block_refinement_accepts_a_higher_likelihood_span():
    tokenizer = SMILESTokenizer(
        ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>", "C", "O"]
    )
    model = AlternatingModel(tokenizer)
    model.calls = 1
    sequence = torch.tensor(
        [[tokenizer.bos_id, tokenizer.vocab["C"], tokenizer.eos_id]],
        dtype=torch.long,
    )
    non_special = torch.tensor([[False, True, False]])
    scores = torch.tensor([[0.0, -10.0, 0.0]])

    refined, refined_scores, diagnostics = _block_refine_tokens(
        model=model,
        x=sequence,
        output_scores=scores,
        non_special=non_special,
        tk=tokenizer,
        steps=1,
        span_max=1,
        candidates=1,
        temperature=1.0,
        fsm_tracker=None,
    )

    assert refined[0, 1].item() == tokenizer.vocab["O"]
    assert refined_scores[0, 1].item() > scores[0, 1].item()
    assert diagnostics["accepted_rows"] == 1

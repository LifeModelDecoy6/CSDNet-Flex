import pytest
import torch

from CSDNet.util.sampling import _all_position_refine_tokens
from CSDNet.util.tokenizer import SMILESTokenizer


class AllPositionRepairModel(torch.nn.Module):
    corruption_level_conditioning = True

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.levels = []

    def forward(self, input_ids, attention_mask, corruption_level=None):
        del attention_mask
        self.levels.append(corruption_level.detach().cpu())
        logits = torch.full(
            (*input_ids.shape, self.tokenizer.vocab_size),
            -10.0,
            device=input_ids.device,
        )
        logits[:, :, self.tokenizer.vocab["O"]] = 0.0
        logits[:, :, self.tokenizer.vocab["C"]] = 5.0
        return logits


class UnconditionedRepairModel(AllPositionRepairModel):
    corruption_level_conditioning = False


class MaskVerifierRejectsModel(AllPositionRepairModel):
    def forward(self, input_ids, attention_mask, corruption_level=None):
        logits = super().forward(
            input_ids,
            attention_mask,
            corruption_level=corruption_level,
        )
        masked = input_ids.eq(self.tokenizer.mask_id)
        logits[:, :, self.tokenizer.vocab["O"]] = torch.where(
            masked,
            torch.full_like(logits[:, :, 0], 8.0),
            logits[:, :, self.tokenizer.vocab["O"]],
        )
        logits[:, :, self.tokenizer.vocab["C"]] = torch.where(
            masked,
            torch.full_like(logits[:, :, 0], -2.0),
            logits[:, :, self.tokenizer.vocab["C"]],
        )
        return logits


def make_tokenizer():
    return SMILESTokenizer(
        ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>", "C", "O"]
    )


def test_all_position_refinement_repairs_visible_low_likelihood_token():
    tokenizer = make_tokenizer()
    model = AllPositionRepairModel(tokenizer)
    tokens = torch.tensor(
        [[
            tokenizer.bos_id,
            tokenizer.vocab["O"],
            tokenizer.vocab["C"],
            tokenizer.eos_id,
        ]]
    )
    non_special = torch.tensor([[False, True, True, False]])

    refined, _, diagnostics = _all_position_refine_tokens(
        model=model,
        x=tokens,
        output_scores=torch.zeros_like(tokens, dtype=torch.float),
        non_special=non_special,
        tk=tokenizer,
        steps=3,
        corruption_start=0.25,
        corruption_end=0.05,
        max_edits=1,
        min_logprob_gain=0.05,
    )

    assert tokenizer.decode(refined[0].tolist()) == "CC"
    assert diagnostics["accepted_edits"] == 1
    assert diagnostics["converged_early"] is True
    assert model.levels[0].item() == pytest.approx(0.25)


def test_all_position_refinement_requires_conditioned_checkpoint():
    tokenizer = make_tokenizer()
    model = UnconditionedRepairModel(tokenizer)
    tokens = torch.tensor(
        [[tokenizer.bos_id, tokenizer.vocab["O"], tokenizer.eos_id]]
    )

    with pytest.raises(ValueError, match="refinement-trained checkpoint"):
        _all_position_refine_tokens(
            model=model,
            x=tokens,
            output_scores=torch.zeros_like(tokens, dtype=torch.float),
            non_special=torch.tensor([[False, True, False]]),
            tk=tokenizer,
            steps=1,
        )


def test_masked_verification_rejects_unsupported_visible_proposal():
    tokenizer = make_tokenizer()
    model = MaskVerifierRejectsModel(tokenizer)
    tokens = torch.tensor(
        [[tokenizer.bos_id, tokenizer.vocab["O"], tokenizer.eos_id]]
    )

    refined, _, diagnostics = _all_position_refine_tokens(
        model=model,
        x=tokens,
        output_scores=torch.zeros_like(tokens, dtype=torch.float),
        non_special=torch.tensor([[False, True, False]]),
        tk=tokenizer,
        steps=2,
        max_edits=1,
        verify_masked=True,
        verify_min_logprob_gain=0.25,
        prevent_revisit=True,
        patience=1,
    )

    assert torch.equal(refined, tokens)
    assert diagnostics["accepted_edits"] == 0
    assert diagnostics["verification_rejected_rows"] == 1


def test_all_position_refinement_respects_per_molecule_edit_cap():
    tokenizer = make_tokenizer()
    model = AllPositionRepairModel(tokenizer)
    tokens = torch.tensor(
        [[
            tokenizer.bos_id,
            tokenizer.vocab["O"],
            tokenizer.vocab["O"],
            tokenizer.eos_id,
        ]]
    )

    refined, _, diagnostics = _all_position_refine_tokens(
        model=model,
        x=tokens,
        output_scores=torch.zeros_like(tokens, dtype=torch.float),
        non_special=torch.tensor([[False, True, True, False]]),
        tk=tokenizer,
        steps=4,
        max_edits=1,
        max_total_edits=1,
        prevent_revisit=True,
    )

    decoded = tokenizer.decode(refined[0].tolist())
    assert decoded in {"CO", "OC"}
    assert diagnostics["accepted_edits"] == 1
    assert diagnostics["max_total_edits"] == 1
    assert diagnostics["rows_at_edit_cap"] == 1

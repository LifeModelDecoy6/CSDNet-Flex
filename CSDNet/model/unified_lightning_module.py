import lightning as L
import torch
import torch.nn.functional as F

from CSDNet.model.lightning_module import SimpleEMA
from CSDNet.model.unified_backbone import UnifiedCSDNetBackbone
from CSDNet.model.unified_corruption import MODE_NAMES


class UnifiedCSDNetLightningModule(L.LightningModule):
    """Joint masked, refinement, infill, insertion and deletion training."""

    def __init__(
        self,
        vocab_size,
        pad_id,
        mask_id,
        bos_id,
        eos_id,
        unk_id,
        aromatic_ids=None,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate=3072,
        max_position_embeddings=512,
        max_gap_count=8,
        position_embedding_type="rotary",
        gradient_checkpointing=True,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        lr=3e-4,
        warmup_steps=2500,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        ema_decay=0.9999,
        use_ema=True,
        aromatic_weight=1.2,
        aromatic_anneal_steps=25000,
        curriculum_steps=10000,
        gap_loss_weight=0.45,
        gap_ordinal_weight=0.10,
        delete_loss_weight=0.30,
        confidence_loss_weight=0.05,
        initial_mode_weights=(0.55, 0.20, 0.15, 0.10),
        final_mode_weights=(0.35, 0.25, 0.25, 0.15),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["aromatic_ids"])
        self.architecture_type = "unified_csdnet"
        self.pad_id = int(pad_id)
        self.mask_id = int(mask_id)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.unk_id = int(unk_id)
        self.max_gap_count = int(max_gap_count)
        self.lr = float(lr)
        self.warmup_steps = int(warmup_steps)
        self.weight_decay = float(weight_decay)
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_eps = float(adam_eps)
        self.aromatic_weight = float(aromatic_weight)
        self.aromatic_anneal_steps = int(aromatic_anneal_steps)
        self.curriculum_steps = int(curriculum_steps)
        self.gap_loss_weight = float(gap_loss_weight)
        self.gap_ordinal_weight = float(gap_ordinal_weight)
        self.delete_loss_weight = float(delete_loss_weight)
        self.confidence_loss_weight = float(confidence_loss_weight)
        self.initial_mode_weights = tuple(float(x) for x in initial_mode_weights)
        self.final_mode_weights = tuple(float(x) for x in final_mode_weights)
        if len(self.initial_mode_weights) != 4 or len(self.final_mode_weights) != 4:
            raise ValueError("Mode weights must have four entries.")
        if self.aromatic_weight < 1.0:
            raise ValueError("aromatic_weight must be at least one.")

        aromatic_lookup = torch.zeros(int(vocab_size), dtype=torch.bool)
        for token_id in aromatic_ids or ():
            if 0 <= int(token_id) < int(vocab_size):
                aromatic_lookup[int(token_id)] = True
        self.register_buffer("aromatic_lookup", aromatic_lookup)

        self.backbone = UnifiedCSDNetBackbone(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate=intermediate,
            pad_token_id=pad_id,
            mask_token_id=mask_id,
            max_position_embeddings=max_position_embeddings,
            max_gap_count=max_gap_count,
            position_embedding_type=position_embedding_type,
            gradient_checkpointing=gradient_checkpointing,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            initializer_range=initializer_range,
        )
        self.ema = (
            SimpleEMA(self.backbone, decay=float(ema_decay))
            if bool(use_ema)
            else None
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        corruption_level=None,
        return_aux=False,
    ):
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            corruption_level=corruption_level,
            return_aux=return_aux,
        )

    def _progress(self, steps):
        if steps <= 0:
            return 1.0
        return min(1.0, max(0.0, float(self.global_step) / float(steps)))

    def _mode_weights(self):
        progress = self._progress(self.curriculum_steps)
        values = [
            start + progress * (end - start)
            for start, end in zip(
                self.initial_mode_weights, self.final_mode_weights
            )
        ]
        total = max(sum(values), 1e-8)
        return [value / total for value in values]

    def _aromatic_factor(self):
        progress = self._progress(self.aromatic_anneal_steps)
        return 1.0 + (self.aromatic_weight - 1.0) * (1.0 - progress)

    @staticmethod
    def _row_average(values, valid, weights=None):
        valid_float = valid.to(dtype=values.dtype)
        if weights is not None:
            valid_float = valid_float * weights.to(dtype=values.dtype)
        denominator = valid_float.sum(dim=1)
        row_loss = (values * valid_float).sum(dim=1) / denominator.clamp(min=1e-8)
        return row_loss, denominator > 0

    @staticmethod
    def _mode_mean(row_loss, row_valid, mode_ids, mode):
        selected = row_valid & mode_ids.eq(int(mode))
        if not selected.any():
            return row_loss.sum() * 0.0
        return row_loss[selected].mean()

    def _normalized_aromatic_weights(self, batch):
        base = batch["token_weights"].float()
        valid = batch["token_labels"].ne(-100)
        factor = torch.ones_like(base)
        aromatic_factor = self._aromatic_factor()
        if aromatic_factor > 1.0:
            aromatic = batch.get("aromatic_mask")
            if aromatic is None:
                safe_labels = batch["token_labels"].clamp(min=0)
                aromatic = self.aromatic_lookup[safe_labels]
            factor = factor + (aromatic_factor - 1.0) * aromatic.float()
        weighted = base * factor * valid.float()
        original_mass = (base * valid.float()).sum(dim=1, keepdim=True)
        new_mass = weighted.sum(dim=1, keepdim=True)
        scale = torch.where(
            new_mass > 0,
            original_mass / new_mass.clamp(min=1e-8),
            torch.ones_like(new_mass),
        )
        return weighted * scale

    def _compute_losses(self, batch):
        output = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            corruption_level=batch["corruption_level"],
            return_aux=True,
        )
        mode_ids = batch["mode_ids"]

        token_labels = batch["token_labels"]
        token_valid = token_labels.ne(-100)
        safe_token_labels = token_labels.masked_fill(~token_valid, 0)
        token_ce = F.cross_entropy(
            output["logits"].transpose(1, 2),
            safe_token_labels,
            reduction="none",
        )
        token_weights = self._normalized_aromatic_weights(batch)
        token_rows, token_row_valid = self._row_average(
            token_ce, token_valid, token_weights
        )

        gap_labels = batch["gap_labels"]
        gap_valid = gap_labels.ne(-100)
        if gap_valid.any() and int(gap_labels[gap_valid].max().item()) > self.max_gap_count:
            raise ValueError(
                "Batch gap labels exceed the backbone max_gap_count; "
                "the collator and model configurations must match."
            )
        safe_gap_labels = gap_labels.masked_fill(~gap_valid, 0)
        gap_ce = F.cross_entropy(
            output["gap_logits"].transpose(1, 2),
            safe_gap_labels,
            reduction="none",
        )
        positive_gap_weight = 1.0 + 3.0 * safe_gap_labels.gt(0).float()
        gap_rows, gap_row_valid = self._row_average(
            gap_ce, gap_valid, positive_gap_weight
        )
        count_values = torch.arange(
            self.max_gap_count + 1,
            device=output["gap_logits"].device,
            dtype=output["gap_logits"].dtype,
        )
        expected_count = (
            output["gap_logits"].softmax(dim=-1) * count_values
        ).sum(dim=-1)
        ordinal = F.smooth_l1_loss(
            expected_count.float(), batch["gap_exact"].float(), reduction="none"
        )
        ordinal_rows, ordinal_row_valid = self._row_average(
            ordinal, gap_valid
        )

        delete_labels = batch["delete_labels"]
        delete_valid = delete_labels.ge(0.0)
        safe_delete_labels = delete_labels.clamp(min=0.0)
        delete_bce = F.binary_cross_entropy_with_logits(
            output["delete_logits"].float(),
            safe_delete_labels.float(),
            reduction="none",
        )
        positive_delete_weight = 1.0 + 7.0 * safe_delete_labels
        delete_rows, delete_row_valid = self._row_average(
            delete_bce, delete_valid, positive_delete_weight
        )

        with torch.no_grad():
            target_probability = output["logits"].float().softmax(dim=-1)
            target_probability = target_probability.gather(
                -1, safe_token_labels.unsqueeze(-1)
            ).squeeze(-1)
        confidence_bce = F.binary_cross_entropy_with_logits(
            output["confidence_logits"].float(),
            target_probability,
            reduction="none",
        )
        confidence_rows, confidence_row_valid = self._row_average(
            confidence_bce, token_valid
        )

        mode_weights = self._mode_weights()
        total = output["logits"].sum() * 0.0
        metrics = {}
        for mode, name in enumerate(MODE_NAMES):
            token_loss = self._mode_mean(
                token_rows, token_row_valid, mode_ids, mode
            )
            gap_loss = self._mode_mean(
                gap_rows, gap_row_valid, mode_ids, mode
            )
            ordinal_loss = self._mode_mean(
                ordinal_rows, ordinal_row_valid, mode_ids, mode
            )
            delete_loss = self._mode_mean(
                delete_rows, delete_row_valid, mode_ids, mode
            )
            confidence_loss = self._mode_mean(
                confidence_rows, confidence_row_valid, mode_ids, mode
            )
            combined = (
                token_loss
                + self.gap_loss_weight * gap_loss
                + self.gap_ordinal_weight * ordinal_loss
                + self.delete_loss_weight * delete_loss
                + self.confidence_loss_weight * confidence_loss
            )
            total = total + mode_weights[mode] * combined
            metrics[f"{name}_loss"] = combined.detach()
            metrics[f"{name}_token"] = token_loss.detach()
            metrics[f"{name}_gap"] = gap_loss.detach()
            metrics[f"{name}_delete"] = delete_loss.detach()

        metrics["gap_positive_fraction"] = (
            (safe_gap_labels.gt(0) & gap_valid).float().sum()
            / gap_valid.float().sum().clamp(min=1.0)
        ).detach()
        metrics["delete_positive_fraction"] = (
            (safe_delete_labels.gt(0) & delete_valid).float().sum()
            / delete_valid.float().sum().clamp(min=1.0)
        ).detach()
        metrics["aromatic_factor"] = torch.tensor(
            self._aromatic_factor(), device=total.device
        )
        return total, metrics

    def training_step(self, batch, _batch_idx):
        loss, metrics = self._compute_losses(batch)
        self.log(
            "train_loss", loss, on_step=True, prog_bar=True, sync_dist=True
        )
        for name, value in metrics.items():
            self.log(
                f"train_{name}",
                value,
                on_step=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss

    def validation_step(self, batch, _batch_idx):
        loss, _ = self._compute_losses(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            self.ema.update(self.backbone)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_eps,
        )

        def schedule(step):
            if step < self.warmup_steps:
                return float(step + 1) / float(max(1, self.warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

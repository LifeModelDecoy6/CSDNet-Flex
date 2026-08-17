import math

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F


class EditSchedulerLightningModule(L.LightningModule):
    """Train an external edit scheduler against a frozen CSDNet teacher."""

    def __init__(
        self,
        scheduler,
        teacher,
        teacher_checkpoint,
        lr=3e-4,
        warmup_steps=2500,
        lr_schedule="cosine",
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_eps=1e-8,
        order_loss_weight=0.25,
        calibration_loss_weight=0.10,
        rate_regularizer_weight=0.01,
        order_temperature=0.7,
    ):
        super().__init__()
        self.scheduler = scheduler
        # The frozen teacher is deliberately not registered as a child module.
        # It is moved to each rank manually and excluded from scheduler
        # checkpoints, keeping the trained artifact lightweight.
        self.__dict__["_teacher_module"] = teacher
        self.teacher_checkpoint = str(teacher_checkpoint)
        self.lr = float(lr)
        self.warmup_steps = int(warmup_steps)
        self.lr_schedule = str(lr_schedule)
        self.weight_decay = float(weight_decay)
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_eps = float(adam_eps)
        self.order_loss_weight = float(order_loss_weight)
        self.calibration_loss_weight = float(calibration_loss_weight)
        self.rate_regularizer_weight = float(rate_regularizer_weight)
        self.order_temperature = float(order_temperature)
        if self.lr_schedule not in {"cosine", "constant_with_warmup"}:
            raise ValueError(
                "lr_schedule must be 'cosine' or 'constant_with_warmup'."
            )
        if self.order_temperature <= 0:
            raise ValueError("order_temperature must be positive.")

        self.save_hyperparameters(
            {
                "architecture_type": "edit_schedule_net",
                "scheduler_config": scheduler.config_dict(),
                "teacher_checkpoint": self.teacher_checkpoint,
                "lr": self.lr,
                "warmup_steps": self.warmup_steps,
                "lr_schedule": self.lr_schedule,
                "weight_decay": self.weight_decay,
                "adam_beta1": self.adam_beta1,
                "adam_beta2": self.adam_beta2,
                "adam_eps": self.adam_eps,
                "order_loss_weight": self.order_loss_weight,
                "calibration_loss_weight": (
                    self.calibration_loss_weight
                ),
                "rate_regularizer_weight": (
                    self.rate_regularizer_weight
                ),
                "order_temperature": self.order_temperature,
            }
        )

    @property
    def teacher(self):
        return self.__dict__["_teacher_module"]

    def on_fit_start(self):
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.teacher.to(self.device)

    @staticmethod
    def _scatter_gap_values(values, gap_ids, valid, gap_count):
        batch_size = values.size(0)
        sums = values.new_zeros(batch_size, gap_count)
        counts = values.new_zeros(batch_size, gap_count)
        safe_ids = gap_ids.clamp(min=0, max=max(0, gap_count - 1))
        sums.scatter_add_(
            1,
            safe_ids,
            values * valid.to(values.dtype),
        )
        counts.scatter_add_(
            1,
            safe_ids,
            valid.to(values.dtype),
        )
        return sums / counts.clamp(min=1.0), counts

    @torch.no_grad()
    def _teacher_difficulty(self, batch, gap_count):
        self.teacher.eval()
        teacher_kwargs = {}
        if bool(
            getattr(
                self.teacher,
                "corruption_level_conditioning",
                False,
            )
        ):
            teacher_kwargs["corruption_level"] = batch[
                "corruption_fraction"
            ]
        output = self.teacher(
            batch["teacher_input_ids"],
            batch["teacher_attention_mask"],
            **teacher_kwargs,
        )
        logits = (
            output["logits"]
            if isinstance(output, dict)
            else output
        )
        labels = batch["teacher_labels"]
        valid = labels.ne(-100)
        safe_labels = labels.masked_fill(~valid, 0)
        token_loss = F.cross_entropy(
            logits.float().transpose(1, 2),
            safe_labels,
            reduction="none",
        )
        return self._scatter_gap_values(
            token_loss,
            batch["teacher_gap_ids"],
            valid,
            gap_count,
        )

    @staticmethod
    def _scheduler_gap_table(values, gap_ids, gap_mask, gap_count):
        table = values.new_full((values.size(0), gap_count), -torch.inf)
        rows, columns = gap_mask.nonzero(as_tuple=True)
        if rows.numel():
            destinations = gap_ids[rows, columns].clamp(
                min=0,
                max=max(0, gap_count - 1),
            )
            table[rows, destinations] = values[rows, columns]
        return table

    @staticmethod
    def _gap_target_lengths(batch, gap_count):
        labels = batch["length_labels"]
        valid = labels.ne(-100)
        table = labels.new_full((labels.size(0), gap_count), -1)
        rows, columns = valid.nonzero(as_tuple=True)
        if rows.numel():
            destinations = batch["gap_ids"][rows, columns].clamp(
                min=0,
                max=max(0, gap_count - 1),
            )
            table[rows, destinations] = labels[rows, columns]
        return table

    def _step(self, batch):
        output = self.scheduler(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            gap_mask=batch["gap_mask"],
            removed_lengths=batch["removed_lengths"],
            corruption_fraction=batch["corruption_fraction"],
        )
        gap_mask = batch["gap_mask"].bool()
        length_labels = batch["length_labels"]
        length_loss = F.cross_entropy(
            output["length_logits"][gap_mask].float(),
            length_labels[gap_mask],
        )

        probabilities = torch.softmax(
            output["length_logits"][gap_mask].float(),
            dim=-1,
        )
        predictions = probabilities.argmax(dim=-1)
        targets = length_labels[gap_mask]
        confidence = probabilities.amax(dim=-1)
        correct = predictions.eq(targets).float()
        calibration_loss = (confidence - correct).square().mean()

        gap_count = int(batch["gap_count"].max().item())
        target_lengths = self._gap_target_lengths(batch, gap_count)
        positive_gaps = target_lengths.gt(0)
        difficulty, teacher_counts = self._teacher_difficulty(
            batch,
            gap_count,
        )
        positive_gaps = positive_gaps & teacher_counts.gt(0)

        scheduler_order = self._scheduler_gap_table(
            output["order_logits"],
            batch["gap_ids"],
            gap_mask,
            gap_count,
        )
        teacher_order = (
            -difficulty / self.order_temperature
        ).masked_fill(~positive_gaps, -torch.inf)
        scheduler_order = scheduler_order.masked_fill(
            ~positive_gaps,
            -torch.inf,
        )
        order_target = torch.softmax(teacher_order, dim=-1)
        order_log_probability = torch.log_softmax(
            scheduler_order,
            dim=-1,
        )
        order_loss_by_row = -(
            order_target * order_log_probability
        ).masked_fill(~positive_gaps, 0.0).sum(dim=-1)
        order_loss = order_loss_by_row.mean()

        positive_rate = self._scheduler_gap_table(
            output["insertion_rate"].float(),
            batch["gap_ids"],
            gap_mask,
            gap_count,
        )
        rate_values = positive_rate[positive_gaps]
        rate_regularizer = (
            torch.log(rate_values.clamp(min=1e-8)).square().mean()
            if rate_values.numel()
            else length_loss.new_zeros(())
        )

        total_loss = (
            length_loss
            + self.order_loss_weight * order_loss
            + self.calibration_loss_weight * calibration_loss
            + self.rate_regularizer_weight * rate_regularizer
        )

        top_k = min(3, probabilities.size(-1))
        top3 = probabilities.topk(top_k, dim=-1).indices
        exact = correct.mean()
        top3_accuracy = top3.eq(targets.unsqueeze(-1)).any(dim=-1).float().mean()
        length_mae = predictions.sub(targets).abs().float().mean()
        zero = targets.eq(0)
        zero_accuracy = (
            predictions[zero].eq(0).float().mean()
            if zero.any()
            else exact.new_zeros(())
        )
        multiple = positive_gaps.sum(dim=-1).gt(1)
        order_accuracy = (
            scheduler_order[multiple].argmax(dim=-1).eq(
                teacher_order[multiple].argmax(dim=-1)
            ).float().mean()
            if multiple.any()
            else exact.new_zeros(())
        )
        return {
            "loss": total_loss,
            "length_loss": length_loss.detach(),
            "order_loss": order_loss.detach(),
            "calibration_loss": calibration_loss.detach(),
            "rate_regularizer": rate_regularizer.detach(),
            "length_exact": exact.detach(),
            "length_top3": top3_accuracy.detach(),
            "length_mae": length_mae.detach(),
            "zero_accuracy": zero_accuracy.detach(),
            "order_top1": order_accuracy.detach(),
            "mean_rate": (
                rate_values.mean().detach()
                if rate_values.numel()
                else exact.new_zeros(())
            ),
        }

    def training_step(self, batch, _):
        metrics = self._step(batch)
        self.log(
            "train_loss",
            metrics["loss"],
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_length_top3",
            metrics["length_top3"],
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        for name, value in metrics.items():
            if name not in {"loss", "length_top3"}:
                self.log(
                    f"train_{name}",
                    value,
                    on_step=True,
                    sync_dist=True,
                )
        return metrics["loss"]

    def validation_step(self, batch, _):
        metrics = self._step(batch)
        for name, value in metrics.items():
            self.log(
                f"val_{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name in {"loss", "length_top3"},
                sync_dist=True,
            )
        return metrics["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.scheduler.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_eps,
        )

        def learning_rate_multiplier(step):
            if step < self.warmup_steps:
                return (step + 1) / max(1, self.warmup_steps)
            if self.lr_schedule == "constant_with_warmup":
                return 1.0
            try:
                total_steps = self.trainer.estimated_stepping_batches
                if total_steps == float("inf"):
                    total_steps = 50000
            except Exception:
                total_steps = 50000
            progress = (step - self.warmup_steps) / max(
                1,
                total_steps - self.warmup_steps,
            )
            return 0.1 + 0.9 * 0.5 * (
                1.0 + np.cos(math.pi * min(progress, 1.0))
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            learning_rate_multiplier,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

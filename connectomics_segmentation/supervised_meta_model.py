from itertools import chain
from typing import Any, Callable, Iterator

import torch
from lightning.pytorch import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAveragePrecision,
    MulticlassCohenKappa,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

opt_factory = Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer]
lr_sched_factory = Callable[
    [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
]


class SupervisedMetaModel(LightningModule):
    def __init__(
        self,
        backbone_model: torch.nn.Module,
        head_model: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer_factory: opt_factory,
        num_classes: int = 6,
        lr_scheduler_factory: lr_sched_factory | None = None,
        compile_model: bool = True,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.backbone_model = backbone_model
        self.head_model = head_model
        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory
        self.compile_model = compile_model

        self.loss = loss

        metrics = MetricCollection(
            [
                MulticlassAccuracy(num_classes=num_classes, ignore_index=num_classes),
                MulticlassPrecision(num_classes=num_classes, ignore_index=num_classes),
                MulticlassRecall(num_classes=num_classes, ignore_index=num_classes),
                MulticlassF1Score(num_classes=num_classes, ignore_index=num_classes),
                MulticlassCohenKappa(num_classes=num_classes, ignore_index=num_classes),
                MulticlassAveragePrecision(
                    num_classes=num_classes, ignore_index=num_classes
                ),
            ]
        )
        self.train_metrics = metrics.clone("train/")
        self.valid_metrics = metrics.clone("val/")
        self.test_metrics = metrics.clone("test/")

    def setup(self, stage: str) -> None:
        if self.compile_model and stage == "fit":
            self.backbone_model = torch.compile(self.backbone_model)
            self.head_model = torch.compile(self.head_model)
        return super().setup(stage)

    def configure_optimizers(self) -> dict[str, Any]:
        backbone_params = self.backbone_model.parameters()
        head_params = self.head_model.parameters()
        params = chain(backbone_params, head_params)
        optimizer = self.optimizer_factory(params)

        if self.lr_scheduler_factory is not None:
            lr_scheduler = self.lr_scheduler_factory(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        features = self.backbone_model(data)
        return self.head_model(features)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        preds = self.forward(batch["data"])
        target = batch["label"]

        loss = self.loss(preds, target)
        self.log("train/loss", loss)

        if not torch.all(target == self.num_classes):
            self.train_metrics(preds, target)
            self.log_dict(self.train_metrics, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        preds = self.forward(batch["data"])
        target = batch["label"]

        loss = self.loss(preds, target)
        self.log("val/loss", loss, on_epoch=True, on_step=False)

        if not torch.all(target == self.num_classes):
            self.valid_metrics(preds, target)
            self.log_dict(self.valid_metrics, on_epoch=True, on_step=False)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        preds = self.forward(batch["data"])
        target = batch["label"]

        if not torch.all(target == self.num_classes):
            self.test_metrics(preds, target)
            self.log_dict(self.test_metrics, on_epoch=True, on_step=False)

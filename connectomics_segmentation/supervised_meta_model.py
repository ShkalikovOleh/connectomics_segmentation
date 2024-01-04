from typing import Callable, Iterator, Any

import torch
from lightning.pytorch import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAveragePrecision,
    MulticlassCohenKappa,
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
        model: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer_factory: opt_factory,
        num_classes: int = 6,
        lr_scheduler_factory: lr_sched_factory | None = None,
        compile_model: bool = True,
    ) -> None:
        super().__init__()

        self.model = model
        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory
        self.compile_model = compile_model

        self.loss = loss

        metrics = MetricCollection(
            [
                MulticlassAccuracy(num_classes=num_classes),
                MulticlassPrecision(num_classes=num_classes),
                MulticlassCohenKappa(num_classes=num_classes),
                MulticlassRecall(num_classes=num_classes),
                MulticlassAveragePrecision(num_classes=num_classes),
            ]
        )
        self.train_metrics = metrics.clone("train/")
        self.valid_metrics = metrics.clone("val/")
        self.test_metrics = metrics.clone("test/")

    def setup(self, stage: str) -> None:
        if self.compile_model and stage == "fit":
            self.model = torch.compile(self.model)
        return super().setup(stage)

    def configure_optimizers(self) -> dict[str, Any]:
        params = self.model.parameters()
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
        return self.model(data)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        preds = self.model(batch["data"])
        target = batch["label"]

        loss = self.loss(preds, target)
        self.log("train/loss", loss)

        self.train_metrics(preds, target)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        preds = self.model(batch["data"])
        target = batch["label"]

        loss = self.loss(preds, target)
        self.log("val/loss", loss, on_epoch=True, on_step=False)

        self.valid_metrics(preds, target)
        self.log_dict(self.valid_metrics, on_epoch=True, on_step=False)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        preds = self.model(batch["data"])
        target = batch["label"]

        self.test_metrics(preds, target)
        self.log_dict(self.test_metrics, on_epoch=True, on_step=False)

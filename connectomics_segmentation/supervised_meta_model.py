from itertools import chain
from typing import Any, Callable, Iterator

import torch
from lightning.pytorch import LightningModule
from torchmetrics import ClasswiseWrapper, MetricCollection
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
        class_names: list[str] | None = None,
        lr_scheduler_factory: lr_sched_factory | None = None,
        compile_model: bool = True,
        log_train_metrics: bool = True,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.backbone_model = backbone_model
        self.head_model = head_model
        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory
        self.compile_model = compile_model
        self.log_train_metrics = log_train_metrics
        self.log_valid_metrics = False
        self.log_test_metrics = False

        self.loss = loss

        metrics = self.create_metrics(num_classes, class_names)
        if log_train_metrics:
            self.train_metrics = metrics.clone("train/")
        self.valid_metrics = metrics.clone("val/")
        self.test_metrics = metrics.clone("test/")

    @staticmethod
    def create_metrics(
        num_classes: int, class_names: list[str] | None
    ) -> MetricCollection:
        overall_metrics_kwargs = {
            "num_classes": num_classes,
            "ignore_index": num_classes,
        }
        classwise_metrics_kwargs = {
            "num_classes": num_classes,
            "ignore_index": num_classes,
            "average": None,
        }
        metrics = MetricCollection(
            {
                "accuracy_overall": MulticlassAccuracy(**overall_metrics_kwargs),  # type: ignore # noqa
                "precision_overall": MulticlassPrecision(**overall_metrics_kwargs),  # type: ignore # noqa
                "recall_overall": MulticlassRecall(**overall_metrics_kwargs),  # type: ignore # noqa
                "f1_overall": MulticlassF1Score(**overall_metrics_kwargs),  # type: ignore # noqa
                "Cohen_Kappa_overall": MulticlassCohenKappa(**overall_metrics_kwargs),  # type: ignore # noqa
                "AP_overall": MulticlassAveragePrecision(**overall_metrics_kwargs),  # type: ignore # noqa
                "classwise_accuracy": ClasswiseWrapper(
                    MulticlassAccuracy(**classwise_metrics_kwargs),
                    class_names,
                    prefix="accuracy_",
                ),
                "classwise_precision": ClasswiseWrapper(
                    MulticlassPrecision(**classwise_metrics_kwargs),
                    class_names,
                    prefix="precision_",
                ),
                "classwise_recall": ClasswiseWrapper(
                    MulticlassRecall(**classwise_metrics_kwargs),
                    class_names,
                    prefix="recall_",
                ),
                "classwise_f1": ClasswiseWrapper(
                    MulticlassF1Score(**classwise_metrics_kwargs),
                    class_names,
                    prefix="f1_",
                ),
            }
        )

        return metrics

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

        if self.log_train_metrics and not torch.all(target == self.num_classes):
            metrics = self.train_metrics(preds, target)
            self.log_dict(metrics)

        return loss

    def on_train_epoch_end(self) -> None:
        if self.log_train_metrics:
            self.train_metrics.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        preds = self.forward(batch["data"])
        target = batch["label"]

        loss = self.loss(preds, target)
        self.log("val/loss", loss, on_epoch=True, on_step=False)

        if not torch.all(target == self.num_classes):
            self.log_valid_metrics = True
            self.valid_metrics.update(preds, target)

    def on_validation_epoch_end(self) -> None:
        if self.log_valid_metrics:
            self.log_dict(self.valid_metrics.compute())
        self.valid_metrics.reset()

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        preds = self.forward(batch["data"])
        target = batch["label"]

        if not torch.all(target == self.num_classes):
            self.log_test_metrics = True
            self.test_metrics.update(preds, target)

        return {"predictions": preds.detach().cpu()}

    def on_test_epoch_end(self) -> None:
        if self.log_test_metrics:
            self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

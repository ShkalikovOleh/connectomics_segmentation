from typing import Any

import numpy as np
import torch
import wandb
from lightning import Callback, LightningModule, Trainer
from lightning import pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from torchmetrics import ClasswiseWrapper, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCohenKappa,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from connectomics_segmentation.postprocessing.crf import apply_crf


class DenseCRFPostprocessingCallback(Callback):
    """This callback gather and store all prediction of center voxel class
    and generates images of semantic segmentation based on the these labels
    when all pixels of image have been saved in the internal buffer.
    Requires from dataloader that it doesn't shuffle the test dataset"""

    def __init__(
        self,
        image_width: int,
        image_height: int,
        n_classes: int,
        position_theta: list[float],
        bilateral_theta: float,
        compat_position: float,
        compat_bilateral: float,
        num_steps: int,
        calculate_metrics: bool = True,
        class_names: list[str] | None = None,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.n_classes = n_classes

        self._class_names = class_names
        self._position_theta = position_theta
        self._bilateral_theta = bilateral_theta
        self._compat_position = compat_position
        self._compat_bilateral = compat_bilateral
        self._num_steps = num_steps

        self._proba_buffer = np.empty(
            (self.n_classes, image_height * image_width), dtype=np.float32
        )
        self._intensities_buffer = np.empty(
            image_height * image_width, dtype=np.float32
        )
        self._remain_unfilled = image_width * image_height
        self._num_image = 1

        self.calculate_metrics = calculate_metrics
        self._metrics_ready = False
        if calculate_metrics:
            self.metrics = self.create_metrics()
            self._true_label_buffer = np.empty(
                image_height * image_width, dtype=np.uint8
            )

    def create_metrics(self) -> MetricCollection:
        overall_metrics_kwargs = {
            "num_classes": self.n_classes,
            "ignore_index": self.n_classes,
        }
        classwise_metrics_kwargs = {
            "num_classes": self.n_classes,
            "ignore_index": self.n_classes,
            "average": None,
        }
        metrics = MetricCollection(
            {
                "CRF/accuracy_overall": MulticlassAccuracy(**overall_metrics_kwargs),  # type: ignore # noqa
                "CRF/precision_overall": MulticlassPrecision(**overall_metrics_kwargs),  # type: ignore # noqa
                "CRF/recall_overall": MulticlassRecall(**overall_metrics_kwargs),  # type: ignore # noqa
                "CRF/f1_overall": MulticlassF1Score(**overall_metrics_kwargs),  # type: ignore # noqa
                "CRF/Cohen_Kappa_overall": MulticlassCohenKappa(**overall_metrics_kwargs),  # type: ignore # noqa
                "CRF/classwise_accuracy": ClasswiseWrapper(
                    MulticlassAccuracy(**classwise_metrics_kwargs),
                    self._class_names,
                    prefix="CRF/accuracy_",
                ),
                "CRF/classwise_precision": ClasswiseWrapper(
                    MulticlassPrecision(**classwise_metrics_kwargs),
                    self._class_names,
                    prefix="CRF/precision_",
                ),
                "CRF/classwise_recall": ClasswiseWrapper(
                    MulticlassRecall(**classwise_metrics_kwargs),
                    self._class_names,
                    prefix="CRF/recall_",
                ),
                "CRF/classwise_f1": ClasswiseWrapper(
                    MulticlassF1Score(**classwise_metrics_kwargs),
                    self._class_names,
                    prefix="CRF/f1_",
                ),
            }
        )

        return metrics

    def __apply_crf_and_log(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        proba = self._proba_buffer.reshape(
            (self.n_classes, self.image_height, self.image_width, 1)
        )
        intensities = self._intensities_buffer.reshape(
            (1, self.image_height, self.image_width, 1)
        )

        crf_labels = apply_crf(
            proba=proba,
            intensities=intensities,
            pos_weights=self._position_theta,
            inten_weight=self._bilateral_theta,
            compat_position=self._compat_position,
            compat_bilateral=self._compat_bilateral,
            num_steps=self._num_steps,
        )

        crf_labels = crf_labels.reshape((self.image_height, self.image_width))
        raw_pred_labels = proba.argmax(axis=0)[:, :, 0]
        true_labels = self._true_label_buffer.reshape(
            (self.image_height, self.image_width)
        )
        image = intensities[0]

        if self._class_names:
            id2class = {i: name for i, name in enumerate(self._class_names)}
        else:
            id2class = {i: str(i) for i in range(self.n_classes)}

        log_name = f"CRF/image {self._num_image}"
        for logger in pl_module.loggers:
            if isinstance(logger, WandbLogger):
                mask_img = wandb.Image(
                    image,
                    masks={
                        "raw_predictions": {
                            "mask_data": raw_pred_labels,
                            "class_labels": id2class,
                        },
                        "ground_truth": {
                            "mask_data": true_labels,
                            "class_labels": id2class,
                        },
                        "crf_prefictions": {
                            "mask_data": crf_labels,
                            "class_labels": id2class,
                        },
                    },
                )

                logger.experiment.log(
                    {log_name: [mask_img]},
                    step=trainer.global_step,
                )

        if self.calculate_metrics:
            if not np.all(true_labels == self.n_classes):
                preds = torch.from_numpy(crf_labels)
                target = torch.from_numpy(true_labels)

                self.metrics.update(preds, target)

                self._metrics_ready = True

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, Any],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        proba = outputs["predictions"].numpy()
        H = proba.shape[-1]
        intensity = batch["data"][:, 0, H // 2, H // 2, H // 2].detach().cpu().numpy()
        true_labels = batch["label"].detach().cpu().numpy()

        N_preds = intensity.size

        start_idx = -self._remain_unfilled
        if self._remain_unfilled <= N_preds:  # type: ignore # noqa
            self._intensities_buffer[start_idx:] = intensity[:-start_idx]
            self._proba_buffer[:, start_idx:] = proba[:-start_idx].T
            self._true_label_buffer[start_idx:] = true_labels[:-start_idx]  # type: ignore # noqa

            intensity = intensity[-start_idx:]
            proba = proba[-start_idx:]
            true_labels = true_labels[-start_idx:]  # type: ignore # noqa

            self.__apply_crf_and_log(trainer, pl_module)

            N_preds += start_idx
            self._remain_unfilled = self.image_height * self.image_width
            start_idx = -self._remain_unfilled
            self._num_image += 1

        end_idx = -self._remain_unfilled + N_preds

        self._intensities_buffer[start_idx:end_idx] = intensity
        self._proba_buffer[:, start_idx:end_idx] = proba.T
        self._true_label_buffer[start_idx:end_idx] = true_labels  # type: ignore # noqa

        self._remain_unfilled -= N_preds

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._metrics_ready:
            metrics_dict = self.metrics.compute()
            pl_module.log_dict(metrics_dict)
            self.metrics.reset()

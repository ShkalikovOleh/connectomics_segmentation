from typing import Any

import numpy as np
import torch
import wandb
from lightning import Callback
from lightning import pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger

from connectomics_segmentation.utils.aggregator import Aggregator


class TestVisualizationCallback(Callback):
    """This callback gather and store all prediction of center voxel class
    and generates images of semantic segmentation based on the these labels
    when all pixels of image have been saved in the internal buffer.
    Requires from dataloader that it doesn't shuffle the test dataset"""

    def __init__(
        self,
        image_width: int,
        image_height: int,
        n_classes: int,
        subvolume_size: int,
        class_names: list[str] | None = None,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.n_classes = n_classes

        self._class_names = class_names
        self._subvolume_size = subvolume_size

        self._proba_aggr = Aggregator(
            (n_classes, 1, image_height, image_width), np.float32
        )
        self._intensities_aggr = Aggregator(
            (1, 1, image_height, image_width), np.float32
        )
        self._gt_aggr = Aggregator((1, 1, image_height, image_width), dtype=np.uint8)
        self._num_image = 1

    def __log_image(self, pl_module: pl.LightningModule):
        proba = self._proba_aggr.get_volume()
        intensities = self._intensities_aggr.get_volume()

        raw_pred_labels = proba.argmax(axis=0)[0]
        true_labels = self._gt_aggr.get_volume().reshape(
            (self.image_height, self.image_width)
        )
        image = intensities[0, 0]

        if self._class_names:
            id2class = {i: name for i, name in enumerate(self._class_names)}
        else:
            id2class = {i: str(i) for i in range(self.n_classes)}

        log_name = f"test/image {self._num_image}"
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
                    },
                )

                logger.experiment.log({log_name: [mask_img]})

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
        true_labels = batch["label"].detach().cpu().unsqueeze(1).numpy()

        H = batch["data"].shape[3]
        start_idx = (H - self._subvolume_size + 1) // 2
        end_idx = start_idx + self._subvolume_size
        intensities = (
            batch["data"][:, :, start_idx, start_idx:end_idx, start_idx:end_idx]
            .detach()
            .unsqueeze(2)
            .cpu()
            .numpy()
        )

        while len(proba) > 0:
            is_filled, proba = self._proba_aggr.add_batch(proba)
            _, intensities = self._intensities_aggr.add_batch(intensities)
            _, true_labels = self._gt_aggr.add_batch(true_labels)

            if is_filled:
                self.__log_image(pl_module)
                self._num_image += 1
            else:
                break

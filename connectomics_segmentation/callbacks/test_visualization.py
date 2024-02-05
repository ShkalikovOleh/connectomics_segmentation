from typing import Any

import numpy as np
import torch
import wandb
from lightning import Callback
from lightning import pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger


class TestVisualizationCallback(Callback):
    """This callback gather and store all prediction of center voxel class
    and generates images of semantic segmentation based on the these labels
    when all pixels of image have been saved in the internal buffer.
    Requires from dataloader that it doesn't shuffle the test dataset"""

    def __init__(
        self, image_width: int, image_height: int, label_colors: list[tuple[int]]
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width

        self._buffer = np.empty(image_height * image_width)
        self._remain_unfilled = image_width * image_height
        self._num_image = 1

        self._label2color = np.array(label_colors, dtype=np.uint8)

    def map_labels_to_color(self, labels: np.ndarray) -> np.ndarray:
        img_shape = (self.image_height, self.image_width, 3)
        img = np.zeros(img_shape, dtype=np.uint8)

        for label in range(self._label2color.shape[0]):
            idx = labels == label
            img[idx, 0] = self._label2color[label, 0]
            img[idx, 1] = self._label2color[label, 1]
            img[idx, 2] = self._label2color[label, 2]

        return img

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, Any],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        preds = outputs["predictions"]
        pred_labels = torch.argmax(preds, dim=1).numpy()
        N_preds = pred_labels.size

        start_idx = -self._remain_unfilled
        if self._remain_unfilled <= N_preds:  # type: ignore # noqa
            self._buffer[start_idx:] = pred_labels[:-start_idx]
            pred_labels = pred_labels[-start_idx:]

            image = self.map_labels_to_color(
                self._buffer.reshape((self.image_height, self.image_width))
            )

            caption = f"Predicted test image {self._num_image}"
            for logger in pl_module.loggers:
                if isinstance(logger, WandbLogger):
                    logger.experiment.log(
                        {"Visualization": [wandb.Image(image, caption=caption)]},
                        step=trainer.global_step,
                    )

            N_preds += start_idx
            self._remain_unfilled = self.image_height * self.image_width
            start_idx = -self._remain_unfilled
            self._num_image += 1

        end_idx = -self._remain_unfilled + N_preds
        self._buffer[start_idx:end_idx] = pred_labels
        self._remain_unfilled -= N_preds

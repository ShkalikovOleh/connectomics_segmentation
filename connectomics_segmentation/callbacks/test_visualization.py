from typing import Any

import numpy as np
import torch
import wandb
from lightning import Callback
from lightning import pytorch as pl


class TestVisualizationCallback(Callback):
    """Callback for visualizing reconstruction and classification."""

    def __init__(self, image_width: int, image_height: int):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self._buffer = np.empty(image_height * image_width)
        self._remain_unfilled = image_width * image_height
        self._num_image = 1

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

            image = self._buffer.reshape((self.image_height, self.image_width))
            caption = f"Predicted test image {self._num_image}"
            pl_module.logger.experiment.log(
                {"Visualization": [wandb.Image(image, caption=caption)]},
                step=trainer.global_step,
            )
            self._num_image += 1

            N_preds += start_idx
            self._remain_unfilled = self.image_height * self.image_width
            start_idx = -self._remain_unfilled

        end_idx = -self._remain_unfilled + N_preds
        self._buffer[start_idx:end_idx] = pred_labels
        self._remain_unfilled -= N_preds

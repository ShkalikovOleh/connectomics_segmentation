import numpy as np
import tifffile
import torch
from lightning import Callback
from lightning import pytorch as pl

from connectomics_segmentation.utils.aggregator import Aggregator


class SaveToTiffCallback(Callback):
    def __init__(
        self,
        H: int,
        W: int,
        D: int,
    ):
        super().__init__()

        self._label_aggr = Aggregator((1, H, W, D), np.uint8)
        self._num_image = 1

    def __save(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        labels = self._label_aggr.get_volume()

        with tifffile.TiffWriter("test_write") as tifw:
            tifw.write(labels)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        proba = outputs["predictions"].float().numpy()

        while len(proba) > 0:
            is_filled, proba = self._label_aggr.add_batch(proba)

            if is_filled:
                self.__save(trainer, pl_module)
                self._num_image += 1
            else:
                break

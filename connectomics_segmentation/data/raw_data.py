import math
import os

import numpy as np
import tifffile
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from patchify import patchify
from torch.utils.data import DataLoader, Dataset

from connectomics_segmentation.utils.paddings import calculate_tiling_padding
from connectomics_segmentation.utils.pylogger import RankedLogger

from .subdataset import BufferizedRandomSampler, SubDataset

log = RankedLogger(__name__, rank_zero_only=True)


class RawDataset(Dataset):
    def __init__(
        self,
        raw_data_path: str | os.PathLike,
        size_power: int = 6,
        subvolume_size: int = 1,
        padding_mode: str = "symmetric",
    ) -> None:
        super().__init__()

        voxel_size = 2**size_power
        data_subvol_size = voxel_size + subvolume_size - 1
        batch_extent = (data_subvol_size, data_subvol_size, data_subvol_size)

        with tifffile.TiffFile(raw_data_path) as raw_data_tif:
            log.info("Load raw data")
            raw_data = raw_data_tif.asarray()

            half_size = voxel_size // 2
            tiling_padding = calculate_tiling_padding(
                subvolume_size, np.asarray(raw_data.shape) + voxel_size - 1
            )

            self.paddings = tuple(
                (half_size, half_size - 1 + tiling_padding[i]) for i in range(3)
            )

            if np.any(self.paddings):
                log.info("Add padding to raw data")
                raw_data = np.pad(
                    raw_data, pad_width=self.paddings, mode=padding_mode  # type: ignore
                )

            log.info("Split raw data into patches")
            self.raw_data_batches = patchify(raw_data, batch_extent, subvolume_size)

    def __len__(self) -> int:
        sx, sy, sz, _, _, _ = self.raw_data_batches.shape
        return sx * sy * sz

    def __getitem__(self, idx: int) -> torch.Tensor:
        sx, sy, sz, _, _, _ = self.raw_data_batches.shape
        real_idx = np.unravel_index(idx, (sx, sy, sz))

        data = (self.raw_data_batches[real_idx] / 255).astype(np.float32)
        batch = torch.from_numpy(data).unsqueeze(0)

        return batch


class RawDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        assert (
            len(cfg.train_val_test_procents) == 3
            and math.isclose(sum(cfg.train_val_test_procents), 1) == 1
        ), "Please specify correct train, valid, test split ratio"

        self.cfg = cfg

    def setup(self, stage: str) -> None:
        log.info(f"Setup datamodule for {stage} stage")

        if hasattr(self, "train_ds"):
            return

        self.dataset = RawDataset(
            self.cfg.raw_data_path,
            self.cfg.size_power,
            self.cfg.subvolume_size,
            self.cfg.padding_mode,
        )
        self.paddings = self.dataset.paddings

        sizes = [
            int(math.floor(len(self.dataset) * frac))
            for frac in self.cfg.train_val_test_procents
        ]
        sizes[0] = len(self.dataset) - sum(sizes[1:])

        self.train_ds = SubDataset(self.dataset, 0, sizes[0])
        self.val_ds = SubDataset(self.dataset, sizes[0], sizes[1])
        self.test_ds = SubDataset(self.dataset, sizes[0] + sizes[1], sizes[2])

    def train_dataloader(self) -> DataLoader:
        sampler = BufferizedRandomSampler(self.train_ds, self.cfg.sampler_buffer_size)
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.train_batch_size,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.devel_batch_size,
            num_workers=self.cfg.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.devel_batch_size,
            num_workers=self.cfg.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.devel_batch_size,
            num_workers=self.cfg.num_workers,
        )

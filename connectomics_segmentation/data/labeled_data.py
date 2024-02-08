import os
from typing import Any, Tuple

import numpy as np
import tifffile
import torch
import volumentations
from lightning import LightningDataModule
from omegaconf import DictConfig, ListConfig
from patchify import patchify
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from connectomics_segmentation.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class LabeledDataset(Dataset):
    """Dataset which contains batches of the raw data of the size of the power of 2
    along every dimension as well as labels for the center voxel of the batch given by
    tiff file. In order to match data location from raw data tiff file and tiff file
    containing labels one has to specify ranges for which labels is provided.
    Note, that ranges for X and Z in Dokumentatio.txt are swapped.
    """

    def __init__(
        self,
        raw_data_path: str | os.PathLike,
        labels_path: str | os.PathLike,
        x_range: Tuple[int, int],
        y_range: Tuple[int, int],
        z_range: Tuple[int, int],
        size_power: int = 6,
        padding_mode: str = "constant",
        transforms: Any = None,
    ) -> None:
        super().__init__()

        self.transforms = transforms

        extent = np.array(
            [x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]],
            dtype=np.int32,
        )
        assert np.any(extent == 1), "One of the dimension shoud have size 1"

        voxel_size = 2**size_power
        batch_extent = (voxel_size, voxel_size, voxel_size)

        with tifffile.TiffFile(raw_data_path) as raw_data_tif:
            raw_data = raw_data_tif.asarray()

            half_size = voxel_size // 2
            paddings = []
            ranges = [x_range, y_range, z_range]
            slices = []
            for i, range in enumerate(ranges):
                padding = [0, 0]
                dim_size = raw_data.shape[i]

                if range[0] < half_size:
                    padding[0] = half_size - range[0]
                if dim_size < range[1] + half_size - 1:
                    padding[1] = range[1] + half_size - 1 - dim_size

                paddings.append(tuple(padding))

                slices.append(
                    slice(
                        range[0] - half_size + padding[0],
                        range[1] + padding[0] + half_size - 1,
                    )
                )

            # take only data for which we have labels
            raw_data = raw_data[tuple(slices)]
            raw_data = (raw_data / 255).astype(np.float32)

            # pad if labeled data is on the edge of the whole data cube
            padded_raw_data = np.pad(
                raw_data, pad_width=paddings, mode=padding_mode  # type: ignore
            )
            del raw_data
            log.info(f"Add padding {paddings} to raw data for {labels_path}")

            batches = patchify(padded_raw_data, batch_extent)
            self.raw_data_batches = batches

        log.info(f"Loading {labels_path} file with labels")
        with tifffile.TiffFile(labels_path) as labels_tif:
            labels = labels_tif.asarray()
            zero_dim = np.argmin(extent)
            labels = np.expand_dims(labels, zero_dim)

            assert np.array_equal(
                extent, labels.shape  # type: ignore
            ), "Provided ranges and the size of labels array should match"

            self.labels = np.where(labels == 0, 7, labels) - 1

    def __len__(self) -> int:
        sx, sy, sz, _, _, _ = self.raw_data_batches.shape
        return sx * sy * sz

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | np.int8]:
        sx, sy, sz, _, _, _ = self.raw_data_batches.shape
        real_idx = np.unravel_index(idx, (sx, sy, sz))

        data = self.raw_data_batches[real_idx]
        label = self.labels[real_idx]

        if self.transforms:
            data = self.transforms(image=data)["image"]

        batch = {"data": torch.from_numpy(data).unsqueeze(0), "label": label}

        return batch


class LabeledDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        transforms: list[volumentations.Transform] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.augmentations = transforms

    @classmethod
    def load_split(
        cls,
        transforms: volumentations.Compose | None,
        ds_cfgs: ListConfig,
        size_power: int,
        padding_mode: str,
        raw_data_path: str | os.PathLike,
    ) -> ConcatDataset:
        datasets = []
        for cfg in ds_cfgs:
            datasets.append(
                LabeledDataset(
                    raw_data_path,
                    cfg.labels_path,
                    cfg.x_range,
                    cfg.y_range,
                    cfg.z_range,
                    size_power,
                    padding_mode,
                    transforms,
                )
            )
        return ConcatDataset(datasets)

    def setup(self, stage: str) -> None:
        log.info(f"Setup datamodule for {stage} stage")

        if stage == "fit":
            if self.augmentations:
                transforms = volumentations.Compose(self.augmentations)
            else:
                transforms = None

            self.train_ds = LabeledDataModule.load_split(
                transforms,
                self.cfg.train_ds_configs,
                self.cfg.size_power,
                self.cfg.padding_mode,
                self.cfg.raw_data_path,
            )
        if stage == "validate" or stage == "fit":
            self.valid_ds = LabeledDataModule.load_split(
                None,
                self.cfg.valid_ds_configs,
                self.cfg.size_power,
                self.cfg.padding_mode,
                self.cfg.raw_data_path,
            )
        elif stage == "test":
            self.test_ds = LabeledDataModule.load_split(
                None,
                self.cfg.test_ds_configs,
                self.cfg.size_power,
                self.cfg.padding_mode,
                self.cfg.raw_data_path,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_ds,
            batch_size=self.cfg.devel_batch_size,
            num_workers=self.cfg.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.devel_batch_size,
            num_workers=self.cfg.num_workers,
        )

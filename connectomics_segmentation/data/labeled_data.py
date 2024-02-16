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
from connectomics_segmentation.utils.paddings import (
    calculate_full_padding_and_slices,
    calculate_tiling_padding,
)

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
        subvolume_size: int = 1,
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
        axis_order = np.argsort(extent)

        voxel_size = 2**size_power

        with tifffile.TiffFile(raw_data_path) as raw_data_tif:
            raw_data = raw_data_tif.asarray()

            paddings, slices = calculate_full_padding_and_slices(
                raw_data.shape,
                voxel_size=voxel_size,
                subvolume_size=subvolume_size,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
            )

            # take only data for which we have labels
            raw_data = raw_data[slices]
            raw_data = (raw_data / 255).astype(np.float32)

            # pad if labeled data is on the edge of the whole data cube
            if np.any(paddings):
                raw_data = np.pad(
                    raw_data, pad_width=paddings, mode=padding_mode  # type: ignore
                )
                log.info(f"Add padding {paddings} to raw data for {labels_path}")

            # sort axis to make all items from different slices equally
            # shaped for simple batching from concatenated dataset
            raw_data = raw_data.transpose(axis_order)

            data_subvol_size = voxel_size + subvolume_size - 1
            batch_extent = (voxel_size, data_subvol_size, data_subvol_size)
            self.raw_data_batches = patchify(raw_data, batch_extent, subvolume_size)

        log.info(f"Loading {labels_path} file with labels")
        with tifffile.TiffFile(labels_path) as labels_tif:
            labels = labels_tif.asarray()
            labels = np.expand_dims(labels, axis_order[0])
            labels = labels.transpose(axis_order)

            assert np.array_equal(
                extent[axis_order], labels.shape  # type: ignore
            ), "Provided ranges and the size of labels array should match"

            tiling_padding = calculate_tiling_padding(
                subvolume_size, extent[axis_order]
            )

            labels = np.where(labels == 0, 7, labels) - 1
            if np.any(tiling_padding):
                pads = tuple((0, p) for p in tiling_padding)
                labels = np.pad(  # type: ignore
                    labels, pad_width=pads, mode="constant", constant_values=6
                )

            batch_shape = (1, subvolume_size, subvolume_size)
            self.labels = patchify(labels, batch_shape, step=subvolume_size)

    def __len__(self) -> int:
        sx, sy, sz, _, _, _ = self.raw_data_batches.shape
        return sx * sy * sz

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sx, sy, sz, _, _, _ = self.raw_data_batches.shape
        real_idx = np.unravel_index(idx, (sx, sy, sz))

        data = self.raw_data_batches[real_idx]
        labels = self.labels[real_idx]

        if self.transforms:
            augm_data = self.transforms(image=data, mask=labels)
            data = augm_data["image"]
            labels = augm_data["mask"]

        batch = {
            "data": torch.from_numpy(data).unsqueeze(0),
            "label": torch.LongTensor(labels.copy()),
        }

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
        subvolume_size: int,
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
                    subvolume_size,
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
                self.cfg.train_subvolume_size,
                self.cfg.raw_data_path,
            )
        if stage == "validate" or stage == "fit":
            self.valid_ds = LabeledDataModule.load_split(
                None,
                self.cfg.valid_ds_configs,
                self.cfg.size_power,
                self.cfg.padding_mode,
                self.cfg.devel_subvolume_size,
                self.cfg.raw_data_path,
            )
        elif stage == "test":
            self.test_ds = LabeledDataModule.load_split(
                None,
                self.cfg.test_ds_configs,
                self.cfg.size_power,
                self.cfg.padding_mode,
                self.cfg.devel_subvolume_size,
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

import typing
from typing import Iterator, Sized

import torch
from torch.utils.data import Dataset, Sampler


class SubDataset(Dataset):
    """The purpose of this class is to handle the case when
    Subset dataset is not suitable because a number of indices
    causes the OOM error. In order to achieve this SubDataset takes
    an original big dataset, a start index and the lengh of the current split.
    Shuffling is to be performed separatly
    """

    def __init__(self, dataset: Dataset, start_idx: int, lenght: int) -> None:
        assert len(dataset) >= start_idx + lenght  # type: ignore

        self.dataset = dataset
        self.start_idx = start_idx
        self.length = lenght

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> typing.Any:
        return self.dataset[self.start_idx + idx]


class BufferizedRandomSampler(Sampler[int]):
    def __init__(self, data_source: Sized, buffer_size: int) -> None:
        self.data_source = data_source
        self.buffer_size = buffer_size
        self.iter_num = 0

    def __iter__(self) -> Iterator[int]:
        for _ in range(len(self.data_source)):
            start_idx = (self.iter_num // self.buffer_size) * self.buffer_size
            shift_idx = self.iter_num % self.buffer_size
            if shift_idx == 0:
                self.shifts = torch.randperm(self.buffer_size).tolist()

            yield start_idx + self.shifts[shift_idx]

    def __len__(self) -> int:
        return len(self.data_source)

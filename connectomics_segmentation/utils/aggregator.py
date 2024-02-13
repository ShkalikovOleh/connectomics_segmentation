from typing import Any, Tuple

import numpy as np


class Aggregator:

    def __init__(self, shape: Tuple[int, int, int, int], dtype: Any = np.uint8) -> None:
        self._buffer = np.zeros(shape, dtype=dtype)
        self._shape = shape
        self._fill_size = np.zeros(3)
        self._curr_idx = 0

    def get_volume(self) -> np.ndarray:
        self._fill_size[:] = 0
        return self._buffer

    def add_batch(self, batch: np.ndarray) -> Tuple[bool, np.ndarray]:
        is_filled = False

        modulos = map(
            lambda stup: stup[0] % stup[1], zip(self._shape[1:], batch.shape[2:])
        )
        if any(modulos):
            raise ValueError(
                "The input batch should have a shape, such that the desired shape"
                "  is divisible by it"
            )
        elif batch.shape[1] != self._shape[0]:
            raise ValueError("Missmatched number of features")

        tile_shape = np.fromiter(
            map(
                lambda stup: stup[0] / stup[1],
                zip(self._shape[1:], batch.shape[2:]),
            ),
            dtype=np.int32,
        )

        for i, vol in enumerate(batch):
            idxs = np.unravel_index(self._curr_idx, tile_shape)
            start = [idx * s for (idx, s) in zip(idxs, batch.shape[2:])]
            self._curr_idx += 1

            s = [(int(f), int(f + e)) for f, e in zip(start, batch.shape[2:])]
            self._buffer[:, s[0][0] : s[0][1], s[1][0] : s[1][1], s[2][0] : s[2][1]] = (
                vol
            )

            is_filled = all(map(lambda t: t[0] == t[1] + 1, zip(tile_shape, idxs)))
            if is_filled:
                self._curr_idx = 0
                break

        return is_filled, batch[i + 1 :]  # type: ignore

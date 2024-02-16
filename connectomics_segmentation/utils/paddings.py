from typing import Tuple

import numpy as np


def calculate_tiling_padding(subvolume_size: int, extent) -> np.ndarray:
    tiling_padding = np.array(
        [
            (subvolume_size - s % subvolume_size) % subvolume_size if s != 1 else 0
            for s in extent
        ]
    )
    return tiling_padding


def calculate_full_padding_and_slices(
    raw_data_shape,
    voxel_size: int,
    subvolume_size: int,
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
    z_range: Tuple[int, int],
) -> Tuple[np.ndarray, Tuple]:
    ranges = [x_range, y_range, z_range]
    half_size = voxel_size // 2
    paddings = []
    extents = []

    slices = []
    for i, range in enumerate(ranges):
        padding = [0, 0]
        extents.append(range[1] - range[0])

        dim_size = raw_data_shape[i]
        padding[0] = max(half_size - range[0], 0)
        padding[1] = max(range[1] + half_size - 1 - dim_size, 0)

        paddings.append(padding)

    paddings = np.array(paddings)
    tiling_paddings = calculate_tiling_padding(
        subvolume_size, np.asarray(extents) + np.asarray(paddings).sum(axis=1)
    )

    for i, (padding, tile_pad, range) in enumerate(
        zip(paddings, tiling_paddings, ranges)
    ):
        if padding[1] == 0:
            dim_size = raw_data_shape[i]
            effective_tile_pad = max(range[1] + half_size - 1 + tile_pad - dim_size, 0)
            tile_ext = tile_pad - effective_tile_pad
        else:  # already have to pad
            tile_ext = 0
            effective_tile_pad = tile_pad

        slices.append(
            slice(
                range[0] - half_size + padding[0],
                range[1] + half_size - 1 - padding[1] + tile_ext,
            )
        )

        paddings[i, 1] += effective_tile_pad

    return paddings, tuple(slices)

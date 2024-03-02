import torch
from torch import nn

from .modules import ConvNextStage, LayerNorm


class ConvNext(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 4,
        depths: list[int] = [3, 3, 9, 3],
        dims: list[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        activation: str = "GELU",
        layer_scale_init_value: float = 1e-6,
        down_dilation: int = 1,
    ):
        super().__init__()

        dp_rates = [
            x.tolist()
            for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)
        ]

        if down_dilation > 1:
            patch_stride = 1
            down_stride = 1
            patch_dilation = patch_size
            down_dilation = down_dilation
        else:
            patch_stride = patch_size
            down_stride = 2
            patch_dilation = 1
            down_dilation = 1

        patch_layer = nn.Sequential(
            nn.Conv3d(
                in_channels,
                dims[0],
                kernel_size=patch_size,
                stride=patch_stride,
                dilation=patch_dilation,
            ),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )

        stages: list[nn.Module] = [patch_layer]
        prev_dim = dims[0]
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            stages.append(
                ConvNextStage(
                    in_channels=prev_dim,
                    out_channels=dim,
                    depth=depth,
                    activation=activation,
                    layer_scale_init_value=layer_scale_init_value,
                    drop_path_rates=dp_rates[i],
                    down_kernel_size=1 if i == 0 else 2,
                    down_stride=1 if i == 0 else down_stride,
                    down_dilation=1 if i == 0 else down_dilation,
                )
            )
            prev_dim = dim

        self.stages = nn.Sequential(*stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(x)

import torch
from torch import nn

from .conv_blocks import Conv3DBlock


class Upsample2ConvBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "GELU",
        norm: str = "batch",
        up_scale: int = 2,
        up_mode: str = "nearest",
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=up_scale, mode=up_mode),
            Conv3DBlock(
                in_features, out_features, padding=1, norm=norm, activation=activation
            ),
            Conv3DBlock(
                out_features, out_features, padding=1, norm=norm, activation=activation
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

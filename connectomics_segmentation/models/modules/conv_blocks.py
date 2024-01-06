import torch
from torch import nn

from connectomics_segmentation.models.modules.utils import (
    get_activation_layer,
    get_norm_layer,
)


class Conv3DBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 3,
        stride: int = 1,
        dillation: int = 1,
        groups: int = 1,
        padding: int = 0,
        bias: bool = True,
        norm: str = "batch",
        activation: str = "LeakyReLU",
        norm_before_act: bool = True,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        conv_layer = nn.Conv3d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dillation,
            groups=groups,
            bias=bias,
            padding=padding,
        )
        norm_layer = get_norm_layer(norm, out_features)
        act_layer = get_activation_layer(activation)

        layers.append(conv_layer)
        if norm_before_act:
            layers.append(norm_layer)
            layers.append(act_layer)
        else:
            layers.append(act_layer)
            layers.append(norm_layer)

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

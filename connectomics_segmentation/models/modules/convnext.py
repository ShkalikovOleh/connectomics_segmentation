import torch
from torch import nn

from .drop_path import DropPath
from .layer_norm import LayerNorm
from .utils import get_activation_layer


class ConvNextBlock(nn.Module):
    """Rewritten to 3D case ConvNext layer from
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/convnext/modeling_convnext.py
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0,
        activation: str = "GELU",
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv3d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.layernorm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = get_activation_layer(activation)
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.layer_scale_parameter = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        input = hidden_states
        x = self.dwconv(hidden_states)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W, D) -> (N, H, W, D, C)

        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.layer_scale_parameter is not None:
            x = self.layer_scale_parameter * x

        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, D, C) -> (N, C, H, W, D)

        x = input + self.drop_path(x)

        return x


class ConvNextStage(nn.Module):
    """ConvNeXT stage, consisting of an optional downsampling layer + multiple
    residual blocks.

    Adopted from
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/convnext/modeling_convnext.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 2,
        down_kernel_size: int = 2,
        down_stride: int = 2,
        down_dilation: int = 1,
        activation: str = "GELU",
        layer_scale_init_value: float = 1e-6,
        drop_path_rates: list[float] | None = None,
    ):
        super().__init__()

        if in_channels != out_channels or down_stride > 1 or down_dilation > 1:
            self.downsampling_layer = nn.Sequential(
                LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=down_kernel_size,
                    stride=down_stride,
                    dilation=down_dilation,
                ),
            )
        else:
            self.downsampling_layer = nn.Identity()

        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = nn.Sequential(
            *[
                ConvNextBlock(
                    dim=out_channels,
                    drop_path=drop_path_rates[j],
                    activation=activation,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for j in range(depth)
            ]
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        hidden_states = self.downsampling_layer(hidden_states)
        hidden_states = self.layers(hidden_states)
        return hidden_states

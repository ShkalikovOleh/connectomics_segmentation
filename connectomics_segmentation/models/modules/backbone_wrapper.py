import torch
from torch import nn


class BackboneWrapper(nn.Module):
    """Act like nn.Sequential but allows to return
    values from intermediate layers specified by user
    """

    def __init__(self, out_idx: list[int], layers: list[nn.Module]) -> None:
        super().__init__()

        assert max(out_idx) < len(
            layers
        ), "Some out indices greater than number of layer"

        self.out_idx = out_idx
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs: list[torch.Tensor] = []

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_idx:
                outs.append(x)

        return outs

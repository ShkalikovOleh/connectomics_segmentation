import torch
from torch import nn


class ResidualModule(nn.Module):
    """Represents residual module which adds output of the main path to the
    original input or result of another module in case if a residual module
    is specified by user
    """

    def __init__(
        self,
        main_module: nn.Module,
        res_module: nn.Module | None = None,
        op: str = "sum",
    ) -> None:
        super().__init__()
        self.main_module = main_module
        self.res_module = res_module if res_module else nn.Identity()

        match op:
            case "sum":
                self.op = lambda a, b: a + b
            case "concat":
                self.op = lambda a, b: torch.cat([a, b], dim=1)
            case _:
                raise ValueError(f"Unsupported operatoion {op}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = self.res_module(x)
        main_out = self.main_module(x)
        return self.op(main_out, residue)

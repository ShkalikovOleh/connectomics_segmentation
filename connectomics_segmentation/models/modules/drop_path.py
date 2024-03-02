import torch
from torch import nn


class DropPath(nn.Module):

    def __init__(self, drop_prob: float | None = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation is adopted from
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/convnext/modeling_convnext.py
        """

        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob  # type: ignore
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        out = x.div(keep_prob) * random_tensor

        return out

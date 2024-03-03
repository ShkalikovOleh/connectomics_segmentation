import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(
        self,
        gamma,
        weights: torch.Tensor | None = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.register_buffer("weights", weights)
        self.gamma = gamma
        self.ignore_idx = ignore_index
        self.label_smoothing = label_smoothing

    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        flatten_pred = pred.permute(0, 2, 3, 4, 1).flatten(end_dim=3)
        flatten_target = target.flatten()

        mask = flatten_target != self.ignore_idx
        masked_pred = flatten_pred[mask]
        masked_target = flatten_target[mask]

        p = F.softmax(masked_pred, dim=1).gather(1, masked_target.unsqueeze(dim=1))
        mult = (1 - p) ** self.gamma
        ce = F.cross_entropy(
            masked_pred,
            masked_target,
            weight=self.weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        return (ce * mult).mean()

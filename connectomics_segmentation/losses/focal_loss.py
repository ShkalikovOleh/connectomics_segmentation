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

    def __call__(self, pred, target):
        mask = target != self.ignore_idx
        masked_pred = pred[:, :, mask[0]]
        masked_target = target[mask].unsqueeze(dim=0)

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

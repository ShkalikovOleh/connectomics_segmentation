from itertools import chain
from typing import Any, Callable, Iterator

import torch
from lightning.pytorch import LightningModule
from torch import nn

from connectomics_segmentation.models.vae import VAE

opt_factory = Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer]
lr_sched_factory = Callable[
    [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
]


class VAEMetaModel(LightningModule):
    def __init__(
        self,
        model: VAE,
        recon_loss: nn.Module,
        optimizer_factory: opt_factory,
        kl_loss_weight: float,
        lr_scheduler_factory: lr_sched_factory | None = None,
        compile_model: bool = True,
    ) -> None:
        super().__init__()

        self.model = model
        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory
        self.compile_model = compile_model

        self.recon_loss = recon_loss
        self.kl_loss_weight = kl_loss_weight

    def setup(self, stage: str) -> None:
        if self.compile_model and stage == "fit":
            self.model = torch.compile(self.model)
        return super().setup(stage)

    def configure_optimizers(self) -> dict[str, Any]:
        params = self.model.parameters()
        optimizer = self.optimizer_factory(params)

        if self.lr_scheduler_factory is not None:
            lr_scheduler = self.lr_scheduler_factory(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def _step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        mean, logvar = self.model.encode(batch)
        latent = self.model.reparametrize(mean, logvar)
        recon = self.model.decode(latent)

        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=1), dim=0
        )
        rec_loss = self.recon_loss(recon, batch)

        loss = self.kl_loss_weight * kl_loss + rec_loss

        self.log(f"{stage}/recon_loss", rec_loss)
        self.log(f"{stage}/kl_loss", kl_loss)
        self.log(f"{stage}/loss", loss)

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self._step(batch, "test")


class CenterVoxelRegressionMetaModel(LightningModule):
    def __init__(
        self,
        backbone_model: torch.nn.Module,
        head_model: torch.nn.Module,
        loss: nn.Module,
        optimizer_factory: opt_factory,
        lr_scheduler_factory: lr_sched_factory | None = None,
        dropout_prob: float = 0.3,
        compile_model: bool = True,
    ) -> None:
        super().__init__()

        self.backbone_model = backbone_model
        self.head_model = head_model
        self.loss_module = loss
        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory
        self.compile_model = compile_model
        self.dropout_prob = dropout_prob

    def setup(self, stage: str) -> None:
        if self.compile_model and stage == "fit":
            self.backbone_model = torch.compile(self.backbone_model)
            self.head_model = torch.compile(self.head_model)
        return super().setup(stage)

    def configure_optimizers(self) -> dict[str, Any]:
        backbone_params = self.backbone_model.parameters()
        head_params = self.head_model.parameters()
        params = chain(backbone_params, head_params)
        optimizer = self.optimizer_factory(params)

        if self.lr_scheduler_factory is not None:
            lr_scheduler = self.lr_scheduler_factory(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        features = self.backbone_model(data)
        return self.head_model(features)

    def _step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        # targets are center voxels
        H = batch.shape[2]
        targets = batch[:, 0, H // 2, H // 2, H // 2].detach().clone().unsqueeze_(1)

        # mask center voxel and several others randomly
        batch[:, 0, H // 2, H // 2, H // 2] = 0
        nn.functional.dropout(
            batch, p=self.dropout_prob, training=self.training, inplace=True
        )
        print(torch.sum(batch == 0))

        preds = self.forward(batch)

        loss = self.loss_module(preds, targets)

        self.log(f"{stage}/loss", loss)

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self._step(batch, "test")

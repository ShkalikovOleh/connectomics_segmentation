import torch
from torch import nn


class VAE(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        enc_out_feat_dim: int,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

        self.mean_head = nn.Linear(enc_out_feat_dim, latent_dim)
        self.logvar_head = nn.Linear(enc_out_feat_dim, latent_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)

        features = torch.flatten(features, start_dim=1)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)

        return mean, logvar

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def reparametrize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * logvar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.encode(x)
        latent = self.reparametrize(mean, logvar)
        return self.decode(latent)

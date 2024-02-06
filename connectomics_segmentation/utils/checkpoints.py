from collections import OrderedDict
import torch
from torch import nn


def load_pretrained_backbone(
    backbone_model: nn.Module,
    ckpt_path: str,
    load_vae_mean_head: bool = False,
) -> None:
    ckpt = torch.load(ckpt_path)

    backbone_prefix = "backbone_model."
    back_key_shift = len(backbone_prefix)

    backbone_state_dict = {}

    for key in ckpt["state_dict"]:
        if key.startswith(backbone_prefix):
            value = ckpt["state_dict"][key]
            new_key = key[back_key_shift:]
            backbone_state_dict[new_key] = value

    backbone_state_dict = OrderedDict(backbone_state_dict)
    backbone_model.load_state_dict(backbone_state_dict)

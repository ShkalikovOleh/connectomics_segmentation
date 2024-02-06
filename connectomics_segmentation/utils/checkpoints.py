from collections import OrderedDict
import torch
from torch import nn


def load_pretrained_backbone(
    backbone_model: nn.Module, ckpt_path: str, load_vae_mean_head: bool = False
) -> nn.Module:
    ckpt = torch.load(ckpt_path)
    is_vae = any(
        map(lambda k: k.startswith("model.decoder"), ckpt["state_dict"].keys())
    )

    if is_vae:
        backbone_prefix = "model.encoder."
    else:
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

    if load_vae_mean_head:
        mean_head_w = ckpt["state_dict"]["model.mean_head.weight"]
        mean_head_b = ckpt["state_dict"]["model.mean_head.bias"]

        out_feat, in_feat = mean_head_w.shape
        vae_mean_module = nn.Linear(in_features=in_feat, out_features=out_feat)

        vae_mean_state_dict = OrderedDict({"weight": mean_head_w, "bias": mean_head_b})
        vae_mean_module.load_state_dict(vae_mean_state_dict)

        return nn.Sequential(backbone_model, nn.Flatten(), vae_mean_module)
    else:
        return backbone_model

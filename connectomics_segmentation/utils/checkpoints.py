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

    has_orig_prefix = any(map(lambda k: "_orig_mod" in k, ckpt["state_dict"].keys()))

    if is_vae:
        backbone_prefix = "model.encoder."
    else:
        backbone_prefix = "backbone_model."
    if has_orig_prefix:
        backbone_prefix += "_orig_mod."
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
        weights_key = "model.mean_head.weight"
        bias_key = "model.mean_head.bias"
        if has_orig_prefix:
            weights_key += "_orig_mod."
            bias_key += "_orig_mod."

        mean_head_w = ckpt["state_dict"][weights_key]
        mean_head_b = ckpt["state_dict"][bias_key]

        out_feat, in_feat = mean_head_w.shape
        vae_mean_module = nn.Linear(in_features=in_feat, out_features=out_feat)

        vae_mean_state_dict = OrderedDict({"weight": mean_head_w, "bias": mean_head_b})
        vae_mean_module.load_state_dict(vae_mean_state_dict)

        return nn.Sequential(backbone_model, nn.Flatten(), vae_mean_module)
    else:
        return backbone_model


def load_head(head_model: nn.Module, ckpt_path: str) -> nn.Module:
    ckpt = torch.load(ckpt_path)

    has_orig_prefix = any(map(lambda k: "_orig_mod" in k, ckpt["state_dict"].keys()))

    head_prefix = "head_model."
    if has_orig_prefix:
        head_prefix += "_orig_mod."
    back_key_shift = len(head_prefix)

    head_model_state_dict = {}

    for key in ckpt["state_dict"]:
        if key.startswith(head_prefix):
            value = ckpt["state_dict"][key]
            new_key = key[back_key_shift:]
            head_model_state_dict[new_key] = value

    head_model_state_dict = OrderedDict(head_model_state_dict)
    head_model.load_state_dict(head_model_state_dict)

    return head_model

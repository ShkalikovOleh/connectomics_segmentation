from torch import nn


def get_norm_layer(norm: str, n_features: int) -> nn.Module:
    match norm:
        case "":
            norm_layer = nn.Identity()
        case "batch":
            norm_layer = nn.BatchNorm3d(n_features)
        case "instance":
            norm_layer = nn.InstanceNorm3d(n_features)
        case _:
            raise ValueError(f"Don't support {norm} normalization")
    return norm_layer


def get_activation_layer(activation: str) -> nn.Module:
    match activation:
        case "":
            act_layer = nn.Identity()
        case "ReLU":
            act_layer = nn.ReLU()
        case _ if "LeakyReLU" in activation:
            slope = activation.removeprefix("LeakyReLU")
            if len(slope) > 0:
                slope = float(slope)
                act_layer = nn.LeakyReLU(slope)
            else:
                act_layer = nn.LeakyReLU()
        case "SiLU":
            act_layer = nn.SiLU()
        case "GELU":
            act_layer = nn.GELU()
        case "GLU":
            act_layer = nn.GLU()
        case _:
            raise ValueError(f"Don't support {activation} activation")
    return act_layer

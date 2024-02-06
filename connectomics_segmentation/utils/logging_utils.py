# Source: https://github.com/ashleve/lightning-hydra-template/blob/bddbc24b82ab6ccfa6243e815a49dc5bfe8d4144/src/utils/logging_utils.py # noqa

from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf
from torch import nn

from connectomics_segmentation.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def log_num_param(hparams, model: nn.Module):
    hparams["model/backbone/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/backbone/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/backbone/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    if object_dict.get("backbone_model"):
        backbone_model = object_dict["backbone_model"]
        head_model = object_dict["head_model"]
        # save number of model parameters
        log_num_param(hparams, backbone_model)
        log_num_param(hparams, head_model)
    else:
        log_num_param(hparams, object_dict["model"])

    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]  # type: ignore

    hparams["data"] = cfg["data"]  # type: ignore
    hparams["trainer"] = cfg["trainer"]  # type: ignore

    hparams["callbacks"] = cfg.get("callbacks")  # type: ignore

    hparams["task_name"] = cfg.get("task_name")  # type: ignore
    hparams["ckpt_path"] = cfg.get("ckpt_path")  # type: ignore
    hparams["seed"] = cfg.get("seed")  # type: ignore

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)

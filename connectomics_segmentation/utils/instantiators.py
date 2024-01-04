# Source: https://github.com/ashleve/lightning-hydra-template/blob/bddbc24b82ab6ccfa6243e815a49dc5bfe8d4144/src/utils/instantiators.py  # noqa
import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from connectomics_segmentation.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config.

    Args:
        callbacks_cfg (DictConfig): A DictConfig object containing callback
        configurations.
    Returns:
        list[Callback]: A list of instantiated callbacks.
    """
    callbacks: list[Callback] = []

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(loggers_cfg: DictConfig) -> list[Logger]:
    """Instantiates loggers from config.

    Args:
        loggers_cfg (DictConfig): Logger configs
    Returns:
        list[Logger]: A list of instantiated PL loggers.
    """

    loggers: list[Logger] = []

    for _, log_conf in loggers_cfg.items():
        if isinstance(log_conf, DictConfig) and "_target_" in log_conf:
            log.info(f"Instantiating logger <{log_conf._target_}>")
            loggers.append(hydra.utils.instantiate(log_conf))

    return loggers

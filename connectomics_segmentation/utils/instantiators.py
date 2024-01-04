import hydra
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig


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
            loggers.append(hydra.utils.instantiate(log_conf))

    return loggers

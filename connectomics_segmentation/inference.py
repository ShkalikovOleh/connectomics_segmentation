import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from connectomics_segmentation.data.raw_data import RawDataModule
from connectomics_segmentation.supervised_meta_model import SupervisedMetaModel
from connectomics_segmentation.utils.checkpoints import (
    load_head,
    load_pretrained_backbone,
)
from connectomics_segmentation.utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
)
from connectomics_segmentation.utils.pylogger import RankedLogger
from connectomics_segmentation.utils.resolvers import register_custom_resolvers

log = RankedLogger(__name__, rank_zero_only=True)


def noop(any) -> None:
    pass


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        log.info(f"Set seed to {cfg.seed}")
        L.seed_everything(cfg.seed, workers=True)

    torch.set_float32_matmul_precision("high")

    log.info("Instantiate callbacks")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiate loggers")
    loggers = instantiate_loggers(cfg.get("loggers"))

    log.info("Instantiate model and load checkpoint")
    backbone_model = instantiate(cfg.model.backbone.net)
    head_model = instantiate(cfg.model.head.net)

    backbone_model = load_pretrained_backbone(backbone_model, cfg.ckpt_path)
    head_model = load_head(head_model, cfg.ckpt_path)

    module = SupervisedMetaModel(
        backbone_model=backbone_model,
        head_model=head_model,
        loss=None,  # type: ignore
        optimizer_factory=noop,  # type: ignore
        class_names=OmegaConf.to_object(cfg.class_names),  # type: ignore
        log_train_metrics=False,
    )

    log.info("Create raw data module")
    dm = RawDataModule(cfg.data)

    log.info("Instantiate trainer")
    trainer = instantiate(cfg.trainer, logger=loggers, callbacks=callbacks)

    log.info("Start prediction")
    trainer.predict(model=module, datamodule=dm, return_predictions=False)

    for logger in module.loggers:
        if isinstance(logger, WandbLogger):
            import wandb

            wandb.finish(quiet=True)
            break


register_custom_resolvers()
if __name__ == "__main__":
    main()

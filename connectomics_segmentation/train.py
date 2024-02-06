import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from connectomics_segmentation.data.labeled_data import LabeledDataModule
from connectomics_segmentation.data.raw_data import RawDataModule
from connectomics_segmentation.supervised_meta_model import SupervisedMetaModel
from connectomics_segmentation.unsurervised_meta_models import VAEMetaModel
from connectomics_segmentation.utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
)
from connectomics_segmentation.utils.logging_utils import log_hyperparameters
from connectomics_segmentation.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        log.info(f"Set seed to {cfg.seed}")
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiate callbacks")
    callbacks = instantiate_callbacks(cfg.callbacks)

    log.info("Instantiate loggers")
    loggers = instantiate_loggers(cfg.loggers)

    log.info("Instantiate loss")
    loss = instantiate(cfg.model.loss)

    log.info("Instantiate optimizer")
    optim_factory = instantiate(cfg.model.optimizer)
    if cfg.model.get("lr_scheduler"):
        log.info("Instantiate LR scheduler")
        sched_factory = instantiate(cfg.model.lr_scheduler)
    else:
        sched_factory = None

    torch.set_float32_matmul_precision("high")

    hparams = {"cfg": cfg}

    if cfg.supervised:
        log.info("Instantiate model")
        backbone_model = instantiate(cfg.model.supervised.backbone)
        head_model = instantiate(cfg.model.supervised.head)

        module = SupervisedMetaModel(
            backbone_model=backbone_model,
            head_model=head_model,
            loss=loss,
            optimizer_factory=optim_factory,
            num_classes=cfg.model.num_classes,
            lr_scheduler_factory=sched_factory,
            class_names=OmegaConf.to_object(cfg.data.class_names),  # type: ignore
            compile_model=cfg.model.compile_model,
        )

        if cfg.data.get("augmentations"):
            log.info("Instantiate augmentations")
            augmentations = instantiate(cfg.data.augmentations)
        else:
            augmentations = None

        log.info("Create labeled data module")
        dm = LabeledDataModule(cfg.data, augmentations)

        hparams["backbone_model"] = backbone_model
        hparams["head_model"] = head_model
    else:
        log.info("Instantiate model")
        if cfg.model.unsupervised.get("vae"):
            model = instantiate(cfg.model.unsupervised.vae)

            module = VAEMetaModel(
                model=model,
                recon_loss=loss,
                optimizer_factory=optim_factory,
                kl_loss_weight=cfg.model.unsupervised.kl_loss_weight,
                lr_scheduler_factory=sched_factory,
                compile_model=cfg.model.compile_model,
            )

            hparams["model"] = model
        else:
            return

        log.info("Create raw data module")
        dm = RawDataModule(cfg.data)

    log.info("Instantiate trainer")
    trainer = instantiate(cfg.trainer, logger=loggers, callbacks=callbacks)
    hparams["trainer"] = trainer

    log.info("Logging hyperparameters")
    log_hyperparameters(hparams)

    log.info("Start training")
    trainer.fit(module, datamodule=dm, ckpt_path=cfg.get("ckpt_path"))

    if cfg.run_test:
        log.info("Start testing")
        trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    main()

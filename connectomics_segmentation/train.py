import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig

from connectomics_segmentation.data.labeled_data import LabeledDataModule
from connectomics_segmentation.data.raw_data import RawDataModule
from connectomics_segmentation.supervised_meta_model import SupervisedMetaModel
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

    log.info("Instantiate model")
    net = instantiate(cfg.model.net)

    log.info("Instantiate loss")
    loss = instantiate(cfg.model.loss)

    log.info("Instantiate optimizer")
    optim_factory = instantiate(cfg.model.optimizer)
    if cfg.model.get("lr_scheduler"):
        log.info("Instantiate LR scheduler")
        sched_factory = instantiate(cfg.model.lr_scheduler)
    else:
        sched_factory = None

    if cfg.supervised:
        module = SupervisedMetaModel(
            model=net,
            loss=loss,
            optimizer_factory=optim_factory,
            num_classes=cfg.model.num_classes,
            lr_scheduler_factory=sched_factory,
            compile_model=cfg.model.compile_model,
        )

        if cfg.data.get("augmentations"):
            log.info("Instantiate augmentations")
            augm_factory = instantiate(cfg.data.augmentations)
        else:
            augm_factory = None

        log.info("Create labeled data module")
        dm = LabeledDataModule(cfg.data, augm_factory)
    else:
        log.info("Create raw data module")
        dm = RawDataModule(cfg.data)
        return

    log.info("Instantiate trainer")
    trainer = instantiate(cfg.trainer, logger=loggers, callbacks=callbacks)

    log.info("Logging hyperparameters")
    hparams = {
        "cfg": cfg,
        "model": net,
        "trainer": trainer,
    }
    log_hyperparameters(hparams)

    log.info("Start training")
    trainer.fit(module, datamodule=dm, ckpt_path=cfg.get("ckpt_path"))

    if cfg.run_test:
        log.info("Start testing")
        trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    main()

from hydra.core.config_store import ConfigStore

from connectomics_segmentation.data.labeled_data import LabeledDataModuleConfig


def register_configs():
    cs = ConfigStore.instance()
    cs.store(group="data", name="labeled_data_config", node=LabeledDataModuleConfig)

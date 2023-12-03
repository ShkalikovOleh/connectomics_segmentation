from hydra.core.config_store import ConfigStore

from connectomics_segmentation.data.labeled_data import LabeledDataModuleConfig
from connectomics_segmentation.data.raw_data import RawDataModuleConfig


def register_configs():
    cs = ConfigStore.instance()
    cs.store(group="data", name="labeled_data_config", node=LabeledDataModuleConfig)
    cs.store(group="data", name="raw_data_config", node=RawDataModuleConfig)

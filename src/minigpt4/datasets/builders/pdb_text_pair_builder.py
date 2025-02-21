import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.pdb_base_dataset_builder import PDB_BaseDatasetBuilder
from minigpt4.datasets.datasets.pdb_dataset import ESMDataset

@registry.register_builder("pdb")
class PDBBuilder(PDB_BaseDatasetBuilder):
    train_dataset_cls = ESMDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/pdb/pdb.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'filter_cap.json')],
            pdb_root=os.path.join(storage_path, 'pdb'),
        )

        return datasets

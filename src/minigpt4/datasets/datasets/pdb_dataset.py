import os
import sys
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import json

import esm
import esm.inverse_folding

class ESMDataset(Dataset):
    def __init__(self, pdb_root, ann_paths, chain="C"):
        """
        protein (string): Root directory of protein (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(pdb_root, ann_paths)
        self.pdb_root = pdb_root
        self.annotation = json.load(open(ann_paths, "r"))
        self.pdb_ids = {}
        self.chain = chain
        self.pdb_root = pdb_root

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        return None

        # ann = self.annotation[index]
        #
        # pdb_file = '{}.pdb'.format(ann["pdb_id"])
        #
        # pdb_path = os.path.join(self.pdb_root, pdb_file)
        # # read in the pdb from pdb_path
        # coords, native_seq = esm.inverse_folding.util.load_coords(pdb_path, self.chain)
        #
        # caption = ann["caption"]
        #
        # return {
        #     "pdb_coords": coords,
        #     "native_seq": native_seq,
        #     "text_input": caption,
        #     "pdb_id": ann["pdb_id"]
        # }
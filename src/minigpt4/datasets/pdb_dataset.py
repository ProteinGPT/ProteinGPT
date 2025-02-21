import os
import sys

import torch
from torch.utils.data import Dataset
import json
import numpy as np
from torch.utils.data.dataloader import default_collate

import time


class ESMDataset(Dataset):
    def __init__(self, pdb_root, seq_root, ann_paths, dataset_description, chain="A"):
        """
        pdb_root (string): Root directory of protein pdb embeddings (e.g. xyz/pdb/)
        seq_root (string): Root directory of sequences embeddings (e.g. xyz/seq/)
        ann_root (string): directory to store the annotation file
        dataset_description (string): json file that describes what data are used for training/testing
		"""
        data_describe = json.load(open(dataset_description, "r"))
        train_set = set(data_describe["train"])
        self.pdb_root = pdb_root
        self.seq_root = seq_root
        self.annotation = json.load(open(ann_paths, "r"))
        keep = []
        for i in range(0, len(self.annotation)):
            if (self.annotation[i]["pdb_id"] in train_set):
                keep.append(self.annotation[i])
        self.annotation = keep
        self.pdb_ids = {}
        self.chain = chain

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        pdb_embedding = '{}.pt'.format(ann["pdb_id"])
        pdb_embedding_path = os.path.join(self.pdb_root, pdb_embedding)
        pdb_embedding = torch.load(
            pdb_embedding_path, map_location=torch.device('cpu'))
            # pdb_embedding_path, map_location=torch.device('cuda'))
        pdb_embedding.requires_grad = False

        seq_embedding = '{}.pt'.format(ann["pdb_id"])
        seq_embedding_path = os.path.join(self.seq_root, seq_embedding)
        seq_embedding = torch.load(
            seq_embedding_path, map_location=torch.device('cpu'))
            # seq_embedding_path, map_location=torch.device('cuda'))
        seq_embedding.requires_grad = False

        caption = ann["caption"]

        return {
            "text_input": caption,
            "pdb_encoder_out": pdb_embedding,
            "seq_encoder_out": seq_embedding,
            "chain": self.chain,
            "pdb_id": ann["pdb_id"]
        }

    # Yijia please check :)
    # def collater(self, samples):
    #     # print(samples)
    #     max_len_pdb_dim0 = -1
    #     max_len_seq_dim0 = -1

    #     for pdb_json in samples:
    #         pdb_embeddings = pdb_json["pdb_encoder_out"]
    #         shape_dim0 = pdb_embeddings.shape[0]
    #         max_len_pdb_dim0 = max(max_len_pdb_dim0, shape_dim0)

    #         seq_embeddings = pdb_json["seq_encoder_out"]
    #         shape_dim0 = seq_embeddings.shape[0]
    #         max_len_seq_dim0 = max(max_len_seq_dim0, shape_dim0)

    #     for pdb_json in samples:
    #         pdb_embeddings = pdb_json["pdb_encoder_out"]
    #         shape_dim0 = pdb_embeddings.shape[0]
    #         pad1 = ((0, max_len_pdb_dim0 - shape_dim0), (0, 0), (0, 0))
    #         arr1_padded = np.pad(pdb_embeddings, pad1, mode='constant', )
    #         pdb_json["pdb_encoder_out"] = arr1_padded

    #         seq_embeddings = pdb_json["seq_encoder_out"]
    #         shape_dim0 = seq_embeddings.shape[0]
    #         pad1 = ((0, max_len_seq_dim0 - shape_dim0), (0, 0), (0, 0))
    #         arr1_padded = np.pad(seq_embeddings, pad1, mode='constant', )
    #         pdb_json["seq_encoder_out"] = arr1_padded

    #     print(samples[0].keys())
    #     return default_collate(samples)

def collater(self, samples):
    max_len_pdb_dim0 = max(pdb_json["pdb_encoder_out"].shape[0] for pdb_json in samples)
    max_len_seq_dim0 = max(pdb_json["seq_encoder_out"].shape[0] for pdb_json in samples)

    for pdb_json in samples:
        pdb_embeddings = pdb_json["pdb_encoder_out"]
        pad_pdb = ((0, max_len_pdb_dim0 - pdb_embeddings.shape[0]), (0, 0), (0, 0))
        pdb_json["pdb_encoder_out"] = torch.tensor(np.pad(pdb_embeddings, pad_pdb, mode='constant'))

        seq_embeddings = pdb_json["seq_encoder_out"]
        pad_seq = ((0, max_len_seq_dim0 - seq_embeddings.shape[0]), (0, 0), (0, 0))
        pdb_json["seq_encoder_out"] = torch.tensor(np.pad(seq_embeddings, pad_seq, mode='constant'))

    return default_collate(samples)

# import os
# import sys

# import torch
# from torch.utils.data import Dataset
# import json
# import numpy as np
# from torch.utils.data.dataloader import default_collate

# import time

# class ESMDataset(Dataset):
#     def __init__(self, pdb_root, ann_paths, chain="A"):
#         """
#         protein (string): Root directory of protein (e.g. coco/images/)
#         ann_root (string): directory to store the annotation file
#         """
#         self.pdb_root = pdb_root
#         self.annotation = json.load(open(ann_paths, "r"))
#         self.pdb_ids = {}
#         self.chain = chain

#     def __len__(self):
#         return len(self.annotation)

#     def __getitem__(self, index):

#         ann = self.annotation[index]

#         protein_embedding = '{}.pt'.format(ann["pdb_id"])

#         protein_embedding_path = os.path.join(self.pdb_root, protein_embedding)
#         protein_embedding = torch.load(protein_embedding_path, map_location=torch.device('cpu'))
#         protein_embedding.requires_grad = False
#         caption = ann["caption"]

#         return {
#             "text_input": caption,
#             "encoder_out": protein_embedding,
#             "chain": self.chain,
#             "pdb_id": ann["pdb_id"]
#         }

#     def collater(self, samples):
#         max_len_protein_dim0 = -1
#         for pdb_json in samples:
#             pdb_embeddings = pdb_json["encoder_out"]
#             shape_dim0 = pdb_embeddings.shape[0]
#             max_len_protein_dim0 = max(max_len_protein_dim0, shape_dim0)
#         for pdb_json in samples:
#             pdb_embeddings = pdb_json["encoder_out"]
#             shape_dim0 = pdb_embeddings.shape[0]
#             pad1 = ((0, max_len_protein_dim0 - shape_dim0), (0, 0), (0, 0))
#             arr1_padded = np.pad(pdb_embeddings, pad1, mode='constant', )
#             pdb_json["encoder_out"] = arr1_padded

#         return default_collate(samples)
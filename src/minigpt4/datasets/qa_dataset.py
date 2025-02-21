import os
import sys

import torch
from torch.utils.data import Dataset
import json
import numpy as np
from torch.utils.data.dataloader import default_collate

import time

class QADataset(Dataset):
    # def __init__(self, pdb_root, seq_root, ann_paths, dataset_description, chain="A"):
    def __init__(self, pdb_root, seq_root, ann_paths, chain="A"):
        """
        pdb_root (string): Root directory of protein pdb embeddings (e.g. xyz/pdb/)
        seq_root (string): Root directory of sequences embeddings (e.g. xyz/seq/)
        ann_root (string): directory to store the annotation file
        dataset_description (string): json file that describes what data are used for training/testing
		"""
        # data_describe = json.load(open(dataset_description, "r"))
        # train_set = set(data_describe["train"])
        self.pdb_root = pdb_root
        self.seq_root = seq_root
        self.qa = json.load(open(ann_paths, "r"))
        self.qa_keys = list(self.qa.keys())
        keep = {}
        # for i in range(0, len(self.qa_keys)):
        #     if (self.qa_keys[i] in train_set):
        #         keep[self.qa_keys[i]] = self.qa[self.qa_keys[i]]
        # self.qa = keep

        self.qa_keys = list(self.qa.keys()) # update qa keys to reflect what was saved after keep
        self.questions = []
        for key in self.qa_keys:
            for j in range(0, len(self.qa[key])):
                self.questions.append((self.qa[key][j], key))
        self.chain = chain

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        qa = self.questions[index]
        pdb_id = qa[1]
        
        pdb_embedding = '{}.pt'.format(pdb_id)
        pdb_embedding_path = os.path.join(self.pdb_root, pdb_embedding)
        pdb_embedding = torch.load(
            pdb_embedding_path, map_location=torch.device('cpu'))
            # pdb_embedding_path, map_location=torch.device('cuda'))
        pdb_embedding.requires_grad = False

        seq_embedding = '{}.pt'.format(pdb_id)
        seq_embedding_path = os.path.join(self.seq_root, seq_embedding)
        seq_embedding = torch.load(
            seq_embedding_path, map_location=torch.device('cpu'))
            # seq_embedding_path, map_location=torch.device('cuda'))
        seq_embedding.requires_grad = False

        return {
            "q_input": str(qa[0]['Q']),
            "a_input": str(qa[0]['A']),
            "pdb_encoder_out": pdb_embedding,
            "seq_encoder_out": seq_embedding,
            "chain": self.chain,
            "pdb_id": pdb_id 
        }
    
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

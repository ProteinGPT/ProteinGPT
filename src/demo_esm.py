import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation_esm import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
import sys

import esm


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--pdb", help="specifiy where the protein file is (.pt)")
    parser.add_argument("--seq", help="specifiy where the sequence file is (.pt)")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

chat_state = CONV_VISION.copy()
img_list = []

pdb_path = args.pdb
seq_path = args.seq
if pdb_path[-3:] == ".pt":
    pdb_embedding = torch.load(pdb_path, map_location=torch.device('cpu'))
    sample_pdb = pdb_embedding.to('cuda:{}'.format(args.gpu_id))
if seq_path[-3:] == ".pt":
    seq_embedding = torch.load(seq_path, map_location=torch.device('cpu'))
    sample_seq = seq_embedding.to('cuda:{}'.format(args.gpu_id))

llm_message = chat.upload_protein(sample_pdb, sample_seq, chat_state, img_list)
print(llm_message)

img_list = [mat.half() for mat in img_list]
while True:
    user_input = input(">")
    if (len(user_input) == 0):
        print("USER INPUT CANNOT BE EMPTY!")
        continue
    elif (user_input.lower() == "exit()"):
        break
    chat.ask(user_input, chat_state)
    llm_message = chat.answer(conv=chat_state,
                            img_list=img_list,
                            num_beams=1,
                            temperature=0.7,
                            max_new_tokens=300,
                            max_length=2000)[0]
    print("B: ", llm_message)

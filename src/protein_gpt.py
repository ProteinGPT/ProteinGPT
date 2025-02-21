"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
# import esm
import minigpt4.tasks as tasks
from minigpt4.esm.esm_config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import now

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.datasets.pdb_dataset import ESMDataset 
from minigpt4.datasets.qa_dataset import QADataset 
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=False, help="path to configuration file.", 
                        default='configs/train_modality_alignment.yaml')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls

def is_stage_1_training(cfg):
    return cfg.to_dict()["run"]["stage"] == 1


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    cfg = Config(parse_args())
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()
    task = tasks.setup_task(cfg)

    datasets_raw = []
    if (is_stage_1_training(cfg)):
        datasets_raw = ESMDataset(pdb_root="/home/ubuntu/pt/",
                                seq_root="/home/ubuntu/seq/",
                                ann_paths="/home/ubuntu/ProteinGPT/data/esm_subset/abstract.json",
                                dataset_description="/home/ubuntu/dataset.json",
                                chain="A")
    else:
        datasets_raw = QADataset(pdb_root="/home/ubuntu/pt/",
                                seq_root="/home/ubuntu/seq/",
                                ann_paths="/home/ubuntu/ProteinGPT/data/esm_subset/GPT_merged_summary.json",
                                chain="A")

    datasets = {'esm': {'train': datasets_raw}}
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )

    runner.train()


if __name__ == "__main__":
    main()

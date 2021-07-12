"""
-*- coding:utf-8 -*-
@author  : jiangmingchao@joyy.sg
@datetime: 2021-06-30
@describe: inference code
"""
from random import choice
import warnings
warnings.filterwarnings('ignore')

import os
import time
import json
import argparse
import numpy as np

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data.dataloader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model.CNN.resnet import resnet50
# swin-transformers
from model.Transformers.swin_transformers.models.build import build_model, build_model_features
from model.Transformers.swin_transformers.configs.config import get_config
# features
from model.CNN.resnet_features import R50Features
from data.ImagenetDataset import ImageDatasetTest
from utils.precise_bn import *

parser = argparse.ArgumentParser()
# ----- data ------
parser.add_argument('--test_file', type=str, default="")
parser.add_argument('--num-classes', type=int, default=1000)

# ----- model -----
parser.add_argument('--checkpoints-path', default='', type=str)
parser.add_argument('--input_size', default=256, type=int)
parser.add_argument('--crop_size', default=224, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--num-workers', default=32, type=int)
parser.add_argument('--save_folder', default="", type=str)
parser.add_argument('--FP16', default=0, type=int)
parser.add_argument('--get_features', default=0, type=int)


# ------swin-transformers------
parser.add_argument('--cfg', default="/data/jiangmingchao/data/code/ImageClassification/model/Transformers/swin_transformers/configs/swin_large_patch4_window12_384.yaml", type=str)
parser.add_argument('--swin', default=0, type=int)

# ------ddp -------
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--world-size', type=int, default=-1,
                    help="number of nodes for distributed training")
parser.add_argument('--rank', default=-1, type=int,
                help='node rank for distributed training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                help='distributed backend')
parser.add_argument('--multiprocessing-distributed', default=1, type=int,
                help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')
parser.add_argument('--local_rank', default=1)

args = parser.parse_args()

config = get_config(args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])

    if args.multiprocessing_distributed:
        args.rank = args.rank * ngpus_per_node + gpu

    if args.gpu is not None:
        print("Use GPU: {} for Testing".format(args.gpu))
    print('rank: {} / {}'.format(args.rank, args.world_size))

    if args.distributed:
        dist.init_process_group(
                                backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
        torch.cuda.set_device(args.gpu)

    if args.rank == 0:
        if not os.path.isfile(args.checkpoints_path):
            os.makedirs(args.checkpoints_path)
    
    if args.get_features:
        if args.swin:
            # swin-transformers
            model = build_model_features(config)
            if args.rank == 0:
                print(model)
        else:
            model = R50Features(args.checkpoints_path)
    
    else:
        if args.swin:
            # swin-transformers
            model = build_model(config)
            state_dict = torch.load(args.checkpoints_path, map_location="cpu")['model']
            model.load_state_dict(state_dict, strict=True)
            # if args.rank == 0:
            #     print(model)
        else:
            model = resnet50(pretrained=False, num_classes=args.num_classes)
            state_dict = torch.load(args.checkpoints_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
    if args.rank == 0:
        print(model)
    
    if args.FP16:
        model = model.half()
        for bn in get_bn_modules(model):
            bn.float()

    if torch.cuda.is_available():
        model.cuda(args.gpu)

    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    
    dataset = ImageDatasetTest(
        image_file = args.test_file,
        train_phase= False, 
        input_size = args.input_size,
        crop_size = args.crop_size,
        shuffle = False,
        mode = "cnn"
    )

    if args.rank == 0:
        print("Validation dataset length: ", len(dataset))

    if args.distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None 

    criterion = nn.CrossEntropyLoss()
    length = len(dataset)

    dataloader = DataLoader(
        dataset = dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers= args.num_workers,
        sampler = sampler, 
        drop_last = False 
    )
    validation(args, dataloader, model, criterion, length)


def validation(args, dataloader, model, criterion, length):
    model.eval()
    device = model.device 
    total_batch = int(length / (args.batch_size*8))

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    file = open(os.path.join(args.save_folder +'r50_features_'+ str(args.rank) + '.log') , "w")
    for batch_idx, data in enumerate(dataloader):
        batch_data, batch_label, batch_path = data[0], data[1], data[2]

        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)

        with torch.no_grad():
            start_time = time.time()
            
            if args.FP16:
                batch_data = batch_data.half()
            
            batch_output = model(batch_data)
            batch_losses = criterion(batch_output, batch_label)
            batch_logits = batch_output.cpu().numpy()

            for i in range(batch_logits.shape[0]):
                image_path = batch_path[i]
                output = batch_logits[i].tolist()
                gt = batch_label[i].data.item()
                result = {
                    "path"         : image_path,
                    "pred_logits"  : output,
                    "real_label"   : gt 
                }
                file.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            batch_time = time.time() - start_time
            if args.rank == 0:
                print(f"Validation Iter: [{batch_idx+1}/{total_batch}] losses: {batch_losses}  , batchtime: {batch_time}")

    file.close()


if __name__ == '__main__':
    args = parser.parse_args()

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        print("ngpus_per_node", ngpus_per_node)
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        print("ngpus_per_node", ngpus_per_node)
        main_worker(args.gpu, ngpus_per_node, args)
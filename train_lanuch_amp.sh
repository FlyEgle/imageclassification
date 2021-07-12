#!/bin/bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
export OMP_NUM_THREADS
export MKL_NUM_THREADS
cd /data/jiangmingchao/data/code/ImageClassification;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore -m torch.distributed.launch --nproc_per_node 8 train_lanuch.py \
--batch_size 512 \
--num_workers 48 \
--lr 1.6 \
--max_epochs 30 \
--warmup_epochs 5 \
--num-classes 1000 \
--input_size 256 \
--crop_size 224 \
--FP16 0 \
--mode O2 \
--amp 1 \
--apex 0 \
--train_file /data/jiangmingchao/data/dataset/imagenet/train_oss_imagenet_128w.txt \
--val_file /data/jiangmingchao/data/dataset/imagenet/val_oss_imagenet_128w.txt \
--log-dir /data/jiangmingchao/data/AICutDataset/imagenet/r50_amp_4k_1.6_30/log_dir \
--checkpoints-path /data/jiangmingchao/data/AICutDataset/imagenet/r50_amp_4k_1.6_30/checkpoints
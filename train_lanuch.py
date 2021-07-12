"""
-*- coding:utf-8 -*-
@author  : jiangmingchao@joyy.sg
@datetime: 2021-0628
@describe: Training loop 
"""
import warnings

from apex.amp import scaler
warnings.filterwarnings('ignore')

import os 
import time 
import math 
import random 
import argparse
import numpy as np 

# apex
try:
    from apex import amp 
    from apex.parallel import convert_syncbn_model
    from apex.parallel import DistributedDataParallel as DDP 
except Exception as e:
    print("amp have not been import !!!")


# actnn
try:
    import actnn
    actnn.set_optimization_level("L3")
except Exception as e:
    print("actnn have no import !!!")

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.cuda.amp import autocast as autocast

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DataParallel
from torch.utils.data.distributed import DistributedSampler

from model.CNN.resnet import resnet50
from data.ImagenetDataset import ImageDataset

from tensorboardX import SummaryWriter
from datetime import datetime
from utils.precise_bn import *


parser = argparse.ArgumentParser()
# ------ddp
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--distributed', default=1, type=int, help="use distributed method to training!!")
# ----- data
parser.add_argument('--train_file', type=str, default="/data/jiangmingchao/data/dataset/imagenet/train_oss_imagenet_128w.txt")
parser.add_argument('--val_file', type=str, default="/data/jiangmingchao/data/dataset/imagenet/val_oss_imagenet_128w.txt")
parser.add_argument('--num-classes', type=int)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--num_classes', type=int, default=1000)

# ----- checkpoints log dir 
parser.add_argument('--checkpoints-path', default='checkpoints', type=str)
parser.add_argument('--log-dir', default='logs', type=str)

# ---- optimizer
parser.add_argument('--lr', default=1e-1, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=8, type=int)

# ---- actnn 2-bit
parser.add_argument('--actnn', default=0, type=int)

# ---- train
parser.add_argument('--warmup_epochs', default=5, type=int)
parser.add_argument('--max_epochs', default=90, type=int)
parser.add_argument('--FP16', default=0, type=int)
parser.add_argument('--apex', default=0, type=int)
parser.add_argument('--mode', default='O1', type=str)
parser.add_argument('--amp', default=1, type=int)

# random seed
def setup_seed(seed=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def translate_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module' in key:
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        crr = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(1/batch_size).item()
            res.append(acc)  #  unit: percentage (%)
            crr.append(correct_k)
        return res, crr


class Metric_rank:
    def __init__(self, name):
        self.name = name
        self.sum = 0.0
        self.n = 0

    def update(self, val):
        self.sum += val
        self.n += 1

    @property
    def average(self):
        return self.sum / self.n


# main func
def main_worker(args):
    total_rank = torch.cuda.device_count()
    print('rank: {} / {}'.format(args.local_rank, total_rank))
    dist.init_process_group(backend=args.dist_backend)
    torch.cuda.set_device(args.local_rank)

    ngpus_per_node = total_rank

    if args.local_rank == 0:
        if not os.path.exists(args.checkpoints_path):
            os.makedirs(args.checkpoints_path)

    # metric
    train_losses_metric = Metric_rank("train_losses")
    train_accuracy_metric = Metric_rank("train_accuracy")
    train_metric = {"losses": train_losses_metric,
                    "accuracy": train_accuracy_metric}

    
    # model 
    model = resnet50(
        pretrained=False,
        num_classes=args.num_classes
        )
    if args.local_rank == 0:
        print(f"===============model arch ===============")
        print(model)
    
    # model mode
    model.train()
    
    if args.actnn:
        model = actnn.QModule(model)
        if args.local_rank == 0:
            print(model)

    if args.apex:
        model = convert_syncbn_model(model)

    # FP16
    if args.FP16:
        model = model.half()
        for bn in get_bn_modules(model):
            bn.float()

    if torch.cuda.is_available():
        model.cuda(args.local_rank)

    # loss 
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.SGD(
                            model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            momentum=args.momentum)
    
    if args.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.mode) 
        model = DDP(model, delay_allreduce=True)
    
    else:
        if args.distributed:
            model = DataParallel(model, 
                        device_ids=[args.local_rank], 
                        find_unused_parameters=True)

    # dataset & dataloader
    train_dataset = ImageDataset(
        image_file = args.train_file,
        train_phase= True,
        input_size = args.input_size,
        crop_size = args.crop_size,
        shuffle = True 
        )
    validation_dataset = ImageDataset(
        image_file = args.val_file,
        train_phase = False, 
        input_size = args.input_size,
        crop_size = args.crop_size,
        shuffle = False 
    )

    if args.local_rank == 0:
        print("Trainig dataset length: ", len(train_dataset))
        print("Validation dataset length: ", len(validation_dataset))

    # sampler
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        validation_sampler = DistributedSampler(validation_dataset)
    else:
        train_sampler = None
        validation_sampler = None

    # logs
    log_writer = SummaryWriter(args.log_dir)

    # dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=args.batch_size,
        shuffle=(validation_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=validation_sampler,
        drop_last=True
    )

    start_epoch = 1
    batch_iter = 0
    train_batch = math.ceil(len(train_dataset) / (args.batch_size * ngpus_per_node))
    total_batch = train_batch * args.max_epochs
    no_warmup_total_batch = int(args.max_epochs - args.warmup_epochs) * train_batch

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    best_loss, best_acc = np.inf, 0.0
    # training loop
    for epoch in range(start_epoch, args.max_epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for epoch
        batch_iter, scaler = train(args, scaler, train_loader, model, criterion, optimizer, epoch, batch_iter, total_batch, train_batch, log_writer, train_metric)

        # calculate the validation with the batch iter
        if epoch % 1 == 0:
            val_loss, val_acc = val(args, validation_loader, model, criterion, epoch, log_writer) 
            # recored & write 
            if args.local_rank == 0:
                best_loss = val_loss
                state_dict = translate_state_dict(model.state_dict())
                state_dict = {
                        'epoch': epoch,
                        'state_dict': state_dict,
                        'optimizer': optimizer.state_dict(),
                    }
                torch.save(
                    state_dict,
                    args.checkpoints_path + '/' 'r50' + f'_losses_{best_loss}' + '.pth'
                )

                best_acc = val_acc
                state_dict = translate_state_dict(model.state_dict())
                state_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state_dict,
                            args.checkpoints_path + '/' + 'r50' + f'_accuracy_{best_acc}' + '.pth')
        # model mode
        model.train()


# train function
def train(args,
        scaler,
        train_loader,
        model,
        criterion,
        optimizer,
        epoch,
        batch_iter,
        total_batch,
        train_batch,
        log_writer,
        train_metric,
        ):
    """Traing with the batch iter for get the metric
    """
    model.train()
    # device = model.device
    loader_length = len(train_loader)

    for batch_idx, data in enumerate(train_loader):
        batch_start = time.time()
        # step learning rate
        lr = adjust_learning_rate(
            args, epoch, batch_iter, optimizer, train_batch
        )

        # forward
        batch_data, batch_label = data[0], data[1]

        if args.FP16:
            batch_data = batch_data.half()

        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()

        if args.amp:
            with autocast():
                batch_output = model(batch_data)
                losses = criterion(batch_output, batch_label)
        else:
            batch_output = model(batch_data)
            losses = criterion(batch_output, batch_label)        

        optimizer.zero_grad()
        if args.apex:
            with amp.scale_loss(losses, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        
        elif args.amp:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        
        else:
            losses.backward()
            optimizer.step()

        # calculate the accuracy
        batch_acc, _ = accuracy(batch_output, batch_label)

        # record the average momentum result 
        train_metric["losses"].update(losses.data.item())
        train_metric["accuracy"].update(batch_acc[0])

        batch_time = time.time() - batch_start

        batch_iter += 1

        if args.local_rank == 0:
            print("[Training] Time: {} Epoch: [{}/{}] batch_idx: [{}/{}] batch_iter: [{}/{}] batch_losses: {:.4f} batch_accuracy: {:.4f} LearningRate: {:.6f} BatchTime: {:.4f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                epoch,
                args.max_epochs,
                batch_idx,
                train_batch,
                batch_iter,
                total_batch,
                losses.data.item(),
                batch_acc[0],
                lr,
                batch_time
            ))

        if args.local_rank == 0:
            # batch record
            record_log(log_writer, losses, batch_acc[0], lr, batch_iter, batch_time)

    # epoch record
    record_scalars(log_writer, train_metric["losses"].average, train_metric["accuracy"].average, epoch, flag="train")

    return batch_iter, scaler


def val(
        args,
        val_loader,
        model,
        criterion,
        epoch,
        log_writer,
        ):
    """Validation and get the metric
    """
    model.eval()
    # device = model.device

    epoch_losses, epoch_accuracy = 0.0, 0.0

    batch_acc_list = []
    batch_loss_list = []

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            batch_data, batch_label = data[0], data[1]

            if args.FP16:
                batch_data = batch_data.half()

            batch_data = batch_data.cuda()
            batch_label = batch_label.cuda()

            if args.amp:
                with autocast():
                    batch_output = model(batch_data)
                    batch_losses = criterion(batch_output, batch_label)
            else:
                batch_output = model(batch_data)
                batch_losses = criterion(batch_output, batch_label)

            batch_accuracy, _ = accuracy(batch_output, batch_label)

            batch_acc_list.append(batch_accuracy[0])
            batch_loss_list.append(batch_losses.data.item())

    epoch_acc = np.mean(batch_acc_list)
    epoch_loss = np.mean(batch_loss_list)

    # all reduce the correct number
    # dist.all_reduce(epoch_accuracy, op=dist.ReduceOp.SUM)

    if args.local_rank == 0:
        print(f"Validation Epoch: [{epoch}/{args.max_epochs}] Epoch_mean_losses: {epoch_loss} Epoch_mean_accuracy: {epoch_acc}")

        record_scalars(log_writer, epoch_loss, epoch_acc, epoch, flag="val")

    return epoch_loss, epoch_acc


def record_scalars(log_writer, mean_loss, mean_acc, epoch, flag="train"):
    log_writer.add_scalar(f"{flag}/epoch_average_loss", mean_loss, epoch)
    log_writer.add_scalar(f"{flag}/epoch_average_acc", mean_acc, epoch)


# batch scalar record
def record_log(log_writer, losses, acc, lr, batch_iter, batch_time, flag="Train"):
    log_writer.add_scalar(f"{flag}/batch_loss", losses.data.item(), batch_iter)
    log_writer.add_scalar(f"{flag}/batch_acc", acc, batch_iter)
    log_writer.add_scalar(f"{flag}/learning_rate", lr, batch_iter)
    log_writer.add_scalar(f"{flag}/batch_time", batch_time, batch_iter)


def adjust_learning_rate(args, epoch, batch_iter, optimizer, train_batch):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    total_epochs = args.max_epochs
    warm_epochs = args.warmup_epochs
    if epoch <= warm_epochs:
        lr_adj = (batch_iter + 1) / (warm_epochs * train_batch)
    elif epoch < int(0.3 * total_epochs):
        lr_adj = 1.
    elif epoch < int(0.6 * total_epochs):
        lr_adj = 1e-1
    elif epoch < int(0.8 * total_epochs):
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj


if __name__ == "__main__":
    args = parser.parse_args()
    setup_seed()

    main_worker(args)
from __future__ import division
import argparse
import os
import time
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import yaml
import sys
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
import models
import os.path as osp
import torchvision.models as PREmodels
sys.path.append(osp.abspath(osp.join(__file__, '../')))
import numpy as np
from devkit.sparse_ops import SparseConv,SparseLinear
from devkit.core import (init_dist, broadcast_params, average_gradients, load_state_ckpt, load_state, save_checkpoint, LRScheduler, set_sparse_scheme)

from devkit.core import load_state_file
from devkit.dataset.imagenet_dataset import ColorAugmentation, ImagenetDataset



# Sparse
import ast # for read schemes from txt file
import random


parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='configs/config_resnet50_2:4.yaml')
parser.add_argument('--schemes_file', default='schemes/test.txt')
parser.add_argument("--local_rank", type=int)
parser.add_argument(
    '--port', default=29500, type=int, help='port of server')
parser.add_argument('--world-size', default=1, type=int)
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--label_smoothing', default=0.0,type=float)
parser.add_argument('--momentum', default=0.9,type=float)
parser.add_argument('--base_lr', default=0.1,type=float)
parser.add_argument('--weight_decay', default=0.00005,type=float)
parser.add_argument('--model_dir', type=str,  default='resnet56_cifar/resnet56_M')
parser.add_argument('--resume_from', default='', help='resume_from')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

args = parser.parse_args()






def main():
    global args, best_prec1
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    # print(args.Ns)
    # exit(0)
    print('Enabled distributed training.')

    port = args.port 
    rank, world_size = init_dist(
        backend='nccl', port=args.port )
    args.rank = rank
    args.world_size = world_size

    # create model
    decay = args.decay
    epochs = args.epochs
    base_lr = args.base_lr
    if rank == 0:
        print("=> creating model '{}'".format(args.model))


    # if args.resume_from=='':
    #     model = models.__dict__[args.model](pretrained=True,N = args.N, M = args.M) #NHWC
    # else:
    #     model = models.__dict__[args.model](pretrained=False,N = args.N, M = args.M) #NHWC
    #     load_state_file(args.resume_from,model)

    model = models.__dict__[args.model](pretrained=True,N = args.N, M = args.M)

    #model.set_datalayout('NHWC')

    # read 
    with open(args.schemes_file) as f:
        first_line = f.readline()

    # read sparse scheme
    sparse_schemes = ast.literal_eval(first_line)


    # set layer-wise sparse scheme    
    set_sparse_scheme(model,sparse_schemes)

    set_weight_penalty(model,decay)


    # summary(model, input_size=(3, 224, 224))

    # set_flops(model)

    if rank == 0:
        print('Use schemes file {}'.format(args.schemes_file))
        print("Start to train mixed Sparse NN")
        print(model)
        # print(model.named_layers)
        #print(model.dense_layers)
        print("Sparse Scheme")
        print(model.check_N_M())


    model.cuda()
    broadcast_params(model)

        
    #print(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    # auto resume from a checkpoint
    model_dir = args.model_dir
    start_epoch = 0
    if args.rank == 0 and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.evaluate:
        load_state_ckpt(args.checkpoint_path, model)
    else:
        best_prec1, start_epoch = load_state(model_dir, model, optimizer=optimizer)
    if args.rank == 0:
        writer = SummaryWriter(model_dir)
    else:
        writer = None

    cudnn.benchmark = True





    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = ImagenetDataset(
        args.train_root,
        args.train_source,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ColorAugmentation(),
            normalize,
        ]))
    val_dataset = ImagenetDataset(
        args.val_root,
        args.val_source,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))


    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size//args.world_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size//args.world_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, 0, writer)
        return

    niters = len(train_loader)

    lr_scheduler = LRScheduler(optimizer, niters, args)



    


    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, writer)


        prec1 = validate(val_loader, model, criterion, epoch, writer)

        if rank == 0:
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if epoch > 1:
                save_checkpoint(model_dir, {
                    'epoch': epoch + 1,
                    'model': args.model,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                    #'arch_optimizer': arch_optimizer.state_dict(),
                }, is_best)
    if rank == 0:
        print("Best accuracy is ",best_prec1 )

def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #complexity_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    SAD = AverageMeter()

    # switch to train mode
    model.train()
    world_size = args.world_size
    rank = args.rank

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        lr_scheduler.update(i, epoch)

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var) / world_size
        current_lr = lr_scheduler.get_lr()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        reduced_loss = loss.data.clone()
        reduced_prec1 = prec1.clone() / world_size
        reduced_prec5 = prec5.clone() / world_size

        dist.all_reduce_multigpu([reduced_loss])
        dist.all_reduce_multigpu([reduced_prec1])
        dist.all_reduce_multigpu([reduced_prec5])

        losses.update(reduced_loss.item(), input.size(0))
        top1.update(reduced_prec1.item(), input.size(0))
        top5.update(reduced_prec5.item(), input.size(0))

        
        optimizer.zero_grad()
        
        loss.backward()
        
        average_gradients(model)
        optimizer.step()
       

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and rank == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Learning Rate {current_lr:.4f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5,current_lr=current_lr))
            niter = epoch * len(train_loader) + i
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], niter)
            writer.add_scalar('Train/Avg_Loss', losses.avg, niter)
            writer.add_scalar('Train/Avg_Top1', top1.avg / 100.0, niter)
            writer.add_scalar('Train/Avg_Top5', top5.avg / 100.0, niter)


def validate(val_loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    world_size = args.world_size
    rank = args.rank

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var) / world_size

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            reduced_loss = loss.data.clone()
            reduced_prec1 = prec1.clone() / world_size
            reduced_prec5 = prec5.clone() / world_size

            dist.all_reduce_multigpu([reduced_loss])
            dist.all_reduce_multigpu([reduced_prec1])
            dist.all_reduce_multigpu([reduced_prec5])

            losses.update(reduced_loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and rank == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
        if rank == 0:
            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

            niter = (epoch + 1)
            writer.add_scalar('Eval/Avg_Loss', losses.avg, niter)
            writer.add_scalar('Eval/Avg_Top1', top1.avg / 100.0, niter)
            writer.add_scalar('Eval/Avg_Top5', top5.avg / 100.0, niter)

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




def set_weight_penalty(model, decay):
    for mod in model.modules():
        if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear) : 
            mod.decay = decay



if __name__ == '__main__':
    main()

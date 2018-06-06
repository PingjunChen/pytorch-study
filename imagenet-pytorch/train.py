# -*- coding: utf-8 -*-

import os, sys, pdb

import argparse, shutil
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from densetnet import DenseNet201
from train_eng import train_imagenet

def set_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet2012 classification')
    # mini_batch and epoch information
    parser.add_argument('--batch-size',      type=int,   default=64)
    parser.add_argument('--epochs',          type=int,   default=90)
    # Optimization parameters
    parser.add_argument('--lr',              type=float, default=0.1)
    parser.add_argument('--momentum',        type=float, default=0.9)
    parser.add_argument('--weight_decay',    type=float, default=5.0e-4)

    #
    parser.add_argument('--seed',            type=int,   default=1)
    parser.add_argument('--device_id',       type=int,   default=3)
    parser.add_argument('--log-interval',    type=int,   default=20)

    # model directory and name
    parser.add_argument('--model-dir',       type=str,   default="../models/ImageNet")
    parser.add_argument('--model-name',      type=str,   default="imagenet")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = set_args()
    # Config model and gpu
    torch.manual_seed(args.seed)
    model = DenseNet201()
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        model.cuda(args.device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    # Start training
    train_imagenet(model, args)

# -*- coding: utf-8 -*-

import os, sys, pdb
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


from loader import test_imagenet_loader
from loader import imagenet_num_class
from densetnet import DenseNet121, DenseNet201
from train_eng import validate

def set_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size',      type=int,   default=32)
    parser.add_argument('--seed',            type=int,   default=3)
    parser.add_argument('--device_id',       type=int,   default=0)
    # model directory and name
    parser.add_argument('--model-dir',       type=str,   default="../models/ImageNet/DenseNet121")
    parser.add_argument('--model-name',      type=str,   default="imagenet-90.pth")

    args = parser.parse_args()
    return args

def test(data_loader, model, args):
    criterion = nn.CrossEntropyLoss()
    top1_acc, top5_acc = validate(data_loader, model, criterion, args)
    print("Top 1 acc is {:.3f}".format(top1_acc))
    print("Top 5 acc is {:.3f}".format(top5_acc))

if __name__ == '__main__':
    args = set_args()
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    model = DenseNet121(num_classes=imagenet_num_class)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        model.cuda(args.device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    # dataloader
    data_loader = test_imagenet_loader(args)
    # start testing
    test(data_loader, model, args)

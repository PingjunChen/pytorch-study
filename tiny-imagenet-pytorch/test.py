# -*- coding: utf-8 -*-

import os, sys, pdb
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


from loader import test_imagenet_loader
from loader import val_imagenet_loader
from loader import imagenet_num_class
from densetnet import DenseNet121, DenseNet201
from resnet import ResNet50
from vgg import VGG
from train_eng import validate

def set_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size',      type=int,   default=50)
    parser.add_argument('--device_id',       type=int,   default=1)
    parser.add_argument('--log-interval',    type=int,   default=20)
    # model directory and name
    parser.add_argument('--model-dir',       type=str,   default="../models/TinyImageNet200/bestmodel")
    parser.add_argument('--model-name',      type=str,   default="densenet121-62-58.210.pth")
    parser.add_argument('--seed',            type=int,   default=1234)

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
    # model = ResNet50(num_classes=imagenet_num_class)
    # model = VGG("VGG19", imagenet_num_class)
    weights_path = os.path.join(args.model_dir, args.model_name)
    model.load_state_dict(torch.load(weights_path))

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        model.cuda(args.device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    # dataloader
    # data_loader = test_imagenet_loader(args)
    data_loader = val_imagenet_loader(args)
    # start testing
    test(data_loader, model, args)

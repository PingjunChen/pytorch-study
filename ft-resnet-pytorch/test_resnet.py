# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np
import argparse

import torch
from torch.autograd import Variable
from dataset import prepare_testset

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def set_args():
    parser = argparse.ArgumentParser(description='PyTorch Fine-Tune ResNet Testing')
    parser.add_argument('--model_dir',       type=str,  default="./models/")
    parser.add_argument('--data_dir',        type=str,  default="./data/test")
    parser.add_argument('--model_name',      type=str,  default="resnet101-00000.pth")
    parser.add_argument('--batch_size',      type=int,   default=4)
    parser.add_argument('--device_id',       type=int,   default=0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print("--Phase 0: arguments settings...")
    args = set_args()
    args.use_gpu = torch.cuda.is_available()

    print("--Phase 1: load fineturned model...")
    model_path = os.path.join(args.model_dir, args.model_name)
    assert os.path.exists(model_path), 'Error: No model found!'
    checkpoint = torch.load(model_path)
    model = checkpoint['model']
    if args.use_gpu:
        model.cuda(args.device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
    model.eval()

    print("--Phase 2: Prepare testing data...")
    test_loss, correct, total = 0, 0, 0
    loader, num_cases = prepare_testset(args)

    print("--Phase 3: Inference...")
    pdb.set_trace()
    for batch_idx, (inputs, targets) in enumerate(loader):

        if args.use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)

        softmax_res = softmax(outputs.data.cpu().numpy()[0])
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct/total
    print("---Test Acc: %.2f%%" %(acc))

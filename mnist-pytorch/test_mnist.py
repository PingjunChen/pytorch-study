# -*- coding: utf-8 -*-

import os, sys, pdb
import argparse

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from network import Net
from loader import test_loader


def set_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--test-batch-size', type=int,   default=1000)
    parser.add_argument('--seed',            type=int,   default=1)
    parser.add_argument('--device_id',       type=int,   default=0)
    # model directory and name
    parser.add_argument('--model-dir',       type=str,   default="./models")
    parser.add_argument('--model-name',      type=str,   default="mnist-10.pth")

    args = parser.parse_args()
    return args


def test(data_loader, model, args):
    # Load model
    weights_path = os.path.join(args.model_dir, args.model_name)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    test_loss, correct = 0.0, 0
    for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(data_loader.dataset)
    print("\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n".format(
        test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)
    ))


if __name__ == '__main__':
    args = set_args()
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    model = Net()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        model.cuda(args.device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    # dataloader
    data_loader = test_loader(args)

    # start training
    test(data_loader, model, args)

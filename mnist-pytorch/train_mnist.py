# -*- coding: utf-8 -*-

import os, sys, pdb
import argparse, shutil
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from network import Net
from loader import train_loader


def set_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # mini_batch and epoch information
    parser.add_argument('--batch-size',      type=int,   default=64)
    parser.add_argument('--epochs',          type=int,   default=10)
    # Optimization parameters
    parser.add_argument('--lr',              type=float, default=0.01)
    parser.add_argument('--momentum',        type=float, default=0.9)
    #
    parser.add_argument('--seed',            type=int,   default=1)
    parser.add_argument('--device_id',       type=int,   default=3)
    parser.add_argument('--log-interval',    type=int,   default=10)
    # model directory and name
    parser.add_argument('--model-dir',       type=str,   default="../models/MNIST")
    parser.add_argument('--model-name',      type=str,   default="mnist")

    args = parser.parse_args()
    return args


def train(data_loader, model, args):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if os.path.exists(args.model_dir):
        shutil.rmtree(args.model_dir)
    os.makedirs(args.model_dir)

    for epoch in range(1, args.epochs+1):
        for batch_idx, (data, target) in enumerate(data_loader):
            if args.cuda:
                data, target = data.cuda(args.device_id), target.cuda(args.device_id)
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx > 0 and batch_idx % args.log_interval == 0:
                batch_progress = 100. * batch_idx / len(data_loader)
                print("Train Epoch: {} [{}/{} ({:.1f})%)]\t Loss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                    batch_progress, loss.data[0]
                ))
        cur_model_name = args.model_name + "-" + str(epoch).zfill(2) + ".pth"
        torch.save(model.state_dict(), os.path.join(args.model_dir, cur_model_name))
        print('Save weights at {}/{}'.format(args.model_dir, cur_model_name))


if __name__ == '__main__':
    args = set_args()
    # Config model and gpu
    torch.manual_seed(args.seed)
    model = Net()
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        model.cuda(args.device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    # dataloader
    data_loader = train_loader(args)

    # Start training
    train(data_loader, model, args)

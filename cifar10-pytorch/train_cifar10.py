# -*- coding: utf-8 -*-

import os, sys, pdb
import argparse, shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from models import alexnet, vgg, resnet, densenet
from loader import train_loader, cifar10_classes


def set_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    # mini_batch and epoch information
    parser.add_argument('--batch-size',      type=int,   default=64)
    parser.add_argument('--epochs',          type=int,   default=300)
    # Optimization parameters
    parser.add_argument('--weight_decay',    type=float, default=5.0e-4)
    parser.add_argument('--lr',              type=float, default=0.1)
    parser.add_argument('--lr_decay_epochs', type=int,   default=100)
    parser.add_argument('--momentum',        type=float, default=0.9)
    #
    parser.add_argument('--seed',            type=int,   default=1)
    parser.add_argument('--device_id',       type=int,   default=0)
    parser.add_argument('--log-interval',    type=int,   default=20)
    # model directory and name
    parser.add_argument('--model-dir',       type=str,   default="../models/CIFAR10/DenseNet201")
    parser.add_argument('--model-name',      type=str,   default="cifar10")

    args = parser.parse_args()
    return args



def train(data_loader, model, args):
    model.train()
    optimizer = optim.SGD(model.parameters(), weight_decay=args.weight_decay,
                          lr=args.lr, momentum=args.momentum)
    criterion =nn.CrossEntropyLoss()
    def adjust_learning_rate(optimizer, epoch):
        lr = args.lr * (0.1 ** (epoch // args.lr_decay_epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print("Learning rate is {}".format(lr))

    if os.path.exists(args.model_dir):
        shutil.rmtree(args.model_dir)
    os.makedirs(args.model_dir)

    for epoch in range(1, args.epochs+1):
        correct, total = 0, 0
        adjust_learning_rate(optimizer, epoch)
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if args.cuda:
                inputs, targets = inputs.cuda(args.device_id), targets.cuda(args.device_id)
            inputs, targets = Variable(inputs), Variable(targets)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx > 0 and batch_idx % args.log_interval == 0:
                batch_progress = 100. * batch_idx / len(data_loader)
                print("Train Epoch: {} [{}/{} ({:.1f})%)]\t Loss: {:.6f}".format(
                    epoch, batch_idx * len(inputs), len(data_loader.dataset),
                    batch_progress, loss.item()))

        print("Accuracy on Epoch {} is [{}/{} {:.3f})]".format(
            epoch, correct, total, correct*1.0/total))
        if epoch % 10 == 0:
            cur_model_name = args.model_name + "-" + str(epoch).zfill(2) + ".pth"
            torch.save(model.state_dict(), os.path.join(args.model_dir, cur_model_name))
            print('Save weights at {}/{}'.format(args.model_dir, cur_model_name))


if __name__ == '__main__':
    args = set_args()
    # Config model and gpu
    torch.manual_seed(args.seed)
    model = alexnet.AlexNet(num_classes=len(cifar10_classes))
    # model = vgg.VGG('VGG16', num_classes=len(cifar10_classes))
    # model = vgg.VGG('VGG19', num_classes=len(cifar10_classes))
    # model = resnet.ResNet50(num_classes=len(cifar10_classes))
    # model = resnet.ResNet101(num_classes=len(cifar10_classes))
    # model = densenet.DenseNet121(num_classes=len(cifar10_classes))
    model = densenet.DenseNet201(num_classes=len(cifar10_classes))

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

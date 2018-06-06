# -*- coding: utf-8 -*-

import torch.utils.data
from torchvision import datasets, transforms

cifar10_mean, cifar10_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def train_loader(args):
    kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}

    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(cifar10_mean, cifar10_std),
                             ])),
                         batch_size=args.batch_size, shuffle=True, **kwargs)
    return loader

def test_loader(args):
    kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}

    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(cifar10_mean, cifar10_std),
                             ])),
                         batch_size=args.batch_size, shuffle=False, **kwargs)
    return loader

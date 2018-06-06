# -*- coding: utf-8 -*-

import torch.utils.data
from torchvision import datasets, transforms

def train_loader(args):
    kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
    loader = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    return loader


def test_loader(args):
    kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
    loader = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return loader

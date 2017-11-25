# -*- coding: utf-8 -*-

import os, sys
import torch
from torchvision import datasets, transforms


def prepare_dataset(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor()
        ]),
    }

    dsets = { x : datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x]) for x in ['train', 'val'] }
    dset_loaders = { x : torch.utils.data.DataLoader(dsets[x], batch_size = args.batch_size,
                                                     shuffle=(x=='train'), num_workers=4) for x in ['train', 'val'] }
    dset_classes = dsets['train'].classes
    num_class = len(dset_classes)

    return dset_loaders, num_class


def prepare_testset(args):
    data_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor()
    ])

    dset = datasets.ImageFolder(args.data_dir, data_transform)
    dset_loader = torch.utils.data.DataLoader(dset, batch_size = args.batch_size,
                                              shuffle=False, num_workers=4)
    return dset_loader, len(dset)

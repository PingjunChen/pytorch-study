# -*- coding: utf-8 -*-

import os, sys, pdb

import torch.utils.data
from torchvision import datasets, transforms

imagenet_root = ""
imagenet_num_class = 1000
imagenet_traindir = os.path.join(imagenet_root, 'train')
imagenet_valdir = os.path.join(imagenet_root, 'val')
imagenet_testdir = os.path.join(imagenet_root, 'test')
imagenet_mean, imagenet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def train_imagenet_loader(args):
    kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}

    train_dataset = datasets.ImageFolder(
        imagenet_traindir,
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)])
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    return train_loader


def val_imagenet_loader(args):
    kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}

    val_dataset = datasets.ImageFolder(
        imagenet_valdir,
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)])
        )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    return val_loader


def test_imagenet_loader(args):
    kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}

    test_dataset = datasets.ImageFolder(
        imagenet_testdir,
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)])
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    return test_loader

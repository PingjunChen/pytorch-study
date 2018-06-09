# -*- coding: utf-8 -*-

import os, sys, pdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import shutil, time

from loader import train_imagenet_loader
from loader import val_imagenet_loader
from utils import adjust_learning_rate
from utils import accuracy, AverageMeter

def train_imagenet(model, args):
    optimizer = optim.SGD(model.parameters(), weight_decay=args.weight_decay,
                          lr=args.lr, momentum=args.momentum)
    criterion =nn.CrossEntropyLoss()
    train_loader = train_imagenet_loader(args)
    val_loader = val_imagenet_loader(args)

    if os.path.exists(args.model_dir):
        shutil.rmtree(args.model_dir)
    os.makedirs(args.model_dir)

    best_prec = 0.0
    for epoch in range(1, args.epochs+1):
        adjust_learning_rate(optimizer, epoch, args)
       # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
       # evaluate on validation set
        cur_prec, _ = validate(val_loader, model, criterion, args)

        # remember best prec@1 and save checkpoint
        is_best = cur_prec > best_prec
        if is_best == True:
            best_prec = cur_prec
            cur_model_name = args.model_name + "-" + str(epoch).zfill(2) + "-" + str(cur_prec) + ".pth"
            torch.save(model.state_dict(), os.path.join(args.model_dir, cur_model_name))
            print('Save weights at {}/{}'.format(args.model_dir, cur_model_name))

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.cuda:
            inputs, targets = inputs.cuda(args.device_id), targets.cuda(args.device_id)
        inputs, targets = Variable(inputs), Variable(targets)
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()


    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets) in enumerate(val_loader):
            if args.cuda:
                inputs, targets = inputs.cuda(args.device_id), targets.cuda(args.device_id)
            inputs, targets = Variable(inputs), Variable(targets)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_interval == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg.item(), top5.avg.item()

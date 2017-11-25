# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np
import time, copy
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.autograd import Variable

from dataset import prepare_dataset
from model import get_network, reset_classifier


def set_args():
    parser = argparse.ArgumentParser(description='PyTorch Fine-Tune ResNet Training')
    parser.add_argument('--net_type',        type=str,   default='resnet')
    parser.add_argument('--depth',           type=int,   default=101)
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--lr_decay_epoch',  type=int,   default=10)
    parser.add_argument('--num_epoch',       type=int,   default=50)
    parser.add_argument('--batch_size',      type=int,   default=4)
    parser.add_argument('--weight_decay',    type=float, default=5e-4)
    parser.add_argument('--data_dir',        type=str,  default="./data/")
    parser.add_argument('--model_dir',       type=str,  default="./models/")
    parser.add_argument('--finetune',        type=bool,  default=True)
    parser.add_argument('--device_id',       type=int,   default=0)

    args = parser.parse_args()
    return args


def train_model(model_ft, data_loader, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ft.parameters(), lr=args.lr,
                             momentum=0.9, weight_decay=args.weight_decay)

    def exp_lr_scheduler(optimizer, epoch, init_lr=args.lr,
                         weight_decay=args.weight_decay,
                         lr_decay_epoch=args.lr_decay_epoch):
        lr = init_lr * (0.5**(epoch // lr_decay_epoch))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay

        return optimizer, lr

    since = time.time()
    best_model, best_acc = model_ft, 0.0

    print("-"*4 + "| Training Epochs = {}".format(args.num_epoch))
    print("-"*4 + "| Initial Learning Rate = {}".format(args.lr))
    print("-"*4 + "| Optimizer = SGD")

    # Starting training and validation
    for epoch in range(args.num_epoch):
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer, lr = exp_lr_scheduler(optimizer, epoch)
                print('=> Training Epoch #%d, LR=%f' %(epoch+1, lr))
                model_ft.train(True)
            else:
                model_ft.train(False)
                model_ft.eval()

            running_loss, running_corrects, tot = 0.0, 0, 0

            num_examples = 0
            for batch_idx, (inputs, labels) in enumerate(data_loader[phase]):
                if args.use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)


                optimizer.zero_grad()
                # Forward Propagation
                outputs = model_ft(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                # Backward Propagation
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Statistics
                running_loss += loss.data[0]
                running_corrects += preds.eq(labels.data).cpu().sum()
                tot += labels.size(0)

            epoch_loss = running_loss / tot
            epoch_acc  = running_corrects / tot

            if phase == "val":
                print('| Validation Epoch #%d\t\t\tLoss %.4f\tAcc %.2f%%'
                    %(epoch+1, epoch_loss, 100.*epoch_acc))

                if epoch_acc > best_acc:
                    print('| Saving Best model...\t\t\tTop1 %.2f%%' %(100.*epoch_acc))
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model_ft)
                    state = {
                        'model': best_model,
                        'acc':   epoch_acc,
                        'epoch': epoch,
                    }
                    if not os.path.isdir(args.model_dir):
                        os.mkdir(args.model_dir)
                    weights_name = args.net_type + str(args.depth) + "-" + str(epoch).zfill(5) + ".pth"
                    torch.save(state, os.path.join(args.model_dir, weights_name))

    time_elapsed = time.time() - since
    print("Training completed in {:.0f} min {:.0f} sec".format(time_elapsed // 60, time_elapsed % 60))
    print("Best validation Acc {:.2f}%".format(best_acc*100))


if __name__ == '__main__':
    print("--Phase 0: arguments settings...")
    args = set_args()
    args.use_gpu = torch.cuda.is_available()

    print("--Phase 1: Data prepration")
    data_loader, num_class = prepare_dataset(args)
    args.num_class = num_class

    print("--Phase 2: Model setup")
    model_ft, file_name = get_network(args)
    model_ft = reset_classifier(model_ft, args)
    if args.use_gpu:
        model_ft.cuda(args.device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    print("--Phase 3: Training Model")
    train_model(model_ft, data_loader, args)

# -*- coding: utf-8 -*-

import os, sys
import torch.nn as nn
from network import resnet

def get_network(args):
    net = resnet(args.finetune, args.depth)
    file_name = 'resnet-%s' %(args.depth)

    return net, file_name

def reset_classifier(model_ft, args):
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, args.num_class)
    return model_ft

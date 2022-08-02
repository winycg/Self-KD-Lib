import os
import sys
import time
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init


def mixup_data(x, y, alpha=0.4):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 0.5

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam, index

def Mixup(net, inputs, targets, criterion_cls, alpha):
    mixed_x, y_a, y_b, lam_mixup, _ = mixup_data(inputs, targets, alpha=alpha)
    logit = net(mixed_x)
    if isinstance(logit, list) or isinstance(logit, tuple):
        logit = logit[0] 
    loss = criterion_cls(logit, y_a) * lam_mixup + criterion_cls(logit, y_b) * (1. - lam_mixup)
    return logit, loss


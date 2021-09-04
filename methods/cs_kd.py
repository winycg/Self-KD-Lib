import os
import sys
import time
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init

def CS_KD(net, inputs, targets, criterion_cls, criterion_div):
    loss_div = torch.tensor(0.).cuda()
    loss_cls = torch.tensor(0.).cuda()

    batch_size = inputs.size(0)
    targets = targets[:batch_size//2]
    logit = net(inputs[:batch_size//2])
    with torch.no_grad():
        outputs_cls = net(inputs[batch_size//2:])
    loss_cls += criterion_cls(logit, targets)
    loss_div += criterion_div(logit, outputs_cls.detach())

    return logit, loss_cls, loss_div
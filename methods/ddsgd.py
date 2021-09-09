import os
import sys
import time
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init

def DDGSD(net, inputs, targets, criterion_cls, criterion_div):
    loss_div = torch.tensor(0.).cuda()
    loss_cls = torch.tensor(0.).cuda()

    inputs = torch.cat(inputs, dim=0)
    batch_size = inputs.size(0) // 2
    logit, features = net(inputs, embedding=True)
    loss_cls += criterion_cls(logit, torch.cat([targets, targets], dim=0)) / 2
    loss_div += criterion_div(logit[:batch_size], logit[batch_size:].detach())
    loss_div += criterion_div(logit[batch_size:], logit[:batch_size].detach())
    loss_div += 5e-4 * (features[:batch_size].mean()-features[batch_size:].mean()) ** 2
    logit = (logit[batch_size:] + logit[:batch_size]) / 2
    return logit, loss_cls, loss_div
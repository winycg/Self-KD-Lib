import os
import sys
import time
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init


def BYOT(net, inputs, targets, criterion_cls, criterion_div):
    loss_div = torch.tensor(0.).cuda()
    loss_cls = torch.tensor(0.).cuda()
    logits, features = net(inputs, feature=True)
    for i in range(len(logits)):
        loss_cls += criterion_cls(logits[i], targets)
        if i != 0:
            loss_div += criterion_div(logits[i], logits[0].detach())
    
    for i in range(1, len(features)):
        if i != 1:
            loss_div += 0.5 * 0.1 * ((features[i] - features[1].detach()) ** 2).mean()
    
    logit = logits[0]   
    return logit, loss_cls, loss_div
            

import os
import sys
import time
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.p = 2

    def forward(self, f):
        loss = sum([self.at_loss(f[i], F.interpolate(f[i+1].detach(), scale_factor=2., mode="bilinear")) for i in range(len(f)-1)])
        return loss

    def at_loss(self, f_s, f_t):
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        B, C, H, W = f.size()
        return F.softmax(f.pow(self.p).mean(1).view(B, -1), dim=1).view(B, H, W)

AT = Attention()

def SAD(net, inputs, targets, criterion_cls, criterion_div):
    loss_div = torch.tensor(0.).cuda()
    loss_cls = torch.tensor(0.).cuda()

    logit, features = net(inputs, feature=True)

    loss_cls += criterion_cls(logit, targets)
    loss_div += 1000 * AT(features)
    return logit, loss_cls, loss_div



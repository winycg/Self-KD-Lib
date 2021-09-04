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

    def forward(self, g_s, g_t):
        loss = sum([self.at_loss(f_s, f_t.detach()) for f_s, f_t in zip(g_s, g_t)])
        return loss

    def at_loss(self, f_s, f_t):
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


AT = Attention()

def FRSKD(net, inputs, targets, criterion_cls, criterion_div):
    loss_div = torch.tensor(0.).cuda()
    loss_cls = torch.tensor(0.).cuda()

    logit, features, bi_feats, bi_logits = net(inputs)

    loss_cls += criterion_cls(logit, targets)
    loss_cls += criterion_cls(bi_logits, targets)
    loss_div += 2 * criterion_div(logit, bi_logits)
    loss_div += 100 * AT(features, bi_feats)

    return logit, loss_cls, loss_div
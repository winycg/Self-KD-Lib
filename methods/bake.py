import os
import sys
import time
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init


def knowledge_ensemble(feats, logits, args):
    batch_size = logits.size(0)
    masks = torch.eye(batch_size).cuda()
    feats = nn.functional.normalize(feats, p=2, dim=1)
    logits = nn.functional.softmax(logits/args.T, dim=1)
    W = torch.matmul(feats, feats.permute(1, 0)) - masks * 1e9
    W = nn.functional.softmax(W, dim=1)
    W = (1 - args.omega) * torch.inverse(masks - args.omega * W)
    return torch.matmul(W, logits)


def BAKE(net, inputs, targets, criterion_cls, criterion_div, args):
    loss_div = torch.tensor(0.).cuda()
    loss_cls = torch.tensor(0.).cuda()
    logits, features = net(inputs, embedding=True)
    with torch.no_grad():
        kd_targets = knowledge_ensemble(features.detach(), logits.detach(), args)
    
    loss_cls += criterion_cls(logits, targets)
    loss_div += criterion_div(logits, kd_targets.detach())
    return logits, loss_cls, loss_div



import os
import sys
import time
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init


def FocalLoss(inputs, targets):
    gamma = 2
    N = inputs.size(0)
    C = inputs.size(1)
    P = F.softmax(inputs, dim=1)

    class_mask = inputs.new(N, C).fill_(0)
    ids = targets.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)
    #print(class_mask)

    probs = (P*class_mask).sum(1).view(-1,1)

    log_p = probs.log()
    #print('probs size= {}'.format(probs.size()))
    #print(probs)

    batch_loss = -(torch.pow((1-probs), gamma))*log_p 
    #print('-----bacth_loss------')
    #print(batch_loss)
    loss = batch_loss.mean()
    return loss
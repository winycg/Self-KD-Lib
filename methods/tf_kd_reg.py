import os
import sys
import time
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init


def TF_KD_reg(outputs, targets, num_classes=100, epsilon=0.1, T=20):
    N = targets.size(0)
    smoothed_labels = torch.full(size=(N, num_classes),
                                 fill_value=epsilon / (num_classes - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1),
                             value=1-epsilon)
    log_prob = F.log_softmax(outputs / T, dim=1)
    loss = - torch.sum(log_prob * smoothed_labels/ T) / N
    return loss
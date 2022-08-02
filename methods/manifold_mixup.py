import os
import sys
import time
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init


def ManifoldMixup(net, inputs, targets, criterion_cls, alpha):
    logit, y_a, y_b, lam = net(inputs, targets, alpha)
    loss = criterion_cls(logit, y_a) * lam + criterion_cls(logit, y_b) * (1. - lam)
    return logit, loss


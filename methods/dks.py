import os
import sys
import time
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init


def DKS(net, inputs, targets, criterion_cls, criterion_div):
    loss_div = torch.tensor(0.).cuda()
    loss_cls = torch.tensor(0.).cuda()
    outputs = net(inputs)
    for j, output in enumerate(outputs):
        loss_cls += criterion_cls(output, targets)
        for output_counterpart in outputs:
            if output_counterpart is not output:
                loss_div += criterion_div(output, output_counterpart.detach())
            else:
                pass
    logit = outputs[0]

    return logit, loss_cls, loss_div
import os
import sys
import time
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init


class Custom_CrossEntropy_PSKD(nn.Module):
    def __init__(self):
        super(Custom_CrossEntropy_PSKD, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, output, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(output)
        loss = (- targets * log_probs).mean(0).sum()
        return loss     


criterion_CE_pskd = Custom_CrossEntropy_PSKD().cuda()


def PSKD(net, inputs, targets, input_indices, epoch, all_predictions, num_classes, args):
    alpha_t = args.alpha_T * ((epoch + 1) / args.epochs)
    alpha_t = max(0, alpha_t)

    targets_numpy = targets.cpu().detach().numpy()
    identity_matrix = torch.eye(num_classes) 
    targets_one_hot = identity_matrix[targets_numpy]

    if epoch == 0:
        all_predictions[input_indices] = targets_one_hot

    soft_targets = ((1 - alpha_t) * targets_one_hot) + (alpha_t * all_predictions[input_indices])
    soft_targets = soft_targets.cuda()

    outputs = net(inputs)
    softmax_output = F.softmax(outputs, dim=1) 
    loss = criterion_CE_pskd(outputs, soft_targets)

    all_predictions[input_indices] = softmax_output.cpu().detach()

    return outputs, loss

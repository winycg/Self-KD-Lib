import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, dim_in=1024, dim_out=2):
        super(Discriminator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim_in, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, dim_out),
        )
        
    def forward(self, x):
        x = self.classifier(x)
        return x


class discriminatorLoss(nn.Module):
    def __init__(self, dim_ins, loss=nn.BCEWithLogitsLoss()):
        super(discriminatorLoss, self).__init__()
        self.classifier = []
        for dim in dim_ins:
            self.classifier.append(Discriminator(dim_in=dim, dim_out=2).cuda())
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.loss = loss

    def forward(self, features1, features2):
        gan_loss = torch.tensor(0.).cuda()
        if isinstance(features1, list) is False:
            features1 = [features1]
            features2 = [features2]
        for i in range(len(self.classifier)):
            inputs = torch.cat((features1[i],features2[i]),0)
            if len(inputs.size())> 2:
                inputs = self.avg_pool(inputs).view(inputs.size(0), -1)
            batch_size = inputs.size(0)
            target = torch.FloatTensor([[1, 0] for _ in range(batch_size//2)] + [[0, 1] for _ in range(batch_size//2)]).cuda()
            outputs = self.classifier[i](inputs)
            gan_loss += self.loss(outputs, target)
        return gan_loss

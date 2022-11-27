import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

import sys

sys.path.append('.')
sys.path.append('..')

from .resnet import CIFAR_ResNet18, CIFAR_ResNet50
from .resnet_imagenet import resnet18_imagenet

__all__ = ['BiFPNc', 'CIFAR_ResNet18_BiFPN', 'CIFAR_ResNet50_BiFPN',
            'ResNet18_BiFPN']


class MaxPool2dStaticSamePadding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x


class DepthConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, depth=1):
        super(DepthConvBlock, self).__init__()
        conv = []
        if kernel_size == 1:
            conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ))
        else:
            conv.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=False, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            ))
            for i in range(depth-1):
                conv.append(nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False, groups=out_channels),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                ))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class BiFPNc(nn.Module):
    def __init__(self, network_channel, num_classes, repeat, depth, width):
        super(BiFPNc, self).__init__()
        self.layers = nn.ModuleList()

        self.net_channels = [x * width for x in network_channel]
        for i in range(repeat):
            self.layers.append(BiFPN_layer(i == 0, DepthConvBlock, network_channel, depth, width))

        self.fc = nn.Linear(self.net_channels[-1], num_classes)

    def forward(self, feats, preact=True):

        for i in range(len(self.layers)):
            layer_preact = preact and i == len(self.layers) - 1
            feats = self.layers[i](feats, layer_preact)

        out = F.adaptive_avg_pool2d(F.relu(feats[-1]), (1, 1)) # for preact case
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return feats, out

    def get_bn_before_relu(self):
        layer = self.layers[-1]
        bn = [layer.up_conv[0].conv[-1][-1]]
        for down_conv in layer.down_conv:
            bn.append(down_conv.conv[-1][-1])
        return bn


class BiFPN_layer(nn.Module):
    def __init__(self, first_time, block, network_channel, depth, width):
        super(BiFPN_layer, self).__init__()
        lat_depth, up_depth, down_depth = depth
        self.first_time = first_time

        self.lat_conv = nn.ModuleList()
        self.lat_conv2 = nn.ModuleList()

        self.up_conv = nn.ModuleList()
        self.up_weight = nn.ParameterList()

        self.down_conv = nn.ModuleList()
        self.down_weight = nn.ParameterList()
        self.down_sample = nn.ModuleList()
        self.up_sample = nn.ModuleList()

        for i, channels in enumerate(network_channel):
            if self.first_time:
                self.lat_conv.append(block(channels, channels * width, 1, 1, 0, lat_depth))

            if i != 0:
                self.lat_conv2.append(block(channels, channels * width, 1, 1, 0, lat_depth))
                self.down_conv.append(block(channels * width, channels * width, 3, 1, 1, down_depth))
                num_input = 3 if i < len(network_channel) - 1 else 2

                self.down_weight.append(nn.Parameter(torch.ones(num_input, dtype=torch.float32), requires_grad=True))
                self.down_sample.append(nn.Sequential(MaxPool2dStaticSamePadding(3, 2),
                                                      block(network_channel[i-1] * width, channels * width, 1, 1, 0, 1)))

            if i != len(network_channel) - 1:
                self.up_sample.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                                    block(network_channel[i+1] * width, channels * width, 1, 1, 0, 1)))
                self.up_conv.append(block(channels * width, channels * width, 3, 1, 1, up_depth))
                self.up_weight.append(nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True))

        self.relu = nn.ReLU()

        self.epsilon = 1e-6

    def forward(self, inputs, preact=True):
        input_trans = [self.lat_conv2[i - 1](F.relu(inputs[i])) for i in range(1, len(inputs))]
        if self.first_time:
            inputs = [self.lat_conv[i](F.relu(inputs[i])) for i in range(len(inputs))] # for od case

        # up
        up_sample = [inputs[-1]]
        out_layer = []
        for i in range(1, len(inputs)):
            w = self.relu(self.up_weight[-i])
            w = w / (torch.sum(w, dim=0) + self.epsilon)

            up_sample.insert(0,
                             self.up_conv[-i](w[0] * F.relu(inputs[-i - 1])
                                              + w[1] * self.up_sample[-i](F.relu(up_sample[0]))))

        out_layer.append(up_sample[0])

        # down
        for i in range(1, len(inputs)):
            w = self.relu(self.down_weight[i - 1])
            w = w / (torch.sum(w, dim=0) + self.epsilon)
            if i < len(inputs) - 1:
                out_layer.append(self.down_conv[i - 1](w[0] * F.relu(input_trans[i - 1])
                                                       + w[1] * F.relu(up_sample[i])
                                                       + w[2] * self.down_sample[i - 1](F.relu(out_layer[-1]))
                                                       )
                                 )
            else:
                out_layer.append(
                    self.down_conv[i - 1](w[0] * F.relu(input_trans[i - 1])
                                          + w[1] * self.down_sample[i - 1](F.relu(out_layer[-1]))
                                          )
                )

        if not preact:
            return [F.relu(f) for f in out_layer]
        return out_layer


class CIFAR_ResNet18_BiFPN(nn.Module):
    def __init__(self, num_classes=100):
        super(CIFAR_ResNet18_BiFPN, self).__init__()
        self.backbone = CIFAR_ResNet18(num_classes=100)
        self.bifpn = BiFPNc(self.backbone.network_channels, num_classes, repeat=1, depth=[2] * 3, width=2)

    def forward(self, x):
        logit, features = self.backbone(x, feature=True)
        bi_feats, bi_logits = self.bifpn(features, preact=True)
        return logit, features, bi_feats, bi_logits


class ResNet18_BiFPN(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18_BiFPN, self).__init__()
        self.backbone = resnet18_imagenet(num_classes=num_classes)
        self.bifpn = BiFPNc(self.backbone.network_channels, num_classes, repeat=1, depth=[1] * 3, width=1)

    def forward(self, x):
        logit, features = self.backbone(x, feature=True)
        bi_feats, bi_logits = self.bifpn(features, preact=True)
        return logit, features, bi_feats, bi_logits


class CIFAR_ResNet50_BiFPN(nn.Module):
    def __init__(self, num_classes=100):
        super(CIFAR_ResNet50_BiFPN, self).__init__()
        self.backbone = CIFAR_ResNet50(num_classes=num_classes)
        self.bifpn = BiFPNc(self.backbone.network_channels, num_classes, repeat=1, depth=[1] * 3, width=1)

    def forward(self, x):
        logit, features = self.backbone(x, feature=True)
        bi_feats, bi_logits = self.bifpn(features, preact=False)
        return logit, features, bi_feats, bi_logits



if __name__ == '__main__':
    net = ResNet18_BiFPN(num_classes=100)
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    import sys
    sys.path.append('..')
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 32, 32)) / 1e6))
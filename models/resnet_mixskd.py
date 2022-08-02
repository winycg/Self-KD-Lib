import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math

__all__ = ['MixSKD_CIFAR_ResNet18']




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion*planes)
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion*planes)
            )
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class CIFAR_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(CIFAR_ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        out = x
        out = self.conv1(out)
        out = self.bn1(out)

        out = self.layer1(out)
        features.append(out)
        out = self.layer2(out)
        features.append(out)
        out = self.layer3(out)
        features.append(out)
        out = self.layer4(out)
        features.append(out)

        out = self.avgpool(out)
        embedding0 = out.view(out.size(0), -1)
        
        logits = self.fc(embedding0)

        return logits, features


class Auxiliary_Classifier(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100,):
        super(Auxiliary_Classifier, self).__init__()
        self.in_planes = 64 * block.expansion
        self.block_extractor1 = nn.Sequential(*[self._make_layer(block, 128, num_blocks[1], stride=2),
                                                self._make_layer(block, 256, num_blocks[2], stride=2),
                                                self._make_layer(block, 512, num_blocks[3], stride=2)])

        self.in_planes = 128 * block.expansion
        self.block_extractor2 = nn.Sequential(*[self._make_layer(block, 256, num_blocks[2], stride=2),
                                                self._make_layer(block, 512, num_blocks[3], stride=2)])
        self.in_planes = 256 * block.expansion
        self.block_extractor3 = nn.Sequential(*[self._make_layer(block, 512, num_blocks[3], stride=2)])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        aux_logits = []
        aux_feats = []
        for i in range(len(x)):
            idx = i + 1
            out = getattr(self, 'block_extractor'+str(idx))(x[i])
            aux_feats.append(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            
            out = getattr(self, 'fc'+str(idx))(out)
            aux_logits.append(out)
            
        return aux_logits, aux_feats



class ResNet_Final_Auxiliary_Classifer(nn.Module):
    def __init__(self, block, num_classes=100):
        super(ResNet_Final_Auxiliary_Classifer, self).__init__()
        self.conv = conv1x1(512 * block.expansion * 4, 512 * block.expansion)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def forward(self, x):
        sum_fea = torch.cat(x, dim=1)
        out = self.conv(sum_fea)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class ResNet_Auxiliary(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet_Auxiliary, self).__init__()
        self.backbone = CIFAR_ResNet(block, num_blocks, num_classes)
        self.auxiliary_classifier = Auxiliary_Classifier(block, num_blocks, num_classes)
        self.final_aux_classifier = ResNet_Final_Auxiliary_Classifer(block, num_classes) 
        
    def forward(self, x, lam=0.5, index=None):
        logits, features = self.backbone(x)
        aux_logits, aux_feats = self.auxiliary_classifier(features[:-1])
        aux_feats.append(features[-1])
        bs = features[0].size(0)

        aux_logits.append(logits)

        if self.training is False:
            return aux_logits, aux_feats

        ensemble_features = [lam * (fea[:bs//2]) + (1 - lam) * (fea[index]) for fea in aux_feats]
        ensemble_mixup_features = [fea[bs//2:] for fea in aux_feats]

        ensemle_logits = self.final_aux_classifier(ensemble_features)
        ensemble_mixup_logits = self.final_aux_classifier(ensemble_mixup_features)

        return aux_logits, aux_feats, ensemle_logits, ensemble_mixup_logits


def MixSKD_CIFAR_ResNet18(**kwargs):
    return ResNet_Auxiliary(BasicBlock, [2,2,2,2], **kwargs)

def MixSKD_CIFAR_ResNet50(**kwargs):
    return ResNet_Auxiliary(Bottleneck, [3,4,6,3], **kwargs)


if __name__ == '__main__':
    net = MixSKD_CIFAR_ResNet18(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    net.eval()
    y = net(x)
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (1, 3, 32, 32)) / 1e6))

#ResNet for training CIFAR10 & CIFAR100

import math
import torch
import torch.nn as nn


def conv3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                        padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = conv3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        preact = out.clone()
        out = self.relu(out)
        
        if self.is_last:
            return out, preact
        else:
            return out

    

class ResNet(nn.Module):
    def __init__(self, depth, num_classes=10):
        super(ResNet, self).__init__()

        assert (depth - 2) % 6 == 0, 'depth shoule be one of 20, 32, 44, 56, 110'
        block_num = (depth - 2) // 6
        self.in_planes = 16
        self.conv1 = conv3(in_planes=3, out_planes=16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(planes=16, block_num=block_num)
        self.layer2 = self._make_layer(planes=32, block_num=block_num, stride=2)
        self.layer3 = self._make_layer(planes=64, block_num=block_num, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, 
                            stride=stride, bias=False),
                nn.BatchNorm2d(planes),
                )
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride, downsample, is_last=(block_num==1)))
        self.in_planes = planes
        for i in range(1, block_num):
            layers.append(BasicBlock(self.in_planes, planes, is_last=(i==block_num-1)))
        return nn.Sequential(*layers)
    
    def forward(self, x, is_feat=False, preact=False):
        x = self.conv1(x)
        x = self.bn1(x)
        f0_pre = x
        x = self.relu(x)
        f0 = x
        
        x, f1_pre = self.layer1(x)
        f1 = x
        x, f2_pre = self.layer2(x)
        f2 = x
        x, f3_pre = self.layer3(x)
        f3 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f4 = x
        x = self.fc(x)
        
        
        if is_feat:
            if preact:
                return [f0_pre, f1_pre, f2_pre, f3_pre, f4], x
            else:
                return [f0, f1, f2, f3, f4], x   # after relu
        else:
            return x


def resnet20(class_num = 10):
    return ResNet(20, class_num)

def resnet32(class_num = 10):
    return ResNet(32, class_num)

def resnet44(class_num = 10):
    return ResNet(44, class_num)

def resnet56(class_num = 10):
    return ResNet(56, class_num)

def resnet110(class_num = 10):
    return ResNet(110, class_num)


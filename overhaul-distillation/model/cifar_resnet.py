#ResNet for training CIFAR10 & CIFAR100
import torch
import torch.nn as nn

def conv3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                        padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        x = self.relu(x)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        
        return out

class ResNet(nn.Module):
    def __init__(self, depth, num_classes=10):
        super(ResNet, self).__init__()

        assert (depth - 2) % 6 == 0, 'depth shoule be one of 20, 32, 44, 56, 110'
        block_num = (depth - 2) // 6
        self.in_planes = 16
        self.conv1 = conv3(in_planes=3, out_planes=16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
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
        layers.append(BasicBlock(self.in_planes, planes, stride, downsample))
        self.in_planes = planes
        for i in range(1, block_num):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def get_bn_before_relu(self):
        
        if isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            print('ResNet unknown block error !!!')

        return [bn1, bn2, bn3]
    
    def get_channel_num(self):

        return [16, 32, 64]

    def extract_feature(self, x):

        x = self.conv1(x)
        x = self.bn1(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)

        x = nn.ReLU(inplace=False)(feat3)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return [feat1, feat2, feat3], out


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

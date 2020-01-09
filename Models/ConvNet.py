import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

class ConvBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''

    def __init__(self, in_planes, planes, stride=1):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(F.relu(out))
        out = self.bn2(out)
        out = self.conv2(F.relu(out))
        return out

class ConvNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ConvNet_CIFAR, self).__init__()
        self.in_planes = 16

        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.last_act = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear = nn.Linear(64, num_classes)

        init.orthogonal_(self.linear.weight)
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.orthogonal_(self.state_dict()[key])
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0 

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.last_act(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ConvNet20():
    return ConvNet_CIFAR(ConvBlock, [3,3,3], 10)

def ConvNet14():
    return ConvNet_CIFAR(ConvBlock, [2,2,2], 10)

def ConvNet8():
    return ConvNet_CIFAR(ConvBlock, [1,1,1], 10)


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn.modules.utils import _single, _pair, _triple

class _DBN(nn.Module):
    _version = 2

    def __init__(self, num_features):
        super(_DBN, self).__init__()
        self.num_features = num_features
        #self.weight = Parameter(torch.Tensor(num_features))
        self.weight = Parameter(torch.Tensor(1))
        self.bias = Parameter(torch.Tensor(self.num_features))
        self.running_mean = torch.zeros(self.num_features).cuda()
        self.running_var = torch.ones(self.num_features).cuda()
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long).cuda()
        self.momentum = 0.9
        self.eps = 1e-5
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input):
        self._check_input_dim(input)
        #self.weight_expand = self.weight
        self.weight_expand = self.weight.expand(self.num_features)
        #self.weight_expand = torch.ones(self.num_features).cuda()
        if self.training:
            #sample_mean = input.transpose(0,1).contiguous().view(self.num_features,-1).mean(1)
            sample_var = input.transpose(0,1).contiguous().view(self.num_features,-1).var(1)
            sample_var = sample_var.mean().unsqueeze(0).expand(self.num_features)
            sample_mean = torch.zeros(self.num_features).cuda()
            #sample_var = torch.ones(self.num_features).cuda()

            out_ = (input - sample_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)) / torch.sqrt(sample_var.unsqueeze(0).unsqueeze(2).unsqueeze(3) + self.eps)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * sample_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * sample_var

            out = self.weight_expand.unsqueeze(0).unsqueeze(2).unsqueeze(3) * out_ + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            scale = self.weight_expand.unsqueeze(0).unsqueeze(2).unsqueeze(3) / torch.sqrt(self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3) + self.eps)
            out = input * scale + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3) - self.running_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3) * scale

        return out

class _DBN2d(_DBN):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class ConvBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''

    def __init__(self, in_planes, planes, stride=1):
        super(ConvBlock, self).__init__()
        self.dbn1 = _DBN2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dbn2= _DBN2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.dbn1(x)
        out = self.conv1(F.relu(out))
        out = self.dbn2(out)
        out = self.conv2(F.relu(out))
        return out

class ConvNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ConvNet_CIFAR, self).__init__()
        self.in_planes = 16

        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.last_act = nn.Sequential(_DBN2d(64), nn.ReLU(inplace=True))
        
        self.last_act = nn.Sequential(nn.ReLU(inplace=True))
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

def ConvNet20_WOBN():
    return ConvNet_CIFAR(ConvBlock, [3,3,3], 10)

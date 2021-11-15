import torch.nn as nn
import math
import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../../')))
#from devkit.ops import SyncBatchNorm2d
import torch
import torch.nn.functional as F
from torch import autograd

from torch.nn import init
from devkit.sparse_ops import SparseConv, SparseLinear

# from devkit.sparse_ops import MixedSparseConv


__all__ = ['ResNet', 'resnet20_cifar_sparse', 'resnet32_cifar_sparse', 'resnet44_cifar_sparse', 'resnet56_cifar_sparse',
           'resnet110_cifar_sparse']

def conv3x3(in_planes, out_planes, stride=1, N=2, M=4,search=False):
    "3x3 convolution with padding"
    return SparseConv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, N=N, M=M,search=search)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, N=2, M=4,search = False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, N=N, M=M,search=search)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, N=N, M=M,search=search)
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
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,N=2, M=4,search=False):
        super(Bottleneck, self).__init__()
        self.conv1 = SparseConv(inplanes, planes, kernel_size=1, bias=False, N=N, M=M,search=search)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SparseConv(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, N=N, M=M,search=search)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SparseConv(planes, planes * 4, kernel_size=1, bias=False, N=N, M=M,search=search)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, depth, num_classes=100, N=2, M=4,search = False):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6


        self.named_layers = {} # layers have parameters like conv2d and linear
        self.dense_layers = {} # layers which will be kept as dense
        
        block = Bottleneck if depth >=54 else BasicBlock
        self.N = N
        self.M = M
        self.inplanes = 16
        self.conv1 = SparseConv(3, 16, kernel_size=3, padding=1,
                               bias=False,N=N, M=M,search=search)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n,stride=1,N=N,M=M,search=search)
        self.layer2 = self._make_layer(block, 32, n, stride=2,N=N,M=M,search=search)
        self.layer3 = self._make_layer(block, 64, n, stride=2,N=N,M=M,search=search)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = SparseLinear(64 * block.expansion, num_classes,N=N,M=M,search=search)


        self._set_sparse_layer_names()

        #initialization

        for m in self.modules():
            if isinstance(m, SparseConv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,N=2, M=4,search=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SparseConv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,N=N, M=M,search=search),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,N=N,M=M,search=search))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,N=N,M=M,search=search))

        return nn.Sequential(*layers)


    def set_weight_decay(self, weight_decay):
        for mod in self.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear) :
                mod.decay = weight_decay


    def _set_sparse_layer_names(self):
        conv2d_idx = 0
        linear_idx = 0

        for mod in self.modules():
            if isinstance(mod, SparseConv):
                layer_name = 'SparseConv{}_{}-{}-{}'.format(
                    conv2d_idx, mod.in_channels, mod.out_channels,mod.kernel_size
                )
                
                mod.set_layer_name(layer_name)

                Cout = mod.weight.data.size()[0]
                C = mod.weight.data.size()[1]
                Kw = mod.weight.data.size()[2]
                Kh = mod.weight.data.size()[3]
                
                mod.layer_ind = conv2d_idx
                self.named_layers[layer_name] = list([Cout,C,Kw,Kh])

                conv2d_idx += 1
            # elif isinstance(mod, torch.nn.BatchNorm2d):
            #     layer_name = 'BatchNorm2D{}_{}'.format(
            #         batchnorm2d_idx, mod.num_features)
            #     named_layers[layer_name] = mod
            #     batchnorm2d_idx += 1
            elif isinstance(mod, SparseLinear):
                layer_name = 'Linear{}_{}-{}'.format(
                    linear_idx, mod.in_features, mod.out_features
                )
                
                Cout = mod.weight.data.size()[0]
                C = mod.weight.data.size()[1]

                mod.set_layer_name(layer_name)
                mod.layer_ind = linear_idx

                self.named_layers[layer_name] = list([Cout,C])
                #self.dense_layers[layer_name] = list([Cout,C])

                linear_idx += 1

    def set_datalayout(self,layout):
        for mod in self.modules():
            if isinstance(mod, SparseConv): # for Linear Layer, data layout does not matter
                mod.change_layout(layout)

    def check_N_M(self):
        sparse_scheme = {}

        for mod in self.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
                sparse_scheme[mod.get_name()] = list([mod.N,mod.M])
            #elif isinstance(mod, torch.nn.Linear): TODOs
            #    pass
        return sparse_scheme


    def get_overall_sparsity(self):
        dense_paras = 0
        sparse_paras = 0
        for mod in self.modules():
            if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
                dense_paras += mod.dense_parameters 
                sparse_paras += mod.get_sparse_parameters()    # number(M) of non-zeros
            # elif isinstance(mod, torch.nn.Linear): # at this moment we keep fully connected layer as dense, and does not account this layer
            #     dense_paras += mod.weight.data.size()[0] * mod.weight.data.size()[1]
            #     sparse_paras += 0
        
        return 1.0 - (sparse_paras/dense_paras)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnet20_cifar_sparse(**kwargs):
    model = ResNet(20,  **kwargs)
    return model


def resnet32_cifar_sparse(**kwargs):
    model = ResNet(32, **kwargs)
    return model


def resnet44_cifar_sparse(**kwargs):
    model = ResNet(44,  **kwargs)
    return model


def resnet56_cifar_sparse(**kwargs):
    model = ResNet(56, **kwargs)
    return model


def resnet110_cifar_sparse(**kwargs):
    model = ResNet(110, **kwargs)
    return model
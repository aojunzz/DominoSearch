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
from devkit.sparse_ops import SparseConv



__all__ = ['ResNet', 'resnet20_cifar_mixed', 'resnet32_cifar_mixed', 'resnet44_cifar_mixed', 'resnet56_cifar_mixed',
           'resnet110_cifar_mixed']



def conv3x3(in_planes, out_planes, stride=1, N=2, M=4):
    "3x3 convolution with padding"
    return SparseConv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, N=N, M=M)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, N=2, M=4):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, N=N, M=M)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, N=N, M=M)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None,N=2, M=4):
        super(Bottleneck, self).__init__()
        self.conv1 = SparseConv(inplanes, planes, kernel_size=1, bias=False, N=N, M=M)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SparseConv(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, N=N, M=M)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SparseConv(planes, planes * 4, kernel_size=1, bias=False, N=N, M=M)
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

    def __init__(self, depth, num_classes=100, Ns=[1,2], M=4):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        self.input_size = 32

        # layer dicts
        self.named_layers = {} # layers have parameters like conv2d and linear
        self.dense_layers = {} # layers which will be kept as dense
        

        block = Bottleneck if depth >=54 else BasicBlock
        self.N = Ns[0]
        self.M = M
        self.inplanes = 16
        self.conv1 = SparseConv(3, 16, kernel_size=3, padding=1,
                               bias=False,N=self.N, M=M)

        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n,stride=1,N=self.N,M=M) #2*3
        self.layer2 = self._make_layer(block, 32, n, stride=2,N=self.N,M=M) #2*3
        self.layer3 = self._make_layer(block, 64, n, stride=2,N=self.N,M=M) #2*3
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)




        num_sparse_conv = 0


        for m in self.modules():
            if isinstance(m, SparseConv):
                num_sparse_conv = num_sparse_conv + 1

        i = 0

        # set the name of each layer
        self._set_sparse_layer_names()

        for m in self.modules():
            if isinstance(m, SparseConv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
                if(i == num_sparse_conv-1 or i==1 or i==num_sparse_conv-2):
                    # first and final we use 2:4 and keep others as 1:4
                    m.apply_N_M(Ns[1],M)  # Ns[1] = 2
                elif i==0 :
                    m.apply_N_M(M,M)
                else:
                    m.apply_N_M(Ns[0],M)  # Ns[0] = 1
                i = i + 1

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            #elif isinstance(m, nn.Linear):

    def _make_layer(self, block, planes, blocks, stride=1,N=2, M=4):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SparseConv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,N=N, M=M),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,N=N,M=M))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,N=N,M=M))

        return nn.Sequential(*layers)


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
                

                self.named_layers[layer_name] = list([Cout,C,Kw,Kh])
                conv2d_idx += 1
            # elif isinstance(mod, torch.nn.BatchNorm2d):
            #     layer_name = 'BatchNorm2D{}_{}'.format(
            #         batchnorm2d_idx, mod.num_features)
            #     named_layers[layer_name] = mod
            #     batchnorm2d_idx += 1
            elif isinstance(mod, torch.nn.Linear):
                layer_name = 'Linear{}_{}-{}'.format(
                    linear_idx, mod.in_features, mod.out_features
                )
                
                Cout = mod.weight.data.size()[0]
                C = mod.weight.data.size()[1]


                self.named_layers[layer_name] = list([Cout,C])
                self.dense_layers[layer_name] = list([Cout,C])

                linear_idx += 1



    def calculate_FLOPs(self,in_size):
        input_size = in_size
        output_size = in_size
        pass


    def check_num_parameters(self):
        original_paramters_list= {}
        sparse_parameters_list = {}

        for mod in self.modules():
            if isinstance(mod, SparseConv):
                #sparse_scheme[mod.get_name()] = list([mod.N,mod.M])
                sparse_parameters_list[mod.get_name()] = mod.get_sparse_parameters() * 	1e-6 #M
                original_paramters_list[mod.get_name()] = mod.dense_parameters * 1e-6
        
        return sparse_parameters_list, original_paramters_list



    def check_N_M(self):
        sparse_scheme = {}

        for mod in self.modules():
            if isinstance(mod, SparseConv):
                sparse_scheme[mod.get_name()] = list([mod.N,mod.M])
            #elif isinstance(mod, torch.nn.Linear): TODOs
            #    pass
        return sparse_scheme
    
    def set_datalayout(self,layout):
        for mod in self.modules():
            if isinstance(mod, SparseConv):
                mod.change_layout(layout)
            #elif isinstance(mod, torch.nn.Linear): TODOs
            #    pass



    def get_overall_sparsity(self):
        dense_paras = 0
        sparse_paras = 0
        for mod in self.modules():
            if isinstance(mod, SparseConv):
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



def resnet20_cifar_mixed(**kwargs):
    model = ResNet(20,  **kwargs)
    return model


def resnet32_cifar_mixed(**kwargs):
    model = ResNet(32, **kwargs)
    return model


def resnet44_cifar_mixed(**kwargs):
    model = ResNet(44,  **kwargs)
    return model


def resnet56_cifar_mixed(**kwargs):
    model = ResNet(56, **kwargs)
    return model


def resnet110_cifar_mixed(**kwargs):
    model = ResNet(110, **kwargs)
    return model
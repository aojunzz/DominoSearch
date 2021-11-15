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




'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = [
    'VGG','vgg16_mixed', 'vgg16_bn_mixed',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=100, Ns=[1,2], M=4):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

        self.N = Ns[0]
        self.M = M

        self.named_layers = {} # layers have parameters like conv2d and linear
        self.dense_layers = {} # layers which will be kept as dense

        num_sparse_conv = 0


        for m in self.modules():
            if isinstance(m, SparseConv):
                num_sparse_conv = num_sparse_conv + 1

        i = 0

        # set the name of each layer
        self._set_sparse_layer_names()


    

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


    def _set_sparse_layer_names(self):
        conv2d_idx = 0
        linear_idx = 0

        for mod in self.modules():
            if isinstance(mod, SparseConv):
                layer_name = 'SparseConv{}_{}-{}'.format(
                    conv2d_idx, mod.in_channels, mod.out_channels
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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels)
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, N=2, M=4):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = SparseConv(in_channels, v, kernel_size=3, stride=1,
                     padding=1, bias=False, N=N, M=M) #(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    # 'E': [64, 128, 'M', 128, 256, 'M', 64, 128, 256, 512, 1024, 'M', 64, 128, 256, 512, 1024, 2048,'M',256, 512, 1024, 512,'M']
}


# def vgg11(**kwargs):
#     """VGG 11-layer model (configuration "A")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = VGG(make_layers(cfg['A']), **kwargs)
#     return model


# def vgg11_bn(**kwargs):
#     """VGG 11-layer model (configuration "A") with batch normalization"""
#     model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
#     return model


# def vgg13(**kwargs):
#     """VGG 13-layer model (configuration "B")
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = VGG(make_layers(cfg['B']), **kwargs)
#     return model


# def vgg13_bn(**kwargs):
#     """VGG 13-layer model (configuration "B") with batch normalization"""
#     model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
#     return model


def vgg16_mixed(**kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn_mixed(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True, N=2, M=4), **kwargs)
    return model


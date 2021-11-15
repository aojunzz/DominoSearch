import torch.nn as nn
import math
import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../../')))
#from devkit.ops import SyncBatchNorm2d
import torch
import torch.nn.functional as F
from torch import autograd
from torch.nn.modules.utils import _pair as pair
from torch.nn import init
from torch.jit.annotations import Optional, Tuple
from torch import Tensor
from collections import namedtuple
from devkit.sparse_ops import SparseConv, SparseLinear
#from devkit.sparse_ops import MixedSparseConv
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url




__all__ = ['GoogLeNet', 'googlenet_sparse', "GoogLeNetOutputs", "_GoogLeNetOutputs"]

model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
GoogLeNetOutputs.__annotations__ = {'logits': Tensor, 'aux_logits2': Optional[Tensor],
                                    'aux_logits1': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat
_GoogLeNetOutputs = GoogLeNetOutputs


def googlenet_sparse(pretrained=False, progress=True, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' not in kwargs:
            kwargs['aux_logits'] = False
        if kwargs['aux_logits']:
            warnings.warn('auxiliary heads in the pretrained googlenet model are NOT pretrained, '
                          'so make sure to train them')
        original_aux_logits = kwargs['aux_logits']
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False
        model = GoogLeNet(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['googlenet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.aux1, model.aux2
        return model

    return GoogLeNet(**kwargs)



class GoogLeNet(nn.Module):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False, init_weights=True,
                 blocks=None,N=2, M=4,search=True):
        super(GoogLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]


        self.N = N
        self.M = M

        self.named_layers = {} # layers have parameters like conv2d and linear
        self.dense_layers = {} # layers which will be kept as dense

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes)
            self.aux2 = inception_aux_block(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = SparseLinear(1024, num_classes, N=16,M=16,search=True)


        self._set_sparse_layer_names()

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, SparseConv) or isinstance(m, SparseLinear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        # type: (Tensor) -> Tensor
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x
    

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




    def set_newidea_flag(self,flag):
        for m in self.modules():
            if isinstance(m, SparseConv) or isinstance(m, SparseLinear):
                self.new_idea = flag

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

    def _forward(self, x):
        # type: (Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux1 = self.aux1(x)
        else:
            aux1 = None

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if aux_defined:
            aux2 = self.aux2(x)
        else:
            aux2 = None

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux2, aux1

    @torch.jit.unused
    def eager_outputs(self, x, aux2, aux1):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> GoogLeNetOutputs
        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x

    def forward(self, x):
        # type: (Tensor) -> GoogLeNetOutputs
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)


class Inception(nn.Module):
    __constants__ = ['branch2', 'branch3', 'branch4']

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj,
                 conv_block=None):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d_dense
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels,N=16, M=16,search=True, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = SparseConv(in_channels, out_channels, bias=False,N=N, M=M,search=search, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)



class BasicConv2d_dense(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d_dense, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

























# __all__ = ['AlexNet', 'alexnet_sparse']


# model_urls = {
#     'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
# }


# class AlexNet(nn.Module):

#     def __init__(self, num_classes=1000,N=2, M=4,search=False):
#         super(AlexNet, self).__init__()


#         self.N = N
#         self.M = M

#         self.named_layers = {} # layers have parameters like conv2d and linear
#         self.dense_layers = {} # layers which will be kept as dense


#         self.features = nn.Sequential(
#             #nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             SparseConv(3,64,kernel_size=11, stride=4, padding=2,N=N, M=M,search=search)
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             #nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             SparseConv(64,192,kernel_size=5, padding=2,N=N, M=M,search=search)
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             #nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             SparseConv(192,384,kernel_size=3, padding=1,N=N, M=M,search=search)
#             nn.ReLU(inplace=True),
#             #nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             SparseConv(384,256,kernel_size=3, padding=1,N=N, M=M,search=search)
#             nn.ReLU(inplace=True),
#             #nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             SparseConv(256,256,kernel_size=3, padding=1,N=N, M=M,search=search)
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             #nn.Linear(256 * 6 * 6, 4096),
#             SparseLinear(256 * 6 * 6,4096,N=N, M=M,search=search)
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             #nn.Linear(4096, 4096),
#             SparseLinear(4096,4096,N=N, M=M,search=search)
#             nn.ReLU(inplace=True),
#             #nn.Linear(4096, num_classes),
#             SparseLinear(4096,num_classes,N=N, M=M,search=search)
#         )

#         self._set_sparse_layer_names()


#     def set_weight_decay(self, weight_decay):
#         for mod in self.modules():
#             if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear) :
#                 mod.decay = weight_decay

#     def _set_sparse_layer_names(self):
#         conv2d_idx = 0
#         linear_idx = 0

#         for mod in self.modules():
#             if isinstance(mod, SparseConv):
#                 layer_name = 'SparseConv{}_{}-{}-{}'.format(
#                     conv2d_idx, mod.in_channels, mod.out_channels,mod.kernel_size
#                 )
                
#                 mod.set_layer_name(layer_name)

#                 Cout = mod.weight.data.size()[0]
#                 C = mod.weight.data.size()[1]
#                 Kw = mod.weight.data.size()[2]
#                 Kh = mod.weight.data.size()[3]
                
#                 mod.layer_ind = conv2d_idx
#                 self.named_layers[layer_name] = list([Cout,C,Kw,Kh])

#                 conv2d_idx += 1
#             # elif isinstance(mod, torch.nn.BatchNorm2d):
#             #     layer_name = 'BatchNorm2D{}_{}'.format(
#             #         batchnorm2d_idx, mod.num_features)
#             #     named_layers[layer_name] = mod
#             #     batchnorm2d_idx += 1
#             elif isinstance(mod, SparseLinear):
#                 layer_name = 'Linear{}_{}-{}'.format(
#                     linear_idx, mod.in_features, mod.out_features
#                 )
                
#                 Cout = mod.weight.data.size()[0]
#                 C = mod.weight.data.size()[1]

#                 mod.set_layer_name(layer_name)
#                 mod.layer_ind = linear_idx

#                 self.named_layers[layer_name] = list([Cout,C])
#                 #self.dense_layers[layer_name] = list([Cout,C])

#                 linear_idx += 1




#     def set_newidea_flag(self,flag):
#         for m in self.modules():
#             if isinstance(m, SparseConv) or isinstance(m, SparseLinear):
#                 self.new_idea = flag

#     def set_datalayout(self,layout):
#         for mod in self.modules():
#             if isinstance(mod, SparseConv): # for Linear Layer, data layout does not matter
#                 mod.change_layout(layout)



#     def check_N_M(self):
#         sparse_scheme = {}

#         for mod in self.modules():
#             if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
#                 sparse_scheme[mod.get_name()] = list([mod.N,mod.M])
#             #elif isinstance(mod, torch.nn.Linear): TODOs
#             #    pass
#         return sparse_scheme



#     def get_overall_sparsity(self):
#         dense_paras = 0
#         sparse_paras = 0
#         for mod in self.modules():
#             if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
#                 dense_paras += mod.dense_parameters 
#                 sparse_paras += mod.get_sparse_parameters()    # number(M) of non-zeros
#             # elif isinstance(mod, torch.nn.Linear): # at this moment we keep fully connected layer as dense, and does not account this layer
#             #     dense_paras += mod.weight.data.size()[0] * mod.weight.data.size()[1]
#             #     sparse_paras += 0
        
#         return 1.0 - (sparse_paras/dense_paras)


#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x


# def alexnet_sparse(pretrained=False, progress=True, **kwargs):

#     model = AlexNet(**kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['alexnet'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model


import torch
from torch import autograd, nn
import torch.nn.functional as F
import numpy as np
from itertools import repeat


# we need two dictionaries 
# {layer_name :layer module}
# {layer_name :}

# this is pruned based on Pytorch standard data layout
# input : N,C,H,W
# weight tensor: Cout,Cin,Kh,Kw , 0,1,2,3 | 0,3,1,2
# we also have another commonly-used data layout
# input N,H,W,C
# weight: Cout, Kh, Kw,Cin 0,2,3,1 | 0,1,2,3

global partition_grad_weight_penalty


class Sparse(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""
    @staticmethod
    def forward(ctx, weight, N, M, decay):
        ctx.save_for_backward(weight)
        output = weight.clone()

        ctx.M = M
        ctx.N = N 

        if(M==N):
            return output #*w_b

        # if N==M:
        #     return output
        length = weight.numel() #number of papameters
        group = int(length/M)

        weight_temp = weight.detach().abs().reshape(group, M)


        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)] # indicate which will be zeros (pruned) 

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape) #

        ctx.mask = w_b
        ctx.decay = decay

        return output*w_b #,w_b

    @staticmethod
    def backward(ctx, grad_output):

        

        if (ctx.M == ctx.N):
            res = grad_output
        else:
            weight, = ctx.saved_tensors
            res = grad_output + ctx.decay * (1-ctx.mask) * weight

        return res, None, None, None


class Sparse_NHWC(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""
    @staticmethod
    def forward(ctx, weight, N, M, decay ):
        ctx.save_for_backward(weight)
        output = weight.clone()

        ctx.M = M
        ctx.N = N 

        if(M==N):
            #w_b = torch.ones(weight_temp.shape, device=weight_temp.device).reshape(weight.shape) #
            #ctx.mask = w_b
            # ctx.decay = decay
            return output #*w_b

        length = weight.numel() #number of papameters
        group = int(length/M)

        weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.permute(0,2,3,1).shape)
        w_b = w_b.permute(0,3,1,2)

        #assert w_b.size()[0] == Cout and w_b.size()[1] == Cin and w_b.size()[2] == Kh and w_b.size()[3] == Kw

        ctx.mask = w_b
        ctx.decay = decay

        return output*w_b #,w_b

    @staticmethod
    def backward(ctx, grad_output):

        
        if (ctx.M == ctx.N):
            res = grad_output
        else:
            weight, = ctx.saved_tensors
            res = grad_output + ctx.decay * (1-ctx.mask) * weight

        
        return res, None, None, None



class SparseConv(nn.Conv2d):
    """" implement N:M sparse convolution layer """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', N=2, M=4, layout='NHWC', search = False, **kwargs):
        self.N = N # number of non-zeros
        self.M = M

        self.name = "deault name"
        self.layout = layout
        self.decay =  0.0002   #0.0002
        self.print_flag = False

        self.flops = 0
        self.input_shape = None
        self.output_shape = None


        self.layer_ind = None

        self.k_ = kernel_size

        if bias == True:
            self.dense_parameters = in_channels * out_channels * kernel_size * kernel_size
        else:
            self.dense_parameters = out_channels * (kernel_size * kernel_size * in_channels + 1)
        
        super(SparseConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)
        

    def update_decay(self,updated_decay):
        self.decay = updated_decay
        



    # TODO: change the sparse scheme
    def apply_N_M(self,N,M):
        self.N = N 
        self.M = M
        

    def change_layout(self,layout):
        if layout not in ['NCHW','NHWC']:
            print("Unsupported layout")
            exit(0)
        self.layout = layout


    def get_sparse_weights(self):
        if self.layout == 'NCHW' or self.k_ == 1:
            #print("use NCHW to train")
            return Sparse.apply(self.weight, self.N, self.M, self.decay)

        elif self.layout == 'NHWC':
            return Sparse_NHWC.apply(self.weight, self.N, self.M,self.decay)


    def set_layer_name(self,name):
        self.name = name

    def get_name(self):
        return self.name
    
    def get_sparse_parameters(self):
        param_size = int(self.dense_parameters * self.N/self.M)  # dense parameters * sparsity (N/M)
        return param_size
    
    # def get_FLOPs(self):
    #     param_size = int(self.dense_parameters * N/M)
    #     out_h = int ()

    def forward(self, x):
        #
        w = self.get_sparse_weights()
        # setattr(self.weight, "mask", mask)
        #self.spare_weight = w.clone() # store the spare weight
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

# ?

class SparseLinear(nn.Linear):

        # def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', N=2, M=4, layout='NCHW', search = False, **kwargs):

    def __init__(self, in_features, out_features, bias = True, N=2, M=2, search = False, **kwargs):
        self.N = N # number of non-zeros
        self.M = M

        self.name = "deault name"
        self.layout = 'NCHW'
        self.decay =  0.0002   #0.0002
        self.print_flag = False

        
        self.layer_ind = None
        self.flops = 0
        self.input_shape = None
        self.output_shape = None



        #self.spare_weight = None
        if bias == True:
            self.dense_parameters = in_features * out_features
        else:
            self.dense_parameters = out_features * (in_features + 1)
        
        super(SparseLinear, self).__init__(in_features, out_features, bias, **kwargs)


    # TODO: update decay

    def update_decay(self,updated_decay):
        self.decay = updated_decay
        



    # TODO: change the sparse scheme
    def apply_N_M(self,N,M):
        self.N = N 
        self.M = M


    # layout has to be initialized
    def change_layout(self,layout):
        if layout not in ['NCHW','NHWC']:
            print("Unsupported layout")
            exit(0)
        self.layout = layout

    def get_sparse_weights(self):

        return Sparse.apply(self.weight, self.N, self.M, self.decay)# support N=M case


    def set_layer_name(self,name):
        self.name = name

    def get_name(self):
        return self.name
    
    def get_sparse_parameters(self):
        param_size = int(self.dense_parameters * self.N/self.M)  # dense parameters * sparsity (N/M)
        return param_size



    def forward(self, x):

        w = self.get_sparse_weights()
        x = F.linear(x, w,self.bias)
        return x

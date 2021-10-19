import torch
from torch import autograd, nn
import torch.nn.functional as F
import numpy as np
from itertools import repeat
from torch._six import container_abcs
from torch import linalg as LA

global partition_grad_weight_penalty


class Sparse_find_mix_from_dense(autograd.Function):
    """" Find mixed N:M from dense pre-trained mode,
    
    """
    @staticmethod
    def forward(ctx, weight, N_intermediate, M, decay, learned_threshold, normalized_factor,layer_name,print_flag,data_layout,apply_penalty):

        ctx.layer_name = layer_name
        ctx.print_flag = print_flag
        ctx.decay = decay
        ctx.normalized_factor = normalized_factor
        ctx.layout = data_layout
        ctx.apply_penalty = apply_penalty
        ctx.save_for_backward(weight)
        output = weight.clone()
        length = weight.numel() #number of papameters
        ctx.M = M
        ctx.N = N_intermediate
        group = int(length/M)


        if ctx.layout == 'NHWC':

            Cout = weight.size()[0]
            Cin = weight.size()[1]
            Kh = weight.size()[2]
            Kw = weight.size()[3]
            weight_t = weight.clone().permute(0,2,3,1)
            weight_temp = weight_t.detach().abs().reshape(group, M)
        else:    
            weight_temp = weight.detach().abs().reshape(group, M)

        mask_b = (weight_temp > learned_threshold) #* 1.0

        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N_intermediate)] # indicate which will be zeros (pruned) 
        #
        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0) #.reshape(weight.shape) #



        if ctx.layout == 'NHWC':
            #penalty_factor = penalty_factor.reshape(weight_t.shape)

            #w_b = mask_b.reshape(weight_t.shape)
            # transpose NHWC to NCHW (default Pytorch Layout) for 
            w_b = w_b.reshape(weight_t.shape)
            w_b = w_b.permute(0,3,1,2)
            #penalty_factor = penalty_factor.permute(0,3,1,2)


        else:
            #penalty_factor = penalty_factor.reshape(weight.shape)

            #w_b = mask_b.reshape(weight.shape)

            w_b = w_b.reshape(weight.shape)

        ctx.mask = w_b 
        
        # ctx.penalty_factor = penalty_factor
        
        return output*w_b #,w_b

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        

        grad_survival = grad_output  * ctx.mask  # only backward the survival gradients
        if ctx.apply_penalty == True:
            #print('apply penalty')
            penalty = weight  * ctx.decay * ctx.normalized_factor 
            penalty = penalty * ctx.mask 
        else:
            penalty = 0

        return    (grad_survival + penalty )  , None, None, None, None,None,None,None,None,None




class SparseConv(nn.Conv2d):
    """" implement N:M sparse convolution layer """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', N=2, M=4, layout='NHWC', search = False, **kwargs):
        self.N = N # number of non-zeros
        self.M = M

        self.N_intermediate = M # inialize as M
        self.apply_penalty = False

        self.name = "deault name"
        self.layout = layout
        self.decay =  0.1   #0.0002
        self.print_flag = False

        self.flops = 0
        self.input_shape = None
        self.output_shape = None

        self.mix_from_dense = search
        self.normal_train = False # This has to be initialized

        self.normalized_factor = 1.0
        self.learned_threshold = None
        self.smallest_survival = None

        self.layer_ind = None

        self.k_ = kernel_size

        self.learned_threshold_m = None # T in used in paper

        self.log = None
        #self.spare_weight = None
        if bias == True:
            self.dense_parameters = in_channels * out_channels * kernel_size * kernel_size
        else:
            self.dense_parameters = out_channels * (kernel_size * kernel_size * in_channels + 1)
        
        super(SparseConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)
        
    

    def get_mask(self):
        N = self.N 
        M = self.M

        weight = self.weight
        length = weight.numel() #number of papameters
        group = int(length/M)
        
        if self.layout == 'NHWC':
            weight_t = weight.clone().permute(0,2,3,1)
            weight_temp = weight_t.detach().abs().reshape(group, M)
        else:    
            weight_temp = weight.detach().abs().reshape(group, M)

        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)] # indicate which will be zeros (pruned) 
        #
        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0) #.reshape(weight.shape) #

        if self.layout == 'NHWC':
            w_b = w_b.reshape(weight_t.shape)
            w_b = w_b.permute(0,3,1,2)

        else:
            w_b = w_b.reshape(weight.shape)

        return w_b 


    def update_decay(self,updated_decay):
        self.decay = updated_decay
        pass


    def smallest_among_survival(self):
        M = self.M
        N = self.N_intermediate
        length = self.weight.numel() #number of papameters
        group = int(length/M)

        #weight_temp = None

        if self.layout == 'NCHW' or self.k_ == 1:
            weight_temp = self.weight.detach().abs().reshape(group, M)

        elif self.layout == 'NHWC': # 

            Cout = self.weight.size()[0]
            Cin = self.weight.size()[1]
            Kh = self.weight.size()[2]
            Kw = self.weight.size()[3]
            weight_t = self.weight.clone().permute(0,2,3,1)
            weight_temp = weight_t.detach().abs().reshape(group, M)
            #index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)] # indicate which will be zeros (pruned) 

  
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)] # indicate which will be zeros (pruned) 


        #np.where(diff_ >= 0 ,diff_ ,np.inf)
        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0) #.reshape(self.weight.shape) #

        weights_survival = w_b * weight_temp
        #print(weights_survival.dtype)
        smallest_of_survival_ = torch.where(weights_survival > 0.0 ,weights_survival ,torch.tensor(float('inf'),dtype=weights_survival.dtype,device=weights_survival.get_device()))
        smallest_of_survival,inds = torch.min(smallest_of_survival_,dim = 1)
        smallest_of_survival_col = smallest_of_survival.reshape(smallest_of_survival.numel(),1)
        #assert smallest_of_survival.numel() == group

        #self.smallest_survival = smallest_of_survival_col
        return smallest_of_survival_col


    def update_learned_sparsity(self):

        M = self.M
        N = self.N_intermediate
        length = self.weight.numel() #number of papameters
        group = int(length/M)

        #weight_temp = None
        if self.layout == 'NCHW' or self.k_ == 1:
            weight_temp = self.weight.detach().abs().reshape(group, M)

            #assert largetest_among_pruned.numel() == group

        elif self.layout == 'NHWC': #

            Cout = self.weight.size()[0]
            Cin = self.weight.size()[1]
            Kh = self.weight.size()[2]
            Kw = self.weight.size()[3]
            weight_t = self.weight.clone().permute(0,2,3,1)
            weight_temp = weight_t.detach().abs().reshape(group, M)


        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)] # indicate which will be zeros (pruned) 

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0) #.reshape(self.weight.shape) #

        weights_pruned = (1-w_b) * weight_temp
        largetest_among_pruned,inds = torch.max(weights_pruned,dim = 1)
        largetest_among_pruned_col = largetest_among_pruned.reshape(largetest_among_pruned.numel(),1)

        self.smallest_survival = self.smallest_among_survival()
        
        self.learned_threshold = largetest_among_pruned_col

        if N==M:
            self.learned_threshold_m = self.intialize_threshold_with_average_lowest()   #self.smallest_survival, intialize
        else:## will never go to else
            pass


    # initialize with average smallest M/2 elements, see paper Section 3.2
    def intialize_threshold_with_average_lowest(self):
        M = self.M

        N = int(M/2)

        length = self.weight.numel() #number of papameters
        group = int(length/M)

        #weight_temp = None

        if self.layout == 'NCHW' or self.k_ == 1:
            weight_temp = self.weight.detach().abs().reshape(group, M)

        elif self.layout == 'NHWC': #
            Cout = self.weight.size()[0]
            Cin = self.weight.size()[1]
            Kh = self.weight.size()[2]
            Kw = self.weight.size()[3]
            weight_t = self.weight.clone().permute(0,2,3,1)
            weight_temp = weight_t.detach().abs().reshape(group, M)

        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)] # indicate of lowest M/2

        w_b = torch.zeros(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=1) #.reshape(self.weight.shape) #

        weights_lower = weight_temp * w_b

        initial_values = torch.sum(weights_lower,dim = 1, keepdim =True) # sum of smallest M/2 elements

        initial_values = initial_values/N # average smallest M/2 elements

        return initial_values

  
    def check_num_survival_parameters(self):

        if self.smallest_survival == None: # this means has not initialized, still dense
            return self.dense_parameters

        # abs : magnitude
        M = self.M
        length = self.weight.numel()
        group = int(length/M)

        if self.layout == 'NCHW' or self.k_ == 1:
            mask_ = (self.weight.detach().abs().reshape(group, M) >= self.smallest_survival) * 1.0

        elif self.layout == 'NHWC': #
            Cout = self.weight.size()[0]
            Cin = self.weight.size()[1]
            Kh = self.weight.size()[2]
            Kw = self.weight.size()[3]
            weight_t = self.weight.clone().permute(0,2,3,1)
            mask_ = (weight_t.detach().abs().reshape(group, M) > self.smallest_survival) * 1.0


        return torch.sum(mask_).cpu().detach().numpy()



    # check sparsity of each group, 
    def check_sparsity_each_group(self,prob = 0.75):
        N_inter = self.N_intermediate
        M = self.M
        n = int(np.log2(M)) + 1
        Ns = [2**x for x in range(n)]
        #Ns = [x for x in range(1,self.M+1)]
        length = self.weight.numel()
        group = int(length/M)
        if self.layout == 'NCHW' or self.k_ == 1: 
            weight_group = (self.weight.detach().abs().reshape(group, M) > self.learned_threshold_m) #* 1.0
            N_every_group = torch.sum(weight_group, 1,keepdim=True)    #X.sum(1, keepdim=True)

        elif self.layout == 'NHWC': 
            Cout = self.weight.size()[0]
            Cin = self.weight.size()[1]
            Kh = self.weight.size()[2]
            Kw = self.weight.size()[3]
            weight_t = self.weight.clone().permute(0,2,3,1)
            weight_group = (weight_t.detach().abs().reshape(group, M) > self.learned_threshold_m) # * 1.0
            N_every_group = torch.sum(weight_group, 1,keepdim=True) # survival N


        survival_elements = torch.sum(N_every_group)
        survival_rate = survival_elements/length

        N_inter_change = False
        if (self.N > 1):
            satisfied_each_group = N_every_group < N_inter
            c_prob = torch.sum(satisfied_each_group)/satisfied_each_group.numel()
            if c_prob >= prob:
                N_inter = N_inter -1
                #self.N_intermediate = N_inter
                N_inter_change = True


        # update self.N_intermediate
        self.N_intermediate = N_inter
        # if N_inter is valide and != current N, then update N
        # return N_inter and True if updated

        if N_inter in Ns and N_inter!=self.N:
            self.N = N_inter  #do not update

            return N_inter, True, 1.0 - survival_rate, N_inter_change
        # return N_inter and False if does not update
        return N_inter, False , 1.0 - survival_rate, N_inter_change


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


        if self.mix_from_dense == True:

            return Sparse_find_mix_from_dense.apply(self.weight, self.N_intermediate, self.M, self.decay, self.learned_threshold_m,self.normalized_factor,self.name,self.print_flag,self.layout,self.apply_penalty)


        elif (self.M==self.N or self.normal_train==True or self.k_ == 1): 
            #self.spare_weight = self.weight
            #print("dense train")
            # Pytorch default layout, because 
            return Sparse.apply(self.weight, self.N, self.M, self.decay)# support N=M case
            #return self.weight
        else:
            if self.layout == 'NCHW':
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

        self.N_intermediate = M # inialize as M

        self.apply_penalty = False

        self.name = "deault name"
        self.layout = 'NCHW'
        self.decay =  0.0002   #0.0002
        self.print_flag = False

        

        self.layer_ind = None
        self.flops = 0
        self.input_shape = None
        self.output_shape = None

        self.mix_from_dense = search
        self.normal_train = False # This has to be initialized

        self.normalized_factor = 1.0
        self.learned_threshold = None
        self.smallest_survival = None

        self.learned_threshold_m = None #smallest among survival and largest among pruned

        self.log = None
        #self.spare_weight = None
        if bias == True:
            self.dense_parameters = in_features * out_features
        else:
            self.dense_parameters = out_features * (in_features + 1)
        
        super(SparseLinear, self).__init__(in_features, out_features, bias, **kwargs)



    # TODO: check the TC compatibility, now keep silence,
    def check_TC_compatibility(self):
        Cout, C, Kh, Kw = self.weight.size()
        if (Cout % 8) != 0 or (C % 16) != 0 :
            # need a smart way to print this message, especially for distributed training
            print("The weight shapes of this layer (%d,%d,%d,%d)-(Cout,C,Kh,Kw) does not meet TC_compatibility, pruning should be skipped, use normal Conv2D" % (Cout,C,Kh,Kw))
            return False # set true here temporally 
        else:
            return True
    
    # # TODO: check the sparsity of real weight
    # def check_weight_sparsity(self):
    #     pass

    # TODO: update decay

    def get_mask(self):
        N = self.N 
        M = self.M

        weight = self.weight
        length = weight.numel() #number of papameters
        group = int(length/M)
        
        if self.layout == 'NHWC':
            weight_t = weight.clone().permute(0,2,3,1)
            weight_temp = weight_t.detach().abs().reshape(group, M)
        else:    
            weight_temp = weight.detach().abs().reshape(group, M)

        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)] # indicate which will be zeros (pruned) 
        #
        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0) #.reshape(weight.shape) #

        if self.layout == 'NHWC':
            w_b = w_b.reshape(weight_t.shape)
            w_b = w_b.permute(0,3,1,2)

        else:
            w_b = w_b.reshape(weight.shape)

        return w_b 


    def update_decay(self,updated_decay):
        self.decay = updated_decay
        pass

    
    def smallest_among_survival(self):

        M = self.M
        N = self.N_intermediate #N = self.N
        length = self.weight.numel() #number of papameters
        group = int(length/M)

        weight_temp = self.weight.detach().abs().reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)] # indicate which will be zeros (pruned) 


        #np.where(diff_ >= 0 ,diff_ ,np.inf)
        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0) #.reshape(self.weight.shape) #

        weights_survival = w_b * weight_temp
        #print(weights_survival.dtype)
        smallest_of_survival_ = torch.where(weights_survival > 0.0 ,weights_survival ,torch.tensor(float('inf'),dtype=weights_survival.dtype,device=weights_survival.get_device()))
        smallest_of_survival,inds = torch.min(smallest_of_survival_,dim = 1)
        smallest_of_survival_col = smallest_of_survival.reshape(smallest_of_survival.numel(),1)
        assert smallest_of_survival.numel() == group

        #self.smallest_survival = smallest_of_survival_col

        return smallest_of_survival_col


    def update_learned_sparsity(self):

        M = self.M
        N = self.N_intermediate #N = self.N
        length = self.weight.numel() #number of papameters
        group = int(length/M)

        weight_temp = self.weight.detach().abs().reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)] # indicate which will be zeros (pruned) 

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0) #.reshape(self.weight.shape) #

        weights_pruned = (1-w_b) * weight_temp
        largetest_among_pruned,inds = torch.max(weights_pruned,dim = 1)
        largetest_among_pruned_col = largetest_among_pruned.reshape(largetest_among_pruned.numel(),1)
        assert largetest_among_pruned.numel() == group

        self.smallest_survival = self.smallest_among_survival()
        
        self.learned_threshold = largetest_among_pruned_col

        if N==M:
            self.learned_threshold_m = self.intialize_threshold_with_average_lowest()   #self.smallest_survival
        else:
            self.learned_threshold_m = (self.smallest_survival + self.learned_threshold_m)/2.0


    # TODOs: Initialize the threshold with average of 2/M lowest values to avoid extreme point 
    def intialize_threshold_with_average_lowest(self):
        M = self.M

        #N = int(3*M/4)

        N = int(M/2)
        length = self.weight.numel() #number of papameters
        group = int(length/M)

        weight_temp = self.weight.detach().abs().reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)] # indicate of lowest 2/M 

        w_b = torch.zeros(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=1) #.reshape(self.weight.shape) #

        weights_lower = weight_temp * w_b

        initial_values = torch.sum(weights_lower,dim = 1, keepdim =True)

        initial_values = initial_values/N

        return initial_values

    # #TODO: Think about which is better, smallest survival or largest pruned
    # def check_sparsity_with_learned_threshold(self):
    #     assert self.learned_threshold != None, 'thresh hold has not been updated'

    #     #assert self.smallest_survival != None, 'smallest_survival has not been updated'
    #     # abs : magnitude
    #     M = self.M
    #     length = self.weight.numel()
    #     group = int(length/M)

    #     #TODO check here
    #     # values >= threshold will be true * 1.0 --> 1.0
    #     # m
    #     mask_ = (self.weight.detach().abs().reshape(group, M) > self.learned_threshold) * 1.0 
    #     sparsity = 1.0 - torch.sum(mask_)/mask_.numel() # note sparsity is number of zeros

    #     return sparsity

    
    # def check_sparsity_with_learned_threshold_M(self):
    #     #assert self.learned_threshold != None, 'thresh hold has not been updated'

    #     assert self.learned_threshold_m != None, 'smallest_survival has not been updated'
    #     # abs : magnitude
    #     M = self.M
    #     length = self.weight.numel()
    #     group = int(length/M)

    #     #TODO check here
    #     mask_ = (self.weight.detach().abs().reshape(group, M) >= self.learned_threshold_m) * 1.0
    #     sparsity = 1.0 - torch.sum(mask_)/mask_.numel() # note sparsity is number of zeros

    #     return sparsity


    # def check_sparsity_with_smallest_survival(self):
    #     #assert self.learned_threshold != None, 'thresh hold has not been updated'

    #     assert self.smallest_survival != None, 'smallest_survival has not been updated'
    #     # abs : magnitude
    #     M = self.M
    #     length = self.weight.numel()
    #     group = int(length/M)

    #     #TODO check here
    #     mask_ = (self.weight.detach().abs().reshape(group, M) > self.smallest_survival) * 1.0
    #     sparsity = 1.0 - torch.sum(mask_)/mask_.numel() # note sparsity is number of zeros

    #     return sparsity


    def check_sparsity_each_group(self,prob = 0.75):
        N_inter = self.N_intermediate
        M = self.M
        n = int(np.log2(M)) + 1
        Ns = [2**x for x in range(n)]

        length = self.weight.numel()
        group = int(length/M)
        if self.layout == 'NCHW' or self.k_ == 1: 
            weight_group = (self.weight.detach().abs().reshape(group, M) > self.learned_threshold_m) #* 1.0
            N_every_group = torch.sum(weight_group, 1,keepdim=True)    #X.sum(1, keepdim=True)

        elif self.layout == 'NHWC': 
            Cout = self.weight.size()[0]
            Cin = self.weight.size()[1]
            Kh = self.weight.size()[2]
            Kw = self.weight.size()[3]
            weight_t = self.weight.clone().permute(0,2,3,1)
            weight_group = (weight_t.detach().abs().reshape(group, M) > self.learned_threshold_m) # * 1.0
            N_every_group = torch.sum(weight_group, 1,keepdim=True) # survival N


        survival_elements = torch.sum(N_every_group)
        survival_rate = survival_elements/length

        N_inter_change = False
        if (self.N > 1):
            satisfied_each_group = N_every_group < N_inter
            c_prob = torch.sum(satisfied_each_group)/satisfied_each_group.numel()
            if c_prob >= prob:
                N_inter = N_inter -1
                #self.N_intermediate = N_inter
                N_inter_change = True

        # update self.N_intermediate
        self.N_intermediate = N_inter
        # if N_inter is valide and != current N, then update N
        # return N_inter and True if updated
        if N_inter in Ns and N_inter!=self.N:
            self.N = N_inter

            return N_inter, True, 1.0 - survival_rate, N_inter_change
        # return N_inter and False if does not update
        return N_inter, False , 1.0 - survival_rate, N_inter_change


    def check_num_survival_parameters(self):

        if self.smallest_survival == None: # this means has not initialized, still dense
            return self.dense_parameters

        # abs : magnitude
        M = self.M
        length = self.weight.numel()
        group = int(length/M)

        #TODO check here
        mask_ = (self.weight.detach().abs().reshape(group, M) >= self.smallest_survival) * 1.0


        return torch.sum(mask_).cpu().detach().numpy()

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
        #if (self.check_TC_compatibility() == False or self.M==self.N):

        if self.mix_from_dense == True:
            # use self.learned_threshold_m to test, orginal is self.smallest_survival
            # data layout of Linear layer does not matter so set as NCHW (default)
            return Sparse_find_mix_from_dense.apply(self.weight, self.N_intermediate, self.M, self.decay, self.learned_threshold_m,self.normalized_factor,self.name,self.print_flag,'NCHW',self.apply_penalty)
            #return Sparse_penalty.apply(self.weight, self.N, self.M, self.decay, self.smallest_survival,self.normalized_factor)

        else :  # for Linear(fully-connected) layer, the layout does not matter so use Pytorch default
            #self.spare_weight = self.weight
            #print("dense train")
            return Sparse.apply(self.weight, self.N, self.M, self.decay)# support N=M case
            #return self.weight
    
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




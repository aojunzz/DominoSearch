import torch
import os
import shutil
# import re
import numpy as np
from devkit.sparse_ops import SparseConv,SparseLinear
DEFAULT_ERK_SCALE = 1.0

import matplotlib.pyplot as plt
import matplotlib

from matplotlib import colors
from matplotlib.ticker import PercentFormatter


import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np
# def set_custom_sparsity_map():
#   if FLAGS.first_layer_sparsity > 0.:
#     CUSTOM_SPARSITY_MAP[
#         'resnet_model/initial_conv'] = FLAGS.first_layer_sparsity
#   if FLAGS.last_layer_sparsity > 0.:
#     CUSTOM_SPARSITY_MAP[
#         'resnet_model/final_dense'] = FLAGS.last_layer_sparsity



#

def get_n_zeros(size, sparsity):
  return int(np.floor(sparsity * size))




# {'SparseConv0_3-16': [16, 3, 3, 3], 'SparseConv1_16-16': [16, 16, 3, 3], 'SparseConv2_16-16': [16, 16, 3, 3], 'SparseConv3_16-16': [16, 16, 3, 3], 'SparseConv4_16-16': [16, 16, 3, 3], 'SparseConv5_16-16': [16, 16, 3, 3], 'SparseConv6_16-16': [16, 16, 3, 3], 'SparseConv7_16-32': [32, 16, 3, 3], 'SparseConv8_32-32': [32, 32, 3, 3], 'SparseConv9_16-32': [32, 16, 1, 1], 'SparseConv10_32-32': [32, 32, 3, 3], 'SparseConv11_32-32': [32, 32, 3, 3], 'SparseConv12_32-32': [32, 32, 3, 3], 'SparseConv13_32-32': [32, 32, 3, 3], 'SparseConv14_32-64': [64, 32, 3, 3], 'SparseConv15_64-64': [64, 64, 3, 3], 'SparseConv16_32-64': [64, 32, 1, 1], 'SparseConv17_64-64': [64, 64, 3, 3], 'SparseConv18_64-64': [64, 64, 3, 3], 'SparseConv19_64-64': [64, 64, 3, 3], 'SparseConv20_64-64': [64, 64, 3, 3], 'Linear0_64-100': [100, 64]}
# {'Linear0_64-100': [100, 64]}
# {'SparseConv0_3-16': 0.0, 'SparseConv1_16-16': 0.4602400878442545, 'SparseConv2_16-16': 0.4602400878442545, 'SparseConv3_16-16': 0.4602400878442545, 'SparseConv4_16-16': 0.4602400878442545, 'SparseConv5_16-16': 0.4602400878442545, 'SparseConv6_16-16': 0.4602400878442545, 'SparseConv7_16-32': 0.6164863782051282, 'SparseConv8_32-32': 0.7514263562440646, 'SparseConv9_16-32': 0.0, 'SparseConv10_32-32': 0.7514263562440646, 'SparseConv11_32-32': 0.7514263562440646, 'SparseConv12_32-32': 0.7514263562440646, 'SparseConv13_32-32': 0.7514263562440646, 'SparseConv14_32-64': 0.8188963452635327, 'SparseConv15_64-64': 0.8810397562025166, 'SparseConv16_32-64': 0.0, 'SparseConv17_64-64': 0.8810397562025166, 'SparseConv18_64-64': 0.8810397562025166, 'SparseConv19_64-64': 0.8810397562025166, 'SparseConv20_64-64': 0.8810397562025166, 'Linear0_64-100': 0.0}



# Input
# net:input model
# sparsities : dictionaries for sparse schemes key = layer name, value = sparsity
# func: set the sparse schmes of each layer by applying m.apply_N_M()
def set_sparse_scheme(net,sparse_schemes):
  for m in net.modules():
    if isinstance(m, SparseConv) or isinstance(m, SparseLinear): # we only consider sparse conv at this moment
      layer_name = m.get_name()
      if layer_name in sparse_schemes : # key
        N,M = sparse_schemes[layer_name]
        m.apply_N_M(N,M)
      else:
        print("unsupported layer at this moment ",layer_name)

# Input:
# sparsities : dictionaries for sparse schemes key = layer name, value = sparsity
# Ns: list, choices for sparse pattern
# return sparse_schemes dictionary with key = layer name, value = sparse scheme [N,M]
# M = 8, Ns = [1,2,4]  -- (0,12.5,25,50,100)  sparsity (0.0,0.5,0.75,0.875) -- [8,4,2,1]


def get_overall_sparsity_with_NM_schemes(net,sparse_schemes):
  total_params = 0
  sparse_paras = 0
  for m in net.modules():
    if isinstance(m, SparseConv) or isinstance(m, SparseLinear): # we only consider sparse conv at this moment
      layer_name = m.get_name()
      dense_parameters = m.dense_parameters
      total_params += dense_parameters
      if layer_name in sparse_schemes : # key
        N,M = sparse_schemes[layer_name]
        sparse_para = dense_parameters * (N*1.0 / M) 
        sparse_paras += sparse_para
      else:
        print("unsupported layer at this moment ",layer_name)
  
  return 1.0 - (sparse_paras/total_params)






# TODO, check the overall FLOPs
def compute_overall_flops(net):
  pass








# https://github.com/sksq96/pytorch-summary
def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    #print(result)

    return result,params_info


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            if isinstance(module, SparseConv):
              m_key = module.name  #"%s-%i" % (class_name, module_idx + 1)
            else:
              m_key = "%s-%i" % (class_name, module_idx + 1)

            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    for mod in model.modules():
        if isinstance(mod, SparseConv): 
          layer_name = mod.name
          mod.input_shape = summary[layer_name]["input_shape"]
          mod.output_shape = summary[layer_name]["output_shape"]

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>25} {:>15}".format(
        "Layer (type)","Input Shape", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>25} {:>15}".format(
            layer,
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)





def conv_flops_counter_hook(conv_module, input, output):
  if isinstance(conv_module, SparseConv): 
    # Can have multiple inputs, getting the first one
    #input = input[0]

    batch_size = 1 #input.shape[0]
    output_dims = list(output.shape[2:]) # H,W

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    return overall_flops
    #conv_module.__flops__ += int(overall_flops)
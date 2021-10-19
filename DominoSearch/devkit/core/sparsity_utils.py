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



# ERK calculation are taken (with some modifications) from https://github.com/google-research/rigl


def mean_var_group(mod,M):
    weight = mod.weight.clone().detach().abs()
    layout = mod.layout
    length = mod.weight.numel()
    group = int(length/M)
    if layout == 'NHWC':
      weight_t = weight.clone().permute(0,2,3,1)
      weight_group = weight_t.detach().abs().reshape(group, M)
    else:
      weight_group = weight.detach().abs().reshape(group, M)

    weight_group = weight_group.cpu().numpy()
    mean_group = np.mean(weight_group, axis=1)
    
    variance_ = np.var(mean_group)
    std_ = np.std(mean_group) 

    return weight_group,mean_group,variance_,std_
      

def get_n_zeros(size, sparsity):
  return int(np.floor(sparsity * size))

# def mask_extract_name_fn(mask_name):
#   return re.findall('(.+)/mask:0', mask_name)[0]

# erk_dict with original sparsity target, net, current sparsity  
def update_erk_sparsity(erk_dict,net):
  
  new_erk_dict = {}
  for m in net.modules():
    if isinstance(m, SparseConv) or isinstance(m, SparseLinear): # we only consider sparse
      layer_name = m.get_name()
      old_sparsity = erk_dict[layer_name]
      real_sparsity = 1.0 - (m.N*1.0 / m.M)
      new_sparsity = old_sparsity-real_sparsity
      new_erk_dict[layer_name] = new_sparsity

  return new_erk_dict



def normalize_erk_sparsity(sparsity_erk_dict):
  layer_name_list = []
  sparsity_list = []
  for layer_name, sparsity in sparsity_erk_dict.items():
    sparsity_list.append(sparsity)
    layer_name_list.append(layer_name)

  normalized_factor = np.asarray(sparsity_list)
  if np.amax(sparsity_list) < 0.01:
    normalized_factor = sparsity_list * 0.0
  elif np.isnan(np.sum(normalized_factor)):
    return None
  else:
    normalized_factor = sparsity_list/np.amax(np.abs(sparsity_list))

  normalized_factor = normalized_factor.tolist()
  zip_iterator = zip(layer_name_list, normalized_factor)
  decay_normalized = dict(zip_iterator)

  return decay_normalized



def get_sparsities_erdos_renyi_NM(all_layers, # layer dicts, k = name, v = module
                               default_sparsity,
                               custom_sparsity_map, # k = layer names, v = sparsity  
                               dense_layers, # desen layer names
                               include_kernel,
                               current_schemes,
                               #extract_name_fn=mask_extract_name_fn,
                               erk_power_scale=DEFAULT_ERK_SCALE):

  is_eps_valid = False

  while not is_eps_valid:


    divisor = 0
    rhs = 0
    raw_probabilities = {}
    for layer_name,layer_shape in all_layers.items():
      N,M = current_schemes[layer_name]
      #var_name = extract_name_fn(mask.name)
      #shape_list = mask.shape.as_list()
      # cin
      layer_shape[1] = layer_shape[1]*N/M
      n_param = np.prod(layer_shape)
      n_zeros = get_n_zeros(n_param, default_sparsity) # calculate the number of zeros 
      if layer_name in dense_layers:
        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
        rhs -= n_zeros
      elif layer_name in custom_sparsity_map:
        # We ignore custom_sparsities in erdos-renyi calculations.
        pass
      else:
        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
        # equation above.
        n_ones = n_param - n_zeros
        rhs += n_ones
        
        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
        if include_kernel:
          raw_probabilities[layer_name] = (np.sum(layer_shape) /
                                          (np.prod(layer_shape)))**erk_power_scale
        else:
          
          raw_probabilities[layer_name] = (n_in + n_out) / ((n_in * n_out))
        # Note that raw_probabilities[mask] * n_param gives the individual
        # elements of the divisor.
        divisor += raw_probabilities[layer_name] * n_param
    # By multipliying individual probabilites with epsilon, we should get the
    # number of parameters per layer correctly.
    eps = rhs / divisor
    # If eps * raw_probabilities[mask.name] > 1. We set the sparsities of that
    # mask to 0., so they become part of dense_layers sets.
    max_prob = np.max(list(raw_probabilities.values()))
    max_prob_one = max_prob * eps
    if max_prob_one > 1:
      is_eps_valid = False
      for mask_name, mask_raw_prob in raw_probabilities.items():
        if mask_raw_prob == max_prob:
          #var_name = extract_name_fn(mask_name)
        #   tf.logging.info('Sparsity of var: %s had to be set to 0.', var_name)
          dense_layers.append(mask_name)
    else:
      is_eps_valid = True

  sparsities = {}
  # With the valid epsilon, we can set sparsities of the remaning layers.
  for layer_name,layer_shape in all_layers.items():
    #var_name = extract_name_fn(mask.name)
    #shape_list = mask.shape.as_list() # ?  [Cout,C,Kh,Kw]
    #print(layer_shape)
    N,M = current_schemes[layer_name]
    layer_shape[1] = layer_shape[1]*N/M
    n_param = np.prod(layer_shape) # Return the product of array elements over a given axis
    if layer_name in custom_sparsity_map:
      sparsities[layer_name] = custom_sparsity_map[var_name]
    #   tf.logging.info('layer: %s has custom sparsity: %f', var_name,
    #                   sparsities[mask.name])
    elif layer_name in dense_layers:
      sparsities[layer_name] = 0.
    else:
      probability_one = eps * raw_probabilities[layer_name]
      sparsities[layer_name] = 1. - probability_one
    # tf.logging.info('layer: %s, shape: %s, sparsity: %f', var_name, mask.shape,
    #                 sparsities[mask.name])
  return sparsities


def get_sparsities_erdos_renyi(all_layers, # layer dicts, k = name, v = module
                               default_sparsity,
                               custom_sparsity_map, # k = layer names, v = sparsity  
                               dense_layers, # desen layer names
                               include_kernel,
                               
                               #extract_name_fn=mask_extract_name_fn,
                               erk_power_scale=DEFAULT_ERK_SCALE):
  """Given the method, returns the sparsity of individual layers as a dict.

  It ensures that the non-custom layers have a total parameter count as the one
  with uniform sparsities. In other words for the layers which are not in the
  custom_sparsity_map the following equation should be satisfied.

  # eps * (p_1 * N_1 + p_2 * N_2) = (1 - default_sparsity) * (N_1 + N_2)
  Args:
    all_masks: list, of all mask Variables.
    default_sparsity: float, between 0 and 1.
    custom_sparsity_map: dict, <str, float> key/value pairs where the mask
      correspond whose name is '{key}/mask:0' is set to the corresponding
        sparsity value.
    include_kernel: bool, if True kernel dimension are included in the scaling.
    extract_name_fn: function, extracts the variable name.
    erk_power_scale: float, if given used to take power of the ratio. Use
      scale<1 to make the erdos_renyi softer.

  Returns:
    sparsities, dict of where keys() are equal to all_masks and individiual
      masks are mapped to the their sparsities.
  """
  # We have to enforce custom sparsities and then find the correct scaling
  # factor.

  is_eps_valid = False
  # # The following loop will terminate worst case when all masks are in the
  # custom_sparsity_map. This should probably never happen though, since once
  # we have a single variable or more with the same constant, we have a valid
  # epsilon. Note that for each iteration we add at least one variable to the
  # custom_sparsity_map and therefore this while loop should terminate.
  #dense_layers = set()
  while not is_eps_valid:
    # We will start with all layers and try to find right epsilon. However if
    # any probablity exceeds 1, we will make that layer dense and repeat the
    # process (finding epsilon) with the non-dense layers.
    # We want the total number of connections to be the same. Let say we have
    # for layers with N_1, ..., N_4 parameters each. Let say after some
    # iterations probability of some dense layers (3, 4) exceeded 1 and
    # therefore we added them to the dense_layers set. Those layers will not
    # scale with erdos_renyi, however we need to count them so that target
    # paratemeter count is achieved. See below.
    # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
    #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
    # eps * (p_1 * N_1 + p_2 * N_2) =
    #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
    # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

    divisor = 0
    rhs = 0
    raw_probabilities = {}
    for layer_name,layer_shape in all_layers.items():
      #var_name = extract_name_fn(mask.name)
      #shape_list = mask.shape.as_list()
      n_param = np.prod(layer_shape)
      n_zeros = get_n_zeros(n_param, default_sparsity) # calculate the number of zeros 
      if layer_name in dense_layers:
        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
        rhs -= n_zeros
      elif layer_name in custom_sparsity_map:
        # We ignore custom_sparsities in erdos-renyi calculations.
        pass
      else:
        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
        # equation above.
        n_ones = n_param - n_zeros
        rhs += n_ones
        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
        if include_kernel:
          raw_probabilities[layer_name] = (np.sum(layer_shape) /
                                          np.prod(layer_shape))**erk_power_scale
        else:
          n_out,n_in = layer_shape[0], layer_shape[1]
          raw_probabilities[layer_name] = (n_in + n_out) / (n_in * n_out)
        # Note that raw_probabilities[mask] * n_param gives the individual
        # elements of the divisor.
        divisor += raw_probabilities[layer_name] * n_param
    # By multipliying individual probabilites with epsilon, we should get the
    # number of parameters per layer correctly.
    eps = rhs / divisor
    # If eps * raw_probabilities[mask.name] > 1. We set the sparsities of that
    # mask to 0., so they become part of dense_layers sets.
    max_prob = np.max(list(raw_probabilities.values()))
    max_prob_one = max_prob * eps
    if max_prob_one > 1:
      is_eps_valid = False
      for mask_name, mask_raw_prob in raw_probabilities.items():
        if mask_raw_prob == max_prob:
          #var_name = extract_name_fn(mask_name)
        #   tf.logging.info('Sparsity of var: %s had to be set to 0.', var_name)
          dense_layers.append(mask_name)
    else:
      is_eps_valid = True

  sparsities = {}
  # With the valid epsilon, we can set sparsities of the remaning layers.
  for layer_name,layer_shape in all_layers.items():
    #var_name = extract_name_fn(mask.name)
    #shape_list = mask.shape.as_list() # ?  [Cout,C,Kh,Kw]
    #print(layer_shape)
    n_param = np.prod(layer_shape) # Return the product of array elements over a given axis
    if layer_name in custom_sparsity_map:
      sparsities[layer_name] = custom_sparsity_map[var_name]
    #   tf.logging.info('layer: %s has custom sparsity: %f', var_name,
    #                   sparsities[mask.name])
    elif layer_name in dense_layers:
      sparsities[layer_name] = 0.
    else:
      probability_one = eps * raw_probabilities[layer_name]
      sparsities[layer_name] = 1. - probability_one
    # tf.logging.info('layer: %s, shape: %s, sparsity: %f', var_name, mask.shape,
    #                 sparsities[mask.name])
  return sparsities




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

            # if isinstance(module, SparseConv):
            #   mod.input_shape = summary[m_key]["input_shape"]
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size
            # if isinstance(module, SparseConv):
            #   mod.output_shape = summary[m_key]["output_shape"]
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
    output_dims = output[2] #list(output.shape[2:]) # H,W

    kernel_dims = conv_module.k_ #list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = kernel_dims*kernel_dims * in_channels * filters_per_channel
        

    active_elements_count = batch_size * output_dims*output_dims

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    return overall_flops
    #conv_module.__flops__ += int(overall_flops)



  



def get_layer_wise_dense_flops_params(net):
  dict_flops = {}
  dict_params = {}
  for mod in net.modules(): 
    if isinstance(mod, SparseConv) or isinstance(mod, SparseLinear):
      layer_name = mod.name
      dense_flops = mod.flops
      dense_params = mod.dense_parameters

      dict_flops[layer_name] = dense_flops
      dict_params[layer_name] = dense_params
  
  return dict_flops,dict_params


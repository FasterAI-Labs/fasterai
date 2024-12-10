# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/misc/06a_misc.bn_folding.ipynb.

# %% auto 0
__all__ = ['BN_Folder']

# %% ../../nbs/misc/06a_misc.bn_folding.ipynb 3
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# %% ../../nbs/misc/06a_misc.bn_folding.ipynb 12
class BN_Folder():
    def __init__(self):
        super().__init__()
        
    def fold(self, model):

        new_model = copy.deepcopy(model)

        module_names = list(new_model._modules)

        for k, name in enumerate(module_names):

            if len(list(new_model._modules[name]._modules)) > 0:
                new_model._modules[name] = self.fold(new_model._modules[name])

            else:
                if isinstance(new_model._modules[name], nn.BatchNorm2d):
                    if isinstance(new_model._modules[module_names[k-1]], nn.Conv2d):

                        # Folded BN
                        folded_conv = self._fold_conv_bn_eval(new_model._modules[module_names[k-1]], new_model._modules[name])

                        # Replace old weight values
                        #new_model._modules.pop(name) # Remove the BN layer
                        new_model._modules[module_names[k]] = nn.Identity()
                        new_model._modules[module_names[k-1]] = folded_conv # Replace the Convolutional Layer by the folded version

        return new_model


    def _bn_folding(self, conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
        if conv_b is None:
            conv_b = bn_rm.new_zeros(bn_rm.shape)
        bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

        w_fold = conv_w * (bn_w * bn_var_rsqrt).view(-1, 1, 1, 1)
        b_fold = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

        return torch.nn.Parameter(w_fold), torch.nn.Parameter(b_fold)


    def _fold_conv_bn_eval(self, conv, bn):
        assert(not (conv.training or bn.training)), "Fusion only for eval!"
        fused_conv = copy.deepcopy(conv)

        fused_conv.weight, fused_conv.bias = self._bn_folding(fused_conv.weight, fused_conv.bias,
                                 bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

        return fused_conv

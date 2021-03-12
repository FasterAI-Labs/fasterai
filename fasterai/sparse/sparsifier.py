# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01a_sparsifier.ipynb (unless otherwise specified).

__all__ = ['Sparsifier']

# Cell
from fastai.vision.all import *

import torch
import torch.nn as nn

from ..criteria import *

# Cell
class Sparsifier():
    '''
    Make a neural network sparse using the `prune` method
    '''
    def __init__(self, model, granularity, method, criteria):
        store_attr()
        self._save_weights() # Save the original weights

    def prune(self, sparsity):
        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.Conv2d):
                weight = self.criteria(m, self.granularity)
                mask = self._compute_mask(self.model, weight, sparsity)
                m.register_buffer("_mask", mask) # Put the mask into a buffer
                self._apply(m)

    def _apply(self, module):
        '''
        Apply the mask and freeze the gradient so the corresponding weights are not updated anymore
        '''
        mask = getattr(module, "_mask")
        module.weight.data.mul_(mask)

        if self.granularity == 'filter': # If we remove complete filters, we want to remove the bias as well
            if module.bias is not None:
                module.bias.data.mul_(mask.squeeze())
                if module.bias.grad is not None: # In case some layers are freezed
                    module.bias.grad.mul_(mask.squeeze())

    def mask_grad(self):
        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.Conv2d):
                mask = getattr(m, "_mask")
                if m.weight.grad is not None: # In case some layers are freezed
                    m.weight.grad.mul_(mask)

                if self.granularity == 'filter': # If we remove complete filters, we want to remove the bias as well
                        if m.bias.grad is not None: # In case some layers are freezed
                            m.bias.grad.mul_(mask.squeeze())


    def _reset_weights(self):
        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.Linear):
                init_weights = getattr(m, "_init_weights")
                m.weight.data = init_weights.clone()
            if isinstance(m, nn.Conv2d):
                init_weights = getattr(m, "_init_weights")
                m.weight.data = init_weights.clone()
                self._apply(m) # Reset the weights and apply the current mask

    def _save_weights(self):
        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.register_buffer("_init_weights", m.weight.clone())

    def _clean_buffers(self):
        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.Conv2d):
                del m._buffers["_mask"]

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                del m._buffers["_init_weights"]


    def _compute_mask(self, model, weight, sparsity):
        '''
        Compute the binary masks
        '''
        if self.method == 'global':
            global_weight = torch.cat([self.criteria(m, self.granularity).view(-1) for m in model.modules() if isinstance(m, nn.Conv2d)])
            threshold = torch.quantile(global_weight, sparsity/100) # Compute the threshold globally

        elif self.method == 'local':
            threshold = torch.quantile(weight.view(-1), sparsity/100) # Compute the threshold locally

        else: raise NameError('Invalid Method')

        # Make sure we don't remove every weight of a given layer
        if threshold > weight.max(): threshold = weight.max()

        mask = weight.ge(threshold).to(dtype=weight.dtype)

        return mask
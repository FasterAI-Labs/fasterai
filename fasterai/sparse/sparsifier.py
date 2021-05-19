# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01a_sparsifier.ipynb (unless otherwise specified).

__all__ = ['Sparsifier']

# Cell
import torch
import torch.nn as nn
from fastcore.basics import store_attr
from .criteria import *

# Cell
class Sparsifier():

    def __init__(self, model, granularity, method, criteria):
        store_attr()
        self._save_weights() # Save the original weights

    def prune_layer(self, module, sparsity):
        weight = self.criteria(module, self.granularity)
        mask = self._compute_mask(self.model, weight, sparsity)
        module.register_buffer("_mask", mask) # Put the mask into a buffer
        self._apply(module)

    def prune_model(self, sparsity):
        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.Conv2d):
                self.prune_layer(m, sparsity)

    def _apply(self, module):
        mask = getattr(module, "_mask")
        module.weight.data.mul_(mask)

        if self.granularity == 'filter': # If we remove complete filters, we want to remove the bias as well
            if module.bias is not None:
                module.bias.data.mul_(mask.squeeze())

    def _mask_grad(self):
        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.Conv2d) and hasattr(m, '_mask'):
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
            if isinstance(m, nn.Conv2d) and hasattr(m, '_mask'):
                del m._buffers["_mask"]

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                del m._buffers["_init_weights"]


    def _compute_mask(self, model, weight, sparsity):
        if self.method == 'global':
            global_weight = torch.cat([self.criteria(m, self.granularity).view(-1) for m in model.modules() if isinstance(m, nn.Conv2d)])
            threshold = torch.quantile(global_weight, sparsity/100) # Compute the threshold globally

        elif self.method == 'local':
            threshold = torch.quantile(weight.view(-1), sparsity/100) # Compute the threshold locally

        else: raise NameError('Invalid Method')

        if threshold > weight.max(): threshold = weight.max() # Make sure we don't remove every weight of a given layer

        mask = weight.ge(threshold).to(dtype=weight.dtype)

        return mask
# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/core/granularity.ipynb.

# %% auto 0
__all__ = ['Granularities']

# %% ../../nbs/core/granularity.ipynb 2
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastcore.basics import *
from fastcore.imports import *

# %% ../../nbs/core/granularity.ipynb 5
class Granularities:
    _granularities_Conv2d = {'weight':0, 'shared_weight':1, 'channel':2, 'column':3, 'row':4, 'kernel':(3,4), 'filter':(2,3,4), 'shared_channel':(1,2), 'shared_column': (1,3), 'shared_row': (1,4), 'vertical_slice': (2,3), 'horizontal_slice': (2,4), 'shared_vertical_slice': (1,2,3), 'shared_horizontal_slice': (1,2,4), 'shared_kernel': (1,3,4), 'layer':(1,2,3,4)}
    _granularities_ConvT2d = {'weight':0, 'shared_weight':2, 'channel':1, 'column':3, 'row':4, 'kernel':(3,4), 'filter':(1,3,4), 'shared_channel':(1,2), 'shared_column': (2,3), 'shared_row': (2,4), 'vertical_slice': (1,3), 'horizontal_slice': (1,4), 'shared_vertical_slice': (1,2,3), 'shared_horizontal_slice': (1,2,4), 'shared_kernel': (2,3,4), 'layer':(1,2,3,4)}
    _granularities_Linear = {'weight':0, 'column':1, 'row':2, 'layer':(1,2)}
    _granularities = {
        torch.nn.Conv2d: _granularities_Conv2d,
        torch.nn.ConvTranspose2d: _granularities_ConvT2d,
        torch.nn.Conv1d: _granularities_Linear,
        torch.nn.Linear: _granularities_Linear
    }
    
    @classmethod
    def get_dim(cls, m, g):
        for k in cls._granularities:
            if isinstance(m, k):
                return listify(cls._granularities[k][g])
        raise NotImplementedError("Unsupported module type")
    
    @classmethod
    def add_granularity(cls, name, g):
        cls._granularities[name] = g
        
    @classmethod
    def allowed_granularities(cls, m):
        for k in cls._granularities:
            if isinstance(m, k):
                print(cls._granularities[k])
                return
        raise NotImplementedError("Unsupported module type")
    
    @classmethod
    def available_granularities(cls):
        print(cls._granularities.keys())

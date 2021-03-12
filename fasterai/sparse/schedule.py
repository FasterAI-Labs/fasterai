# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/0c_schedules.ipynb (unless otherwise specified).

__all__ = ['iterative', 'one_shot', 'sched_agp', 'dsd']

# Cell
import numpy as np

# Cell

def iterative(start, end, pos, n_steps=3):
    "Perform iterative pruning, and pruning in `n_steps` steps"
    return start + ((end-start)/n_steps)*(np.ceil((pos)*n_steps))

def one_shot(start, end, pos): return end

def sched_agp(start, end, pos): return end + start - end * (1 - pos)**3

def dsd(start, end, pos):
    if pos<0.5:
        return start + (1 + math.cos(math.pi*(1-pos*2))) * (end-start) / 2
    else:
        return end + (1 - math.cos(math.pi*(1-pos*2))) * (start-end) / 2
# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/0c_schedules.ipynb (unless otherwise specified).

__all__ = ['Schedule', 'sched_oneshot', 'one_shot', 'sched_iterative', 'iterative', 'sched_agp', 'agp',
           'sched_onecycle', 'one_cycle', 'cos', 'lin', 'sched_dsd', 'dsd']

# Cell
import numpy as np
import matplotlib.pyplot as plt
from fastcore.basics import *
from fastai.callback.schedule import *

# Cell
class Schedule():
    def __init__(self, sched_func, start_pct=0., end_pct=None, start_sparsity=0.):
        store_attr()
        self.current_sparsity, self.previous_sparsity = map(listify, [start_sparsity, start_sparsity])
        if self.end_pct is None: self.end_pct=1.

    def __call__(self, end_sparsity, pct_train):
        if type(end_sparsity)!= 'list': end_sparsity = listify(end_sparsity)
        if pct_train>=self.start_pct and pct_train <= self.end_pct:
            self.current_sparsity = [self.sched_func(self.start_sparsity, sp, (pct_train-self.start_pct)/(self.end_pct-self.start_pct)) for sp in end_sparsity]
        return self.current_sparsity

    @property
    def pruned(self):
        return self.previous_sparsity!=self.current_sparsity

    def after_pruned(self):
        self.previous_sparsity=self.current_sparsity

    def plot(self, end_sparsity):
        prune = np.linspace(0, 1, 1000)
        sps = [self([end_sparsity], p) for p in prune]
        fig, ax = plt.subplots(1, 1, figsize=(8,6), dpi=100)
        plt.plot(prune, sps, c='teal', linewidth=2)
        plt.xlabel('training iterations (Normalized)')
        plt.ylabel('sparsity')
        self.current_sparsity = self.previous_sparsity

    def reset(self):
        self.current_sparsity, self.previous_sparsity = map(listify, [self.start_sparsity, self.start_sparsity])

# Cell
def sched_oneshot(start, end, pos): return end

one_shot = Schedule(sched_oneshot, start_pct=0.5)

# Cell
def sched_iterative(start, end, pos, n_steps=3):
    "Perform iterative pruning, and pruning in `n_steps` steps"
    return start + ((end-start)/n_steps)*(np.ceil((pos)*n_steps))

iterative = Schedule(sched_iterative, start_pct=0.2)

# Cell
def sched_agp(start, end, pos): return end + (start - end) * (1 - pos)**3

agp = Schedule(sched_agp, start_pct=0.2)

# Cell
def sched_onecycle(start, end, pos, α=14, β=6):
    out = (1+np.exp(-α+β)) / (1 + (np.exp((-α*pos)+β)))
    return start + (end-start)*out

one_cycle = Schedule(sched_onecycle)

# Cell
cos = Schedule(sched_cos)
lin = Schedule(sched_lin)

# Cell
def sched_dsd(start, end, pos):
    if pos<0.5:
        return start + (1 + math.cos(math.pi*(1-pos*2))) * (end-start) / 2
    else:
        return end + (1 - math.cos(math.pi*(1-pos*2))) * (start-end) / 2

dsd = Schedule(sched_dsd)
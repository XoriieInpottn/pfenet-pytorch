#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-07-19
"""
import math

import torch
from torch import nn
from torch.nn import init
from torch.optim.lr_scheduler import LambdaLR


class LayerNorm(nn.Module):

    def __init__(self, shape, eps=1e-7):
        """Layer normalization

        :param shape: Feature shape.
        :param eps: Used to prevent divide zero.

        References:
            Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton, Layer Normalization,
            https://arxiv.org/pdf/1607.06450.pdf
        """
        super(LayerNorm, self).__init__()
        self._eps = eps

        self.weight = nn.Parameter(torch.empty(shape))
        self.bias = nn.Parameter(torch.empty(shape))

        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        axes = tuple(range(1, len(x.shape)))
        mean = x.mean(axes, keepdim=True)
        var = (x ** 2).mean(axes, keepdim=True) - mean ** 2
        h = (x - mean) / (torch.sqrt(var) + self._eps)
        h = h * self.weight + self.bias
        return h


class CosineWarmUpAnnealingLR(LambdaLR):

    def __init__(self,
                 optimizer,
                 num_loops,
                 warm_up_proportion=0.01,
                 max_factor=1.0,
                 min_factor=0.0,
                 pow_warm_up=None,
                 pow_annealing=2.0,
                 last_epoch=-1):
        self._num_loops = num_loops
        self._warm_up_proportion = warm_up_proportion
        self._max_factor = max_factor
        self._min_factor = min_factor
        self._pow_warm_up = pow_warm_up
        self._pow_annealing = pow_annealing
        self._warm_up_loops = int(self._warm_up_proportion * self._num_loops)
        super(CosineWarmUpAnnealingLR, self).__init__(
            optimizer=optimizer,
            lr_lambda=self._lr_lambda,
            last_epoch=last_epoch
        )

    def _lr_lambda(self, i: int) -> float:
        if self._warm_up_loops == 0:
            return self._max_factor
        if i <= self._warm_up_loops:
            i = i - self._warm_up_loops + 1
            value = (math.cos(i * math.pi / self._warm_up_loops) + 1.0) * 0.5
            if self._pow_warm_up is not None and self._pow_warm_up != 1.0:
                value = math.pow(value, self._pow_warm_up)
            value = value * (self._max_factor - self._min_factor) + self._min_factor
        else:
            if i >= self._num_loops:
                i = self._num_loops - 1
            i = i - self._warm_up_loops
            value = (math.cos(i * math.pi / (self._num_loops - self._warm_up_loops)) + 1.0) * 0.5
            if self._pow_annealing is not None and self._pow_annealing != 1.0:
                value = math.pow(value, self._pow_annealing)
            value = value * (self._max_factor - self._min_factor) + self._min_factor
        return value

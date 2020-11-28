import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
import torch as torch

class StaticLayer(nn.Module):
    def __init__(self,in_channels, out_channels, dropout, Trnn):
        self.Trnn = Trnn
        self.dropout = dropout
        self.static = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.static(x)
        return x.unsqueeze(1).repeat(1, self.Trnn, 1)


class ConvLayer(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size = 2):
    
        c1 = nn.Conv1d(30, 30, 2, dilation = 1)
        c2 = nn.Conv1d(30, 30, 2, dilation = 2)
        c3 = nn.Conv1d(30, 30, 2,  dilation = 4)
        c4 = nn.Conv1d(30, 30, 2, dilation = 8)
        c5 = nn.Conv1d(30, 30, 2, dilation = 16)
        c6 = nn.Conv1d(30, 30, 2, dilation = 32)

    def forward(self, x_t):
        x_t = x_t.permute(0, 2, 1))
        x_t = F.pad(x_t, (0,0), "constant", 0)
        x_t = c1(x_t)
        x_t = F.pad(x_t, (2,0), "constant", 0)
        x_t = c2(x_t)
        x_t = F.pad(x_t, (4,0), "constant", 0)
        x_t = c3(x_t)
        x_t = F.pad(x_t, (8,0), "constant", 0)
        x_t = c4(x_t)
        x_t = F.pad(x_t, (16,0), "constant", 0)
        x_t = c5(x_t)
        x_t = F.pad(x_t, (32,0), "constant", 0)
        x_t = c6(x_t)
        
        return x_t.permute(0, 2, 1)

class ExpandLayer(nn.Module):
    """Expands the dimension referred to as `expand_axis` into two
    dimensions by applying a sliding window. For example, a tensor of
    shape (1, 4, 2) as follows:

    [[[0. 1.]
      [2. 3.]
      [4. 5.]
      [6. 7.]]]

    where `expand_axis` = 1 and `Trnn` = 3 (number of windows) and
    `lead_future` = 2 (window length) will become:

    [[[[0. 1.]
       [2. 3.]]

      [[2. 3.]
       [4. 5.]]

      [[4. 5.]
       [6. 7.]]]]

    Used for expanding future information tensors

    Parameters
    ----------
    Trnn : int
        Length of the time sequence (number of windows)
    lead_future : int
        Number of future time points (window length)
    expand_axis : int
        Axis to expand"""

    def __init__(self, Trnn, lead_future, **kwargs):
        super(ExpandLayer, self).__init__(**kwargs)
    
        self.Trnn = Trnn
        self.lead_future = lead_future

    def forward(self, x):

        # First create a matrix of indices, which we will use to slice
        # `input` along `expand_axis`. For example, for Trnn=3 and
        # lead_future=2,
        # idx = [[0. 1.]
        #        [1. 2.]
        #        [2. 3.]]
        # We achieve this by doing a broadcast add of
        # [[0.] [1.] [2.]] and [[0. 1.]]
        idx = torch.add(torch.arange(self.Trnn).unsqueeze(axis = 1), 
                        torch.arange(self.lead_future).unsqueeze(axis = 0))
        # Now we slice `input`, taking elements from `input` that correspond to
        # the indices in `idx` along the `expand_axis` dimension
        return x[:, idx, :]


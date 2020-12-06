import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
import torch as torch

class StaticLayer(nn.Module):
    def __init__(self, in_channels, time_step, static_features, out_channels = 30, dropout = 0.4):
        super().__init__()
        self.time_step = time_step
        #self.static_features = static_features
        self.dropout = nn.Dropout(dropout)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.static = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x):
        x = x['static_features'][:,:1,:]
        x = self.dropout(x)
        x = self.static(x)
        return x.repeat(1, self.time_step, 1)

class ConvLayer(nn.Module):
    def __init__(self, time_step, timevarying_features, in_channels, out_channels = 30, kernel_size = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.timevarying_features = timevarying_features
        self.time_step = time_step

        self.c1 = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, dilation = 1)
        self.c2 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, dilation = 2)
        self.c3 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size,  dilation = 4)
        self.c4 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, dilation = 8)
        self.c5 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, dilation = 16)
        self.c6 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, dilation = 32)

    def forward(self, x):
        x_t = x['timevarying_features'][:, :self.time_step, :]
        x_t = x_t.permute(0, 2, 1)
        x_t = F.pad(x_t, (1,0), "constant", 0)
        x_t = self.c1(x_t)
        x_t = F.pad(x_t, (2,0), "constant", 0)
        x_t = self.c2(x_t)
        x_t = F.pad(x_t, (4,0), "constant", 0)
        x_t = self.c3(x_t)
        x_t = F.pad(x_t, (8,0), "constant", 0)
        x_t = self.c4(x_t)
        x_t = F.pad(x_t, (16,0), "constant", 0)
        x_t = self.c5(x_t)
        
        return x_t.permute(0, 2, 1)

class ExpandLayer(nn.Module):
    """Expands the dimension referred to as `expand_axis` into two
    dimensions by applying a sliding window. For example, a tensor of
    shape (1, 4, 2) as follows:

    [[[0. 1.]
      [2. 3.]
      [4. 5.]
      [6. 7.]]]

    where `expand_axis` = 1 and `time_step` = 3 (number of windows) and
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
    time_step : int
        Length of the time sequence (number of windows)
    lead_future : int
        Number of future time points (window length)
    expand_axis : int
        Axis to expand"""

    def __init__(self, time_step, lead_future, future_information, **kwargs):
        super(ExpandLayer, self).__init__(**kwargs)
    
        self.time_step = time_step
        self.future_information = future_information
        self.lead_future = lead_future

    def forward(self, x):

        # First create a matrix of indices, which we will use to slice
        # `input` along `expand_axis`. For example, for time_step=3 and
        # lead_future=2,
        # idx = [[0. 1.]
        #        [1. 2.]
        #        [2. 3.]]
        # We achieve this by doing a broadcast add of
        # [[0.] [1.] [2.]] and [[0. 1.]]
        idx = torch.add(torch.arange(self.time_step).unsqueeze(axis = 1),
                        torch.arange(self.lead_future).unsqueeze(axis = 0))
        # Now we slice `input`, taking elements from `input` that correspond to
        # the indices in `idx` along the `expand_axis` dimension
        return x[:, idx, :]

        
class GlobalFutureLayer(nn.Module):
    def __init__(self, time_step, lead_future, future_features_count, out_channels = 30):
        super().__init__()
        self.time_step = time_step
        self.lead_future = lead_future
        self.future_features_count = future_features_count
        self.out_channels = out_channels

        self.l1 = nn.Linear(self.lead_future * self.future_features_count, out_channels)
        
    def forward(self, x):
        x = x.contiguous().view(-1, self.time_step, self.lead_future * self.future_features_count)
        
        return self.l1(x)
    
class HorizonSpecific(nn.Module):
    def __init__(self, Tpred, time_step, num = 20):
        super().__init__()
        self.Tpred = Tpred
        self.time_step = time_step
        self.num = num
        
    def forward(self, x):
        x = nn.Linear(x.size(-1), self.Tpred * self.num)(x)
        x = F.relu(x)

        return x.view(-1, self.time_step, self.Tpred, 20)

class HorizonAgnostic(nn.Module):
    def __init__(self, out_channels, lead_future):
        super().__init__()
        self.out_channels = out_channels
        self.lead_future = lead_future
        
    def forward(self, x):
        x = nn.Linear(x.size(-1), self.out_channels)(x)
        x = F.relu(x)
        x = x.unsqueeze(axis = 2)
        x = x.repeat(1,1, self.lead_future, 1)

        return x
    
class LocalMlp(nn.Module):
    def __init__(self, hidden, output):
        super().__init__()
        self.hidden = hidden
        self.output = output
        
    def forward(self, x):
        x = nn.Linear(x.size(-1), self.hidden)(x)
        x = F.relu(x)
        x = nn.Linear(self.hidden, self.output)(x)
        x = F.relu(x)

        return x


class Span1(nn.Module):
    def __init__(self, time_step, lead_future, num_quantiles):
        super().__init__()
        self.time_step = time_step
        self.lead_future = lead_future
        self.num_quantiles = num_quantiles
        
    def forward(self, x):
        x = nn.Linear(x.size(-1), self.num_quantiles)(x)
        x = F.relu(x.contiguous().view(-1, x.size(-2), x.size(-1)))
        x = x.view(-1, self.time_step, self.lead_future, self.num_quantiles)
        x = x.view(-1, self.time_step, self.lead_future*self.num_quantiles)

        return x


class SpanN(nn.Module):
    def __init__(self, time_step, lead_future, num_quantiles, spanN_count):
        super().__init__()
        self.time_step = time_step
        self.lead_future = lead_future
        self.num_quantiles = num_quantiles
        self.spanN_count = spanN_count
        
    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous().view(-1, self.time_step, x.size(-2) * x.size(-1))

        x = nn.Linear(x.size(-1), self.spanN_count * self.num_quantiles)(x)

        return x
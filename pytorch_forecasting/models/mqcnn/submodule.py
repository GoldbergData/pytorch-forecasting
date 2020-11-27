import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
import torch as torch

class StaticLayer(nn.Module):
    pass

class ConvLayer(nn.Module):
    pass

class ExpandLayer(nn.Module):
    pass


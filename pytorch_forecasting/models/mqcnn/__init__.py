import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
import torch as torch
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, QuantileLoss
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.mqcnn.sub_modules import (
    StaticLayer, ConvLayer, ExpandLayer
)


class MQCNN(BaseModelWithCovariates):
    pass

class MQCNNEncoder(nn.Module):
    def __init__(self, ):
        self.static = StaticLayer()
        self.conv = ConvLayer()

    def forward(self, x):
        x_s = self.static(x)
        x_t = self.conv(x)

        return torch.cat((x_s, x_t), axis = 1)


class MQCNNDecoder(nn.Module):
    pass
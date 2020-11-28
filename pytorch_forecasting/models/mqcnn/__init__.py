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
    """Decoder implementation for MQCNN

    Parameters
    ----------
    config
        Configurations
    ltsp : list of tuple of int
        List of lead-time / span tuples to make predictions for
    expander : HybridBlock
        Overrides default future data expander if not None
    hf1 : HybridBlock
        Overrides default global future layer if not None
    hf2 : HybridBlock
        Overrides default local future layer if not None
    ht1 : HybridBlock
        Overrides horizon-specific layer if not None
    ht2 : HybridBlock
        Overrides horizon-agnostic layer if not None
    h : HybridBlock
        Overrides local MLP if not None
    span_1 : HybridBlock
        Overrides span 1 layer if not None
    span_N : HybridBlock
        Overrides span N layer if not None

    Inputs:
        - **xf** : Future data of shape
            (batch_size, Trnn + lead_future - 1, num_future_ts_features)
        - **encoded** : Encoded input tensor of shape
            (batch_size, Trnn, n) for some n
    Outputs:
        - **pred_1** :  Span 1 predictions of shape
            (batch_size, Trnn, Tpred * num_quantiles)
        - **pred_N** : Span N predictions of shape
            (batch_size, Trnn, span_N_count * num_quantiles)

        In both outputs, the last dimensions has the predictions grouped
        together by quantile. For example, the quantiles are P10 and P90
        then the span 1 predictions will be:
        Tpred_0_p50, Tpred_1_p50, ..., Tpred_N_p50, Tpred_0_p90,
        Tpred_1_p90, ... Tpred_N_90


    """

    def __init__(self, config, ltsp, expander=None, hf1=None, hf2=None,
                 ht1=None, ht2=None, h=None, span_1=None, span_N=None,
                 **kwargs):
        super(MQCNNDecoder, self).__init__(**kwargs)
        self.future_features_count = len(config.future_information)
        self.Trnn = config.Trnn
        self.lead_future = config.lead_future
        self.ltsp = ltsp

        # We assume that Tpred == span1_count.
        self.Tpred = max(map(lambda x: x[0] + x[1], self.ltsp))
        span1_count = len(list(filter(lambda x: x[1] == 1, self.ltsp)))
        assert span1_count == self.Tpred, "Number of span 1 horizons: {} " \
                                          "does not match Tpred: {}" \
                                          .format(span1_count, self.Tpred)

        self.spanN_count = len(list(filter(lambda x: x[1] != 1, self.ltsp)))
        self.num_quantiles = len(config.quantiles)
        with self.name_scope():
            # Setting default components:
            if expander is None:
                expander = ExpandLayer(self.Trnn, config.lead_future)
            if hf1 is None:
                hf1 = self._get_global_future_layer()
            if hf2 is None:
                hf2 = self._get_local_future_layer()
            if ht1 is None:
                ht1 = self._get_horizon_specific()
            if ht2 is None:
                ht2 = self._get_horizon_agnostic()
            if h is None:
                h = self._get_local_mlp()
            if span_1 is None:
                span_1 = self._get_span_1()
            if span_N is None:
                span_N = self._get_span_N()

            self.expander = expander
            self.hf1 = hf1
            self.hf2 = hf2
            self.ht1 = ht1
            self.ht2 = ht2
            self.h = h
            self.span_1 = span_1
            self.span_N = span_N

    def forward(self, F, xf, encoded):
        expanded = self.expander(xf)
        hf1 = self.hf1(expanded)
        hf2 = self.hf2(expanded)

        ht = torch.cat(encoded, hf1, dim=-1)
        ht1 = self.ht1(ht)
        ht2 = self.ht2(ht)
        h = torch.cat(ht1, ht2, hf2, dim=-1)
        h = self.h(h)
        return self.span_1(h), self.span_N(h)

    def _get_global_future_layer(self, x):
        x = x.view(-1, self.Trnn, self.lead_future * self.future_features_count)
        
        return nn.Linear(self.lead_feature * self_future_features_count, 30)(x)

    def _get_local_future_layer(self, x):
        return nn.Tanh(x)

    def _get_horizon_specific(self, x):
        x = nn.Linear(self.Tpred * 20)(x)
        x = nn.ReLU(x)

        return x.view(-1, self.Trnn, self.Tpred, 20)

    def _get_horizon_agnostic(self, x, in_channels, out_channels):
        x = nn.Linear(in_channels, out_channels)(x)
        x = nn.ReLU(x)
        x = x.unsqueeze(axis = 2)
        x = x.repeat(1,1, self.lead_future, 1)

        return x

    def _get_local_mlp(self,x):
        x = nn.Linear(in_channels, 50)(x)
        x = nn.ReLU(x)
        x = nn.Linear(in_channels, 10)(x)
        x = nn.ReLU(x)

        return x

    def _get_span_1(self):
        pass

    def _get_span_N(self):
        pass
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
import torch as torch
from pytorch_forecasting.data impgit ort TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, QuantileLoss
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.mqcnn.sub_modules import (
    StaticLayer, ConvLayer, ExpandLayer, GlobalFutureLayer, HorizonAgnostic, HorizonSpecific,
    Span1, SpanN, LocalMlp
)

class MQCNNModel(BaseModelWithCovariates):
    def __init__(self, static_features, timevarying_features, time_step, future_information, ltsp, lead_future,
                 global_hidden_units, horizon_specific_hidden_units,
                 horizon_agnostic_hidden_units, local_mlp_hidden_units, local_mlp_output_units):
        super(MQCNNModel, self).__init__()
        self.time_step = time_step
        self.static_features = static_features
        self.timevarying_features = timevarying_features
        self.future_information = future_information
        self.ltsp = ltsp
        self.lead_future = lead_future
        self.global_hidden_units = global_hidden_units
        self.horizon_specific_hidden_units = horizon_specific_hidden_units
        self.horizon_agnostic_hidden_units = horizon_agnostic_hidden_units
        self.local_mlp_hidden_units = local_mlp_hidden_units
        self.local_mlp_output_units = local_mlp_output_units

        self.encoder = MQCNNEncoder(self.time_step, self.static_features, self.timevarying_features)
        self.decoder = MQCNNDecoder(self.time_step, self.lead_future, self.ltsp, self.future_information,
                                    self.global_hidden_units, self.horizon_specific_hidden_units,
                                    self.horizon_agnostic_hidden_units, self.local_mlp_hidden_units,
                                    self.local_mlp_output_units)

    def forward(self, x):
        encoding = self.encoder(x)
        output = self.decoder(encoding, x)

        return output

class MQCNNEncoder(nn.Module):
    def __init__(self, time_step, static_features, timevarying_features):
        self.time_step = time_step
        self.static_features = static_features
        self.timevarying_features = timevarying_features
        self.static = StaticLayer(in_channels = len(self.static_features),
                                  Trnn = self.time_step,
                                  static_features = self.static_features)

        self.conv = ConvLayer(in_channels = len(self.timevarying_features),
                              timevarying_features = self.timevarying_features)

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

    def __init__(self, time_step, lead_future, future_information, ltsp,
                 global_hidden_units, horizon_specific_hidden_units, horizon_agnostic_hidden_units,
                 local_mlp_hidden_units, local_mlp_output_units,
                 num_quantiles = 2,expander=None, hf1=None, hf2=None,
                 ht1=None, ht2=None, h=None, span_1=None, span_N=None,
                 **kwargs):
        super(MQCNNDecoder, self).__init__(**kwargs)
        self.future_features_count = len(future_information)
        self.future_information = future_information
        self.time_step = time_step
        self.lead_future = lead_future
        self.ltsp = ltsp
        self.num_quantiles = num_quantiles
        self.global_hidden_units = global_hidden_units
        self.horizon_specific_hidden_units = horizon_specific_hidden_units
        self.horizon_agnostic_hidden_units = horizon_agnostic_hidden_units
        self.local_mlp_hidden_units = local_mlp_hidden_units
        self.local_mlp_output_units = local_mlp_output_units

        # We assume that Tpred == span1_count.
        # Tpred = forecast_end_index
        self.Tpred = max(map(lambda x: x[0] + x[1], self.ltsp))
        span1_count = len(list(filter(lambda x: x[1] == 1, self.ltsp)))
        #print(self.Tpred, span1_count)
        #assert span1_count == self.Tpred, f"Number of span 1 horizons: {span1_count}\
                                            #does not match Tpred: {self.Tpred}" 

        self.spanN_count = len(list(filter(lambda x: x[1] != 1, self.ltsp)))
        # Setting default components:
        if expander is None:
            expander = ExpandLayer(self.time_step, self.lead_future, self.future_information)
        if hf1 is None:
            hf1 = GlobalFutureLayer(self.lead_future, self.future_features_count, out_channels=self.global_hidden_units)
        if ht1 is None:
            ht1 = HorizonSpecific(self.Tpred, self.time_step, num = self.horizon_specific_hidden_units)
        if ht2 is None:
            ht2 = HorizonAgnostic(self.horizon_agnostic_hidden_units, self.lead_future)
        if h is None:
            h = LocalMlp(self.local_mlp_hidden_units, self.local_mlp_output_units)
        if span_1 is None:
            span_1 = Span1(self.time_step, self.lead_future, self.num_quantiles)
        if span_N is None:
            span_N = SpanN(self.time_step, self.lead_future, self.num_quantiles, self.spanN_count)

        self.expander = expander
        self.hf1 = hf1
        self.hf2 = hf2
        self.ht1 = ht1
        self.ht2 = ht2
        self.h = h
        self.span_1 = span_1
        self.span_N = span_N

    def forward(self, F, x, encoded):
        xf = x[[self.future_information]]
        expanded = self.expander(xf)
        hf1 = self.hf1(expanded)
        hf2 = F.tanh(expanded)

        ht = torch.cat(encoded, hf1, dim=-1)
        ht1 = self.ht1(ht)
        ht2 = self.ht2(ht)
        h = torch.cat(ht1, ht2, hf2, dim=-1)
        h = self.h(h)
        return self.span_1(h), self.span_N(h)



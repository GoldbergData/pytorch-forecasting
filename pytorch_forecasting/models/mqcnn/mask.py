import torch
from torch import nn
import torch.functional as F

cclass InstockMask(nn.Module):
    def __init__(self, time_step, ltsp, min_instock_ratio = 0.5, eps_instock_dph = 1e-3,
                eps_total_dph = 1e-3, **kwargs):

        super(InstockMask, self).__init__(**kwargs)

        if not eps_total_dph > 0:
            raise ValueError(f"epsilon_total_dph of {eps_total_dph} is invalid! \
                              This parameter must be > 0 to avoid division by 0.")

        self.min_instock_ratio = min_instock_ratio
        self.eps_instock_dph = eps_instock_dph
        self.eps_total_dph = eps_total_dph

    def forward(self, F, demand, total_dph, instock_dph):

        if total_dph is not None and instock_dph is not None:

            total_dph = total_dph + self.eps_total_dph
            instock_dph = instock_dph + self.eps_instock_dph
            instock_rate = torch.round(instock_dph/total_dph)

            demand = torch.where(instock_rate >= self.min_instock_ratio, demand,
                                 -torch.ones_like(demand))

        return demand


class _BaseInstockMask(nn.Module):      
    def __init__(self, time_step, ltsp, min_instock_ratio = 0.5, eps_total_dph = 1e-3,
                eps_instock_dph = 1e-3, **kwargs):

        super(_BaseInstockMask, self).__init__(**kwargs)

        if not eps_total_dph > 0:
            raise ValueError(f"epsilon_total_dph of {eps_total_dph} is invalid! \
                              This parameter must be > 0 to avoid division by 0.")

        self.instock_mask = InstockMask(time_step, ltsp, min_instock_ratio=min_instock_ratio,
                                        eps_instock_dph = eps_instock_dph, 
                                        eps_total_dph = eps_total_dph)

    def forward(self, F):
        raise NotImplementedError

class HorizonMask(_BaseInstockMask):
    def __init__(self, time_step, ltsp, min_instock_ratio = 0.5, eps_instock_dph=1e-3,
                eps_total_dph=1e-3, **kwargs):

        super(HorizonMask, self).__init__(time_step, ltsp, 
                                          min_instock_ratio = min_instock_ratio,
                                          eps_instock_dph=eps_instock_dph,
                                          eps_total_dph=eps_total_dph, **kwargs)
        
        self.mask_idx = _compute_horizon_mask(time_step, ltsp)

    def forward(self, F, demand, total_dph, instock_dph):
        demand_instock = self.instock_mask(demand, total_dph, instock_dph)

        mask = torch.broadcast_tensors(self.mask_idx, demand_instock)

        masked_demand = torch.where(mask, demand_instock, -torch.ones_like(demand_instock))

        return masked_demand
        

def _compute_horizon_mask(time_step, ltsp):

    horizon = np.array(list(map(lambda _ltsp: _ltsp[0] + _ltsp[1], ltsp))).\
                        reshape((1, len(ltsp)))

    forecast_date_range = np.arange(time_step).reshape((time_step, 1))
    relative_distance = forecast_date_range + horizon
    mask = relative_distance < time_step
    return torch.tensor(mask).unsqueeze(0)
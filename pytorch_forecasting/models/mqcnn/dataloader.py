from torch import nn.Module
from torch.functionla import F
from .mask import HorizonMask


class DemandExpander(nn.Module):

    def __init__(self, time_step, ltsp, normalize = True,
                mask_func = HorizonMask, min_instock_ratio=0.5,
                eps_instock_dph = 1e-3, eps_total_dph = 1e-3, **kwargs):

        super(DemandExpander, self).__init__(**kwargs)
        if not eps_total_dph > 0:
            raise ValueError("eps_total_dph can't be 0")

        Tpred = max(map(lambda x: x[0] + x[1], ltsp))
        pos_sp1 = [i for i, x in enumerate(ltsp) if x[1] == 1]
        pos_spN = [i for i, x in enumerate(ltsp) if x[1] != 1]

        self.pos_sp1 = pos_sp1
        self.pos_spN = pos_spN

        self.ltsp_kernel = _ltsp_kernel(Tpred, ltsp, normalize)
        self.ltsp_idx = _ltsp_idx(time_step, Tpred)
        self.demand_mask = mask_func(time_step, ltsp, min_instock_ratio=min_instock_ratio,
                                     eps_instock_dph=eps_instock_dph,
                                     eps_total_dph = eps_total_dph)

    def forward(self, demand, total_dph, instock_dph):
        ltsp_demand = _apply_ltsp_kernel(demand, self.ltsp_idx, self.ltsp_kernel)

        ltsp_idph = _apply_ltsp_kernel(instock_dph, self.ltsp_idx, self.ltsp_kernel)
        ltsp_dph = _apply_ltsp_kernel(total_dph, self.ltsp_idx, self.ltsp_kernel)

        masked_demand = self.demand_mask(ltsp_demand, ltsp_dph, ltsp_idph)
        masked_demand_sp1 = masked_demand[:, :, self.pos_sp1]
        masked_demand_spN = masked_demand[:, :, self.pos_spN]

        return masked_demand_sp1, masked_demand_spN

def _ltsp_idx(time_step, Tpred):
        idx = np.arange(time_step).reshape(-1, 1) + np.arange(Tpred)
        return torch.tensor(idx)

def _ltsp_kernel(Tpred, ltsp, normalize = True):
    
        ltsp_count = len(ltsp)
        kernel = np.zeros((Tpred, ltsp_count), dtype = 'float32')
        for i in range(len(ltsp)):
            lead_time = ltsp[i][0]
            span = ltsp[i][1]
            if normalize:
                kernel[lead_time:lead_time + span, i] = 1.0/span
            else:
                kernel[lead_time:lead_time + span, i] = 1.0

        return torch.tensor(kernel)

def _apply_ltsp_kernel(s, ltsp_idx, ltsp_kernel):
        s_ltsp = s[:, ltsp_idx].float()
        
        return s_ltsp @ ltsp_kernel 
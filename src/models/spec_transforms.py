import torch
import torch.nn as nn
import torchaudio
import numpy as np


class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        norm_min, norm_max = Xstd.min(-1)[0].min(-1)[0], Xstd.max(-1)[0].max(-1)[0]
        fix_ind = (norm_max - norm_min) > self.eps * torch.ones_like(
            (norm_max - norm_min)
        )
        V = torch.zeros_like(Xstd)
        if fix_ind.sum():
            V_fix = Xstd[fix_ind]
            norm_max_fix = norm_max[fix_ind, None, None]
            norm_min_fix = norm_min[fix_ind, None, None]
            V_fix = torch.max(
                torch.min(V_fix, norm_max_fix),
                norm_min_fix,
            )
            # print(V_fix.shape, norm_min_fix.shape, norm_max_fix.shape)
            V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
            V[fix_ind] = V_fix
        return V
    
    
class TimeMasking(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=25)
        self.p = p
        
    def forward(self, spec):
        if torch.rand(1) < self.p:
            spec = self.time_mask(spec)
        return spec
    

class FrequencyMasking(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        self.p = p
        
    def forward(self, spec):
        if torch.rand(1) < self.p:
            spec = self.freq_mask(spec)
        return spec

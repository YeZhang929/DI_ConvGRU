# 作者：张叶
# 时间：2021/10/11  16:18

import torch
import torch.nn as nn
import numpy as np
import math

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
    def forward(self, output, target):
        loss = 0
        p0 = output
        t0 = target
        mask = t0 == t0
        p = p0[mask].cuda()
        t = t0[mask].cuda()
        temp = ((p - t)**2).mean()
        loss = loss + temp
        return loss
        


class RmseLoss(torch.nn.Module):
    def __init__(self):
        super(RmseLoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[1]
        loss = 0
        for k in range(ny):
            p0 = output[:, k, :, :]
            t0 = target[:, k, :, :]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = torch.sqrt(((p - t)**2).mean())
            loss = loss + temp
        return loss

class ubRmseLoss(torch.nn.Module):
    def __init__(self):
        super(ubRmseLoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k, :, :]
            t0 = target[:, :, k, :, :]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            #t = torch.where(torch.isnan(t), torch.full_like(t, 0), t)
            pmean = p.mean()
            tmean = t.mean()
            p_ub = p-pmean
            t_ub = t-tmean
            temp = torch.sqrt(((p_ub - t_ub)**2).mean())
            loss = loss + temp
        return loss

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from run0 import *


def Criterion(num_f_classes, num_c_classes, num_perclass_train, num_perclass_train_c, switch_reweighting, defer_epoch,
              beta_EffectNum_f, beta_EffectNum_c, switch_criterion, gamma_Focal, epoch):
    if switch_reweighting == 'Reweighting' and epoch == defer_epoch:
        per_cls_weights_f = 1 / np.array(num_perclass_train)
        per_cls_weights_c = 1 / np.array(num_perclass_train_c)

        per_cls_weights_f = torch.FloatTensor(per_cls_weights_f).to(device)
        per_cls_weights_c = torch.FloatTensor(per_cls_weights_c).to(device)
    elif switch_reweighting == 'EffectNum' and epoch == defer_epoch:
        effective_num_f = 1.0 - np.power(beta_EffectNum_f, num_perclass_train)
        per_cls_weights_f = (1.0 - beta_EffectNum_f) / np.array(effective_num_f)
        per_cls_weights_f = per_cls_weights_f / np.sum(per_cls_weights_f) * len(num_perclass_train)

        effective_num_c = 1.0 - np.power(beta_EffectNum_c, num_perclass_train_c)
        per_cls_weights_c = (1.0 - beta_EffectNum_c) / np.array(effective_num_c)
        per_cls_weights_c = per_cls_weights_c / np.sum(per_cls_weights_c) * len(num_perclass_train_c)

        per_cls_weights_f = torch.FloatTensor(per_cls_weights_f).to(device)
        per_cls_weights_c = torch.FloatTensor(per_cls_weights_c).to(device)
    else:
        per_cls_weights_f = torch.ones(num_f_classes).to(device)
        per_cls_weights_c = torch.ones(num_c_classes).to(device)
# =====================================================================================================================

    if switch_criterion == 'CrossEntropy':
        criterion_f = nn.CrossEntropyLoss(weight=per_cls_weights_f).to(device)
        criterion_c = nn.CrossEntropyLoss(weight=per_cls_weights_c).to(device)
    elif switch_criterion == 'Focal':
        criterion_f = FocalLoss(weight=per_cls_weights_f, gamma=gamma_Focal).to(device)
        criterion_c = FocalLoss(weight=per_cls_weights_c, gamma=gamma_Focal).to(device)
    elif switch_criterion == 'LDAM':
        criterion_f = LDAMLoss(cls_num_list=num_perclass_train, max_m=0.5, s=30, weight=per_cls_weights_f).to(device)
        criterion_c = LDAMLoss(cls_num_list=num_perclass_train_c, max_m=0.5, s=30, weight=per_cls_weights_c).to(device)
    else:
        raise ValueError(switch_criterion)

    return criterion_f, criterion_c


def focal_loss(input_values, gamma):
    """Computes the focal loss

    Reference: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
    """
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss


class FocalLoss(nn.Module):
    """Reference: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py"""
    def __init__(self, weight=None, gamma=0., reduction='mean'):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, weight=self.weight, reduction=self.reduction), self.gamma)


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(device)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.FloatTensor).to(device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)
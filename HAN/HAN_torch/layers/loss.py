# -*- coding: utf-8 -*-
# @Time : 2023/3/22 15:29
# @Author : yysgz
# @File : loss.py
# @Project : HAN_torch
# @Description :

import torch
import torch.nn as nn
import torch.nn.functional as F

class HAN_Loss(nn.Module):
    def __init__(self):
        super(HAN_Loss, self).__init__()

    def masked_softmax_cross_entropy(self, logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = nn.CrossEntropyLoss(logits, labels)  # 返回交叉熵向量

        mask = torch.FloatTensor(mask)  # 改变tensor数据类型
        mask /= torch.mean(mask)  # 通过均值求loss
        loss *= mask

        return torch.mean(loss)  # 用于计算tensor某一维度的mean

# -*- coding: utf-8 -*-
# @Time : 2023/3/23 14:45
# @Author : yysgz
# @File : Attn_Head.py
# @Project : HAN_torch
# @Description :

import torch
import torch.nn as nn

class Attn_Head(nn.Module):
    def __init__(self, out_sz,  in_drop=0.0, coef_drop=0.0, residual=False):
        super(Attn_Head, self).__init__()
        self.out_sz = out_sz
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.return_coef = False
        self.act1 = nn.ReLU()
        self.act2 = nn.ELU()

    def forward(self, features, bias_mat):
        """[summary]
        multi-head attention计算
        [description]
        # forward；model = HeteGAT_multi
        attns.append(layers.attn_head(features,            # list:3, tensor（1， 3025， 1870）
                                    bias_mat=bias_mat,     # list:2, tensor(1, 3025, 3025)
                                    out_sz=hid_units[0],   # hid_units:[8]，卷积核的个数
                                    activation=activation, # nonlinearity:torch.nn.elu
                                    in_drop=ffd_drop,      # tensor, ()
                                    coef_drop=attn_drop,   # tensor, ()
                                    residual=False))
        Arguments:
            features {[type]} -- shape=(batch_size, nb_nodes, fea_size))
        """
        if self.in_drop != 0.0:
            features = nn.Dropout(features, 1.0 - self.in_drop)  # 以rate置0
        features_fts = nn.Conv1d(features, self.out_sz, 1, use_bias=False)  # 一维卷积操作, out: (1, 3025, 8)

        f_1 = nn.Conv1d(features_fts, 1, 1)  # (1, 3025, 1)
        f_2 = nn.Conv1d(features_fts, 1, 1)  # (1, 3025, 1)

        logits = f_1 + torch.transpose(f_2, [0, 2, 1])  # 转置         # (1, 3025, 3025)
        coefs = nn.Softmax(self.act1(logits) + bias_mat)  # (1, 3025, 3025)
        coefs = torch.add(coefs, bias_mat)

        if self.coef_drop != 0.0:
            coefs = nn.Dropout(coefs, 1.0 - self.coef_drop)
        if self.in_drop != 0.0:
            features_fts = torch.nn.Dropout(features_fts, 1.0 - self.in_drop)

        vals = torch.matmul(coefs, features_fts)  # (1, 3025, 8)
        # ret = torch.add(vals, bias_mat) # 将bias向量加到value矩阵上      # (1. 3025， 8)
        ret = vals

        # residual connection 残差连接
        if self.residual:
            if features.shape[-1] != ret.shape[-1]:
                ret = ret + nn.Conv1d(features, ret.shape[-1], 1)  # activation
            else:
                features_fts = ret + features
        if self.return_coef:
            return self.act2(ret), coefs
        else:
            return self.act2(ret)  # activation
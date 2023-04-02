# -*- coding: utf-8 -*-
# @Time : 2023/3/21 20:38
# @Author : yysgz
# @File : HeteGAT_multi.py
# @Project : HAN_torch
# @Description :

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Attn_Head import Attn_Head, SimpleAttnLayer

class HeteGAT_multi(nn.Module):
    '''
    inputs_list=feas_list, nb_classes=nb_classes, nb_nodes=nb_nodes, attn_drop=0.5,
                              ffd_drop=0.0, biases_list=biases_list, hid_units=args.hid_units, n_heads=args.n_heads,
                              activation=nn.ELU(), residual=args.residual)

    '''
    def __init__(self, feature_size, nb_classes, nb_nodes, attn_drop, ffd_drop,
                 nb_biases, hid_units, n_heads, activation=nn.ELU()):
        super(HeteGAT_multi, self).__init__()
        self.feature_size = feature_size  # list:3, (3025, 1864)
        self.nb_classes = nb_classes  # 3
        self.nb_nodes = nb_nodes  # 3025
        self.attn_drop = attn_drop  # 0.5
        self.ffd_drop = ffd_drop  # 0.0
        self.nb_biases = nb_biases  # list:2, (3025,3025)
        self.hid_units = hid_units  # [8]
        self.n_heads = n_heads  # [8,1]
        self.activation = activation  # nn.ELU
        # self.residual = residual
        self.mlp_attn_size = 128

        self.layers = self._make_attn_head()
        self.simpleAttnLayer = SimpleAttnLayer(64, self.mlp_attn_size, time_major=False, return_alphas=True)  # 64, 128
        self.fc = nn.Linear(64, nb_classes)  # 64, 3


    def _make_attn_head(self):
        layers = []
        for i in range(self.nb_biases):  # (3025,1864); (3025,3025)
            attn_list = []
            for j in range(self.n_heads[0]):  # 8-head
                attn_list.append(Attn_Head(in_channel=int(self.feature_size/self.n_heads[0]), out_sz=self.hid_units[0],  # in_channel,233; out_sz,8
                                in_drop=self.ffd_drop, coef_drop=self.attn_drop, activation=self.activation))

            layers.append(nn.Sequential(*list(m for m in attn_list)))
        return nn.Sequential(*list(m for m in layers))

    def forward(self, features, batch_bias_list, batch_nodes):
        embed_list = []

        # multi-head attention in a hierarchical manner
        for i, biases in enumerate(batch_bias_list):
            attns = []

            batch_feature = features[batch_nodes]  # (100, 1864)
            batch_bias = batch_bias_list[i]  # (100, 100)
            attn_embed_size = int(batch_feature.shape[1] / self.n_heads[0])
            jhy_embeds = []
            for n in range(self.n_heads[0]):  # [8,1], 8个head
                # multi-head attention 计算
                attns.append(self.layers[i][n](batch_feature[:, n*attn_embed_size: (n+1)*attn_embed_size], batch_bias))

            h_1 = torch.cat(attns, dim=-1)  # shape=(1, 100, 64)
            embed_list.append(torch.transpose(h_1,1,0))  # list:2. 其中每个元素tensor, (100, 1, 64)

        multi_embed = torch.cat(embed_list, dim=1)   # tensor, (100, 2, 64)
        # simple attention 合并多个meta-based homo-graph embedding
        final_embed, att_val = self.simpleAttnLayer(multi_embed)  # (100, 64)

        out = []
        # 添加一个全连接层做预测(final_embedding, prediction) -> (100, 3)
        out.append(self.fc(final_embed))

        return out[0]

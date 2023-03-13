# -*- coding: utf-8 -*-
# @Time : 2023/3/21 20:38
# @Author : yysgz
# @File : HeteGAT_multi.py
# @Project : HAN_torch
# @Description :

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Attn_Head import Attn_Head
from utils.SimpleAttLayer import SimpleAttLayer

class HeteGAT_multi(nn.Module):
    '''
    # forward；model = HeteGAT_multi
    logits, final_embedding, att_val = model.inference(fea_in_list,  # list:3, tensor（1， 3025， 1870）
                                                       nb_classes,   # 3
                                                       nb_nodes,     # 3025
                                                       is_train,     # bool
                                                       attn_drop,    # tensor, ()
                                                       ffd_drop,     # tensor, ()
                                                       batch_bias_list=bias_in_list,  # list:2, tensor(1, 3025, 3025)
                                                       hid_units=hid_units,   # hid_units: [8]
                                                       n_heads=n_heads,       # n_heads: [8, 1]
                                                       residual=residual,     # residual: False
                                                       activation=nonlinearity)  # nonlinearity:tf.nn.elu

    '''
    def __init__(self, nb_classes, hid_units, n_heads, residual, mlp_attn_size, attn_drop, ffd_drop):
        super(HeteGAT_multi, self).__init__()
        self.nb_classes = nb_classes
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.residual = residual
        self.mlp_attn_size = mlp_attn_size
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        # 4 params
        self.attn_head1 = Attn_Head(out_sz=n_heads[0], in_drop=ffd_drop,
                                    coef_drop=attn_drop, residual=residual)
        self.attn_head2 = Attn_Head(out_sz=n_heads[1], in_drop=ffd_drop,
                                    coef_drop=attn_drop, residual=residual)

    def forward(self, batch_feature, batch_bias):
        embed_list = []
        attns = []
        jhy_embeds = []
        for _ in range(self.n_heads[0]):  # [8,1], 8个head
            # multi-head attention 计算
            attns.append(self.attn_head1(batch_feature, batch_bias=batch_bias))

        h_1 = torch.concat(attns, dim=-1)  # shape=(1, 3025, 64)

        for i in range(1, len(self.hid_units)):
            h_old = h_1
            attns = []
            for _ in range(self.n_heads[i]):
                attns.append(self.attn_head2(h_1, batch_bias=batch_bias))
            h_1 = torch.cat(attns, dim=-1)
        embed_list.append(torch.unsqueeze(torch.squeeze(h_1), dim=1))  # list:2. 其中每个元素tensor, (3025, 1, 64)


        multi_embed = torch.concat(embed_list, dim=1)   # tensor, (2, 3025, 64)
        # attention输出：tensor(3025, 64)、softmax概率
        final_embed, att_val = SimpleAttLayer(multi_embed,
                                              self.mlp_attn_size,
                                              time_major=False,
                                              return_alphas=True)

        out = []
        for i in range(self.n_heads[-1]):  # 1
            # 用于添加一个全连接层(input, output) -> (3025, 3)
            out.append(nn.Linear(final_embed, self.nb_classes))
        #     out.append(attn_head(h_1, batch_bias=batch_bias,
        #                                 out_sz=nb_classes, activation=lambda x: x,
        #                                 in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = torch.sum(out) / self.n_heads[-1]  # add_n是列表相加。tensor,(3025, 3)
        # logits_list.append(logits)
        print('de')

        logits = torch.unsqueeze(logits, dim=0)  # (1, 3025, 3)
        # attention通过全连接层预测(1, 3025, 3)、attention final_embedding tensor(3025, 64)、attention 概率
        return logits, final_embed, att_val

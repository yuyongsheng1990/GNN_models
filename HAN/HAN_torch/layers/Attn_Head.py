# -*- coding: utf-8 -*-
# @Time : 2023/3/23 14:45
# @Author : yysgz
# @File : Attn_Head.py
# @Project : HAN_torch
# @Description :
import numpy as np
import torch
import torch.nn as nn

class Attn_Head(nn.Module):
    def __init__(self, in_channel, out_sz, in_drop=0.0, coef_drop=0.0, activation=None, return_coef=False):
        super(Attn_Head, self).__init__()
        # self.bias_mat = bias_mat  # (3025,3025)
        self.in_drop = in_drop  # 0.0
        self.coef_drop = coef_drop  # 0.5
        self.return_coef = return_coef
        self.conv1 = nn.Conv1d(in_channel, out_sz, 1, bias=False)  # (233,8)
        self.conv2_1 = nn.Conv1d(out_sz, 1, 1, bias=False)  # (8,1)
        self.conv2_2 = nn.Conv1d(out_sz, 1, 1, bias=False)  # (8,1)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.in_dropout = nn.Dropout(in_drop)
        self.coef_dropout = nn.Dropout(coef_drop)
        self.activation = activation

    def forward(self, x, bias_mx):  # (100, 233); bias_mx, 残差, (100, 100)
        seq = x.float()
        if self.in_drop != 0.0:
            seq = self.in_drop(x)  # 以rate置0
            seq = seq.float()
        # reshape x and bias_mx for nn.Conv1d, (1, 233, 100)
        seq = torch.transpose(seq[np.newaxis], 2, 1)  # (1, 233, 100)
        bias_mx = bias_mx[np.newaxis]
        seq_fts = self.conv1(seq)  # 一维卷积操作, out: (1, 8, 100)

        f_1 = self.conv2_1(seq_fts)  # (1, 1, 100)
        f_2 = self.conv2_2(seq_fts)  # (1, 1, 100)

        logits = f_1 + torch.transpose(f_2, 2, 1)  # 转置 (1, 100, 100)
        logits = self.leakyrelu(logits)

        coefs = self.softmax(logits + bias_mx.float())  # add残差, (1, 100, 100)

        if self.coef_drop != 0.0:
            coefs = self.coef_dropout(coefs)
        if self.in_drop != 0.0:
            seq_fts = self.in_dropout(seq_fts)

        ret = torch.matmul(coefs, torch.transpose(seq_fts, 2, 1))  # (1, 100, 8)

        if self.return_coef:
            return self.activation(ret), coefs
        else:
            return self.activation(ret)  # activation


class SimpleAttnLayer(nn.Module):
    def __init__(self, inputs, attn_size, time_major=False, return_alphas=False):  # inputs, 64; attention_size,128; return_alphas=True
        super(SimpleAttnLayer, self).__init__()
        self.hidden_size = inputs  # 64
        self.return_alphas = return_alphas  # True
        self.time_major = time_major
        self.w_omega = nn.Parameter(torch.Tensor(self.hidden_size, attn_size))  # (64, 128)
        self.b_omega = nn.Parameter(torch.Tensor(attn_size))  # (128,)
        self.u_omega = nn.Parameter(torch.Tensor(attn_size, 1))  # (128,)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_omega)
        nn.init.zeros_(self.b_omega)
        nn.init.xavier_uniform_(self.u_omega)

    def forward(self, x):  # (100,2,64)
        '''
        inputs: tensor, (3025, 64)
        attention_size: 128
        '''
        if isinstance(x, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = torch.concat(x, 2)  # 表示在shape第2个维度上拼接

        v = self.tanh(torch.matmul(x, self.w_omega) + self.b_omega)  # (100,2,128)
        vu = torch.matmul(v, self.u_omega)  # (100,2,1)
        alphas = self.softmax(vu)

        output = torch.sum(x * alphas.reshape(alphas.shape[0],-1,1), dim=1)  # (100,2,64)*(100,1,2) -> (100,64)

        if not self.return_alphas:
            return output
        else:
            return output, alphas  # attention输出、softmax概率
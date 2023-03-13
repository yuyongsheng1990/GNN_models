# -*- coding: utf-8 -*-
# @Time : 2023/3/23 15:06
# @Author : yysgz
# @File : SimpleAttLayer.py
# @Project : HAN_torch
# @Description :

import torch
import torch.nn as nn
from torch.autograd import Variable

def SimpleAttLayer(inputs, attention_size, time_major=False, return_alphas=False):
    '''
    inputs: tensor, (3025, 2, 64)
    attention_size: 128
    '''
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = torch.concat(inputs, 2)  # 表示在shape第2个维度上拼接

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = inputs.permute(1, 0, 2)

    hidden_size = inputs.shape[2]  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = Variable(torch.randn(hidden_size, attention_size, requires_grad=True))  # (64, 128)
    b_omega = Variable(torch.randn(attention_size, requires_grad=True))               # (128, )
    u_omega = Variable(torch.randn(attention_size, requires_grad=True))               # (128, )

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    v = torch.tanh(torch.tensordot(inputs, w_omega, dims=1) + b_omega)   # (3025, 2, 128)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = torch.tensordot(v, u_omega, dims=1)  # 任意维度的tensor矩阵相乘；(B,T) shape   tensor, (3025, 2)
    alphas = nn.Softmax(vu)         # (B,T) shape   tensor, (3025, 2)

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = torch.mean(inputs * torch.unsqueeze(alphas, -1), 1)  # (3025, 2, 64) * (3025, 2, 1) = (3025, 2, 64) -> (3025, 64)

    if not return_alphas:
        return output
    else:
        return output, alphas  # attention输出、softmax概率
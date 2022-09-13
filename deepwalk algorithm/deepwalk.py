# -*- coding: utf-8 -*-
# @Time : 2022/7/28 11:26
# @Author : yysgz
# @File : DeepWalk.py
# @Project : DeepWalk Demo.ipynb

import os
project_path = os.getcwd()

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import random

adj_list = [[1,2,3],[0,2,3],[0,1,3],[0,1,2],[5,6],[4,6],[4,5],[1,3]]
size_vertex = len(adj_list) # number of vertices

w = 3 # window size
d = 2 # embedding size
y = 200 # walks per ventext
t = 6 # walk length
lr = 0.025 # learning rate

v = [0,1,2,3,4,5,6] # labels of available vertices

def RandomWalk(node, t):
    walk = [node]

    for i in range(t - 1):
        list_length = len(adj_list[node])-1
        node_index = random.randint(0, list_length)
        node = adj_list[node][node_index]
        walk.append(node)
    return walk


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.phi = nn.Parameter(torch.rand((size_vertex, d), requires_grad=True))  # size_vertex是one-hot向量维度，d是word embedding维度
        self.phi2 = nn.Parameter(torch.rand((d, size_vertex), requires_grad=True))

    def forward(self, one_hot):
        hidden = torch.matmul(one_hot, self.phi)
        out = torch.matmul(hidden, self.phi2)
        return out


model = Model()

def skip_gram(wvi, w):
    # wvi表示random walk生成的node sequence；w = 3 window size
    for j in range(len(wvi)):

        # 损失函数loss=每个panel的第ti=1那个位置误差error之和
        # generate one hot vector
        one_hot = torch.zeros(size_vertex)
        one_hot[wvi[j]] = 1  # 中心词输入one-hot向量
        # 未激活的输出向量
        out = model(one_hot)
        loss = 0
        # for c-th panel
        for k in range(max(0, j - w), min(j + w + 1, len(wvi))):  # 确定中心点j左右范围，k遍历中心点j的左右位置
            # 损失函数e=每个panel的第j个位置误差之和
            error = torch.log(torch.sum(torch.exp(out))) - out[wvi[k]]
            loss += error
        loss.backward()

        for param in model.parameters():
            param.data.sub_(lr * param.grad)
            param.grad.data.zero_() # param.grad就是每次迭代计算的误差error，需要每次重置为0

# wvi=RandomWalk(0,t) # wvi表示random walk生成的node sequence
# skip_gram(wvi, w)

for i in range(y):
    random.shuffle(v)
    for vi in v:
        wvi=RandomWalk(vi,t)
        skip_gram(wvi, w)
# -*- coding: utf-8 -*-
# @Time : 2023/3/22 12:13
# @Author : yysgz
# @File : data_process.py
# @Project : HAN_torch
# @Description : data process acm_dataset.

import numpy as np
import networkx as nx
import scipy.sparse as sp

"""
 Finally, the matrix is converted to bias vectors.
"""
def adj_to_bias(adj, nb_nodes, nhood=1):  # adj,(3025, 3025); sizes, [3025]

    mt = np.eye(adj.shape[0])
    for _ in range(nhood):
        mt = np.matmul(mt, (adj + np.eye(adj.shape[1])))  # 相乘
    for i in range(nb_nodes):
        for j in range(nb_nodes):
            if mt[i][j] > 0.0:
                mt[i][j] = 1.0

    return -1e9 * (1.0 - mt)  # 科学计数法，2.5 x 10^(-27)表示为：2.5e-27
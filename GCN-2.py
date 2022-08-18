# -*- coding: utf-8 -*-
# @Time : 2022/8/16 11:30
# @Author : yysgz
# @File : GCN.py
# @Project : GNN Algorithms
# @Description : 2022.08.16,实现GCN-2，参考github links:https://github.com/dragen1860/GCN-PyTorch

'------------------------------config------------------------------------'
import argparse

args = argparse.ArgumentParser() # 创建一个参数解析对象
# 然后向该对象中添加你想要的参数和对象
args.add_argument('--dataset', default='cora')
args.add_argument('--model', default='gcn')
args.add_argument('--learning_rate', type=float, default=0.01)
args.add_argument('--epochs', type=int, default=2000)
args.add_argument('--hidden', type=int, default=16)
args.add_argument('--dropout', type=float, default=0.5)
args.add_argument('--weight_decay', type=float, default=5e-4)
args.add_argument('--early_stopping', type=int, default=10)
args.add_argument('--max_degree', type=int, default=3)

# 最后调用parse_args()方法进行解析，解析成功后即可进行调用
args = args.parse_args(args=[])
print(args)

'-----------------------------utils--------------------------------------'
import torch
from torch.nn import functional as F

def masked_loss(out, label, mask):
    loss = F.cross_entropy(out,label, reduction='none')
    mask = mask.float() # 将整数和字符串转换成浮点型
    mask = mask/mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss

def masked_acc(out, label, mask):
    # [node, f]
    pred = out.argmax(dim=1) # argmax返回最大值的索引
    correct = torch.eq(pred, label).float()
    mask = mask.float()
    mask = mask.mean()
    correct *= mask
    acc = correct.mean()
    return acc

def sparse_dropout(x, rate, noise_shape):
    '''
    :param x:
    :param rate:
    :param noise_shape: int_scalar
    :return
    '''
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).bool()  # 取整函数floor；byte数据类型
    i = x._indices()  # [2,49216]，按索引index获取切片
    v = x._values()  # [49216]

    # [2,49216] => [49216,2] => [remained node,2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
    out = out * (1. / (1 - rate))

    return out

# sparse tensor
i = torch.LongTensor([[0, 1, 1],
                     [2, 1, 0]])
d = torch.tensor([3, 6, 9], dtype=torch.float)
a = torch.sparse.FloatTensor(i, d, torch.Size([2, 3]))
print(a)

def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x,y)
    else:
        res = torch.mm(x,y)
    return res

'-------------data---------------------------------------'
import pandas as pd
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

def parse_index_file(filename):
    '''
    parse index file
    '''
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    '''
    create mask
    '''
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def load_data(dataset_str):
    '''
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    '''
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open('data/ind.{}.{}'.format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_recorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_recorder)

    if dataset_str == 'citeseer':
        # fix citeseer dataset (there are some isolated nodes in the graph)
        # find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_recorder), max(test_idx_recorder) - 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_recorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))  # vstack按行拼接
    labels[test_idx_recorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])  # 训练集
    val_mask = sample_mask(idx_val, labels.shape[0])  # 验证集
    test_mask = sample_mask(idx_test, labels.shape[0])  # 测试集

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    '''
    convert sparse matrix to tuple representation.
    '''

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx

def preprocess_features(features):
    """
    Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = np.array(features.sum(1))  # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1).flatten() # 1/rowsum, [2708]
    r_inv[np.isinf(r_inv)] = 0. # zero inf data
    r_mat_inv = sp.diags(r_inv) # sparse diagonal matrix, [2708, 2708]
    features = r_mat_inv.dot(features) # D^-1:[2708, 2708]@X:[2708, 2708]
    return sparse_to_tuple(features) # [coordinates, data, shape], []

def normalize_adj(adj):
    'symmetrically normalize adjacency matrix'
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    'preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.'
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


# 切比雪夫多项式之和最佳线性逼近任意一个函数
def chebyshev_polynomials(adj, k):
    '''
    calculate chebyshev polynomials up to order k.
    return a list of sparse matrices(tuple representation)
    '''
    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

'-------------------------graph convolution layer--------------'
import torch
from torch import nn
from torch.nn import functional as F


# 构建图卷积层
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0,
                 is_sparse_inputs=False,
                 bias=False,
                 activation=F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()

        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

    # 构造前馈网络
    def forward(self, inputs):
        # prints('inputs': inputs)
        x, support = inputs
        if self.training and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless:
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight

        out = torch.sparse.mm(support, xw)
        if self.bias is not None:
            out += self.bias

        return self.activation(out), support

'--------------------------GCN model--------------'
import torch
from torch import nn
from torch.nn import functional as F


# 搭建GCN网络模型
class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_features_nonzero):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        print('num_features_nonzero:', num_features_nonzero)

        self.layers = nn.Sequential(GraphConvolution(self.input_dim, args.hidden, num_features_nonzero,
                                                     activation=F.relu, dropout=args.dropout,
                                                     is_sparse_inputs=True),
                                    GraphConvolution(args.hidden, output_dim, num_features_nonzero,
                                                     activation=F.relu, dropout=args.dropout,
                                                     is_sparse_inputs=False))

    def forward(self, inputs):
        x, support = inputs
        x = self.layers((x, support))
        return x

    def l2_loss(self):
        layer = self.layers.children()
        layer = next(iter(layer))

        loss = None
        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss

'----------------------------train-------------------'
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import numpy as np

seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
print('adj:', adj.shape)
print('faetures:', features.shape)
print('y:', y_train.shape, y_val.shape, y_test.shape)
print('mask:', train_mask.shape, val_mask.shape, test_mask.shape)

features = preprocess_features(features)
supports = preprocess_adj(adj)

device = torch.device('cuda')
train_label = torch.from_numpy(y_train).long()
num_classes = train_label.shape[1]
train_label = train_label.argmax(dim=1)
train_mask = torch.from_numpy(train_mask.astype(int))

val_label = torch.from_numpy(y_val).long()
val_label = val_label.argmax(dim=1)
val_mask = torch.from_numpy(val_mask.astype(int))

test_label = torch.from_numpy(y_test).long()
test_label = test_label.argmax(dim=1)
test_mask = torch.from_numpy(test_mask.astype(int))

i = torch.from_numpy(features[0]).long()
v = torch.from_numpy(features[1])
feature = torch.sparse.FloatTensor(i.t(), v, features[2])

i = torch.from_numpy(features[0]).long()
v = torch.from_numpy(features[1])
support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float()

print('x:', feature)
print('sp:', support)
num_features_nonzero = feature._nnz()
feat_dim = feature.shape[1]

# training
net = GCN(feat_dim, num_classes, num_features_nonzero)
# net
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

net.train()
for epoch in range(args.epochs):
    out = net((feature, support))
    out = out[0]
    loss = masked_loss(out, train_label, train_mask)
    loss += args.weight_decay * net.l2_loss()

    acc = masked_acc(out, train_label, train_mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(epoch, loss.item(), acc.item())
# test
net.eval()
out = net((feature, support))
out = out[0]
acc = masked_acc(out, test_label, test_mask)
print('test:', acc.item())
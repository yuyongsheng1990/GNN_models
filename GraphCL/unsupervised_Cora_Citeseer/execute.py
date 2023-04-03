import numpy as np
import scipy.sparse as sp
import pickle as pkl
import networkx as nx

import torch
import torch.nn as nn
import random
from models import DGI, LogReg
from utils import process
import pdb
import aug
import os
import sys
import argparse


parser = argparse.ArgumentParser("My DGI")

parser.add_argument('--dataset',          type=str,           default="citeseer",                help='data')
parser.add_argument('--aug_type',         type=str,           default="subgraph",                help='aug type: mask or edge')
parser.add_argument('--drop_percent',     type=float,         default=0.1,               help='drop percent')
parser.add_argument('--seed',             type=int,           default=39,                help='seed')
parser.add_argument('--gpu',              type=int,           default=0,                 help='gpu')
parser.add_argument('--save_name',        type=str,           default='try.pkl',                help='save ckpt name')

args = parser.parse_args()

print('-' * 100)
print(args)
print('-' * 100)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    # x,csr_mx,(120,3703); y,ndarray,(120,6); tx,csr_mx,(1000,3703); ty,ndarray,(1000,6); allx,csr_matrix,(2312,3703), ally,ndarray,(2312,6)
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/ind.{}.test.index".format(dataset_str))  # list:1000
    test_idx_range = np.sort(test_idx_reorder)  # sorted test idx, ndarray,(1000,), min:2312; max:3326

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)  # range(2312,3327), len:1015
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))  # 创建稀疏矩阵, lil_matrx,(1015,3703)
        tx_extended[test_idx_range-min(test_idx_range), :] = tx  # [0,1014]; tx,csr_mx,(1000,3703)
        tx = tx_extended  # tx, lil_mx,(1015,3703)
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))  # 创建ndarray, (1015,6)
        ty_extended[test_idx_range-min(test_idx_range), :] = ty  # [0,1014]; ty,ndarray,(1000,6)
        ty = ty_extended  # ndarray,(1015,6)

    features = sp.vstack((allx, tx)).tolil()  # features,lil_mx,(3327,3703)
    features[test_idx_reorder, :] = features[test_idx_range, :]  # test_idx_recorder range[2312, 3326]; (3327,3703)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))  # csr_mx,(3327,3327)

    labels = np.vstack((ally, ty))  # ndarray,(3327,6)
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # 用sorted labels代替原有labels order

    idx_test = test_idx_range.tolist()  # sorted test_idx
    idx_train = range(len(y))  # (0,120)
    idx_val = range(len(y), len(y)+500)  # (120,620)

    return adj, features, labels, idx_train, idx_val, idx_test  # adj,csr_mx,(3327,3327); features,lil_max,(3327,3703); labels,(3327,6); range(0,120); range(120,620); random_samp[2312,3327],len:1000

dataset = args.dataset
aug_type = args.aug_type
drop_percent = args.drop_percent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# training params


batch_size = 1
nb_epochs = 100
patience = 20
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 512
sparse = True


nonlinearity = 'prelu' # special name to separate parameters
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset)  # adj,csr_mx,(3327,3327); features,lil_mx,(3327,3703); labels,ndarray,(3327,6); (0,120); (120,620); range[2312,3327],len:1000
features, _ = process.preprocess_features(features)  # matrix, (3327,3703)

nb_nodes = features.shape[0]  # node number, 3327
ft_size = features.shape[1]   # node features dim, 3703
nb_classes = labels.shape[1]  # classes = 6

features = torch.FloatTensor(features[np.newaxis])  # tensor, (1,3327,3703)


'''
------------------------------------------------------------
edge node mask subgraph
------------------------------------------------------------
'''
print("Begin Aug:[{}]".format(args.aug_type))
# if args.aug_type == 'edge':
#
#     aug_features1 = features
#     aug_features2 = features
#
#     aug_adj1 = aug.aug_random_edge(adj, drop_percent=drop_percent) # random drop edges
#     aug_adj2 = aug.aug_random_edge(adj, drop_percent=drop_percent) # random drop edges
#
# elif args.aug_type == 'node':
#
#     aug_features1, aug_adj1 = aug.aug_drop_node(features, adj, drop_percent=drop_percent)
#     aug_features2, aug_adj2 = aug.aug_drop_node(features, adj, drop_percent=drop_percent)
#
# elif args.aug_type == 'subgraph':  # 从original feature和adj中抽取subgraph aug_fts and aug_adjs
#     # features,tensor,(1,3327,3703); adj,csr_mx,(3327,3327); drop_p,0.1 ->数据增强后的feature和adj, aug_ft1,tensor,(1,2120,3703), aug_adj1,csr_mx,(2120,2120);
#     aug_features1, aug_adj1 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)
#     aug_features2, aug_adj2 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)  # random sampled subgraph_node_list不同, -> aug_feature不同, aug_adj不同.
#
# elif args.aug_type == 'mask':
#
#     aug_features1 = aug.aug_random_mask(features,  drop_percent=drop_percent)
#     aug_features2 = aug.aug_random_mask(features,  drop_percent=drop_percent)
#
#     aug_adj1 = adj
#     aug_adj2 = adj
#
# else:
#     assert False



'''
------------------------------------------------------------
'''

# adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))  # 原始adj matrix做归一化normalize, coo_matrix, (3327,3327)
# aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))  # aug_adj1做归一化, coo_matrix, (2120,2120)
# aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))  # aug_adj2做归一化, coo_matrix(2120,2120)

# if sparse:
#     sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)  # tensor sparse matrix, (3327,3327)
#     sp_aug_adj1 = process.sparse_mx_to_torch_sparse_tensor(aug_adj1)  # (2120,2120)
#     sp_aug_adj2 = process.sparse_mx_to_torch_sparse_tensor(aug_adj2)  # (2120,2120)
#
# else:
#     adj = (adj + sp.eye(adj.shape[0])).todense()
#     aug_adj1 = (aug_adj1 + sp.eye(aug_adj1.shape[0])).todense()
#     aug_adj2 = (aug_adj2 + sp.eye(aug_adj2.shape[0])).todense()


'''
------------------------------------------------------------
mask
------------------------------------------------------------
'''

'''
------------------------------------------------------------
'''
# if not sparse:
#     adj = torch.FloatTensor(adj[np.newaxis])
#     aug_adj1 = torch.FloatTensor(aug_adj1[np.newaxis])
#     aug_adj2 = torch.FloatTensor(aug_adj2[np.newaxis])


labels = torch.FloatTensor(labels[np.newaxis])  # tensor, (1,3327,6)
idx_train = torch.LongTensor(idx_train)  # tensor,(120,)
idx_val = torch.LongTensor(idx_val)  # (500,)
idx_test = torch.LongTensor(idx_test)  # (1000,)

model = DGI(ft_size, hid_units, nonlinearity)  # 3703; 512; 'prelu'
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

# if torch.cuda.is_available():
#     print('Using CUDA')
#     model.cuda()
#     features = features.cuda()
#     aug_features1 = aug_features1.cuda()
#     aug_features2 = aug_features2.cuda()
#     if sparse:
#         sp_adj = sp_adj.cuda()
#         sp_aug_adj1 = sp_aug_adj1.cuda()
#         sp_aug_adj2 = sp_aug_adj2.cuda()
#     else:
#         adj = adj.cuda()
#         aug_adj1 = aug_adj1.cuda()
#         aug_adj2 = aug_adj2.cuda()
#
#     labels = labels.cuda()
#     idx_train = idx_train.cuda()
#     idx_val = idx_val.cuda()
#     idx_test = idx_test.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

for epoch in range(nb_epochs):
    model.train()
    if args.aug_type == 'subgraph':  # 从original feature和adj中抽取subgraph aug_fts and aug_adjs
    # features,tensor,(1,3327,3703); adj,csr_mx,(3327,3327); drop_p,0.1 ->数据增强后的feature和adj, aug_ft1,tensor,(1,2120,3703), aug_adj1,csr_mx,(2120,2120);
        aug_features1, aug_adj1 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)
        aug_features2, aug_adj2 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)  # random sampled subgraph_node_list不同, -> aug_feature不同, aug_adj不同.

    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))  # 原始adj matrix做归一化normalize, coo_matrix, (3327,3327)
    aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))  # aug_adj1做归一化, coo_matrix, (2120,2120)
    aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))  # aug_adj2做归一化, coo_matrix(2120,2120)

    if sparse:
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)  # tensor sparse matrix, (3327,3327)
        sp_aug_adj1 = process.sparse_mx_to_torch_sparse_tensor(aug_adj1)  # (2120,2120)
        sp_aug_adj2 = process.sparse_mx_to_torch_sparse_tensor(aug_adj2)  # (2120,2120)


    optimiser.zero_grad()

    idx = np.random.permutation(nb_nodes)  # ndarray, (3327,)
    shuf_fts = features[:, idx, :]  # shuffle features, (1,3327,3703)

    lbl_1 = torch.ones(batch_size, nb_nodes)  # labels for aug_1, (1,3327)
    lbl_2 = torch.zeros(batch_size, nb_nodes)  # (1,3327)
    lbl = torch.cat((lbl_1, lbl_2), 1)  # (1,6654)

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()
    # 基于data augmentation生成关于original features和shuffled features的鉴别向量 ret
    logits = model(features, shuf_fts, aug_features1, aug_features2,  # features,tensor,(1,3327,3703); # shuf_fts, (1,3327,3703); aug_ft1,tensor,(1,2120,3703)
                   sp_adj if sparse else adj,            # tensor sparse matrix, (3327,3327)
                   sp_aug_adj1 if sparse else aug_adj1,  # (2120,2120); aug_adj1,csr_mx,(2120,2120)
                   sp_aug_adj2 if sparse else aug_adj2,
                   sparse, None, None, None, aug_type=aug_type)  # True; 'subgraph'
    # logits, (1,6654)
    loss = b_xent(logits, lbl)  # 在augmentation前提下，discriminater学习区分features和shuffle_features. 0.6931
    print('Loss:[{:.4f}]'.format(loss.item()))

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), args.save_name)
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()

print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load(args.save_name))

embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)  # 返回的是gcn向量, (1,3327,512)
'''
我用的话只需要到这就行了
'''
train_embs = embeds[0, idx_train]
val_embs = embeds[0, idx_val]
test_embs = embeds[0, idx_test]

train_lbls = torch.argmax(labels[0, idx_train], dim=1)
val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_lbls = torch.argmax(labels[0, idx_test], dim=1)

tot = torch.zeros(1)
tot = tot.cuda()

accs = []

for _ in range(50):
    log = LogReg(hid_units, nb_classes)  # regression模型
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    log.cuda()

    pat_steps = 0
    best_acc = torch.zeros(1)
    best_acc = best_acc.cuda()
    for _ in range(100):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)
        
        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    accs.append(acc * 100)
    print('acc:[{:.4f}]'.format(acc))
    tot += acc

print('-' * 100)
print('Average accuracy:[{:.4f}]'.format(tot.item() / 50))
accs = torch.stack(accs)
print('Mean:[{:.4f}]'.format(accs.mean().item()))
print('Std :[{:.4f}]'.format(accs.std().item()))
print('-' * 100)



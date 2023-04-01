# -*- coding: utf-8 -*-
# @Time : 2023/3/21 20:32
# @Author : yysgz
# @File : HAN_torch.py
# @Project : HAN_torch
# @Description : 20230321，用pytorch实现HAN model，代替原有的tensorflow version。
import gc

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import argparse

import json
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from data_process.data_process import adj_to_bias
from layers.loss import HAN_Loss
from models.HeteGAT_multi import HeteGAT_multi
from utils.Evaluate import evaluate

# set parameters
def args_register():
    parser = argparse.ArgumentParser()  # 创建参数对象
    # 添加参数
    parser.add_argument('--nb_epochs', default=100, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=100, type=int, help='minibatch size')
    parser.add_argument('--patience', default=20, type=int, help='early stopping')
    parser.add_argument('--lr', default=0.3, help='learning rate')
    parser.add_argument('--l2', default=0.001, help='weight decay')
    parser.add_argument('--hid_units', default=[8], help='hidden dimension')
    parser.add_argument('--out_dim', default=3, help='output dimension of representation')
    parser.add_argument('--n_heads', default=[8, 1], help='number of heads used in Heterogeneous GAT')
    parser.add_argument('--validation_percent', default=0.2, type=float, help='percentage of validation nodes')
    parser.add_argument('--cluster_type', default='kmeans', help='types of clustering algorithms')
    # attention
    parser.add_argument('--residual', default=False, type=bool, help='whether use residual connection for attention')
    parser.add_argument('--mlp_attn_size', default=128, help='the attention size of mlp')
    parser.add_argument('--attn_drop', default=0.0, help='the drop probability of attention')
    parser.add_argument('--ffd_drop', default=0.0, help='the ffd_drop')

    parser.add_argument('--data_path', default='./data', type=str, help='data path')
    # parser.add_argument('--acm_filepath', default='./data/ACM3025.mat', help='acm datapath')

    # 解析参数
    args = parser.parse_args(args=[])

    return args

# create mask
def sample_mask(idx, len):
    mask = np.zeros(len)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)

# load acm data
import scipy.io as sio
def load_data_acm3025(path=None):
    data = sio.loadmat(path)
    '''
    全是ndarray
    PTP: (3025, 3025)，全是1
    PLP: (3025, 3025)，有0有1，有些向量时相同的
    PAP: (3025, 3025)，对角线全是1，其他元素基本是0，很少是1.
    feature: (3025, 1870)，由0、1组成。
    label: (3025, 3),就3列，1061、965、999
    train_idx: (1, 600)，随机抽的索引，范围在0-2225之间
    val_idx: (1, 300)，200-2325之间随机抽取的索引
    test_idx: (1, 2125)，300-3024之间随机抽取的索引
    '''
    labels, features = data['label'], data['feature'].astype(float)  # (3025, 3); (3025, 1870)
    # 为了使用multi_head attention, 将feature dimension transfer into 1864
    features = features[:, :1864]
    # 将labels 3列 转化为 1列
    if labels.shape[0] > 1:
        y = np.argmax(labels, axis=1)

    nb_fea = features.shape[0]  # 3025

    train_idx = data['test_idx']
    val_idx = data['val_idx']
    test_idx = data['train_idx']

    # 邻接矩阵；特征矩阵
    adjs_list = [data['PAP'] - np.eye(nb_fea), data['PLP'] - np.eye(nb_fea)]  # (3025, 3025)
    feas_list = [features, features, features]  # features list:3, (3025, 1864)
    feas_list = [torch.from_numpy(fea) for fea in feas_list]

    train_mask = sample_mask(train_idx, y.shape[0])  # # 3025长度的bool list，train_idx位置为True
    val_mask = sample_mask(val_idx, y.shape[0])
    test_mask = sample_mask(test_idx, y.shape[0])

    # extract y_train, y_val, y_test
    y_train = y[train_mask]  # (2125,)
    y_val = y[val_mask]  # (300,)
    y_test = y[test_mask]  # (600,)

    print('y_train: {}, y_val: {}, y_test: {}, train_idx: {}, val_idx:{}, test_idx: {}'.format(y_train.shape,
                                                                                               y_val.shape,
                                                                                               y_test.shape,
                                                                                               train_idx.shape,
                                                                                               val_idx.shape,
                                                                                               test_idx.shape))

    return adjs_list, feas_list, y, y_train, y_val, y_test, train_mask, val_mask, test_mask

if __name__=='__main__':

    # define args
    args = args_register()
    print('batch size: ', args.batch_size)
    print('nb_epochs: ', args.nb_epochs)

    with open(args.data_path + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)  # __dict__将模型参数保存成字典形式；indent缩进打印

    # load acm data
    acm_filepath = args.data_path + '/ACM3025.mat'
    adjs_list, feas_list, y, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_acm3025(path=acm_filepath)

    nb_nodes = feas_list[0].shape[0]  # 3025
    ft_size = feas_list[0].shape[1]  # 1864
    nb_classes = len(np.unique(y))  # 3

    # 计算偏差矩阵
    biases_list = [torch.from_numpy(adj_to_bias(adj, nb_nodes, nhood=1)) for adj in adjs_list]  # list:2, (3025,3025)
    # 10 params, 需要用到feature_list和biases_list shape去创建 multi-head attention
    model = HeteGAT_multi(inputs_list=feas_list, nb_classes=nb_classes, nb_nodes=nb_nodes, attn_drop=0.5,
                          ffd_drop=0.0, biases_mat_list=biases_list, hid_units=args.hid_units, n_heads=args.n_heads,
                          activation=nn.ELU(), residual=args.residual)
    model.to(device)
    print(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.l2)

    y_train = torch.from_numpy(y_train).long().to(device)
    y_val = torch.from_numpy(y_val).long().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)

    train_idx = np.where(train_mask == 1)[0]
    val_idx = np.where(val_mask == 1)[0]
    test_idx = np.where(test_mask == 1)[0]
    train_idx = torch.from_numpy(train_idx).to(device)
    val_idx = torch.from_numpy(val_idx).to(device)
    test_idx = torch.from_numpy(test_idx).to(device)

    train_num_samples = len(y_train)
    val_num_samples = len(y_val)
    test_num_samples = len(y_test)

    print('训练节点个数: ', train_num_samples)
    print('验证节点个数: ', val_num_samples)
    print('测试节点个数: ', test_num_samples)

    '''---------------------------training-------------------------'''
    best_val_nmi = 1e-9
    best_epoch = 0
    min_loss = 0
    wait = 0
    # all_val_nmi = []  # record validation nmi of all epochs before early stop

    train_loss_history = []
    train_acc_history = []
    val_loss_history =[]
    val_acc_history =[]

    for epoch in range(args.nb_epochs):  # 100
        model.train()

        batch_train_loss = []
        batch_train_acc = []
        batch_val_loss = []
        batch_val_acc = []

        num_batches = int(train_num_samples / args.batch_size)
        for batch in range(num_batches):
            correct = 0
            i_start = batch * args.batch_size
            i_end = min((batch+1) * args.batch_size, train_num_samples)

            batch_nodes = train_idx[i_start: i_end]
            batch_labels = y_train[i_start: i_end]

            optimizer.zero_grad()

            outputs = model(feas_list, batch_nodes)  # (100, 64)
            _, pred = torch.max(outputs.data, 1)  # (100,)

            loss = loss_fn(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            correct += torch.sum(pred == batch_labels).to(torch.float32)
            tran_acc = correct / args.batch_size
            batch_train_loss.append(loss.item())
            batch_train_acc.append(tran_acc)

        train_loss = np.mean(batch_train_loss)
        train_acc = np.mean(batch_train_acc)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        '''-----------------validation--------------------'''
        model.eval()
        num_batches = int(val_num_samples / args.batch_size)
        for batch in range(num_batches):
            correct = 0
            i_start = batch * args.batch_size
            i_end = min((batch+1) * args.batch_size, val_num_samples)

            batch_nodes = val_idx[i_start: i_end]
            batch_labels = y_val[i_start: i_end]

            outputs = model(feas_list, batch_nodes)
            _, pred = torch.max(outputs.data, 1)

            loss = loss_fn(outputs, batch_labels)

            correct += torch.sum(pred == batch_labels).to(torch.float32)
            val_acc = correct / args.batch_size
            batch_val_loss.append(loss.item())
            batch_val_acc.append(val_acc)
        val_loss = np.mean(batch_val_loss)
        val_acc = np.mean(batch_val_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print("epoch: {:03d}, loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, train_loss, train_acc, val_loss, val_acc))

        # early_stop
        if val_acc > best_val_nmi:
            best_val_nmi = val_acc
            best_epoch = epoch
            wait = 0
            # save model
            model_path = args.data_path + '/result'
            if (epoch == 0) and (not os.path.isdir(model_path)):
                os.mkdir(model_path)
            p = model_path + '/best_model.pt'
            torch.save(model.state_dict(), p)
            print('Best model was at epoch ', str(best_epoch))
        else:
            wait += 1
        if wait >= args.patience:
            print('Early stop at epoch: ', best_epoch, '! val_loss: ', val_loss, ', Best acc: ', best_val_nmi)
            break
    # save the train_loss during training
    np.save(args.data_path + '/result/train_loss.npy', np.asarray(train_loss_history))
    # save the val_acc during training
    np.save(args.data_path + '/result/val_acc.npy', np.asarray(val_acc_history))

    '''--------------loading best model--------------'''
    best_model_path = args.data_path + '/result/best_model.pt'
    model.load_state_dict(torch.load(best_model_path))
    print('best model loaded')
    '''------------------test------------------------'''
    test_loss_history = []
    test_acc_history = []

    num_batches = int(test_num_samples / args.batch_size)
    for batch in range(num_batches):
        correct = 0
        i_start = batch * args.batch_size
        i_end = min((batch + 1) * args.batch_size, test_num_samples)

        batch_nodes = test_idx[i_start: i_end]
        batch_labels = y_test[i_start: i_end]

        outputs = model(feas_list, batch_nodes)
        _, pred = torch.max(outputs.data, 1)

        loss = loss_fn(outputs, batch_labels)

        correct += torch.sum(pred == batch_labels).to(torch.float32)
        test_acc = correct / args.batch_size
        print(test_acc)
        test_loss_history.append(loss.item())
        test_acc_history.append(test_acc)

    test_loss = np.mean(test_loss_history)
    test_acc = np.mean(test_acc_history)
    print('test_loss: ', test_loss, ', test_acc: ', test_acc)
    #
    print('model test is done')




















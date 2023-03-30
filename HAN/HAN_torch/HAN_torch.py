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
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

from data_process.data_process import adj_to_bias
from layers.loss import HAN_Loss
from models.HeteGAT_multi import HeteGAT_multi
from utils.Evaluate import evaluate

# set parameters
def args_register():
    parser = argparse.ArgumentParser()  # 创建参数对象
    # 添加参数
    parser.add_argument('--nb_epochs', default=100, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=50, type=int, help='minibatch size')
    parser.add_argument('--patience', default=10, type=int, help='early stopping')
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
    labels, features = data['label'], data['feature'].astype(float)
    # # 将labels 3列 转化为 1列
    # if labels.shape[0] > 1:
    #     y = np.argmax(labels, axis=1)

    nb_fea = features.shape[0]
    adjs_list = [data['PAP'] - np.eye(nb_fea), data['PLP'] - np.eye(nb_fea)]  # (3025, 3025)
    feas_list = [features, features, features]  # features: (3025, 1870)

    y = labels
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    train_mask = sample_mask(train_idx, y.shape[0])  # # 3025长度的bool list，train_idx位置为True
    val_mask = sample_mask(val_idx, y.shape[0])
    test_mask = sample_mask(test_idx, y.shape[0])

    # extract y_train, y_val, y_test
    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y_train[train_mask, :] = y[train_mask, :]
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]

    print('y_train: {}, y_val: {}, y_test: {}, train_idx: {}, val_idx:{}, test_idx: {}'.format(y_train.shape,
                                                                                               y_val.shape,
                                                                                               y_test.shape,
                                                                                               train_idx.shape,
                                                                                               val_idx.shape,
                                                                                               test_idx.shape))

    return adjs_list, feas_list, y_train, y_val, y_test, train_mask, val_mask, test_mask

if __name__=='__main__':

    # define args
    args = args_register()
    print('batch size: ', args.batch_size)
    print('nb_epochs: ', args.nb_epochs)

    with open(args.data_path + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)  # __dict__将模型参数保存成字典形式；indent缩进打印

    acm_filepath = args.data_path + '/ACM3025.mat'
    # load acm data
    adjs_list, feas_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_acm3025(path=acm_filepath)

    nb_nodes = feas_list[0].shape[0]  # 3025
    ft_size = feas_list[0].shape[1]  # 1870
    # nb_classes = len(np.unique(y_train))
    nb_classes = y_train.shape[1]
    activation = nn.ReLU

    biases_list = [adj_to_bias(adj, nb_nodes, nhood=1) for adj in adjs_list]
    # 7 params
    model = HeteGAT_multi(nb_classes, args.hid_units, args.n_heads, args.residual,
                          mlp_attn_size=args.mlp_attn_size, attn_drop=args.attn_drop, ffd_drop=args.ffd_drop)
    print(model.parameters())
    loss_fn = HAN_Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    best_val_nmi = 1e-9
    best_epoch = 0
    wait = 0
    all_val_nmi = []  # record validation nmi of all epochs before early stop

    print('---------------------training-------------------------------')
    for epoch in range(args.nb_epochs):  # 100
        model.train()

        losses = []
        total_loss = 0.0

        train_num_samples, val_num_samples, test_num_samples = y_train.size(0), y_val.size(0), y_test.size(0)
        num_batches = int(train_num_samples / args.batch_size) + 1
        for batch in range(num_batches):
            i_start = batch * args.batch_size
            i_end = min((batch+1) * args.batch_size, train_num_samples)
            for g in zip(feas_list, biases_list):  # graph表示PAP、PLP
                fea_mx = feas_list[g]
                bias_mx = biases_list[g]

                batch_feature = fea_mx[train_mask][i_start:i_end]
                batch_bias = bias_mx[train_mask][i_start:i_end]

                batch_labels = y_train[i_start:i_end]

                # sampling neighbors of batch nodes
                # adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=[-1, -1],
                #                              batch_size=args.batch_size)

                optimizer.zero_grad()  # 将参数置0

                pred = model(batch_feature, batch_bias, residual=args.residual)

                batch_loss = loss_fn(pred, batch_labels)
                losses.append(batch_loss.item())
                total_loss += batch_loss.item()

                del pred
                gc.collect()

                batch_loss.backward()
                optimizer.step()

                del batch_loss
                gc.collect()

        # print loss
        total_loss /= (num_batches)
        message = 'epoch {}/{}. average loss: {:.4f}'.format(epoch, args.nb_epochs, total_loss)

        print('-------------------------validation-----------------------------')
        model.eval()

        val_pred = torch.FloatTensor([])

        num_batches = int(val_num_samples / args.batch_size) + 1
        for batch in range(num_batches):

            # split batch
            i_start = batch * args.batch_size
            i_end = min((batch+1) * args.batch_size, val_num_samples)

            batch_pred = torch.FloatTensor([])
            for g in zip(feas_list, biases_list):
                val_feature = feas_list[g][val_mask][i_start:i_end]
                val_bias = biases_list[g][val_mask][i_start:i_end]

                pre = model(val_feature, val_bias)
                batch_pred += pre
            batch_pred /= min(feas_list.shape[0], biases_list.shape[0])

            val_pred = torch.cat([val_pred, batch_pred], dim=1)

        validation_nmi = evaluate(val_pred,
                                  y_val,
                                  epoch=epoch,
                                  is_validation=True,
                                  cluster_type=args.cluster_type)

        all_val_nmi.append(validation_nmi)

        # judge early stop
        if validation_nmi > best_val_nmi:
            best_val_nmi = validation_nmi
            best_epoch = epoch
            wait = 0
            # save model
            model_path = args.data_path + '/models'
            if (epoch==0) & (not os.path.isdir(model_path)):
                os.mkdir(model_path)
            p = model_path + '/best.pt'
            torch.save(model.state_dict(), p)
            print('Best model was at epoch ', str(best_epoch))
        else:
            wait += 1
        if wait >= args.patience:
            print('Early stopping at epoch ', str(epoch))
            print('Best model was at epoch ', str(best_epoch))
            break

    # save all validation nmi
    np.save(args.data_path + '/all_val_nmi.npy', np.asarray(all_val_nmi))

    # load the best model of the current block
    best_model_path = args.data_path + '/models/best.pt'
    model.load_state_dict(torch.load(best_model_path))
    print('Best model loaded')

    print('-----------------testing------------------------------')
    model.eval()

    test_pred = torch.FloatTensor([])

    num_batches = int(test_num_samples / args.batch_size) + 1

    for batch in range(num_batches):

        # split batch
        i_start = args.batch_size * batch
        i_end = min((batch+1) * args.batch_size, test_num_samples)

        batch_pred = torch.FloatTensor([])
        for g in zip(feas_list, biases_list):
            test_feature = feas_list[g][test_mask][i_start:i_end]
            test_bias = biases_list[g][test_mask][i_start:i_end]

            pre = model(test_feature, test_bias)
            batch_pred += pre
        batch_pred /= min(feas_list.shape[0], biases_list.shape[0])

        test_pred = torch.cat([test_pred, batch_pred], dim=1)

    test_nmi = evaluate(test_pred,
                        y_test,
                        epoch=-1,
                        is_validation=False,
                        cluster_type=args.cluster_type)

    print('model test is done')




















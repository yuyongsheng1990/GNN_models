# -*- coding: utf-8 -*-
# @Time : 2022/12/4 15:27
# @Author : yysgz
# @File : kmeans_model.py
# @Project : process.py
# @Description :
import os
import numpy as np

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

import pickle  # 把训练好的模型存储起来

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import roc_curve, f1_score
from sklearn import manifold  # 一种非线性降维的手段
from sklearn.model_selection import train_test_split

def my_Kmeans(x, y, k=3, time=10, return_NMI=False):

    x = np.array(x)
    x = np.squeeze(x)  # (2125, 64)
    y = np.array(y)  # (2125, 3)

    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)  # (2125, )
    print('xx: {}, yy: {}'.format(x.shape, y.shape))

    # estimator = KMeans(n_clusters=k)
    # ARI_list = []  # adjusted_rand_score(
    # NMI_list = []
    # if time:
    #     for i in range(time):
    #         estimator.fit(x, y)
    #         y_pred = estimator.predict(x)
    #         score = normalized_mutual_info_score(y, y_pred)
    #         NMI_list.append(score)
    #         s2 = adjusted_rand_score(y, y_pred)
    #         ARI_list.append(s2)
    #         print('time {}: NMI: {:.4f}, ARI: {:.4f}'.format(i, score, s2))
    #     # print('NMI_list: {}'.format(NMI_list))
    #     score = sum(NMI_list) / len(NMI_list)
    #     s2 = sum(ARI_list) / len(ARI_list)
    #     print('NMI (10 avg): {:.4f} , ARI (10avg): {:.4f}'.format(score, s2))

    ARI_list = []  # adjusted_rand_score(
    NMI_list = []
    if time:
        # print('KMeans exps {}次 æ±~B平å~]~G '.format(time))
        for i in range(time):
            estimator = KMeans(n_clusters=k, random_state=0).fit(x)
            y_pred = estimator.labels_
            score = normalized_mutual_info_score(y, y_pred)
            NMI_list.append(score)
            s2 = adjusted_rand_score(y, y_pred)
            ARI_list.append(s2)
        # print('NMI_list: {}'.format(NMI_list))
        score = sum(NMI_list) / len(NMI_list)
        s2 = sum(ARI_list) / len(ARI_list)
        print('NMI (10 avg): {:.4f} , ARI (10avg): {:.4f}'.format(score, s2))
    else:
        # estimator = KMeans(n_clusters=k, random_state=0).fit(x)
        estimator.fit(x, y)
        y_pred = estimator.predict(x)
        score = normalized_mutual_info_score(y, y_pred)
        print("NMI on all label data: {:.5f}".format(score))
    if return_NMI:
        return score, s2
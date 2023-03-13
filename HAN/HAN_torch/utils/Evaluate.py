# -*- coding: utf-8 -*-
# @Time : 2023/3/22 17:09
# @Author : yysgz
# @File : Evaluate.py
# @Project : HAN_torch
# @Description :

import numpy as np
from sklearn.cluster import KMeans

from utils.clustering import run_kmeans

def evaluate(extract_features, extract_labels, indices, epoch, num_isolated_nodes, save_path,
             is_validation=True, cluster_type='kmeans'):
    message = ''
    message += '\nEpoch '
    message += str(epoch)
    message += '\n'

    # with isolated nodes
    if cluster_type == 'kmeans':
        n_tweets, n_classes, nmi, ami, ari = run_kmeans(extract_features, extract_labels, indices)
    elif cluster_type == 'dbscan':
        pass

    if is_validation:
        mode = 'validation'
    else:
        mode = 'test'
    message += '\tNumber of ' + mode + ' tweets: '
    message += str(n_tweets)
    message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
    message += str(n_classes)
    message += '\n\t' + mode + ' NMI: '
    message += str(nmi)
    message += '\n\t' + mode + ' AMi: '
    message += str(ami)
    message += '\n\t' + mode + ' ARI: '
    message += str(ari)
    if cluster_type == 'dbscan':
        message += '\n\t' + mode + ' best_eps: '
        message += '\n\t' + mode + ' best_min_Pts: '

    if num_isolated_nodes != 0:
        # without isolated nodes
        message += '\n\tWithout isolated nodes:'
        n_tweets, n_classes, nmi, ami, ari = run_kmeans(extract_features, extract_labels, indices,
                                                        save_path + '/isolated_nodes.pt')
        message += '\tNumber of ' + mode + 'tweets: '
        message += str(n_tweets)
        message += '\n\tNumber of classes covered by ' + mode + ' tweets'
        message += str(n_classes)
        message += '\n\t' + mode + 'NMI: '
        message += str(nmi)
        message += '\n\t' + mode + 'AMI: '
        message += str(ami)
        message += '\n\t' + mode + 'ARI: '
        message += str(ari)
    message += '\n'

    with open(save_path + '/evaluate.txt', 'a') as f:
        f.write(message)
    print(message)

    np.save(save_path + '/%s_metric.npy' % mode, np.asarray([nmi, ami, ari]))
    if is_validation:
        return nmi
    else:
        pass
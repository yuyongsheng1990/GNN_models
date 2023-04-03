import torch
import copy
import random
import pdb
import scipy.sparse as sp
import numpy as np

def main():
    pass


def aug_random_mask(input_feature, drop_percent=0.2):
    
    node_num = input_feature.shape[1]  # attribute masking
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0][0])
    for j in mask_idx:
        aug_feature[0][j] = zeros
    return aug_feature


def aug_random_edge(input_adj, drop_percent=0.2):

    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()

    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))

    single_index_list = []
    for i in list(index_list):
        single_index_list.append(i)
        index_list.remove((i[1], i[0]))
    
    
    edge_num = int(len(row_idx) / 2)      # 9228 / 2
    add_drop_num = int(edge_num * percent / 2) 
    aug_adj = copy.deepcopy(input_adj.todense().tolist())

    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)

    
    for i in drop_idx:
        aug_adj[single_index_list[i][0]][single_index_list[i][1]] = 0
        aug_adj[single_index_list[i][1]][single_index_list[i][0]] = 0
    
    '''
    above finish drop edges
    '''
    node_num = input_adj.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(l, add_drop_num)

    for i in add_list:
        
        aug_adj[i[0]][i[1]] = 1
        aug_adj[i[1]][i[0]] = 1
    
    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csr_matrix(aug_adj)
    return aug_adj


def aug_drop_node(input_fea, input_adj, drop_percent=0.2):

    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)

    node_num = input_fea.shape[0]
    drop_num = int(node_num * drop_percent)    # number of drop nodes
    all_node_list = [i for i in range(node_num)]

    drop_node_list = sorted(random.sample(all_node_list, drop_num))

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj


def aug_subgraph(input_fea, input_adj, drop_percent=0.2):  # features,tensor,(1,3327,3703); adj,csr_mx,(3327,3327); drop_p,0.1
    
    input_adj = torch.tensor(input_adj.todense().tolist())  # tensor, (3327,3327)
    input_fea = input_fea.squeeze(0)  # tensor, (3327,3703)
    node_num = input_fea.shape[0]  # 3327

    all_node_list = [i for i in range(node_num)]  # list: 3327, (0,3326)
    s_node_num = int(node_num * (1 - drop_percent))  # 取9成node, int, 2994
    center_node_id = random.randint(0, node_num - 1)  # randint 随机返回一个整数 -> 作为中心节点, 859
    sub_node_id_list = [center_node_id]  # subgraph中心节点id列表
    all_neighbor_list = []

    for i in range(s_node_num - 1):
        
        all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()  # 第i个中心节点的non-zero neighbors, list:2,[1795,605]; 3,[1795,605,859]
        
        all_neighbor_list = list(set(all_neighbor_list))
        new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]  # 去除subgraph中心节点node的non-zero neighbors list. list:2,[1795,605];
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]  # 若non-zero neighbors list非空，则随机取一个作为中心节点, 605
            sub_node_id_list.append(new_node)  # 加入subgraph 中心节点 list中
        else:
            break

    # sub_node_list, 2120, [859,605,1795,1483...], all_neighbor_list, 2120, [1,5,8,10,12...]; sub_node_id_list,2120,[430,2844,670,924...], all_neighbor_list,2120,[1,5,8,10,12...]
    drop_node_list = sorted([i for i in all_node_list if not i in sub_node_id_list])  # list:1207, 不在subgraph 中心节点node list的 drop node list

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)  # (2120,3703), subgraph node对应的features
    aug_input_adj = delete_row_col(input_adj, drop_node_list)  # tensor,（2120, 2120), subgraph node对应的adj_mx

    aug_input_fea = aug_input_fea.unsqueeze(0)  # tensor,(1,2120,3703)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))  # csr_mx, (2120,2120)

    return aug_input_fea, aug_input_adj  # 抽取子图node features和adjs





def delete_row_col(input_matrix, drop_list, only_row=False):

    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out



    



    

     

    







if __name__ == "__main__":
    main()
    

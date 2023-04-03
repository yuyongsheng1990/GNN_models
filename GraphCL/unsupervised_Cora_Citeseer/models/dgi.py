import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator, Discriminator2
import pdb

# 对比学习是为了增强gcn生成embedding的robustness！
class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):  # 3703; 512; 'prelu'
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.disc2 = Discriminator2(n_h)
                # features,tensor,(1,3327,3703); shuf_fts, (1,3327,3703); aug_ft1,tensor,(1,2120,3703); tensor sparse matrix, (3327,3327); aug_adj1,csr_mx,(2120,2120)
    def forward(self, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2, aug_type):
                # (features, shuf_fts, aug_features1, aug_features2, sp_adj, sp_aug_adj1, sp_aug_adj2, sparse, None, None, None, aug_type=aug_type)
        h_0 = self.gcn(seq1, adj, sparse)  # features,(1,3327,3703); sp_adj,(3327,3327); True -> GCN生成向量tensor, (1,3327,512)
        if aug_type == 'edge':

            h_1 = self.gcn(seq1, aug_adj1, sparse)
            h_3 = self.gcn(seq1, aug_adj2, sparse)

        elif aug_type == 'mask':

            h_1 = self.gcn(seq3, adj, sparse)
            h_3 = self.gcn(seq4, adj, sparse)

        elif aug_type == 'node' or aug_type == 'subgraph':

            h_1 = self.gcn(seq3, aug_adj1, sparse)  # aug_ft1, (1,2120,3703); aut_adj1, csr_mx,(2120,2120); True -> GCN利用fts和adjs生成向量, tensor,(1,2120,512)
            h_3 = self.gcn(seq4, aug_adj2, sparse)  # aug_ft2, (1,2120,3703); aut_adj2, csr_mx,(2120,2120); True -> (1,2120,512)
            
        else:
            assert False
            
        c_1 = self.read(h_1, msk)  # readout压缩维度, tensor,(1,512)
        c_1= self.sigm(c_1)  # (1,512)

        c_3 = self.read(h_3, msk)  # tensor, (1,512)
        c_3= self.sigm(c_3)

        h_2 = self.gcn(seq2, adj, sparse)  # shuffled_features,(1,3327,3703); sp_adj,(3327,3327); True -> (1,3327,512)

        ret1 = self.disc(c_1, h_0, h_2, samp_bias1, samp_bias2)  # 基于aug_ft1生成鉴别向量 1,(1,6654) <- aug_1 embedding, (1,512); original embedding(1,3327,512); shuffle original fts,(1,3327,512)
        ret2 = self.disc(c_3, h_0, h_2, samp_bias1, samp_bias2)  # 基于aug_ft2生成鉴别向量 2,(1,6654) <- aug_1 embedding, (1,512); original embedding(1,3327,512); shuffle original fts,(1,3327,512)

        ret = ret1 + ret2  # (1,6654)
        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):  # features,tensor,(1,3327,3703); tensor sparse matrix, (3327,3327); True
        h_1 = self.gcn(seq, adj, sparse)  # -> (1,3327,512)
        c = self.read(h_1, msk)  # -> (1,512)

        return h_1.detach(), c.detach()


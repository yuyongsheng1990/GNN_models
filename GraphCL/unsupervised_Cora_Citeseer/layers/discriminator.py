import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)  # 512; 512

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):  # aug_1 embedding, (1,512); original embedding(1,3327,512); shuffle original fts,(1,3327,512)
        c_x = torch.unsqueeze(c, 1)  # (1,1,512)
        c_x = c_x.expand_as(h_pl)  # 处理增强特征(1,3327,512)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)  # 处理原始特征 tensor, (1,3327)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)  # 处理shuffled原始特征 tensor, (1,3327)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)  # (1,6654)

        return logits


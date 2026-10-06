import torch
import torch.nn as nn

from config import HIDDEN_LAYERS, HIDDEN_WIDTH

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden=HIDDEN_WIDTH, layers=HIDDEN_LAYERS,
                 positive_output=False):
        super().__init__()
        seq = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.Tanh()]
        seq += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*seq)
        self.positive_output = positive_output

    def forward(self, *feats):
        x = torch.cat(feats, dim=1)
        y = self.net(x)
        if self.positive_output:
            y = nn.functional.softplus(y)
        return y

class RANSPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.U_net = MLP(in_dim=2, positive_output=False)
        self.k_net = MLP(in_dim=2, positive_output=True)
        self.eps_net = MLP(in_dim=2, positive_output=True)
        self.dpdz_net = MLP(in_dim=1, positive_output=False)
    def forward(self, r_hat, re_n):
        U = self.U_net(r_hat, re_n)
        k = self.k_net(r_hat, re_n)
        eps = self.eps_net(r_hat, re_n)
        dpdz = self.dpdz_net(re_n)
        return U, k, eps, dpdz

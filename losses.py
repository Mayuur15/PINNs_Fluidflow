import math
import torch
from config import EPS_FLOOR, N_MASS_QUAD, R_PIPE, REWEIGHT_MOMENTUM

def data_loss(model, data):
    r_hat, re_n = data['r_hat'], data['re_n']
    U_pred = model.U_net(r_hat, re_n)
    k_pred = model.k_net(r_hat, re_n)
    eps_pred = model.eps_net(r_hat, re_n)
    L_U = torch.mean((U_pred - data['U'])**2)
    L_k = torch.mean((k_pred - data['k'])**2)
    L_eps = torch.mean((torch.log(eps_pred + EPS_FLOOR)
                         - torch.log(data['eps'] + EPS_FLOOR))**2) 
    dpdz_pred = model.dpdz_net(data['re_n_cases'])
    L_dpdz = torch.mean((dpdz_pred - data['dpdz_cases'])**2)
    return {'data_U': L_U, 'data_k': L_k, 'data_eps': L_eps, 'data_dpdz': L_dpdz}

def bc_loss(model, re_n_cases, R_HAT_MAX=1.0):
    n = re_n_cases.shape[0]
    device = re_n_cases.device
    ones = torch.ones(n, 1, device=device)*R_HAT_MAX
    zeros = torch.zeros(n, 1, device=device, requires_grad=True)
    # no-slip: U(r_hat=1, Re) = 0
    U_wall = model.U_net(ones, re_n_cases)
    L_noslip = torch.mean(U_wall**2)
    U0 = model.U_net(zeros, re_n_cases)
    k0 = model.k_net(zeros, re_n_cases)
    eps0 = model.eps_net(zeros, re_n_cases)
    dU0 = torch.autograd.grad(U0, zeros, torch.ones_like(U0),
                               create_graph=True, retain_graph=True)[0]
    dk0 = torch.autograd.grad(k0, zeros, torch.ones_like(k0),
                               create_graph=True, retain_graph=True)[0]
    deps0 = torch.autograd.grad(eps0, zeros, torch.ones_like(eps0),
                                 create_graph=True, retain_graph=True)[0]
    L_sym = torch.mean(dU0**2) + torch.mean(dk0**2) + torch.mean(deps0**2)
    return {'bc_noslip': L_noslip, 'bc_symmetry': L_sym}
def mass_loss(model, re_n_cases, Q_target,r_hat_max=1.0):
    n_cases = re_n_cases.shape[0]
    device = re_n_cases.device
    quad = torch.linspace(0.0, r_hat_max, N_MASS_QUAD, device=device).view(1, -1).repeat(n_cases, 1)
    re_rep = re_n_cases.repeat(1, N_MASS_QUAD)
    U_flat = model.U_net(quad.reshape(-1, 1), re_rep.reshape(-1, 1))
    U_grid = U_flat.reshape(n_cases, N_MASS_QUAD)
    integrand = U_grid * quad
    Q_model = 2 * math.pi * R_PIPE**2 * torch.trapz(integrand, quad, dim=1).view(-1, 1)
    L_mass = torch.mean(((Q_model - Q_target) / (Q_target + EPS_FLOOR))**2)
    return {'mass': L_mass}
class AdaptiveWeights:
    def __init__(self, names, momentum=REWEIGHT_MOMENTUM):
        self.lam = {n: 1.0 for n in names}
        self.momentum = momentum
    def update(self, raw_losses):
        for n, v in raw_losses.items():
            val = max(v.detach().item(), EPS_FLOOR)
            new_lam = 1.0 / val
            self.lam[n] = self.momentum * self.lam[n] + (1 - self.momentum) * new_lam
    def weighted_sum(self, raw_losses):
        return sum(self.lam[n] * v for n, v in raw_losses.items())

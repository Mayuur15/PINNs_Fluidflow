import sys

import numpy as np
import torch

from config import (PRETRAIN_EPOCHS, PHYSICS_EPOCHS, LR, N_COLLOCATION,
                     REWEIGHT_EVERY, LOG_EVERY, GRAD_CLIP)
from data import load_dataset, normalize_re
from models import RANSPINN
from physics import pde_residuals
from losses import data_loss, bc_loss, mass_loss, AdaptiveWeights

def sample_collocation(re_cases, n_per_case, device,R_HAT_MAX=1.0):
    """Interior points only: r_hat in [0.01, 0.99], avoids the 1/r singularity
    at the centerline; the wall/centerline behavior is covered by bc_loss."""
    n_cases = re_cases.shape[0]
    r_hat = torch.rand(n_cases, n_per_case, device=device) * (R_HAT_MAX - 0.01) + 0.01
    re_rep = re_cases.repeat(1, n_per_case)
    r_hat = r_hat.reshape(-1, 1).clone().requires_grad_(True)
    re_rep = re_rep.reshape(-1, 1)
    return r_hat, re_rep
def train(csv_path, device='cpu', pretrain_epochs=PRETRAIN_EPOCHS,
          physics_epochs =PHYSICS_EPOCHS, verbose=True):
    data = load_dataset(csv_path, device)

    data['re_n'] = normalize_re(data['Re'])
    data['re_n_cases'] = normalize_re(data['Re_cases'])
    scales = {
    'mom':  float(torch.abs(data["dpdz_cases"]).mean()),               
    'kres': float(data['eps'].mean()),                      
    'eres': float((data['eps']**2 / data['k']).mean()),   
    }
    R_HAT_MAX = data['r_hat'].max().item()
    model = RANSPINN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.95)
    for epoch in range(pretrain_epochs):
        opt.zero_grad()
        d = data_loss(model, data)
        loss = d['data_U'] + d['data_k'] + d['data_eps'] + d['data_dpdz']
        loss.backward()
        opt.step()
        sched.step()
        if verbose and epoch % LOG_EVERY == 0:
            print(f"[pretrain {epoch:5d}] "
                  f"U={d['data_U'].item():.3e} k={d['data_k'].item():.3e} "
                  f"eps(log)={d['data_eps'].item():.3e} dpdz={d['data_dpdz'].item():.3e}")     
    names = ['data_U', 'data_k', 'data_eps', 'data_dpdz',
              'bc_noslip', 'bc_symmetry', 'mass',
              'pde_mom', 'pde_k', 'pde_eps']
    aw = AdaptiveWeights(names)
    for epoch in range(physics_epochs):
        opt.zero_grad()
        d = data_loss(model, data)
        b = bc_loss(model, data['re_n_cases'],R_HAT_MAX)
        m = mass_loss(model, data['re_n_cases'], data['Q_cases'], R_HAT_MAX)
        r_col, re_col = sample_collocation(data['re_n_cases'], N_COLLOCATION, device,R_HAT_MAX=R_HAT_MAX)
        res_mom, res_k, res_eps = pde_residuals(model, r_col, re_col)
        p = {
            'pde_mom': torch.mean((res_mom/scales['mom'])**2),
            'pde_k': torch.mean((res_k/scales['kres'])**2),
            'pde_eps': torch.mean((res_eps/scales['eres'])**2),
        }

        raw = {**d, **b, **m, **p}

        if epoch % REWEIGHT_EVERY == 0:
            aw.update(raw)
        loss = aw.weighted_sum(raw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()

        if verbose and epoch % LOG_EVERY == 0:
            terms = " ".join(f"{k}={v.item():.2e}" for k, v in raw.items())
            print(f"[physics {epoch:5d}] total={loss.item():.3e} | {terms}")

    return model, data


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'pipe_rans_pinn_5case_Re10k_20k.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")
    model, data = train(csv_path, device)
    torch.save(model.state_dict(), 'rans_pinn_weightsv2.pt')
    print("done -- saved rans_pinn_weightsv2.pt")

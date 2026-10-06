
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import load_dataset, normalize_re
from models import RANSPINN


def evaluate(weights_path, csv_path, device='cpu', out_prefix='eval'):
    data = load_dataset(csv_path, device)
    data['re_n_cases'] = normalize_re(data['Re_cases'])

    model = RANSPINN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    re_cases = data['Re_cases'].cpu().numpy().flatten()
    dpdz_true = data['dpdz_cases'].cpu().numpy().flatten()
    dpdz_pred = model.dpdz_net(data['re_n_cases']).detach().cpu().numpy().flatten()

    print(f"{'Re':>8} | {'dp/dz CFD':>12} | {'dp/dz pred':>12} | {'rel. err %':>10}")
    for re, t, p in zip(re_cases, dpdz_true, dpdz_pred):
        err = 100 * abs(p - t) / abs(t)
        print(f"{re:8.0f} | {t:12.3f} | {p:12.3f} | {err:10.2f}")

    r_plot = torch.linspace(0, 1, 200, device=device).view(-1, 1)

    for re_val in re_cases:
        re_n = normalize_re(torch.full_like(r_plot, float(re_val)))
        with torch.no_grad():
            U_p = model.U_net(r_plot, re_n).cpu().numpy().flatten()
            k_p = model.k_net(r_plot, re_n).cpu().numpy().flatten()
            eps_p = model.eps_net(r_plot, re_n).cpu().numpy().flatten()

        mask = (data['Re'].cpu().numpy().flatten() == re_val)
        r_data = data['r_hat'].cpu().numpy().flatten()[mask]
        U_data = data['U'].cpu().numpy().flatten()[mask]
        k_data = data['k'].cpu().numpy().flatten()[mask]
        eps_data = data['eps'].cpu().numpy().flatten()[mask]

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        for ax, pred, true, name in zip(
                axes, [U_p, k_p, eps_p], [U_data, k_data, eps_data],
                ['U [m/s]', 'k [m^2/s^2]', 'epsilon [m^2/s^3]']):
            ax.plot(r_plot.cpu().numpy(), pred, label='PINN')
            ax.plot(r_data, true, 'o', ms=3, label='CFD', alpha=0.6)
            ax.set_xlabel('r_hat')
            ax.set_ylabel(name)
            ax.legend(fontsize=8)
        fig.suptitle(f'Re = {int(re_val)}')
        fig.tight_layout()
        fig.savefig(f'{out_prefix}_Re{int(re_val)}.png', dpi=140)
        plt.close(fig)

    print(f"\nsaved profile plots: {out_prefix}_Re<value>.png")


if __name__ == '__main__':
    weights_path = sys.argv[1] if len(sys.argv) > 1 else 'rans_pinn_weights.pt'
    csv_path = sys.argv[2] if len(sys.argv) > 2 else 'pipe_rans_pinn_5case_Re10k_20k.csv'
    evaluate(weights_path, csv_path)

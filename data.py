import numpy as np
import pandas as pd
import torch
from config import RE_MIN, RE_MAX, R_PIPE ,RHO


def normalize_re(re):

    return 2.0 * (re - RE_MIN) / (RE_MAX - RE_MIN) - 1.0
def load_dataset(csv_path, device):
    df = pd.read_csv(csv_path)

    required = {'r_over_R', 'Re', 'U_m_s', 'k_m2_s2', 'epsilon_m2_s3',
                'dp_dz_Pa_m', 'Q_L_s'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    r_hat = torch.tensor(df['r_over_R'].values, dtype=torch.float32, device=device).view(-1, 1)
    re = torch.tensor(df['Re'].values, dtype=torch.float32, device=device).view(-1, 1)
    U = torch.tensor(df['U_m_s'].values/df['U_bulk_m_s'].values, dtype=torch.float32, device=device).view(-1, 1)
    k = torch.tensor(df['k_m2_s2'].values/df['U_bulk_m_s'].values**2, dtype=torch.float32, device=device).view(-1, 1)
    eps = torch.tensor(df['epsilon_m2_s3'].values*R_PIPE/df['U_bulk_m_s'].values**3, dtype=torch.float32, device=device).view(-1, 1)
    re_unique = np.sort(df['Re'].unique())
    dpdz_by_re = df.groupby('Re')['dp_dz_Pa_m'].first()
    Q_by_re = df.groupby('Re')['Q_L_s'].first() 
    u_bulk = df.groupby('Re')['U_bulk_m_s'].first() 
    re_cases = torch.tensor(re_unique, dtype=torch.float32, device=device).view(-1, 1)
    dpdz_cases = torch.tensor([dpdz_by_re[r]*R_PIPE/(RHO*(u_bulk[r]**2)) for r in re_unique],
                               dtype=torch.float32, device=device).view(-1, 1)
    Q_cases = torch.tensor([Q_by_re[r] / 1000.0 for r in re_unique],  # -> m^3/s
                            dtype=torch.float32, device=device).view(-1, 1)

    return {
        'r_hat': r_hat, 'Re': re, 'U': U, 'k': k, 'eps': eps,
        'Re_cases': re_cases, 'dpdz_cases': dpdz_cases, 'Q_cases': Q_cases,
        'u_bulk': u_bulk
    }

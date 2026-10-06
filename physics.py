import torch
from config import RE_MAX, RE_MIN, RHO, MU, C_MU, C1_EPS, C2_EPS, SIGMA_K, SIGMA_EPS, R_PIPE, EPS_FLOOR

def _d(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                create_graph=True, retain_graph=True)[0]
def ReR_from_re_n(re_n):
    
    re = 0.5 * (re_n + 1.0) * (RE_MAX - RE_MIN) + RE_MIN
    ReR = re * 0.5
    return ReR

def pde_residuals(model, r_hat, re_n):
    U = model.U_net(r_hat, re_n)
    k = model.k_net(r_hat, re_n)
    eps = model.eps_net(r_hat, re_n)
    dpdz = model.dpdz_net(re_n)
    ReR = ReR_from_re_n(re_n)
    r = r_hat * R_PIPE

    dU_dr = _d(U, r_hat) 
    dk_dr = _d(k, r_hat)
    deps_dr = _d(eps, r_hat)

    mu_t = C_MU * k**2 / (eps + EPS_FLOOR)

    # momentum 
    flux_U = r * (1/ReR + mu_t) * dU_dr
    dflux_U_dr = _d(flux_U, r_hat)
    res_mom = -dpdz + dflux_U_dr / r

    # k
    flux_k = r * (1/ReR + mu_t / SIGMA_K) * dk_dr
    dflux_k_dr = _d(flux_k, r_hat) 
    res_k = dflux_k_dr / r -  eps + mu_t*dU_dr**2

    # epsilon
    flux_eps = r * (1/ReR + mu_t / SIGMA_EPS) * deps_dr
    dflux_eps_dr = _d(flux_eps, r_hat)
    res_eps = (dflux_eps_dr / r
               + C1_EPS * (eps / (k + EPS_FLOOR)) * mu_t * dU_dr**2
               - C2_EPS * eps**2 / (k + EPS_FLOOR))

    return res_mom, res_k, res_eps

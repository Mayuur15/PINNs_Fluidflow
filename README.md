# RANS-PINN Pipe Flow Surrogate — Phase 1 (forward model)

Implements exactly the architecture you specified: four independent small
tanh-MLPs (`U_net`, `k_net`, `eps_net`, `dpdz_net`), standard k-epsilon closure,
`L = L_data + L_BC + L_PDE + L_mass`, two-stage training (pretrain on data,
then add physics with adaptive per-term reweighting), log-loss for epsilon only.

## Files
- `config.py` — physical constants, k-epsilon constants, hyperparameters
- `data.py` — loads the CSV, builds per-point and per-Re-case tensors
- `models.py` — the four MLPs
- `physics.py` — momentum / k / epsilon PDE residuals via autograd
- `losses.py` — data loss, BC loss (no-slip + symmetry), mass constraint, adaptive weights
- `train.py` — Stage 1 (data-only) then Stage 2 (+ BC/PDE/mass) training loop
- `evaluate.py` — plots predicted vs CFD profiles per Re, prints a dp/dz table

## Run
```bash
python train.py pipe_rans_pinn_5case_Re10k_20k_v2.csv
python evaluate.py rans_pinn_weights.pt pipe_rans_pinn_5case_Re10k_20k_v2.csv
```
Full run is 5000 + 15000 epochs on 5 Re cases — a few minutes on CPU, faster on GPU
if `torch.cuda.is_available()`.

## Two implementation details not explicitly specified, so flagged rather than hidden
1. **Softplus output on `k_net` / `eps_net`** to guarantee k, epsilon ≥ 0. Hidden
   layers are still plain tanh as specified — this only touches the final
   activation. Without it, `mu_t = rho*C_mu*k^2/eps` and the epsilon PDE (which
   divides by k) blow up or go complex the moment training pushes either field
   negative, and the log-loss on epsilon is undefined for eps ≤ 0.
2. **Re input normalization** to `[-1, 1]` (linear map from `[10000, 20000]`).
   Not stated, but feeding raw Re ~ 1e4 into a tanh-MLP alongside r_hat ~ O(1)
   saturates the first layer immediately — this is standard practice, not a
   change to the network structure itself.

Both are implementation necessities to make the stated architecture actually
train, not changes to what you specified (network split, closure, loss terms,
training stages are all untouched).

## What I verified (smoke-tested here, not fully trained)
Ran the complete two-stage pipeline end-to-end on your actual CSV: no NaNs,
losses decrease, `evaluate.py` produces plots and a dp/dz comparison table.
I did **not** run the full 5000+15000 epoch schedule — that's a several-minute
job better run on your machine (ideally with a GPU) than burned here on a
throwaway smoke test. Don't take the numbers you see from a 300-epoch run as
indicative of final accuracy.

## Honest limitation to watch for once you do the full run
5 Re values is a very thin training set for `dpdz_net` and for how well the PDE
residual can regularize between cases — if `L_mass` and `L_PDE` don't converge
to comparably small values as the data terms once physics turns on, that's the
first place to look (increase `N_COLLOCATION`, or check the adaptive weights
aren't letting one term dominate — print `aw.lam` periodically to check).

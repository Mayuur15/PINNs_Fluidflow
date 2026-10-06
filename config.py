
RHO = 998.2          # kg/m^3
MU = 1.003e-3        # Pa.s
R_PIPE = 0.01082     # m
D_PIPE = 2.0 * R_PIPE
C_MU = 0.09
C1_EPS = 1.44
C2_EPS = 1.92
SIGMA_K = 1.0
SIGMA_EPS = 1.3
RE_MIN = 10000.0
RE_MAX = 20000.0

EPS_FLOOR = 1e-8
HIDDEN_LAYERS = 4
HIDDEN_WIDTH = 128
PRETRAIN_EPOCHS = 5000    
PHYSICS_EPOCHS = 15000     
LR = 1e-3
GRAD_CLIP = 5.0 
N_COLLOCATION = 1024       # PDE collocation points per Re case, per step
N_MASS_QUAD = 201          # quadrature points for the flow-rate (L_mass) integral
REWEIGHT_EVERY = 100       # recompute adaptive lambda_i every N steps
REWEIGHT_MOMENTUM = 0.9    # EMA smoothing on the adaptive weights

LOG_EVERY = 500

# Simulation constraints
import numpy as np
np.random.seed(42)

SIM_DAYS = 730
HISTO_DAYS = 365
N_SIMULATIONS = 100
MC_SIMS = 1000

# Replenishment constraints & constants
WRITE_OFF_RATE = 0.01
WRITE_OFF_FREQUENCY = 7

# Stock constraints
LEAD_TIME = 3
BASE_STOCK = 0
DEFAULT_SERVICE_LEVEL = 0.95

# Demand constraints
GAMMA_SHAPE = np.random.uniform(6, 8) # 7
GAMMA_SCALE = np.random.uniform(14, 18) # 16
POISSON_LAMBDA = np.random.uniform(75, 85) # 80
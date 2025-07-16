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
SHAPE_GAMMA_POISSON = np.random.uniform(6, 8) # 7
SCALE_GAMMA_POISSON = np.random.uniform(14, 18) # 16
LAMBDA_GAMMA_POISSON = np.random.uniform(75, 85) # 80

SHAPE_GAMMA_GAMMA_LOW_MEAN = np.random.uniform(6, 8) # 7
SCALE_GAMMA_GAMMA_LOW_MEAN = np.random.uniform(2, 4) # 3
SHAPE_GAMMA_GAMMA_HIGH_MEAN = np.random.uniform(6, 8) # 7
SCALE_GAMMA_GAMMA_HIGH_MEAN = np.random.uniform(28, 30) # 29

SHAPE_GAMMA_LOW_VAR = np.random.uniform(6, 8) # 7
SCALE_GAMMA_LOW_VAR = np.random.uniform(14, 18) # 16

RATE_SPORADIC_HIGH = np.random.uniform(0.005, 0.1)  # 0.05

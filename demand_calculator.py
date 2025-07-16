import numpy as np
import ciw.dists
from dataclasses import dataclass
from typing import Tuple
from config import LEAD_TIME


np.random.seed(42)

@dataclass
class DailyDemandDistribution:
    day: int
    actual_demand: int
    demand_mean: float
    demand_std: float
    forecast_distribution: ciw.dists.Distribution

class DemandCalculator:
    def __init__(self, sim_days: int):
        self.sim_days = sim_days
        self.daily_demand_distribution = []
        self.environment = None

    def set_environment(self, environment):
        self.environment = environment
        self.daily_demand_distribution = self.environment.generate_distribution()

    def get_daily_demand(self, time_period: int) -> int:
        return self.daily_demand_distribution[time_period].actual_demand

    def sample_lead_time_demand(self, time_period, daily_demand_distribution, mc_sims):
        samples = []
        for i in range(mc_sims):
            total_demand = 0
            for j in range(1, LEAD_TIME + 1):
                total_demand += daily_demand_distribution[time_period + j].forecast_distribution.sample()
            samples.append(total_demand)
        return samples

    @staticmethod
    def get_demand_stats(shape: float, scale: float, rate: float) -> Tuple[float, float]:
        weights = [0.9, 0.1]
        mean_gamma = shape * scale
        mean_poisson = rate

        var_gamma = shape * scale ** 2
        var_poisson = rate

        mixture_mean = weights[0] * mean_gamma + weights[1] * mean_poisson

        mixture_var = (
                weights[0] * (var_gamma + (mean_gamma - mixture_mean) ** 2) +
                weights[1] * (var_poisson + (mean_poisson - mixture_mean) ** 2)
        )
        mixture_std = np.sqrt(mixture_var)

        return mixture_mean, mixture_std


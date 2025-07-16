import numpy as np
import ciw.dists
from dataclasses import dataclass
from typing import List
from config import POISSON_LAMBDA, GAMMA_SHAPE, GAMMA_SCALE, MC_SIMS, LEAD_TIME

np.random.seed(42)

@dataclass
class DailyDemandDistribution:
    day: int
    actual_demand: int
    demand_mean: float
    demand_std: float
    forecast_distribution: ciw.dists.Distribution

class DemandGenerator:
    def __init__(self, SIM_DAYS):
        self.SIM_DAYS = SIM_DAYS
        self.daily_demand_distribution: List[DailyDemandDistribution] = self.generate_distribution()

    def generate_distribution(self) -> List[DailyDemandDistribution]:

        daily_demand_distribution = []

        for day in range(self.SIM_DAYS):
            demand_distribution = ciw.dists.MixtureDistribution(
                [
                    ciw.dists.Gamma(shape=GAMMA_SHAPE, scale=GAMMA_SCALE),
                    ciw.dists.Poisson(rate=POISSON_LAMBDA)
                ],
                [0.9, 0.1]
            )
            demand_mean, demand_std = self.get_demand_stats()
            actual_demand = int(demand_distribution.sample())

            daily_demand_distribution.append(
                DailyDemandDistribution(
                    day=day,
                    actual_demand=actual_demand,
                    demand_mean=demand_mean,
                    demand_std=demand_std,
                    forecast_distribution=demand_distribution
                )
            )
        return daily_demand_distribution

    def get_daily_demand(self, time_period: int) -> int: # CHECK UTILITY OF THIS FUNCTION
        return self.daily_demand_distribution[time_period].actual_demand

    def sample_lead_time_demand(self, time_period, mc_sims=MC_SIMS): # CHECK TO REDUCE TIME COMPLEXITY
        samples = []
        for i in range(mc_sims):
            total_demand = 0
            for j in range(1, LEAD_TIME + 1):
                total_demand += self.daily_demand_distribution[time_period + j].forecast_distribution.sample()
            samples.append(total_demand)
        return samples

    # ciw.dists doesn't have a method to get mean and std, so we calculate it manually
    def get_demand_stats(self) -> (float, float):
        weights = [0.9, 0.1]
        mean_gamma = GAMMA_SHAPE * GAMMA_SCALE
        mean_poisson = POISSON_LAMBDA

        var_gamma = GAMMA_SHAPE * GAMMA_SCALE ** 2
        var_poisson = POISSON_LAMBDA

        mixture_mean = weights[0] * mean_gamma + weights[1] * mean_poisson

        mixture_var = (
                weights[0] * (var_gamma + (mean_gamma - mixture_mean) ** 2) +
                weights[1] * (var_poisson + (mean_poisson - mixture_mean) ** 2)
        )
        mixture_std = np.sqrt(mixture_var)

        return mixture_mean, mixture_std

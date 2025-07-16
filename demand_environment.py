import ciw
from abc import abstractmethod, ABC
from typing import List, Tuple
from config import SHAPE_GAMMA_POISSON, LAMBDA_GAMMA_POISSON, SCALE_GAMMA_POISSON, SHAPE_GAMMA_GAMMA_LOW_MEAN, \
    SCALE_GAMMA_GAMMA_LOW_MEAN, SHAPE_GAMMA_GAMMA_HIGH_MEAN, SCALE_GAMMA_GAMMA_HIGH_MEAN, SHAPE_GAMMA_LOW_VAR, \
    SCALE_GAMMA_LOW_VAR, RATE_SPORADIC_HIGH
from demand_calculator import DailyDemandDistribution, DemandCalculator

class Environment(ABC):
    def __init__(self, sim_days: int):
        self.sim_days = sim_days
        self.demand_calculator = DemandCalculator(sim_days)
        self.demand_distribution = self.generate_distribution()

    @property
    def daily_demand_distribution(self) -> List[DailyDemandDistribution]:
        return self.demand_distribution

    @abstractmethod
    def create_distribution(self) -> ciw.dists.Distribution:
        pass

    @abstractmethod
    def get_distribution_params(self) -> Tuple[float, float, float]:
        pass

    def get_demand_stats(self) -> Tuple[float, float]:
        shape, scale, rate = self.get_distribution_params()
        return self.demand_calculator.get_demand_stats(shape, scale, rate)

    def generate_distribution(self) -> List[DailyDemandDistribution]:
        daily_demand_distribution = []

        for day in range(self.sim_days):
            demand_distribution = self.create_distribution()
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

class GammaPoisson(Environment):

    def create_distribution(self) -> ciw.dists.Distribution:
        demand_distribution = ciw.dists.MixtureDistribution(
            [
                ciw.dists.Gamma(shape=SHAPE_GAMMA_POISSON, scale=SCALE_GAMMA_POISSON),
                ciw.dists.Poisson(rate=LAMBDA_GAMMA_POISSON)
            ],
            [0.9, 0.1]
        )
        return demand_distribution

    def get_distribution_params(self) -> Tuple[float, float, float]:
        return SHAPE_GAMMA_POISSON, SCALE_GAMMA_POISSON, LAMBDA_GAMMA_POISSON

class GammaGammaHighVariance(Environment):

    def create_distribution(self) -> ciw.dists.Distribution:
        demand_distribution = ciw.dists.MixtureDistribution(
            [
                ciw.dists.Gamma(shape=SHAPE_GAMMA_GAMMA_LOW_MEAN, scale=SCALE_GAMMA_GAMMA_LOW_MEAN),
                ciw.dists.Gamma(shape=SHAPE_GAMMA_GAMMA_HIGH_MEAN, scale=SCALE_GAMMA_GAMMA_HIGH_MEAN)
            ],
            [0.5, 0.5]
        )
        return demand_distribution

    def get_distribution_params(self) -> Tuple[float, float, float]:
        return SHAPE_GAMMA_GAMMA_LOW_MEAN, SCALE_GAMMA_GAMMA_LOW_MEAN, SHAPE_GAMMA_GAMMA_HIGH_MEAN

class SpikingDemand(Environment):

    def create_distribution(self) -> str:
        demand_distribution = ciw.dists.MixtureDistribution(
            [
                ciw.dists.Deterministic(value=0),
                ciw.dists.Exponential(rate=RATE_SPORADIC_HIGH)
            ],
            [0.95, 0.05]
        )
        return demand_distribution

    def get_distribution_params(self) -> Tuple[float, float, float]:
        return 0.0, 0.0, RATE_SPORADIC_HIGH

class SingleGammaLowVariance(Environment):

    def create_distribution(self) -> ciw.dists.Distribution:
        demand_distribution = ciw.dists.Gamma(shape=SHAPE_GAMMA_LOW_VAR, scale=SCALE_GAMMA_LOW_VAR)
        return demand_distribution

    def get_distribution_params(self) -> Tuple[float, float, float]:
        return SHAPE_GAMMA_LOW_VAR, SCALE_GAMMA_LOW_VAR, 0.0
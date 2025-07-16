import montecarlo_simulator
from agent_environment import MonteCarloAgent, SafetyStockAgent, ForecastAgent, BaseAgent
from demand_environment import GammaPoisson, GammaGammaHighVariance, SingleGammaLowVariance, SpikingDemand
from demand_calculator import DemandCalculator
from config import SIM_DAYS, N_SIMULATIONS
import ciw
ciw.seed(11)

simulation_version = input("Choose a simulation version (0-3):\n"
                         "0: Historical Demand Agent\n"
                         "1: Safety Stock Agent\n" 
                         "2: Forecast Agent\n"
                         "3: Monte Carlo Agent\n")

print(f"Running simulation version: {simulation_version}")
simulation_version = int(simulation_version)

# Create environment first
environment_version = input("Choose demand distribution (0-3):\n"
                         "0: 90/10 Gamma/Poisson\n"
                         "1: 50/50 Gamma(mu=20)/Gamma(mu=200)\n" 
                         "2: Spiking High Demand\n"
                         "3: Gamma\n")

print(f"Running demand distribution: {environment_version}")
environment_version = int(environment_version)

environment_versions = {
    0: GammaPoisson(SIM_DAYS),
    1: GammaGammaHighVariance(SIM_DAYS),
    2: SpikingDemand(SIM_DAYS),
    3: SingleGammaLowVariance(SIM_DAYS),
}
try:
    selected_environment = environment_versions[environment_version]
except:
    raise ValueError("Invalid demand distribution version")

demand_calculator = DemandCalculator(SIM_DAYS)
demand_calculator.set_environment(selected_environment)
daily_demand_distribution = demand_calculator

demand_mean = [d.demand_mean for d in daily_demand_distribution.daily_demand_distribution]
demand_std = [d.demand_std for d in daily_demand_distribution.daily_demand_distribution]

agent_versions = {
    0: BaseAgent(daily_demand_distribution),
    1: SafetyStockAgent(daily_demand_distribution),
    2: ForecastAgent(daily_demand_distribution, demand_mean, demand_std),
    3: MonteCarloAgent(daily_demand_distribution)
}
try:
    selected_agent = agent_versions[simulation_version]
except:
    raise ValueError("Invalid simulation version")

sim = montecarlo_simulator.MonteCarloSimulator(selected_agent, selected_environment)
sim.run_simulation(N_SIMULATIONS, SIM_DAYS)

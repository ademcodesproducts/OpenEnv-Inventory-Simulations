import montecarlo_simulator
from agent_environment import MonteCarloAgent, SafetyStockAgent, ForecastAgent, Agent, BaseAgent
from demand_generator import DemandGenerator
from config import SIM_DAYS, N_SIMULATIONS
import ciw
ciw.seed(11)

simulation_version = input("Choose a simulation version (0-3):\n"
                         "0: Historical Demand Agent\n"
                         "1: Safety Stock Agent\n" 
                         "2: Forecast Agent\n"
                         "3: Monte Carlo Agent\n")

print(f"Running simulation version: {simulation_version}")
version = int(simulation_version)

# THIS SHIT SUPPOSED TO BE HAPPENING DURING SIMULATION
demand_generator = DemandGenerator(SIM_DAYS)
daily_demand_distribution = demand_generator.daily_demand_distribution

demand_mean = [d.demand_mean for d in daily_demand_distribution]
demand_std = [d.demand_std for d in daily_demand_distribution]

agent_versions = {
    0: BaseAgent(daily_demand_distribution),
    1: SafetyStockAgent(daily_demand_distribution),
    2: ForecastAgent(daily_demand_distribution, demand_mean, demand_std),
    3: MonteCarloAgent(daily_demand_distribution)
}
try:
    selected_agent = agent_versions[version]
except:
    raise ValueError("Invalid simulation version")

sim = montecarlo_simulator.MonteCarloSimulator(selected_agent)
sim.run_simulation(N_SIMULATIONS, SIM_DAYS)
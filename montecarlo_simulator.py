import pandas as pd
from config import LEAD_TIME, HISTO_DAYS
from demand_calculator import DemandCalculator
from order_processor import OrderProcessor
from inventory_manager import InventoryManager
from performance_tracker import PerformanceTracker

class MonteCarloSimulator():
    def __init__(self, agent, environment):
        self.agent = agent
        self.environment = environment

    def run_simulation(self, N_SIMULATIONS, SIM_DAYS):

        all_simulation_results = []

        for sim in range(N_SIMULATIONS):
            time_period = 0
            order_processor = OrderProcessor()
            performance_tracker = PerformanceTracker()
            demand_calculator = DemandCalculator(SIM_DAYS)
            demand_calculator.set_environment(self.environment)
            self.agent.daily_demand_distribution = demand_calculator

            inventory_manager = InventoryManager(
                order_processor=order_processor,
                agent=self.agent
            )

            # Generate historical demand data -> consistency with algo 1 & 2
            for day in range(HISTO_DAYS):
                _ = demand_calculator.get_daily_demand(day)

            for day in range(HISTO_DAYS, SIM_DAYS):
                demand_quantity = demand_calculator.get_daily_demand(day)
                base_inventory = inventory_manager.inventory

                inventory_manager.inventory_update(demand_quantity)

                if day < SIM_DAYS - LEAD_TIME: # no more orders after the last lead time
                    inventory_manager.reorder(day)

                inventory_manager.process_deliveries(day)

                fulfilled_demand = min(demand_quantity, base_inventory)
                daily_writeoff = inventory_manager.apply_writeoff(day)

                performance_tracker.daily_performance(
                    demand_quantity=demand_quantity,
                    fulfilled_demand=fulfilled_demand,
                    daily_writeoff=daily_writeoff
                )

                time_period += 1

            print(f"\n Simulation {sim + 1}")
            print("-" * 30)
            sim_summary = performance_tracker.performance_summary()

            for key, value in sim_summary.items():
                    print(f"{key.title()}: {value}")

            all_simulation_results.append(performance_tracker.performance_summary())

        self.generate_overall_report(all_simulation_results)

    def generate_overall_report(self, results):
        df = pd.DataFrame(results)
        summary_stats = df.agg(['mean', 'std']).transpose()
        summary_stats.columns = ['Mean', 'Standard Deviation']
        print("Cummulatige Simulation Results")
        print("---------------------------------------------")
        print(summary_stats.to_string(float_format="%.6f"))
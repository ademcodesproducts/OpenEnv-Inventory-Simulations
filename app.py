import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SIM_DAYS, HISTO_DAYS, LEAD_TIME
from agent_environment import BaseAgent, SafetyStockAgent, ForecastAgent, MonteCarloAgent
from demand_environment import GammaPoisson, GammaGammaHighVariance, SpikingDemand, SingleGammaLowVariance
from demand_calculator import DemandCalculator
from order_processor import OrderProcessor
from inventory_manager import InventoryManager
from performance_tracker import PerformanceTracker

ENV_MAP = {
    "GammaPoisson (90/10 mixture)": GammaPoisson,
    "GammaGamma High Variance (bimodal)": GammaGammaHighVariance,
    "Spiking Demand": SpikingDemand,
    "Single Gamma Low Variance": SingleGammaLowVariance,
}


def run_simulation(agent_name, env_name):
    env_class = ENV_MAP[env_name]
    environment = env_class(SIM_DAYS)

    dc = DemandCalculator(SIM_DAYS)
    dc.set_environment(environment)

    for i in range(SIM_DAYS):
        dc.get_daily_demand(i)

    demand_mean = [d.demand_mean for d in dc.daily_demand_distribution]
    demand_std = [d.demand_std for d in dc.daily_demand_distribution]

    agent_map = {
        "Base (Historical Mean)": BaseAgent(dc),
        "Safety Stock": SafetyStockAgent(dc),
        "Forecast": ForecastAgent(dc, demand_mean, demand_std),
        "Monte Carlo": MonteCarloAgent(dc),
    }
    agent = agent_map[agent_name]

    order_processor = OrderProcessor()
    performance_tracker = PerformanceTracker()
    inventory_manager = InventoryManager(order_processor=order_processor, agent=agent)

    daily_inventory = []
    running_fill_rate = []
    total_demand = 0
    total_fulfilled = 0

    for day in range(HISTO_DAYS, SIM_DAYS):
        demand_qty = dc.get_daily_demand(day)
        base_inv = inventory_manager.inventory

        inventory_manager.inventory_update(demand_qty)
        if day < SIM_DAYS - LEAD_TIME:
            inventory_manager.reorder(day)
        inventory_manager.process_deliveries(day)

        fulfilled = min(demand_qty, base_inv)
        daily_writeoff = inventory_manager.apply_writeoff(day)

        total_demand += demand_qty
        total_fulfilled += fulfilled
        performance_tracker.daily_performance(demand_qty, int(fulfilled), daily_writeoff)

        daily_inventory.append(inventory_manager.inventory)
        running_fill_rate.append(total_fulfilled / total_demand if total_demand > 0 else 0)

    summary = performance_tracker.performance_summary()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    days = list(range(len(daily_inventory)))
    ax1.plot(days, daily_inventory, color="steelblue", linewidth=0.8)
    ax1.set_ylabel("Inventory Level")
    ax1.set_title(f"{agent_name}  |  {env_name}")
    ax2.plot(days, running_fill_rate, color="seagreen", linewidth=0.8)
    ax2.axhline(y=0.95, color="red", linestyle="--", linewidth=0.6, label="95% target")
    ax2.set_ylabel("Cumulative Fill Rate")
    ax2.set_xlabel("Evaluation Day")
    ax2.set_ylim(0, 1)
    ax2.legend()
    plt.tight_layout()

    metrics = (
        f"**Fill Rate:** {summary['fill_rate']:.2%}  \n"
        f"**Stockouts:** {summary['stock_out_count']}  \n"
        f"**Total Lost Sales:** {summary['total_lost_sales']:.0f}  \n"
        f"**Write-offs:** {summary['write_offs']:.0f}  \n"
        f"**Total Demand:** {summary['total_demand']:.0f}"
    )
    return fig, metrics


with gr.Blocks(title="Inventory Simulation") as demo:
    gr.Markdown("# Inventory Optimization: Agent Comparison")
    gr.Markdown("Select an agent and demand environment, then run a 365-day simulation.")
    with gr.Row():
        agent_dd = gr.Dropdown(
            choices=["Base (Historical Mean)", "Safety Stock", "Forecast", "Monte Carlo"],
            value="Safety Stock",
            label="Agent",
        )
        env_dd = gr.Dropdown(
            choices=list(ENV_MAP.keys()),
            value="GammaPoisson (90/10 mixture)",
            label="Demand Environment",
        )
    run_btn = gr.Button("Run Simulation", variant="primary")
    with gr.Row():
        chart = gr.Plot(label="Simulation Results")
        metrics_md = gr.Markdown(label="Metrics")
    run_btn.click(run_simulation, inputs=[agent_dd, env_dd], outputs=[chart, metrics_md])

demo.launch()

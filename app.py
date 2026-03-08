import json
import re

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import InferenceClient

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

DECISION_INTERVAL = 5

LLM_SYSTEM_PROMPT = """You are an expert inventory optimization agent in a stochastic simulation.

Decide the REORDER POINT (ROP) — the inventory threshold that triggers a new order.

RULES:
- Orders arrive LEAD_TIME=3 days after placement
- Every 7 days, 1% of inventory is written off
- Goal: fill rate >= 95% at end of episode

OUTPUT — respond with this exact JSON (no markdown fences):
{
  "subgoals": ["subgoal 1", "subgoal 2"],
  "state_analysis": "2-3 sentence analysis",
  "recovery_plan": "recovery strategy if fill rate < 95%",
  "reorder_point": <number>,
  "confidence": "high|medium|low"
}"""


# ── Shared chart builder ───────────────────────────────────────────────────────

def build_chart(daily_inventory, running_fill_rate, rop_markers, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    days = list(range(len(daily_inventory)))
    ax1.plot(days, daily_inventory, color="steelblue", linewidth=0.8)
    if rop_markers:
        rop_days, rop_vals = zip(*rop_markers)
        ax1.scatter([d - HISTO_DAYS for d in rop_days], rop_vals,
                    color="orange", s=20, zorder=5, label="ROP set")
        ax1.legend(fontsize=8)
    ax1.set_ylabel("Inventory Level")
    ax1.set_title(title)
    ax2.plot(days, running_fill_rate, color="seagreen", linewidth=0.8)
    ax2.axhline(y=0.95, color="red", linestyle="--", linewidth=0.6, label="95% target")
    ax2.set_ylabel("Cumulative Fill Rate")
    ax2.set_xlabel("Evaluation Day")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    return fig


# ── Tab 1: Baseline agents ─────────────────────────────────────────────────────

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
    daily_inventory, running_fill_rate = [], []
    total_demand, total_fulfilled = 0, 0
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
    fig = build_chart(daily_inventory, running_fill_rate, [], f"{agent_name}  |  {env_name}")
    metrics = (
        f"**Fill Rate:** {summary['fill_rate']:.2%}  \n"
        f"**Stockouts:** {summary['stock_out_count']}  \n"
        f"**Total Lost Sales:** {summary['total_lost_sales']:.0f}  \n"
        f"**Write-offs:** {summary['write_offs']:.0f}  \n"
        f"**Total Demand:** {summary['total_demand']:.0f}"
    )
    return fig, metrics


# ── Tab 2: LLM agent (live) ────────────────────────────────────────────────────

def _parse_decision(raw: str, fallback_rop: float) -> dict:
    try:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        match = re.search(r'"reorder_point"\s*:\s*(\d+\.?\d*)', raw)
        return {
            "subgoals": ["parse error"],
            "state_analysis": raw[:150],
            "recovery_plan": "N/A",
            "reorder_point": float(match.group(1)) if match else fallback_rop,
            "confidence": "low",
        }


def run_llm_simulation(env_name, hf_token):
    env_class = ENV_MAP[env_name]
    environment = env_class(SIM_DAYS)
    dc = DemandCalculator(SIM_DAYS)
    dc.set_environment(environment)
    for i in range(SIM_DAYS):
        dc.get_daily_demand(i)

    order_processor = OrderProcessor()
    performance_tracker = PerformanceTracker()
    inventory_manager = InventoryManager(
        order_processor=order_processor,
        agent=BaseAgent(dc),  # placeholder; we override ROP manually
    )

    client = InferenceClient(token=hf_token or None)
    convo_history = []
    memory_bank = []
    current_rop = dc.daily_demand_distribution[HISTO_DAYS].demand_mean * LEAD_TIME
    daily_inventory, running_fill_rate, rop_markers = [], [], []
    total_demand, total_fulfilled = 0, 0
    decision_log = []

    for day in range(HISTO_DAYS, SIM_DAYS):
        demand_qty = dc.get_daily_demand(day)
        base_inv = inventory_manager.inventory

        inventory_manager.inventory_update(demand_qty)

        # Manual reorder using current_rop
        if day < SIM_DAYS - LEAD_TIME and inventory_manager.inventory <= current_rop:
            hist = [dc.daily_demand_distribution[d].actual_demand
                    for d in range(max(0, day - 30), day)]
            mean_d = sum(hist) / len(hist) if hist else current_rop / LEAD_TIME
            qty = max(0, current_rop - inventory_manager.inventory + mean_d * LEAD_TIME)
            if qty > 0:
                order_processor.place_order(day, int(qty))

        inventory_manager.process_deliveries(day)
        fulfilled = min(demand_qty, base_inv)
        daily_writeoff = inventory_manager.apply_writeoff(day)
        total_demand += demand_qty
        total_fulfilled += fulfilled
        performance_tracker.daily_performance(demand_qty, int(fulfilled), daily_writeoff)
        daily_inventory.append(inventory_manager.inventory)
        fr = total_fulfilled / total_demand if total_demand > 0 else 0
        running_fill_rate.append(fr)

        # LLM decision every DECISION_INTERVAL days
        if (day - HISTO_DAYS) % DECISION_INTERVAL == 0 and day < SIM_DAYS - LEAD_TIME:
            hist30 = [dc.daily_demand_distribution[d].actual_demand
                      for d in range(max(0, day - 30), day)]
            snapshot = {
                "day": day, "days_remaining": SIM_DAYS - day,
                "current_inventory": round(inventory_manager.inventory, 1),
                "demand_mean_30d": round(sum(hist30) / len(hist30), 1) if hist30 else 0,
                "fill_rate_so_far": f"{fr*100:.1f}%",
                "recent_stockouts": performance_tracker.stock_out_count,
                "lead_time": LEAD_TIME,
            }
            if memory_bank:
                snapshot["memory"] = memory_bank[-6:]

            user_msg = (
                f"Day {day}/{SIM_DAYS}\n{json.dumps(snapshot, indent=2)}\n\n"
                f"Set reorder_point for the next {DECISION_INTERVAL} days."
            )
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                *convo_history[-6:],
                {"role": "user", "content": user_msg},
            ]
            try:
                resp = client.chat.completions.create(
                    model="Qwen/Qwen2.5-72B-Instruct",
                    messages=messages,
                    max_tokens=600,
                )
                raw = resp.choices[0].message.content
                decision = _parse_decision(raw, current_rop)
                current_rop = max(0.0, decision["reorder_point"])
                convo_history = [*convo_history[-5:],
                                 {"role": "user", "content": user_msg},
                                 {"role": "assistant", "content": raw}]
                memory_bank = [*memory_bank[-7:], {
                    "day": day, "rop": round(current_rop, 1),
                    "fill_rate": f"{fr*100:.1f}%",
                    "confidence": decision.get("confidence", "?"),
                }]
                rop_markers.append((day, current_rop))
                conf = decision.get("confidence", "?")
                analysis = decision.get("state_analysis", "")[:80]
                decision_log.append(
                    f"**Day {day}** | ROP={current_rop:.0f} | Fill={fr*100:.1f}% "
                    f"| [{conf}] {analysis}"
                )
            except Exception as e:
                decision_log.append(f"**Day {day}** | API error: {str(e)[:60]}")

            # Yield live update
            fig = build_chart(daily_inventory, running_fill_rate, rop_markers,
                              f"Qwen2.5-72B  |  {env_name}  |  Day {day}/{SIM_DAYS}")
            summary = performance_tracker.performance_summary()
            metrics = (
                f"**Fill Rate:** {summary['fill_rate']:.2%}  \n"
                f"**Stockouts:** {summary['stock_out_count']}  \n"
                f"**Lost Sales:** {summary['total_lost_sales']:.0f}  \n"
                f"**Write-offs:** {summary['write_offs']:.0f}  \n"
                f"**Decisions:** {len(decision_log)}"
            )
            log_md = "\n\n".join(decision_log[-20:])
            yield fig, metrics, log_md

    # Final yield
    fig = build_chart(daily_inventory, running_fill_rate, rop_markers,
                      f"Qwen2.5-72B  |  {env_name}  |  COMPLETE")
    summary = performance_tracker.performance_summary()
    metrics = (
        f"**Fill Rate:** {summary['fill_rate']:.2%}  \n"
        f"**Stockouts:** {summary['stock_out_count']}  \n"
        f"**Lost Sales:** {summary['total_lost_sales']:.0f}  \n"
        f"**Write-offs:** {summary['write_offs']:.0f}  \n"
        f"**Decisions:** {len(decision_log)}"
    )
    yield fig, metrics, "\n\n".join(decision_log)


# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Inventory Simulation") as demo:
    gr.Markdown("# Inventory Optimization: Agent Comparison")

    with gr.Tabs():

        with gr.Tab("Baseline Agents"):
            gr.Markdown("Run one of the 4 rule-based agents through a full 365-day simulation.")
            with gr.Row():
                agent_dd = gr.Dropdown(
                    choices=["Base (Historical Mean)", "Safety Stock", "Forecast", "Monte Carlo"],
                    value="Safety Stock", label="Agent",
                )
                env_dd = gr.Dropdown(
                    choices=list(ENV_MAP.keys()),
                    value="GammaPoisson (90/10 mixture)", label="Demand Environment",
                )
            run_btn = gr.Button("Run Simulation", variant="primary")
            with gr.Row():
                chart = gr.Plot(label="Results")
                metrics_md = gr.Markdown(label="Metrics")
            run_btn.click(run_simulation, inputs=[agent_dd, env_dd], outputs=[chart, metrics_md])

        with gr.Tab("LLM Agent — Live"):
            gr.Markdown(
                "Qwen2.5-72B makes a reorder decision every 5 days. "
                "Chart and log update in real-time as the simulation runs."
            )
            with gr.Row():
                llm_env_dd = gr.Dropdown(
                    choices=list(ENV_MAP.keys()),
                    value="GammaPoisson (90/10 mixture)", label="Demand Environment",
                )
                hf_token_box = gr.Textbox(
                    label="HF Token (optional if HF_TOKEN env var is set)",
                    type="password", placeholder="hf_...",
                )
            llm_run_btn = gr.Button("Run LLM Simulation", variant="primary")
            with gr.Row():
                llm_chart = gr.Plot(label="Live Simulation")
                with gr.Column():
                    llm_metrics = gr.Markdown(label="Metrics")
                    llm_log = gr.Markdown(label="Decision Log")
            llm_run_btn.click(
                run_llm_simulation,
                inputs=[llm_env_dd, hf_token_box],
                outputs=[llm_chart, llm_metrics, llm_log],
            )

demo.launch()

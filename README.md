---
title: Inventory Reasoning Environment
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - inventory-optimization
  - long-horizon
license: apache-2.0
---

# Inventory Simulations

Stochastic modeling and Monte Carlo simulations to dimension inventory levels and evaluate reorder strategies under different demand environments.
 
## Overview
 
This project simulates inventory management over a 2-year (730-day) horizon, comparing four reorder-point agents across multiple service level targets. It models realistic demand patterns with seasonality, runs Monte Carlo experiments, and visualizes results via an interactive Dash dashboard.
 
## Agents
 
| Agent | Strategy |
|---|---|
| **Historical Demand Agent** (Baseline) | Reorder point = mean historical demand × lead time |
| **Safety Stock Agent** | Adds a safety buffer using the normal quantile of historical demand std |
| **Forecast Agent** | Uses future distribution means as forecast; adds safety stock based on forecast error |
| **Monte Carlo Agent** | Samples lead-time demand distributions MC_SIMS times; uses the service-level quantile of samples |
 
## Demand Environments
 
| Environment | Distribution |
|---|---|
| `SingleGammaLowVariance` | Single Gamma distribution (active by default) |
| `GammaGammaHighVariance` | 50/50 mixture of two Gamma distributions (low- and high-scale) |
| `GammaPoisson` | 90/10 mixture of Gamma and Poisson distributions |
 
All environments apply **seasonality multipliers** (by month and weekday) to the base distribution scale.
 
## Project Structure
 
```
├── main.py                          # Entry point: runs experiments and launches dashboard
├── config.py                        # Global constants (lead time, sim length, service levels, etc.)
├── simulator.py                     # Core simulation loop
├── montecarlo_simulator.py          # Alternative MC-based simulation runner
├── agent.py                         # Agent classes (Base, SafetyStock, Forecast, MonteCarlo)
├── demand_distribution.py           # Demand environment builders with seasonality
├── demand_distribution_parameters.py # Gamma/Poisson parameter ranges
├── demand_calculator.py             # Demand calculation utilities
├── demand_environment.py            # Demand environment interface
├── inventory_manager.py             # Inventory state, reordering, write-offs
├── order_processor.py               # Order queue management with lead time
├── performance_tracker.py           # Fill rate, service level, write-off tracking
├── simulation_plots.py              # Interactive Dash/Plotly dashboard
└── pyproject.toml                   # Project dependencies (Poetry)
```
 
## Key Parameters (`config.py`)
 
| Parameter | Default | Description |
|---|---|---|
| `SIM_DAYS` | 730 | Total simulation horizon (days) |
| `HISTO_DAYS` | 365 | History window used by agents for demand estimation |
| `LEAD_TIME` | 3 | Order-to-delivery delay (days) |
| `N_SIMULATIONS` | 100 | Number of independent simulation trials per scenario |
| `MC_SIMS` | 1000 | Monte Carlo samples per reorder decision |
| `WRITE_OFF_RATE` | 0.01 | Daily fraction of inventory written off (spoilage/expiry) |
| `DEFAULT_SERVICE_LEVEL` | 0.95 | Default target service level |
 
## Performance Metrics
 
- **Fill rate** — fraction of total demand fulfilled
- **Average service level** — fraction of days without a stockout
- **Write-offs** — cumulative inventory lost to spoilage
- **Lost sales** — total demand not fulfilled due to stockouts
- **Average inventory level** — mean daily on-hand stock
 
## OpenEnv HTTP API

This environment exposes a REST interface for LLM agent evaluation. The server runs on port 7860 (HF Spaces default).

| Endpoint | Method | Description |
|---|---|---|
| `/reset?env_type={0-3}` | POST | Start a new episode, returns `InventoryObservation` |
| `/step` | POST | Send `{"reorder_point": float, "reasoning": str}`, returns `StepResult` |
| `/state` | GET | Episode metadata (day, fill_rate, done) |

**Demand env_type**: 0=GammaPoisson, 1=GammaGammaHighVariance, 2=SpikingDemand, 3=SingleGammaLowVariance

**Reward**: `-0.001` per step; terminal reward = final fill rate at day 730.

### Connecting an agent

```python
import asyncio
from client.inventory_client import InventoryEnvClient, InventoryAction

async def run_agent():
    async with InventoryEnvClient("https://YOUR_USERNAME-inventory-env.hf.space") as env:
        obs = await env.reset(env_type=0)
        while obs.days_remaining > 0:
            rop = obs.demand_mean_30d * 3 + obs.demand_std_30d * 1.65
            result = await env.step(InventoryAction(reorder_point=rop))
            obs = result.observation
        print(f"Final fill rate: {result.reward:.3f}")

asyncio.run(run_agent())
```

### Local server

```bash
uvicorn server.inventory_env:app --reload --port 7860
```

### Docker

```bash
docker build -t inventory-env .
docker run -p 7860:7860 inventory-env
```

### Deploy to HF Spaces

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create inventory-env --type space --space-sdk docker
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/inventory-env
git push hf main
```

## Installation
 
Requires Python 3.11+. Install dependencies with [Poetry](https://python-poetry.org/):
 
```bash
poetry install
```
 
Or with pip:
 
```bash
pip install numpy scipy ciw pandas plotly dash pyinstrument
```
 
## Usage
 
```bash
python main.py
```
 
This runs 100 simulation trials for each combination of agent and service level (90%–98%), prints a summary table, then launches the Dash dashboard at `http://127.0.0.1:8050/`.
 
### Enabling Additional Environments
 
In `main.py`, uncomment the desired entries in `environment_configs`:
 
```python
environment_configs = {
    0: {"name": "90/10 Gamma/Poisson", "class": GammaPoisson},
    1: {"name": "50/50 Gamma(mu=20)/Gamma(mu=200)", "class": GammaGammaHighVariance},
    2: {"name": "Gamma", "class": SingleGammaLowVariance},
}
```
 
## Dashboard
 
The interactive dashboard (Plotly Dash) provides four views:
 
1. **Average Inventory & Demand Over Time** — inventory level vs. realized demand per agent
2. **Observed Service Level vs. Write-Offs** — scatter plot comparing agent efficiency
3. **Fill Rate vs. Write-Offs** — trade-off between stockouts and waste
4. **Write-Offs vs. Service Level Trend** — how write-offs scale with the target service level
 
## Reproducibility
 
The simulation uses fixed seeds for reproducibility:
 
```python
np.random.seed(11)
ciw.random.seed(11)
```

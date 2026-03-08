---
title: Inventory Reasoning Environment
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
tags:
  - openenv
  - reinforcement-learning
  - inventory-optimization
  - long-horizon
license: apache-2.0
app_port: 7860
---

# Inventory Simulations

Stochastic inventory simulation comparing rule-based agents against an LLM agent (Qwen2.5-72B) trained with GRPO reinforcement learning. Runs over a 2-year (730-day) horizon with a live Gradio UI on HF Spaces.

## Overview

The environment simulates day-to-day inventory decisions under stochastic demand. At each step an agent sets a **reorder point** — the inventory level that triggers a replenishment order. The goal is to maximize fill rate (≥95%) while minimizing holding costs, write-offs, and stockout penalties.

A P&L-based reward function is used throughout:

```
daily_reward = (revenue − holding_cost − stockout_penalty − order_cost − writeoff_cost)
               ÷ baseline_profit
```

## Agents

| Agent | Strategy |
|---|---|
| **Historical Mean (Baseline)** | ROP = mean historical demand × lead time |
| **Safety Stock** | Adds normal-quantile safety buffer on top of historical mean |
| **Forecast** | Uses future distribution means + safety stock on forecast error |
| **Monte Carlo** | Samples lead-time demand distributions; uses service-level quantile |
| **LLM (Qwen2.5-72B)** | Calls HF Inference API every 5 days; reasons over demand trend, pending orders, fill rate; outputs `reorder_point` as JSON |

The LLM agent can be fine-tuned locally with GRPO (`agent/train_grpo.py`) to produce a LoRA adapter that replaces the base Qwen model.

## Demand Environments

| Environment | Distribution |
|---|---|
| `GammaPoisson` | 90/10 mixture of Gamma and Poisson |
| `GammaGammaHighVariance` | 50/50 mixture of two Gamma distributions (bimodal) |
| `SpikingDemand` | Gamma with occasional demand spikes |
| `SingleGammaLowVariance` | Single Gamma, low variance |

All environments apply **seasonality multipliers** (by month and weekday) to the base scale.

## Project Structure

```
├── app.py                    # Gradio UI (Baseline + LLM tabs)
├── config.py                 # Global constants
├── reward.py                 # Unified P&L reward function
├── demand_environment.py     # Demand distribution classes
├── demand_calculator.py      # Per-day demand sampling
├── inventory_manager.py      # Inventory state, reordering, write-offs
├── order_processor.py        # Order queue with stochastic lead time
├── performance_tracker.py    # Fill rate, write-offs, lost sales
├── agent_environment.py      # Rule-based agent classes
├── server/inventory_env.py   # FastAPI HTTP environment (OpenEnv API)
├── client/inventory_client.py# Async Python client for the HTTP env
├── agent/
│   ├── train_grpo.py         # GRPO fine-tuning of Qwen2.5-3B-Instruct
│   ├── finetune_agent.py     # Local inference with optional LoRA adapter
│   └── llm_agent_runner.py   # CLI runner for the LLM agent
└── Dockerfile                # HF Spaces container
```

## Key Parameters (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `SIM_DAYS` | 730 | Total simulation horizon (days) |
| `HISTO_DAYS` | 365 | Warm-up history before decisions begin |
| `LEAD_TIME` | 3 | Order-to-delivery delay (days) |
| `LEAD_TIME_JITTER` | 1 | ±stochastic jitter on lead time |
| `WRITE_OFF_RATE` | 0.00143 | Daily spoilage fraction |
| `SELLING_PRICE` | 10.0 | Revenue per unit sold |
| `UNIT_COST` | 4.0 | Cost per unit ordered |
| `FIXED_ORDER_COST` | 50.0 | Fixed cost per order placed |

## GRPO Training (LLM fine-tuning)

Requires a GPU (run on Colab/Kaggle). Start the HTTP environment server first, then:

```bash
# Terminal 1 — start the environment server
uvicorn server.inventory_env:app --port 7860

# Terminal 2 — train
python agent/train_grpo.py \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --base-url http://localhost:7860 \
    --n-iterations 5 \
    --episodes-per-iter 20 \
    --output-dir ./grpo_inventory
```

Each iteration: collect rollouts → compute P&L rewards → GRPO update → save LoRA adapter.

## OpenEnv HTTP API

| Endpoint | Method | Description |
|---|---|---|
| `/reset?env_type={0-3}` | POST | Start a new episode, returns `InventoryObservation` |
| `/step` | POST | Send `{"reorder_point": float, "reasoning": str}`, returns `StepResult` |
| `/state` | GET | Episode metadata (day, fill_rate, done) |

**env_type**: 0=GammaPoisson, 1=GammaGammaHighVariance, 2=SpikingDemand, 3=SingleGammaLowVariance

### Connecting an external agent

```python
import asyncio
from client.inventory_client import InventoryEnvClient, InventoryAction

async def run_agent():
    async with InventoryEnvClient("https://ademarteau-rl-inventory-simulations.hf.space") as env:
        obs = await env.reset(env_type=0)
        while obs.days_remaining > 0:
            rop = obs.demand_mean_30d * 3 + obs.demand_std_30d * 1.65
            result = await env.step(InventoryAction(reorder_point=rop))
            obs = result.observation
        print(f"Final fill rate: {obs.fill_rate_so_far:.3f}")

asyncio.run(run_agent())
```

### Local setup

```bash
pip install -r requirements.txt

# Run the Gradio UI
python app.py

# Or run just the HTTP environment server
uvicorn server.inventory_env:app --reload --port 7860
```

### Docker

```bash
docker build -t inventory-sim .
docker run -p 7860:7860 inventory-sim
```

"""
LLM Agent Runner — Qwen2.5-72B vs Inventory Simulation

Runs the Qwen LLM agent through the FastAPI inventory simulation server
via the HuggingFace Inference API. The model receives reward signals
(fill rate, stockouts, lost sales) each turn and adapts its reorder point
decisions using a rolling memory bank.

Usage:
    set HF_TOKEN=hf_...
    python llm_agent_runner.py --env 0 --episodes 1

    --env: 0=GammaPoisson  1=GammaGamma  2=Spiking  3=SingleGamma
    --episodes: number of full runs (default 1)
"""

import argparse
import json
import re
import time
import threading
import requests
import uvicorn
from huggingface_hub import InferenceClient

from config import SIM_DAYS, HISTO_DAYS, LEAD_TIME

# ── Server ─────────────────────────────────────────────────────────────────────

BASE_URL = "http://127.0.0.1:7861"
DECISION_INTERVAL = 5  # Claude decides every N days

ENV_NAMES = {
    0: "GammaPoisson",
    1: "GammaGamma High Variance",
    2: "Spiking Demand",
    3: "Single Gamma Low Variance",
}

def start_server():
    from server.inventory_env import app
    uvicorn.run(app, host="127.0.0.1", port=7861, log_level="warning")

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert inventory optimization agent embedded in a stochastic simulation environment.

YOUR ROLE:
You receive a JSON snapshot of the current simulation state and must decide the REORDER POINT (ROP) — the inventory threshold that triggers a new order.

ENVIRONMENT RULES:
- Orders arrive exactly LEAD_TIME=3 days after placement
- You place an order whenever inventory <= your ROP
- Order quantity = ROP - current_inventory + mean_demand * LEAD_TIME (already handled)
- Every 7 days, 1% of inventory is written off (waste/expiry)
- Reward = fill_rate at end of simulation (target: >=95%)
- Reward is SPARSE: fill rate only stabilizes after ~50 days

REASONING REQUIREMENTS - you MUST do all 4:
1. SUBGOAL DECOMPOSITION: Break the problem into explicit subgoals (e.g., "build buffer", "survive spike risk", "minimize waste")
2. STATE ANALYSIS: Interpret current inventory, demand trend, stockout risk, fill rate trajectory
3. DECISION: Output a specific numeric ROP with clear justification
4. RECOVERY PLAN: If fill rate < 95% or recent stockouts occurred, state your recovery strategy

CRITICAL: Reason BEYOND the next step. Your ROP today affects inventory 3+ days from now.
For spiking demand: ROP must account for rare but catastrophic spikes.
For high-variance: wider safety buffers needed.
For stable demand: tighter ROP to avoid write-offs.

OUTPUT FORMAT — respond with this exact JSON (no markdown fences):
{
  "subgoals": ["subgoal 1", "subgoal 2", "subgoal 3"],
  "state_analysis": "2-3 sentence analysis of current state and risks",
  "recovery_plan": "what you're doing to recover or maintain performance",
  "reorder_point": <number>,
  "confidence": "high|medium|low",
  "reasoning_depth": "brief note on what makes this decision non-trivial"
}"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def build_snapshot(obs: dict, reward: float, info: dict, memory_bank: list) -> str:
    memory_section = ""
    if memory_bank:
        memory_section = f"\nYOUR MEMORY FROM PREVIOUS DECISIONS:\n{json.dumps(memory_bank[-8:], indent=2)}"

    snapshot = {
        "day": obs["day"],
        "days_remaining": obs["days_remaining"],
        "current_inventory": round(obs["current_inventory"], 1),
        "demand_last_5": [round(d, 1) for d in obs["demand_last_5"]],
        "demand_mean_30d": round(obs["demand_mean_30d"], 1),
        "demand_std_30d": round(obs["demand_std_30d"], 1),
        "fill_rate_so_far": f"{obs['fill_rate_so_far']*100:.1f}%",
        "recent_stockouts": obs["recent_stockouts"],
        "recent_lost_sales": round(obs["recent_lost_sales"], 1),
        "pending_orders": obs["pending_orders"],
        "last_step_reward": round(reward, 4),
        "info": {k: round(v, 4) if isinstance(v, float) else v for k, v in info.items()},
        "lead_time": LEAD_TIME,
        "service_level_target": 0.95,
    }
    return (
        f"ENVIRONMENT SNAPSHOT — Day {obs['day']}/{SIM_DAYS}\n"
        f"{json.dumps(snapshot, indent=2)}"
        f"{memory_section}\n\n"
        f"Decide your reorder_point for the next {DECISION_INTERVAL} days."
    )


def call_llm(content: str, convo_history: list, client: InferenceClient) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *convo_history[-6:],
        {"role": "user", "content": content},
    ]
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=messages,
        max_tokens=1000,
    )
    raw = response.choices[0].message.content
    try:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        return json.loads(cleaned), raw
    except (json.JSONDecodeError, ValueError):
        match = re.search(r'"reorder_point"\s*:\s*(\d+\.?\d*)', raw)
        rop = float(match.group(1)) if match else 300.0
        return {
            "subgoals": ["parse error — fallback"],
            "state_analysis": raw[:200],
            "recovery_plan": "N/A",
            "reorder_point": rop,
            "confidence": "low",
            "reasoning_depth": "parse failed",
        }, raw


def update_memory(memory_bank: list, decision: dict, obs: dict) -> list:
    entry = {
        "day": obs["day"],
        "rop_set": round(decision["reorder_point"], 1),
        "confidence": decision.get("confidence", "?"),
        "fill_rate": obs["fill_rate_so_far"],
        "recent_stockouts": obs["recent_stockouts"],
        "key_insight": (decision.get("state_analysis") or "")[:80],
    }
    return [*memory_bank[-14:], entry]


def print_decision(day: int, decision: dict, obs: dict):
    rop = decision["reorder_point"]
    fr = obs["fill_rate_so_far"] * 100
    conf = decision.get("confidence", "?")
    print(f"  Day {day:3d} | ROP={rop:6.0f} | Fill={fr:5.1f}% | [{conf}] {decision.get('reasoning_depth','')[:60]}")


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(env_type: int, episode_num: int, client: InferenceClient, session: requests.Session) -> dict:
    env_name = ENV_NAMES.get(env_type, str(env_type))
    print(f"\nEpisode {episode_num} — {env_name}")
    print("-" * 60)

    r = session.post(f"{BASE_URL}/reset", params={"env_type": env_type})
    r.raise_for_status()
    obs = r.json()

    memory_bank = []
    convo_history = []
    current_rop = obs["demand_mean_30d"] * LEAD_TIME
    llm_decisions = 0

    for _ in range(SIM_DAYS - HISTO_DAYS):
        day = obs["day"]

        r = session.post(f"{BASE_URL}/step", json={"reorder_point": current_rop, "reasoning": ""})
        r.raise_for_status()
        result = r.json()
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        info = result["info"]

        if (day - HISTO_DAYS) % DECISION_INTERVAL == 0 and day < SIM_DAYS - LEAD_TIME:
            snapshot_str = build_snapshot(obs, reward, info, memory_bank)
            decision, raw_resp = call_llm(snapshot_str, convo_history, client)
            current_rop = max(0.0, decision["reorder_point"])
            memory_bank = update_memory(memory_bank, decision, obs)
            convo_history = [
                *convo_history[-6:],
                {"role": "user", "content": snapshot_str},
                {"role": "assistant", "content": raw_resp},
            ]
            llm_decisions += 1
            print_decision(day, decision, obs)

        if done:
            break

    state = session.get(f"{BASE_URL}/state").json()
    fill_rate = state["fill_rate"]
    stockouts = state["stockouts"]
    lost_sales = state["lost_sales"]

    print(f"\n{'='*60}")
    print(f"  Fill Rate:      {fill_rate*100:.2f}%  {'✓' if fill_rate >= 0.95 else '✗ (target: 95%)'}")
    print(f"  Stockouts:      {stockouts}")
    print(f"  Lost Sales:     {lost_sales:.0f}")
    print(f"  LLM Decisions:  {llm_decisions}")
    print(f"{'='*60}")

    return {"episode": episode_num, "env": env_name, "fill_rate": fill_rate,
            "stockouts": stockouts, "lost_sales": lost_sales, "llm_decisions": llm_decisions}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run LLM agent through inventory simulation")
    parser.add_argument("--env", type=int, default=0, choices=[0, 1, 2, 3],
                        help="0=GammaPoisson 1=GammaGamma 2=Spiking 3=SingleGamma")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    args = parser.parse_args()

    # Start server in background
    print("Starting simulation server...")
    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    time.sleep(2.0)

    client = InferenceClient()  # reads HF_TOKEN from env
    session = requests.Session()

    results = []
    for ep in range(1, args.episodes + 1):
        results.append(run_episode(args.env, ep, client, session))

    if args.episodes > 1:
        avg_fr = sum(r["fill_rate"] for r in results) / len(results)
        print(f"\nAverage fill rate across {args.episodes} episodes: {avg_fr*100:.2f}%")


if __name__ == "__main__":
    main()

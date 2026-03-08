"""
Minimal GRPO training script for the Inventory Reasoning Environment.

Designed to run in Google Colab with a single GPU. Connects to the
OpenEnv-compatible HF Space and fine-tunes Qwen2.5-3B-Instruct using
Unsloth + TRL's GRPOTrainer with a custom rollout_func.

Usage (Colab):
    !pip install "openenv-core[core]>=0.2.1" unsloth trl datasets
    !pip install git+https://huggingface.co/spaces/ademarteau/rl-inventory-simulations
    %run train_colab.py
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# ── Connect to the OpenEnv Inventory Environment on HF Spaces ────────────────

from inventory_env_client import InventoryEnv
from models import InventoryAction

ENV_URL = "https://ademarteau-rl-inventory-simulations.hf.space"

# ── Prompt & parsing ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an inventory optimization agent. Each step you receive the current state \
of a supply-chain simulation and must decide a reorder_point — the inventory level \
that triggers a replenishment order.

Reply with ONLY a JSON object:
{"reorder_point": <number>, "reasoning": "<brief explanation>"}
"""


def format_user_message(obs_dict: dict) -> str:
    return json.dumps({
        "day": obs_dict["day"],
        "days_remaining": obs_dict["days_remaining"],
        "current_inventory": round(obs_dict["current_inventory"], 2),
        "demand_mean_30d": round(obs_dict["demand_mean_30d"], 2),
        "demand_std_30d": round(obs_dict.get("demand_std_30d", 0), 2),
        "fill_rate_so_far": round(obs_dict["fill_rate_so_far"], 4),
        "recent_stockouts": obs_dict["recent_stockouts"],
        "pending_orders": obs_dict.get("pending_orders", []),
    }, separators=(",", ":"))


def parse_rop(text: str) -> float | None:
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
        return float(json.loads(cleaned)["reorder_point"])
    except Exception:
        m = re.search(r'"reorder_point"\s*:\s*([0-9]+\.?[0-9]*)', text)
        return float(m.group(1)) if m else None


# ── Analytical reward (30-day lookahead) ─────────────────────────────────────

SELLING_PRICE = 25.0
UNIT_COST = 10.0
FIXED_ORDER_COST = 150.0
HOLDING_RATE = 0.005
WRITE_OFF_RATE = 0.00143
LEAD_TIME = 3
LOOKAHEAD = 30
TARGET_FILL = 0.95


def analytical_reward(obs: dict, rop: float) -> float:
    inv = obs["current_inventory"]
    mean_d = obs["demand_mean_30d"]
    std_d = obs.get("demand_std_30d", 0.0)
    horizon = min(LOOKAHEAD, obs.get("days_remaining", 365))
    if horizon <= 0 or mean_d <= 0:
        return 0.0

    pending = [(p["arrival_day"], p["quantity"]) for p in obs.get("pending_orders", [])]
    total_profit = total_sold = total_demand = 0.0

    for t in range(horizon):
        day = obs["day"] + t
        delivered = sum(q for a, q in pending if a == day)
        inv += delivered
        pending = [(a, q) for a, q in pending if a > day]
        inv = max(0.0, inv - inv * WRITE_OFF_RATE)
        demand = mean_d + (std_d * 0.3 if t % 5 == 0 else 0.0)
        sold = min(demand, inv)
        lost = max(0.0, demand - inv)
        inv = max(0.0, inv - demand)
        total_sold += sold
        total_demand += demand

        oq = 0.0
        pipe = sum(q for a, q in pending)
        if inv + pipe <= rop:
            oq = max(0.0, rop - (inv + pipe) + mean_d * LEAD_TIME)
            pending.append((day + LEAD_TIME, oq))

        rev = sold * SELLING_PRICE
        hc = inv * UNIT_COST * HOLDING_RATE
        sp = lost * (SELLING_PRICE - UNIT_COST)
        oc = (FIXED_ORDER_COST if oq > 0 else 0) + oq * UNIT_COST
        wc = (inv * WRITE_OFF_RATE) * UNIT_COST
        total_profit += rev - hc - sp - oc - wc

    baseline = mean_d * (SELLING_PRICE - UNIT_COST) * horizon
    pnl_r = total_profit / baseline if baseline > 0 else 0.0
    fr = total_sold / total_demand if total_demand > 0 else 0.0
    fill_r = 1.0 if fr >= TARGET_FILL else -(TARGET_FILL - fr) / TARGET_FILL
    return float(np.clip(0.6 * pnl_r + 0.4 * fill_r, -2.0, 2.0))


# ── Reward function for GRPOTrainer ──────────────────────────────────────────

def reward_fn(completions: list[str], obs_json: list[str] | None = None, **kwargs) -> list[float]:
    rewards = []
    for i, c in enumerate(completions):
        rop = parse_rop(c)
        if rop is None:
            rewards.append(-1.0)
            continue
        if obs_json is not None:
            raw = obs_json[i]
            obs = json.loads(raw[0] if isinstance(raw, list) else raw)
            rewards.append(analytical_reward(obs, rop))
        else:
            rewards.append(0.0)
    return rewards


# ── Collect episode dataset via OpenEnv ──────────────────────────────────────

def collect_episode(tokenizer, env_url: str, env_type: int = 0) -> list[dict]:
    rows = []
    with InventoryEnv(base_url=env_url) as env:
        result = env.reset(env_type=env_type)
        obs = result.observation
        while not result.done:
            obs_dict = {
                "day": obs.day,
                "days_remaining": obs.days_remaining,
                "current_inventory": obs.current_inventory,
                "demand_mean_30d": obs.demand_mean_30d,
                "demand_std_30d": obs.demand_std_30d,
                "fill_rate_so_far": obs.fill_rate_so_far,
                "recent_stockouts": obs.recent_stockouts,
                "recent_lost_sales": obs.recent_lost_sales,
                "pending_orders": [
                    {"arrival_day": p.arrival_day, "quantity": p.quantity}
                    for p in obs.pending_orders
                ],
            }
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_user_message(obs_dict)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            rop = obs.demand_mean_30d * 3 + obs.demand_std_30d * 1.65
            result = env.step(InventoryAction(reorder_point=rop))
            obs = result.observation
            rows.append({"prompt": prompt, "obs_json": json.dumps(obs_dict)})
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading model via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=16,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()

    n_iterations = 3

    for iteration in range(n_iterations):
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration + 1}/{n_iterations}")
        print(f"{'='*50}")

        print("Collecting episode via OpenEnv...")
        rows = collect_episode(tokenizer, ENV_URL, env_type=0)
        dataset = Dataset.from_list(rows)
        print(f"  Collected {len(dataset)} steps")

        grpo_config = GRPOConfig(
            output_dir=f"./grpo_inventory/iter_{iteration}",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_generations=4,
            max_completion_length=128,
            learning_rate=5e-6,
            bf16=True,
            logging_steps=5,
            save_strategy="no",
            report_to="none",
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        result = trainer.train()
        loss = result.training_loss if hasattr(result, "training_loss") else "N/A"
        print(f"  Training loss: {loss}")

        model.save_pretrained(f"./grpo_inventory/iter_{iteration}")
        tokenizer.save_pretrained(f"./grpo_inventory/iter_{iteration}")
        print(f"  Saved adapter to ./grpo_inventory/iter_{iteration}")

    print("\nDone! Final adapter saved.")


if __name__ == "__main__":
    main()

"""
GRPO fine-tuning of a Qwen2.5-Instruct model for inventory optimization.

Training loop (repeated N iterations):
  1. Collect — run current model on HTTP server to gather (prompt, obs_json) tuples
  2. Train   — run one epoch of GRPOTrainer on the collected dataset
  3. Save    — save LoRA adapter checkpoint

Usage:
    python agent/train_grpo.py --base-url http://localhost:7860 --n-iterations 5
    python agent/train_grpo.py --eval --adapter-path ./grpo_inventory/iter_4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Any

from tqdm import tqdm

import datasets
import numpy as np
import torch
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from client.inventory_client import InventoryAction, InventoryEnvClient, InventoryObservation


# ── Logging helpers ───────────────────────────────────────────────────────────

def _gpu_mem_str() -> str:
    if not torch.cuda.is_available():
        return "no GPU"
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved  = torch.cuda.memory_reserved()  / 1024 ** 3
    total     = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return f"{allocated:.1f}GB alloc / {reserved:.1f}GB reserved / {total:.0f}GB total"

def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


# ── Prompt constants ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert inventory optimization agent operating inside a stochastic supply-chain simulation.

YOUR OBJECTIVE:
Maximize the fill rate (fraction of demand fulfilled) while minimizing inventory write-offs over a \
365-day episode. The episode ends at day 730 (after 365 days of decisions following a 365-day warm-up).

ENVIRONMENT RULES:
- Orders arrive exactly 3 days after placement (LEAD_TIME = 3)
- An order is placed automatically whenever inventory <= your chosen reorder_point
- Order quantity = reorder_point - current_inventory + mean_demand * LEAD_TIME (handled by the env)
- Every 7 days, 1% of on-hand inventory is written off (waste/expiry)
- Fill rate = total units fulfilled / total units demanded (target: >= 95%)
- Reward is SPARSE: fill rate only stabilises after many days; plan ahead

YOUR ACTION EACH STEP:
Set `reorder_point` — the inventory level at or below which a replenishment order fires.
A higher ROP builds safety buffer but risks write-offs. A lower ROP conserves stock but risks stockouts.

REASONING GUIDANCE:
- Analyse demand trend and variability before deciding
- Account for pending orders already in the pipeline — they will arrive soon
- After stockouts, raise ROP aggressively to rebuild buffer
- If fill rate is healthy and inventory is high, consider lowering ROP to reduce write-offs
- Think 3+ days ahead; your ROP today only shows its effect after lead time

RESPONSE FORMAT — reply with ONLY a valid JSON object, no markdown fences:
{"reorder_point": <float>, "reasoning": "<concise explanation>", "confidence": <float 0-1>}
"""

MEMORY_SIZE = 15


# ── Prompt formatting ─────────────────────────────────────────────────────────


def format_prompt(obs_dict: dict[str, Any], memory_bank: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Build a list of chat messages (system + user) from an observation dict and memory bank.
    Matches the snapshot format used by ClaudeInventoryAgent._build_user_message.
    """
    snapshot: dict[str, Any] = {
        "day": obs_dict["day"],
        "days_remaining": obs_dict["days_remaining"],
        "current_inventory": round(obs_dict["current_inventory"], 2),
        "demand_last_5": [round(d, 2) for d in obs_dict["demand_last_5"]],
        "demand_mean_30d": round(obs_dict["demand_mean_30d"], 2),
        "demand_std_30d": round(obs_dict["demand_std_30d"], 2),
        "fill_rate_so_far": round(obs_dict["fill_rate_so_far"], 4),
        "recent_stockouts": obs_dict["recent_stockouts"],
        "recent_lost_sales": round(obs_dict["recent_lost_sales"], 2),
        "pending_orders": obs_dict.get("pending_orders", []),
        "memory_bank": memory_bank[-MEMORY_SIZE:],
    }
    user_content = json.dumps(snapshot, separators=(",", ":"))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ── Parsing and reward ────────────────────────────────────────────────────────


def parse_rop(completion_text: str) -> float | None:
    """Extract reorder_point from a model completion. Returns None if unparseable."""
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", completion_text).strip()
        return float(json.loads(cleaned)["reorder_point"])
    except Exception:
        match = re.search(r'"reorder_point"\s*:\s*([0-9]+\.?[0-9]*)', completion_text)
        if match:
            return float(match.group(1))
        return None



# ── Model generation helper ───────────────────────────────────────────────────


def _generate_completion(
    model: Any,
    tokenizer: AutoTokenizer,
    prompt_str: str,
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
) -> str:
    """Run greedy/sample generation for a single prompt string."""
    inputs = tokenizer(prompt_str, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Rollout collection ────────────────────────────────────────────────────────


async def _run_episode_async(
    model: Any | None,
    tokenizer: AutoTokenizer,
    base_url: str,
    env_type: int,
    episode_idx: int,
    device: str,
) -> list[dict[str, str]]:
    """
    Run one full episode. For episode 0 (model=None), uses the safety-stock heuristic
    to seed the dataset quickly without a model forward pass.
    Returns a list of {"prompt": str, "obs_json": str} rows.
    """
    rows: list[dict[str, str]] = []
    memory_bank: list[dict[str, Any]] = []
    use_heuristic = model is None or episode_idx == 0
    parse_failures = 0
    step_times: list[float] = []

    async with InventoryEnvClient(base_url=base_url) as env:
        obs: InventoryObservation = await env.reset(env_type=env_type)
        done = False
        step = 0

        with tqdm(total=365, desc=f"  ep{episode_idx+1}", unit="step",
                  dynamic_ncols=True, leave=False) as pbar:
            while not done:
                t0 = time.time()
                obs_dict: dict[str, Any] = {
                    "day": obs.day,
                    "days_remaining": obs.days_remaining,
                    "current_inventory": round(obs.current_inventory, 2),
                    "demand_last_5": [round(d, 2) for d in obs.demand_last_5],
                    "demand_mean_30d": round(obs.demand_mean_30d, 2),
                    "demand_std_30d": round(obs.demand_std_30d, 2),
                    "fill_rate_so_far": round(obs.fill_rate_so_far, 4),
                    "recent_stockouts": obs.recent_stockouts,
                    "recent_lost_sales": round(obs.recent_lost_sales, 2),
                    "pending_orders": [
                        {"arrival_day": o.arrival_day, "quantity": o.quantity}
                        for o in obs.pending_orders
                    ],
                }

                messages = format_prompt(obs_dict, memory_bank)
                prompt_str: str = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                if use_heuristic:
                    rop = obs.demand_mean_30d * 3 + obs.demand_std_30d * 1.65
                else:
                    completion = _generate_completion(
                        model, tokenizer, prompt_str, device
                    )
                    rop_parsed = parse_rop(completion)
                    if rop_parsed is None:
                        parse_failures += 1
                        rop = obs.demand_mean_30d * 3 + obs.demand_std_30d * 1.65
                    else:
                        rop = rop_parsed

                result = await env.step(InventoryAction(reorder_point=rop))
                obs = result.observation
                done = result.done
                reward = result.reward

                rows.append({
                    "prompt": prompt_str,
                    "obs_json": json.dumps(obs_dict),
                    "step_reward": float(reward),
                })

                memory_bank = (memory_bank + [{
                    "day": obs_dict["day"],
                    "reorder_point": round(rop, 2),
                    "fill_rate_after": round(obs.fill_rate_so_far, 4),
                }])[-MEMORY_SIZE:]

                elapsed = time.time() - t0
                step_times.append(elapsed)
                avg_step = sum(step_times[-20:]) / len(step_times[-20:])

                pbar.update(1)
                pbar.set_postfix({
                    "day":    obs_dict["day"],
                    "fill":   f"{obs.fill_rate_so_far:.3f}",
                    "rop":    f"{rop:.0f}",
                    "rew":    f"{reward:+.2f}",
                    "fails":  parse_failures,
                    "s/step": f"{avg_step:.2f}",
                })
                step += 1

    return rows


def collect_rollout_dataset(
    model: Any | None,
    tokenizer: AutoTokenizer,
    base_url: str,
    env_type: int,
    n_episodes: int,
    device: str,
) -> datasets.Dataset:
    """
    Run n_episodes against the HTTP server and return a Dataset with columns
    ["prompt", "obs_json"]. Episode 0 always uses the heuristic fallback;
    subsequent episodes use the model.
    """
    all_rows: list[dict[str, str]] = []
    collect_start = time.time()

    for ep in range(n_episodes):
        use_heuristic = model is None or ep == 0
        mode = "heuristic" if use_heuristic else "model"
        ep_start = time.time()
        print(f"  Episode {ep + 1}/{n_episodes} [{mode}]...", flush=True)
        rows = asyncio.run(
            _run_episode_async(model, tokenizer, base_url, env_type, ep, device)
        )
        all_rows.extend(rows)
        ep_dur = time.time() - ep_start
        steps_per_sec = len(rows) / ep_dur if ep_dur > 0 else 0
        remaining_eps = n_episodes - (ep + 1)
        eta = _fmt_duration(ep_dur * remaining_eps)
        print(
            f"    ✓ {len(rows)} steps in {_fmt_duration(ep_dur)} "
            f"({steps_per_sec:.1f} steps/s) | total={len(all_rows)} | ETA: {eta}",
            flush=True,
        )

    total_dur = time.time() - collect_start
    print(f"  Collection done: {len(all_rows)} steps in {_fmt_duration(total_dur)}", flush=True)
    return datasets.Dataset.from_list(all_rows)


# ── Analytical reward ─────────────────────────────────────────────────────────

UNIT_COST = 10.0
SELLING_PRICE = 25.0
FIXED_ORDER_COST = 150.0
HOLDING_RATE = 0.005
WRITE_OFF_RATE = 0.00143
LEAD_TIME = 3
LOOKAHEAD_DAYS = 365
TARGET_FILL_RATE = 0.95
FILL_RATE_WEIGHT = 0.4


def _simulate_rop(obs: dict[str, Any], rop: float) -> float:
    """
    Run a deterministic LOOKAHEAD_DAYS-day forward simulation with the
    proposed ROP starting from the observation state. Returns a composite
    reward combining:
      - Normalised cumulative P&L (60% weight)
      - Fill rate vs 95% target (40% weight)
    """
    inv = obs["current_inventory"]
    mean_d = obs["demand_mean_30d"]
    std_d = obs.get("demand_std_30d", 0.0)
    current_day = obs["day"]
    days_remaining = obs.get("days_remaining", 365)

    pending: list[tuple[int, float]] = [
        (p["arrival_day"], p["quantity"])
        for p in obs.get("pending_orders", [])
    ]

    horizon = min(LOOKAHEAD_DAYS, days_remaining)
    if horizon <= 0 or mean_d <= 0:
        return 0.0

    total_profit = 0.0
    total_sold = 0.0
    total_demand = 0.0

    for t in range(horizon):
        day = current_day + t

        delivered = sum(qty for arr, qty in pending if arr == day)
        inv += delivered
        pending = [(arr, qty) for arr, qty in pending if arr > day]

        spoilage = inv * WRITE_OFF_RATE
        inv = max(0.0, inv - spoilage)

        demand = mean_d + (std_d * 0.3 if t % 5 == 0 else 0.0)
        sold = min(demand, inv)
        lost = max(0.0, demand - inv)
        inv = max(0.0, inv - demand)
        total_sold += sold
        total_demand += demand

        order_qty = 0.0
        if inv <= rop:
            order_qty = max(0.0, rop - inv + mean_d * LEAD_TIME)
            pending.append((day + LEAD_TIME, order_qty))

        revenue = sold * SELLING_PRICE
        holding_cost = inv * UNIT_COST * HOLDING_RATE
        stockout_penalty = lost * (SELLING_PRICE - UNIT_COST)
        order_cost = (FIXED_ORDER_COST if order_qty > 0 else 0.0) + order_qty * UNIT_COST
        writeoff_cost = spoilage * UNIT_COST

        total_profit += revenue - holding_cost - stockout_penalty - order_cost - writeoff_cost

    baseline = mean_d * (SELLING_PRICE - UNIT_COST) * horizon
    pnl_reward = total_profit / baseline if baseline > 0 else 0.0

    sim_fill_rate = total_sold / total_demand if total_demand > 0 else 0.0
    if sim_fill_rate >= TARGET_FILL_RATE:
        fill_reward = 1.0
    else:
        fill_reward = -(TARGET_FILL_RATE - sim_fill_rate) / TARGET_FILL_RATE

    return (1.0 - FILL_RATE_WEIGHT) * pnl_reward + FILL_RATE_WEIGHT * fill_reward


def build_reward_fn(tokenizer: AutoTokenizer):  # noqa: ARG001
    def reward_fn(
        prompts: list[str],
        completions: list[str],
        obs_json: list[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        rewards: list[float] = []
        for i, completion in enumerate(completions):
            rop = parse_rop(completion)
            if rop is None:
                rewards.append(-1.0)
                continue

            if obs_json is not None:
                raw = obs_json[i]
                obs = json.loads(raw[0] if isinstance(raw, list) else raw)
                r = _simulate_rop(obs, rop)
            else:
                r = 0.0

            rewards.append(float(np.clip(r, -2.0, 2.0)))
        return rewards
    return reward_fn


# ── Model setup ───────────────────────────────────────────────────────────────


def setup_model_and_tokenizer(
    base_model: str,
    lora_rank: int,
    lora_alpha: int,
    device: str,
) -> tuple[Any, AutoTokenizer]:
    """Load base model + tokenizer via Unsloth and wrap with LoRA adapters."""
    print(f"Loading model and tokenizer from {base_model} via Unsloth (bfloat16)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        load_in_4bit=False,
        fast_inference=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()
    device_used = next(model.parameters()).device
    print(f"  Model device: {device_used}  |  GPU memory: {_gpu_mem_str()}", flush=True)
    return model, tokenizer


# ── Main training loop ────────────────────────────────────────────────────────


def train(
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    output_dir: str = "./grpo_inventory",
    base_url: str = "http://localhost:7860",
    env_type: int = 0,
    n_iterations: int = 5,
    episodes_per_iter: int = 20,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    device: str = "auto",
    num_generations: int = 4,
    per_device_batch_size: int = 2,
    grad_accum: int = 4,
    max_new_tokens: int = 256,
    learning_rate: float = 5e-6,
) -> None:
    """
    Outer GRPO training loop: collect → train → save, repeated n_iterations times.
    """
    os.makedirs(output_dir, exist_ok=True)

    model: Any | None = None
    tokenizer: AutoTokenizer | None = None

    run_start = time.time()

    for iteration in range(n_iterations):
        iter_output_dir = os.path.join(output_dir, f"iter_{iteration}")
        dataset_dir = os.path.join(output_dir, f"dataset_iter_{iteration}")
        adapter_marker = os.path.join(iter_output_dir, "adapter_config.json")

        # ── Skip completed iterations ──────────────────────────────────────────
        if os.path.exists(adapter_marker):
            print(f"\n[SKIP] Iteration {iteration + 1}/{n_iterations} already complete "
                  f"(found {adapter_marker})", flush=True)
            # Still need to load model for subsequent iterations
            if model is None:
                print("[Phase 0] Loading adapter from checkpoint via Unsloth...", flush=True)
                model, _tok = FastLanguageModel.from_pretrained(
                    model_name=iter_output_dir,
                    max_seq_length=2048,
                    load_in_4bit=False,
                    fast_inference=True,
                )
                _tok.pad_token = _tok.eos_token
                _tok.padding_side = "left"
                tokenizer = _tok
                print(f"  Loaded adapter from {iter_output_dir}  |  GPU: {_gpu_mem_str()}", flush=True)
            continue

        iter_start = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"ITERATION {iteration + 1}/{n_iterations}  |  GPU: {_gpu_mem_str()}", flush=True)
        print(f"{'='*60}", flush=True)

        if model is None:
            print("\n[Phase 0] Setting up model and tokenizer...", flush=True)
            t0 = time.time()
            model, tokenizer = setup_model_and_tokenizer(
                base_model, lora_rank, lora_alpha, device
            )
            print(f"  Done in {_fmt_duration(time.time() - t0)}", flush=True)

        # ── Phase 1: collect or reload saved dataset ───────────────────────────
        if os.path.exists(dataset_dir):
            print(f"\n[Phase 1] Loading saved rollout dataset from {dataset_dir}...", flush=True)
            dataset = datasets.load_from_disk(dataset_dir)
            print(f"  Loaded {len(dataset)} steps  |  GPU: {_gpu_mem_str()}", flush=True)
        else:
            print(f"\n[Phase 1] Collecting rollout dataset ({episodes_per_iter} episodes)...", flush=True)
            t0 = time.time()
            dataset = collect_rollout_dataset(
                model, tokenizer, base_url, env_type, episodes_per_iter, device
            )
            print(f"  Dataset: {len(dataset)} steps in {_fmt_duration(time.time() - t0)}  |  GPU: {_gpu_mem_str()}", flush=True)
            print(f"  Saving dataset to {dataset_dir}...", flush=True)
            dataset.save_to_disk(dataset_dir)

        reward_fn = build_reward_fn(tokenizer)

        os.makedirs(iter_output_dir, exist_ok=True)

        print(f"\n[Phase 2] Training (output: {iter_output_dir})...", flush=True)
        t0 = time.time()
        grpo_config = GRPOConfig(
            output_dir=iter_output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=grad_accum,
            num_generations=num_generations,
            max_completion_length=max_new_tokens,
            learning_rate=learning_rate,
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

        train_result = trainer.train()
        train_dur = time.time() - t0
        train_loss = train_result.training_loss if hasattr(train_result, "training_loss") else "N/A"
        print(f"  Training done in {_fmt_duration(train_dur)} | loss={train_loss}  |  GPU: {_gpu_mem_str()}", flush=True)

        print(f"\n[Phase 3] Saving LoRA adapter to {iter_output_dir}...", flush=True)
        model.save_pretrained(iter_output_dir)
        tokenizer.save_pretrained(iter_output_dir)

        iter_dur = time.time() - iter_start
        elapsed_total = time.time() - run_start
        remaining_iters = n_iterations - (iteration + 1)
        eta = _fmt_duration((elapsed_total / (iteration + 1)) * remaining_iters)
        print(
            f"\n── Iteration {iteration + 1}/{n_iterations} done in {_fmt_duration(iter_dur)} "
            f"| loss={train_loss} | ETA remaining: {eta} ──",
            flush=True,
        )

    total_dur = time.time() - run_start
    print(f"\n{'='*60}", flush=True)
    print(f"Training complete in {_fmt_duration(total_dur)}. {n_iterations} iterations.", flush=True)
    print(f"Final adapter: {os.path.join(output_dir, f'iter_{n_iterations - 1}')}", flush=True)


# ── Eval runner ───────────────────────────────────────────────────────────────


async def _eval_episode_async(
    model: Any,
    tokenizer: AutoTokenizer,
    base_url: str,
    env_type: int,
    device: str,
) -> dict[str, Any]:
    """Run one evaluation episode using the fine-tuned model and return a summary."""
    memory_bank: list[dict[str, Any]] = []

    async with InventoryEnvClient(base_url=base_url) as env:
        obs: InventoryObservation = await env.reset(env_type=env_type)
        done = False
        step = 0
        last_info: dict[str, Any] = {}

        while not done:
            obs_dict: dict[str, Any] = {
                "day": obs.day,
                "days_remaining": obs.days_remaining,
                "current_inventory": round(obs.current_inventory, 2),
                "demand_last_5": [round(d, 2) for d in obs.demand_last_5],
                "demand_mean_30d": round(obs.demand_mean_30d, 2),
                "demand_std_30d": round(obs.demand_std_30d, 2),
                "fill_rate_so_far": round(obs.fill_rate_so_far, 4),
                "recent_stockouts": obs.recent_stockouts,
                "recent_lost_sales": round(obs.recent_lost_sales, 2),
                "pending_orders": [
                    {"arrival_day": o.arrival_day, "quantity": o.quantity}
                    for o in obs.pending_orders
                ],
            }

            messages = format_prompt(obs_dict, memory_bank)
            prompt_str: str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            completion = _generate_completion(model, tokenizer, prompt_str, device)
            rop_parsed = parse_rop(completion)
            rop = rop_parsed if rop_parsed is not None else (
                obs.demand_mean_30d * 3 + obs.demand_std_30d * 1.65
            )

            result = await env.step(InventoryAction(reorder_point=rop))
            obs = result.observation
            done = result.done
            last_info = result.info

            memory_bank = (memory_bank + [{
                "day": obs_dict["day"],
                "reorder_point": round(rop, 2),
                "fill_rate_after": round(obs.fill_rate_so_far, 4),
            }])[-MEMORY_SIZE:]

            step += 1
            if step % 30 == 0:
                print(
                    f"Day {obs.day:4d} | "
                    f"inv={obs.current_inventory:7.1f} | "
                    f"ROP={rop:7.1f} | "
                    f"fill={obs.fill_rate_so_far:.3f}"
                )

    summary: dict[str, Any] = {
        "final_fill_rate": last_info.get("fill_rate", obs.fill_rate_so_far),
        "stockouts": last_info.get("stockouts", obs.recent_stockouts),
        "lost_sales": last_info.get("lost_sales", obs.recent_lost_sales),
    }
    print(
        f"\nEval complete — "
        f"fill_rate={summary['final_fill_rate']:.4f}  "
        f"stockouts={summary['stockouts']}  "
        f"lost_sales={summary['lost_sales']:.1f}"
    )
    return summary


def run_eval(
    adapter_path: str,
    base_model: str,
    base_url: str,
    env_type: int,
    device: str,
) -> None:
    """Load a saved LoRA adapter via Unsloth and run one evaluation episode."""
    print(f"Loading adapter from {adapter_path} via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        load_in_4bit=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    FastLanguageModel.for_inference(model)
    print("Model loaded. Running eval episode...")
    asyncio.run(_eval_episode_async(model, tokenizer, base_url, env_type, device))


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning of Qwen2.5-Instruct for inventory optimization."
    )
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", type=str, default="./grpo_inventory")
    parser.add_argument("--base-url", type=str, default="http://localhost:7860")
    parser.add_argument(
        "--env-type",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="0=GammaPoisson, 1=GammaGammaHighVariance, 2=SpikingDemand, 3=SingleGammaLowVariance",
    )
    parser.add_argument("--n-iterations", type=int, default=5)
    parser.add_argument("--episodes-per-iter", type=int, default=20)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Skip training; run one eval episode using a saved adapter.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (required when --eval is set).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.eval:
        if not args.adapter_path:
            print("Error: --adapter-path is required when using --eval.")
            sys.exit(1)
        run_eval(
            adapter_path=args.adapter_path,
            base_model=args.base_model,
            base_url=args.base_url,
            env_type=args.env_type,
            device=args.device,
        )
    else:
        train(
            base_model=args.base_model,
            output_dir=args.output_dir,
            base_url=args.base_url,
            env_type=args.env_type,
            n_iterations=args.n_iterations,
            episodes_per_iter=args.episodes_per_iter,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            device=args.device,
            num_generations=args.num_generations,
            per_device_batch_size=args.per_device_batch_size,
            grad_accum=args.grad_accum,
            max_new_tokens=args.max_new_tokens,
            learning_rate=args.learning_rate,
        )

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
from typing import Any

import datasets
import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from scipy.stats import norm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from client.inventory_client import InventoryAction, InventoryEnvClient, InventoryObservation

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


def compute_proxy_reward(obs_dict: dict[str, Any], rop: float) -> float:
    """
    Compute a local reward (no HTTP needed) given an observation dict and a proposed reorder point.
    Returns a value clipped to [-1, 1].
    """
    LEAD_TIME = 3
    inventory = obs_dict["current_inventory"]
    mean_demand = obs_dict["demand_mean_30d"]
    std_demand = obs_dict["demand_std_30d"]
    pending_qty = sum(o["quantity"] for o in obs_dict.get("pending_orders", []))

    coverage_without_order = inventory + pending_qty

    if coverage_without_order <= rop:
        order_qty = max(0.0, rop - coverage_without_order + mean_demand * LEAD_TIME)
        coverage_with_order = coverage_without_order + order_qty
    else:
        coverage_with_order = coverage_without_order

    days_coverage = coverage_with_order / max(mean_demand, 1.0)

    safety_days = norm.ppf(0.95) * std_demand / max(mean_demand, 1.0) * (LEAD_TIME ** 0.5)
    target_days = LEAD_TIME + safety_days

    deviation = abs(days_coverage - target_days) / max(target_days, 1.0)
    reward = float(max(0.0, 1.0 - 2.0 * deviation))

    if days_coverage < LEAD_TIME:
        reward -= 1.0 * (LEAD_TIME - days_coverage) / LEAD_TIME

    return float(np.clip(reward, -1.0, 1.0))


# ── Model generation helper ───────────────────────────────────────────────────


def _generate_completion(
    model: AutoModelForCausalLM,
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
    model: AutoModelForCausalLM | None,
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

    async with InventoryEnvClient(base_url=base_url) as env:
        obs: InventoryObservation = await env.reset(env_type=env_type)
        done = False
        step = 0

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

            if use_heuristic:
                rop = obs.demand_mean_30d * 3 + obs.demand_std_30d * 1.65
            else:
                completion = _generate_completion(
                    model, tokenizer, prompt_str, device
                )
                rop_parsed = parse_rop(completion)
                rop = rop_parsed if rop_parsed is not None else (
                    obs.demand_mean_30d * 3 + obs.demand_std_30d * 1.65
                )

            rows.append({"prompt": prompt_str, "obs_json": json.dumps(obs_dict)})

            result = await env.step(InventoryAction(reorder_point=rop))
            obs = result.observation
            done = result.done

            memory_bank = (memory_bank + [{
                "day": obs_dict["day"],
                "reorder_point": round(rop, 2),
                "fill_rate_after": round(obs.fill_rate_so_far, 4),
            }])[-MEMORY_SIZE:]

            step += 1

    return rows


def collect_rollout_dataset(
    model: AutoModelForCausalLM | None,
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

    for ep in range(n_episodes):
        use_heuristic = model is None or ep == 0
        mode = "heuristic" if use_heuristic else "model"
        print(f"  Collecting episode {ep + 1}/{n_episodes} ({mode})...")
        rows = asyncio.run(
            _run_episode_async(model, tokenizer, base_url, env_type, ep, device)
        )
        all_rows.extend(rows)
        print(f"    Episode {ep + 1} collected {len(rows)} steps (total so far: {len(all_rows)})")

    return datasets.Dataset.from_list(all_rows)


# ── Reward function factory ───────────────────────────────────────────────────


def build_reward_fn(tokenizer: AutoTokenizer):  # noqa: ARG001
    """
    Returns a closure compatible with TRL's GRPOTrainer reward_funcs interface.
    Handles both list[str] and list[list[str]] batching from TRL internals.
    """

    def reward_fn(
        prompts: list[str],
        completions: list[str],
        obs_json: list[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        if obs_json is None:
            return [-1.0] * len(completions)

        rewards: list[float] = []
        for completion, obs_j in zip(completions, obs_json):
            if isinstance(obs_j, list):
                obs_j = obs_j[0] if obs_j else "{}"
            try:
                obs_dict = json.loads(obs_j)
            except (json.JSONDecodeError, TypeError):
                rewards.append(-1.0)
                continue

            rop = parse_rop(completion)
            if rop is None:
                rewards.append(-1.0)
            else:
                rewards.append(compute_proxy_reward(obs_dict, rop))

        return rewards

    return reward_fn


# ── Model setup ───────────────────────────────────────────────────────────────


def setup_model_and_tokenizer(
    base_model: str,
    lora_rank: int,
    lora_alpha: int,
    device: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model + tokenizer and wrap with LoRA adapters."""
    print(f"Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model from {base_model} (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
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

    model: AutoModelForCausalLM | None = None
    tokenizer: AutoTokenizer | None = None

    for iteration in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{n_iterations}")
        print(f"{'='*60}")

        if iteration == 0:
            print("\n[Phase 0] Setting up model and tokenizer...")
            model, tokenizer = setup_model_and_tokenizer(
                base_model, lora_rank, lora_alpha, device
            )

        print(f"\n[Phase 1] Collecting rollout dataset ({episodes_per_iter} episodes)...")
        dataset = collect_rollout_dataset(
            model, tokenizer, base_url, env_type, episodes_per_iter, device
        )
        print(f"  Dataset size: {len(dataset)} steps")

        print("\n[Phase 2] Building reward function...")
        reward_fn = build_reward_fn(tokenizer)

        iter_output_dir = os.path.join(output_dir, f"iter_{iteration}")
        os.makedirs(iter_output_dir, exist_ok=True)

        print(f"\n[Phase 2] Configuring GRPOTrainer (output: {iter_output_dir})...")
        grpo_config = GRPOConfig(
            output_dir=iter_output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=grad_accum,
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            save_steps=0,
            report_to="none",
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        print("[Phase 2] Training...")
        train_result = trainer.train()

        train_loss = train_result.training_loss if hasattr(train_result, "training_loss") else "N/A"
        print(f"  Training complete. Loss: {train_loss}")

        print(f"\n[Phase 3] Saving LoRA adapter to {iter_output_dir}...")
        model.save_pretrained(iter_output_dir)
        tokenizer.save_pretrained(iter_output_dir)

        print(
            f"\nIteration {iteration + 1} summary: "
            f"dataset_size={len(dataset)}, "
            f"loss={train_loss}, "
            f"adapter_saved={iter_output_dir}"
        )

    print(f"\n{'='*60}")
    print(f"Training complete. {n_iterations} iterations finished.")
    print(f"Final adapter: {os.path.join(output_dir, f'iter_{n_iterations - 1}')}")


# ── Eval runner ───────────────────────────────────────────────────────────────


async def _eval_episode_async(
    model: AutoModelForCausalLM,
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
    """Load a saved LoRA adapter and run one evaluation episode."""
    from peft import PeftModel

    print(f"Loading base model {base_model} for eval...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
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

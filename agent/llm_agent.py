"""
LLM-powered inventory optimization agent using Claude.

Usage:
    python -m agent.llm_agent --env-type 0 --base-url http://localhost:7860

Or directly:
    python agent/llm_agent.py --env-type 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any

import anthropic

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from client.inventory_client import InventoryAction, InventoryEnvClient, InventoryObservation

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


class ClaudeInventoryAgent:
    """Inventory optimization agent backed by Claude claude-sonnet-4-5."""

    MEMORY_SIZE = 15
    HISTORY_TURNS = 6
    MODEL = "claude-sonnet-4-5"

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._memory_bank: list[dict[str, Any]] = []
        self._conversation: list[dict[str, str]] = []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_user_message(self, obs: InventoryObservation) -> str:
        snapshot: dict[str, Any] = {
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
            "memory_bank": self._memory_bank[-self.MEMORY_SIZE :],
        }
        return json.dumps(snapshot, separators=(",", ":"))

    def _fallback_rop(self, obs: InventoryObservation) -> float:
        return obs.demand_mean_30d * 3 + obs.demand_std_30d * 1.65

    def _parse_response(self, text: str) -> dict[str, Any]:
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
        return json.loads(cleaned)

    def _update_memory(
        self,
        obs: InventoryObservation,
        reorder_point: float,
        reasoning: str,
    ) -> None:
        entry: dict[str, Any] = {
            "day": obs.day,
            "reorder_point": round(reorder_point, 2),
            "reasoning_snippet": reasoning[:80],
            "fill_rate_after": round(obs.fill_rate_so_far, 4),
        }
        self._memory_bank = (self._memory_bank + [entry])[-self.MEMORY_SIZE :]

    # ── Public interface ──────────────────────────────────────────────────────

    def decide(self, obs: InventoryObservation) -> tuple[float, str, float]:
        """
        Return (reorder_point, reasoning, confidence) for the given observation.
        Falls back to a safety-stock formula if the LLM response cannot be parsed.
        """
        user_content = self._build_user_message(obs)
        user_msg: dict[str, str] = {"role": "user", "content": user_content}

        trimmed_history = self._conversation[-self.HISTORY_TURNS :]
        messages = trimmed_history + [user_msg]

        reorder_point: float
        reasoning: str
        confidence: float

        try:
            response = self._client.messages.create(
                model=self.MODEL,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            raw_text: str = response.content[0].text  # type: ignore[union-attr]

            try:
                parsed = self._parse_response(raw_text)
                reorder_point = float(parsed["reorder_point"])
                reasoning = str(parsed.get("reasoning", ""))
                confidence = float(parsed.get("confidence", 0.5))
            except (json.JSONDecodeError, KeyError, ValueError):
                match = re.search(r'"reorder_point"\s*:\s*([0-9]+\.?[0-9]*)', raw_text)
                reorder_point = float(match.group(1)) if match else self._fallback_rop(obs)
                reasoning = raw_text[:200]
                confidence = 0.1

            assistant_msg: dict[str, str] = {"role": "assistant", "content": raw_text}
            self._conversation = (self._conversation + [user_msg, assistant_msg])[
                -self.HISTORY_TURNS * 2 :
            ]

        except Exception as exc:  # noqa: BLE001
            reorder_point = self._fallback_rop(obs)
            reasoning = f"API error — fallback used: {exc}"
            confidence = 0.0

        reorder_point = max(0.0, reorder_point)
        self._update_memory(obs, reorder_point, reasoning)
        return reorder_point, reasoning, confidence

    def reset(self) -> None:
        """Clear per-episode state (memory and conversation history)."""
        self._memory_bank = []
        self._conversation = []


# ── Episode runner ────────────────────────────────────────────────────────────


async def run_episode(
    base_url: str,
    env_type: int,
    api_key: str,
) -> dict[str, Any]:
    """
    Run a full episode against the inventory server and return a summary dict.

    Returns
    -------
    dict with keys: final_fill_rate, stockouts, lost_sales
    """
    agent = ClaudeInventoryAgent(api_key=api_key)

    async with InventoryEnvClient(base_url=base_url) as env:
        obs = await env.reset(env_type=env_type)
        agent.reset()

        step = 0
        done = False
        last_info: dict[str, Any] = {}

        while not done:
            rop, reasoning, confidence = agent.decide(obs)

            result = await env.step(InventoryAction(reorder_point=rop, reasoning=reasoning))
            obs = result.observation
            done = result.done
            last_info = result.info
            step += 1

            if step % 30 == 0:
                snippet = reasoning[:60].replace("\n", " ")
                print(
                    f"Day {obs.day:4d} | "
                    f"inv={obs.current_inventory:7.1f} | "
                    f"ROP={rop:7.1f} | "
                    f"fill={obs.fill_rate_so_far:.3f} | "
                    f"conf={confidence:.2f} | "
                    f"{snippet}..."
                )

    summary: dict[str, Any] = {
        "final_fill_rate": last_info.get("fill_rate", obs.fill_rate_so_far),
        "stockouts": last_info.get("stockouts", obs.recent_stockouts),
        "lost_sales": last_info.get("lost_sales", obs.recent_lost_sales),
    }

    print(
        f"\nEpisode complete — "
        f"fill_rate={summary['final_fill_rate']:.4f}  "
        f"stockouts={summary['stockouts']}  "
        f"lost_sales={summary['lost_sales']:.1f}"
    )
    return summary


# ── Entry point ───────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a ClaudeInventoryAgent episode against the inventory server."
    )
    parser.add_argument(
        "--env-type",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help=(
            "Demand environment: "
            "0=GammaPoisson, 1=GammaGammaHighVariance, "
            "2=SpikingDemand, 3=SingleGammaLowVariance"
        ),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:7860",
        help="Base URL of the inventory environment server.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.api_key:
        print("Error: no Anthropic API key provided. Set ANTHROPIC_API_KEY or use --api-key.")
        sys.exit(1)

    asyncio.run(
        run_episode(
            base_url=args.base_url,
            env_type=args.env_type,
            api_key=args.api_key,
        )
    )

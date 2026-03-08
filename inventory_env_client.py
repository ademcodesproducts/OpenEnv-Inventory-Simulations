"""
OpenEnv WebSocket client for the Inventory Reasoning Environment.

Usage:
    from inventory_env_client import InventoryEnv
    from models import InventoryAction

    with InventoryEnv(base_url="https://YOUR-SPACE.hf.space") as env:
        result = env.reset(env_type=0)
        obs = result.observation
        while not result.done:
            result = env.step(InventoryAction(reorder_point=450.0))
        print(f"Final fill rate: {result.observation.fill_rate_so_far:.3f}")
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from models import InventoryAction, InventoryObservation, PendingOrderModel


class InventoryEnv(EnvClient[InventoryAction, InventoryObservation, State]):
    """WebSocket client for the Inventory Reasoning Environment."""

    def _step_payload(self, action: InventoryAction) -> Dict[str, Any]:
        return {
            "reorder_point": action.reorder_point,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[InventoryObservation]:
        obs_data = payload.get("observation", {})
        pending_raw = obs_data.get("pending_orders", [])
        pending = [PendingOrderModel(**p) if isinstance(p, dict) else p for p in pending_raw]

        obs = InventoryObservation(
            day=obs_data.get("day", 0),
            current_inventory=obs_data.get("current_inventory", 0.0),
            demand_last_5=obs_data.get("demand_last_5", []),
            demand_mean_30d=obs_data.get("demand_mean_30d", 0.0),
            demand_std_30d=obs_data.get("demand_std_30d", 0.0),
            fill_rate_so_far=obs_data.get("fill_rate_so_far", 0.0),
            recent_stockouts=obs_data.get("recent_stockouts", 0),
            recent_lost_sales=obs_data.get("recent_lost_sales", 0.0),
            days_remaining=obs_data.get("days_remaining", 0),
            pending_orders=pending,
            demand_last_year_7d=obs_data.get("demand_last_year_7d", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

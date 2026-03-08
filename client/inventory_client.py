"""
Async HTTP client for the Inventory Reasoning Environment.

Usage:
    import asyncio
    from client.inventory_client import InventoryEnvClient, InventoryAction

    async def main():
        async with InventoryEnvClient("http://localhost:7860") as env:
            obs = await env.reset(env_type=0)
            print(obs)

            result = await env.step(InventoryAction(
                reorder_point=350.0,
                reasoning="Safety stock estimate based on 30-day history"
            ))
            print(result.reward, result.done)

    asyncio.run(main())
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import httpx


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class InventoryAction:
    reorder_point: float
    reasoning: str = ""


@dataclass
class PendingOrder:
    arrival_day: int
    quantity: int


@dataclass
class InventoryObservation:
    day: int
    current_inventory: float
    demand_last_5: List[float]
    demand_mean_30d: float
    demand_std_30d: float
    fill_rate_so_far: float
    recent_stockouts: int
    recent_lost_sales: float
    days_remaining: int
    pending_orders: List[PendingOrder]
    demand_last_year_7d: List[float]

    @classmethod
    def from_dict(cls, d: dict) -> "InventoryObservation":
        return cls(
            day=d["day"],
            current_inventory=d["current_inventory"],
            demand_last_5=d["demand_last_5"],
            demand_mean_30d=d["demand_mean_30d"],
            demand_std_30d=d["demand_std_30d"],
            fill_rate_so_far=d["fill_rate_so_far"],
            recent_stockouts=d["recent_stockouts"],
            recent_lost_sales=d["recent_lost_sales"],
            days_remaining=d["days_remaining"],
            pending_orders=[PendingOrder(**o) for o in d["pending_orders"]],
            demand_last_year_7d=d.get("demand_last_year_7d", []),
        )


@dataclass
class StepResult:
    observation: InventoryObservation
    reward: float
    done: bool
    info: dict

    @classmethod
    def from_dict(cls, d: dict) -> "StepResult":
        return cls(
            observation=InventoryObservation.from_dict(d["observation"]),
            reward=d["reward"],
            done=d["done"],
            info=d["info"],
        )


# ── Client ────────────────────────────────────────────────────────────────────

class InventoryEnvClient:
    """
    Async client for the inventory environment server.

    Parameters
    ----------
    base_url : str
        Base URL of the running server, e.g. "http://localhost:7860" or
        "https://YOUR_USERNAME-inventory-env.hf.space"
    timeout : float
        Request timeout in seconds (default 30).
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._timeout = timeout

    async def __aenter__(self) -> "InventoryEnvClient":
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)
        return self

    async def __aexit__(self, *_):
        if self._client:
            await self._client.aclose()

    def _ensure_client(self):
        if self._client is None:
            raise RuntimeError("Use 'async with InventoryEnvClient(...) as env:' context manager")

    async def reset(self, env_type: int = 0) -> InventoryObservation:
        """Initialize a new episode. env_type: 0=GammaPoisson, 1=GammaGammaHighVariance,
        2=SpikingDemand, 3=SingleGammaLowVariance"""
        self._ensure_client()
        r = await self._client.post("/reset", params={"env_type": env_type})
        r.raise_for_status()
        return InventoryObservation.from_dict(r.json())

    async def step(self, action: InventoryAction) -> StepResult:
        """Advance one simulation day with the given reorder point."""
        self._ensure_client()
        r = await self._client.post(
            "/step",
            json={"reorder_point": action.reorder_point, "reasoning": action.reasoning},
        )
        r.raise_for_status()
        return StepResult.from_dict(r.json())

    async def state(self) -> dict:
        """Get current episode metadata without advancing the simulation."""
        self._ensure_client()
        r = await self._client.get("/state")
        r.raise_for_status()
        return r.json()

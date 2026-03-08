"""
OpenEnv Environment implementation for the Inventory Reasoning Environment.

Subclasses openenv.core.env_server.interfaces.Environment to provide a
Gymnasium-style step/reset/state API over the stochastic inventory simulation.
"""

from __future__ import annotations

import sys
import os
from typing import Any, List
from uuid import uuid4

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import InventoryAction, InventoryObservation, PendingOrderModel
from config import SIM_DAYS, HISTO_DAYS, LEAD_TIME, WRITE_OFF_RATE
from reward import compute_daily_pnl
from demand_environment import (
    GammaPoisson,
    GammaGammaHighVariance,
    SpikingDemand,
    SingleGammaLowVariance,
)
from demand_calculator import DemandCalculator
from order_processor import OrderProcessor
from performance_tracker import PerformanceTracker

ENV_TYPES = {
    0: GammaPoisson,
    1: GammaGammaHighVariance,
    2: SpikingDemand,
    3: SingleGammaLowVariance,
}


class InventoryEnvironment(Environment):
    """
    Stochastic inventory simulation exposed as an OpenEnv Environment.

    Each episode generates a 730-day demand series (365-day warm-up + 365 decision
    days). The agent sets a reorder_point each step; the environment computes
    deliveries, spoilage, demand fulfillment, reorders, and a P&L-based reward.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._day: int = 0
        self._inventory: float = 0.0
        self._demand_series: List[int] = []
        self._order_processor = OrderProcessor()
        self._performance_tracker = PerformanceTracker()
        self._total_demand: float = 0.0
        self._total_fulfilled: float = 0.0
        self._stockouts: int = 0
        self._lost_sales: float = 0.0
        self._initialized: bool = False

    # ── helpers ────────────────────────────────────────────────────────────────

    def _get_obs(self, *, reward: float = 0.0, done: bool = False,
                 extra_meta: dict | None = None) -> InventoryObservation:
        hist30 = self._demand_series[max(0, self._day - 30): self._day]
        last5 = self._demand_series[max(0, self._day - 5): self._day]

        pending = [
            PendingOrderModel(arrival_day=o.arrival_day, quantity=o.quantity)
            for o in self._order_processor.order_queue[:5]
        ]

        ly_anchor = self._day - 365
        ly_start = max(0, ly_anchor - 3)
        ly_end = min(len(self._demand_series), ly_anchor + 4)
        demand_last_year_7d = [float(d) for d in self._demand_series[ly_start:ly_end]]

        meta = extra_meta or {}
        return InventoryObservation(
            day=self._day,
            current_inventory=self._inventory,
            demand_last_5=[float(d) for d in last5],
            demand_mean_30d=float(np.mean(hist30)) if hist30 else 0.0,
            demand_std_30d=float(np.std(hist30)) if len(hist30) > 1 else 0.0,
            fill_rate_so_far=(
                self._total_fulfilled / self._total_demand
                if self._total_demand > 0 else 0.0
            ),
            recent_stockouts=self._stockouts,
            recent_lost_sales=self._lost_sales,
            days_remaining=SIM_DAYS - self._day,
            pending_orders=pending,
            demand_last_year_7d=demand_last_year_7d,
            done=done,
            reward=reward,
            metadata=meta,
        )

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, episode_id: str | None = None,
              **kwargs: Any) -> InventoryObservation:
        env_type = int(kwargs.get("env_type", 0))
        if env_type not in ENV_TYPES:
            env_type = 0

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        env_class = ENV_TYPES[env_type]
        environment = env_class(SIM_DAYS)
        dc = DemandCalculator(SIM_DAYS)
        dc.set_environment(environment)
        self._demand_series = [dc.get_daily_demand(i) for i in range(SIM_DAYS)]

        self._day = HISTO_DAYS
        self._inventory = 0.0
        self._order_processor = OrderProcessor()
        self._performance_tracker = PerformanceTracker()
        self._total_demand = 0.0
        self._total_fulfilled = 0.0
        self._stockouts = 0
        self._lost_sales = 0.0
        self._initialized = True

        return self._get_obs(reward=0.0, done=False)

    def step(self, action: InventoryAction, timeout_s: float | None = None,
             **kwargs: Any) -> InventoryObservation:
        if not self._initialized:
            raise RuntimeError("Call reset() before step()")
        if self._day >= SIM_DAYS:
            raise RuntimeError("Episode already done. Call reset().")

        day = self._day
        demand = self._demand_series[day]

        # 1. Deliver pending orders
        delivered = sum(
            o.quantity for o in self._order_processor.order_queue
            if o.arrival_day == day
        )
        self._inventory += delivered
        self._order_processor.order_queue = [
            o for o in self._order_processor.order_queue if o.arrival_day > day
        ]

        # 2. Daily spoilage
        spoilage = self._inventory * WRITE_OFF_RATE
        self._inventory = max(0.0, self._inventory - spoilage)
        self._performance_tracker.write_offs += spoilage

        # 3. Fulfill demand
        units_sold = min(demand, self._inventory)
        self._inventory = max(0.0, self._inventory - demand)
        lost = max(0.0, demand - units_sold)
        if lost > 0:
            self._stockouts += 1
        self._lost_sales += lost
        self._total_demand += demand
        self._total_fulfilled += units_sold

        # 4. Reorder logic
        rop = max(0.0, action.reorder_point)
        qty = 0
        hist = self._demand_series[max(0, day - 30): day]
        mean_demand = float(np.mean(hist)) if hist else 0.0
        pipeline = sum(o.quantity for o in self._order_processor.order_queue)
        inv_position = self._inventory + pipeline
        if day < SIM_DAYS - LEAD_TIME and inv_position <= rop:
            qty = max(0.0, rop - inv_position + mean_demand * LEAD_TIME)
            if qty > 0:
                self._order_processor.place_order(day, int(qty))

        # 5. Track
        self._performance_tracker.daily_performance(
            demand_quantity=demand,
            fulfilled_demand=int(units_sold),
            daily_writeoff=0,
        )

        self._day += 1
        self._state.step_count += 1
        done = self._day >= SIM_DAYS

        fill_rate = (
            self._total_fulfilled / self._total_demand
            if self._total_demand > 0 else 0.0
        )

        pnl = compute_daily_pnl(
            units_sold=units_sold,
            lost=lost,
            inventory_after=self._inventory,
            ordered_qty=qty,
            spoilage=spoilage,
            mean_demand=mean_demand,
        )
        reward = pnl["daily_reward"]

        return self._get_obs(
            reward=reward,
            done=done,
            extra_meta={
                "fill_rate": fill_rate,
                "stockouts": self._stockouts,
                "lost_sales": self._lost_sales,
                "inventory_in": delivered,
                "units_sold": units_sold,
                "daily_profit": pnl["daily_profit"],
                "daily_reward": pnl["daily_reward"],
                "reasoning_logged": action.reasoning[:200] if action.reasoning else "",
            },
        )

    @property
    def state(self) -> State:
        return self._state

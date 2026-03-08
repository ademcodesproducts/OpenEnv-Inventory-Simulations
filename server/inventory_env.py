import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import (
    SIM_DAYS, HISTO_DAYS, LEAD_TIME,
    WRITE_OFF_RATE, WRITE_OFF_FREQUENCY,
)
from reward import compute_daily_pnl
from demand_environment import (
    GammaPoisson, GammaGammaHighVariance, SpikingDemand, SingleGammaLowVariance,
)
from demand_calculator import DemandCalculator
from order_processor import OrderProcessor
from performance_tracker import PerformanceTracker

app = FastAPI(title="Inventory Reasoning Environment")

ENV_TYPES = {
    0: GammaPoisson,
    1: GammaGammaHighVariance,
    2: SpikingDemand,
    3: SingleGammaLowVariance,
}


# ── Pydantic models (request/response) ───────────────────────────────────────

class InventoryAction(BaseModel):
    reorder_point: float
    reasoning: str = ""


class PendingOrder(BaseModel):
    arrival_day: int
    quantity: int


class InventoryObservation(BaseModel):
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


class StepResult(BaseModel):
    observation: InventoryObservation
    reward: float
    done: bool
    info: dict


class StateResponse(BaseModel):
    day: int
    fill_rate: float
    done: bool
    total_demand: float
    total_fulfilled: float
    stockouts: int
    lost_sales: float


# ── Episode state (single global episode for simplicity) ─────────────────────

class EpisodeState:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.day: int = 0
        self.inventory: float = 0.0
        self.demand_series: List[int] = []
        self.order_processor = OrderProcessor()
        self.performance_tracker = PerformanceTracker()
        self.total_demand: float = 0.0
        self.total_fulfilled: float = 0.0
        self.stockouts: int = 0
        self.lost_sales: float = 0.0
        self.initialized: bool = False

    def get_obs(self) -> InventoryObservation:
        hist_start = max(0, self.day - HISTO_DAYS)
        hist = self.demand_series[hist_start:self.day]
        last5 = self.demand_series[max(0, self.day - 5):self.day]
        hist30 = self.demand_series[max(0, self.day - 30):self.day]

        pending = [
            PendingOrder(arrival_day=o.arrival_day, quantity=o.quantity)
            for o in self.order_processor.order_queue[:5]
        ]

        ly_anchor = self.day - 365
        ly_start = max(0, ly_anchor - 3)
        ly_end = min(len(self.demand_series), ly_anchor + 4)
        demand_last_year_7d = [float(d) for d in self.demand_series[ly_start:ly_end]]

        return InventoryObservation(
            day=self.day,
            current_inventory=self.inventory,
            demand_last_5=[float(d) for d in last5],
            demand_mean_30d=float(np.mean(hist30)) if hist30 else 0.0,
            demand_std_30d=float(np.std(hist30)) if len(hist30) > 1 else 0.0,
            fill_rate_so_far=(
                self.total_fulfilled / self.total_demand
                if self.total_demand > 0 else 0.0
            ),
            recent_stockouts=self.stockouts,
            recent_lost_sales=self.lost_sales,
            days_remaining=SIM_DAYS - self.day,
            pending_orders=pending,
            demand_last_year_7d=demand_last_year_7d,
        )


episode = EpisodeState()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=InventoryObservation)
def reset(env_type: int = 0):
    if env_type not in ENV_TYPES:
        raise HTTPException(status_code=400, detail=f"env_type must be 0-{len(ENV_TYPES)-1}")

    episode.reset_state()

    env_class = ENV_TYPES[env_type]
    environment = env_class(SIM_DAYS)
    dc = DemandCalculator(SIM_DAYS)
    dc.set_environment(environment)
    episode.demand_series = [dc.get_daily_demand(i) for i in range(SIM_DAYS)]

    # Warm up history (agents use HISTO_DAYS of history before acting)
    episode.day = HISTO_DAYS
    episode.initialized = True

    return episode.get_obs()


@app.post("/step", response_model=StepResult)
def step(action: InventoryAction):
    if not episode.initialized:
        raise HTTPException(status_code=400, detail="Call /reset before /step")
    if episode.day >= SIM_DAYS:
        raise HTTPException(status_code=400, detail="Episode already done. Call /reset.")

    day = episode.day
    demand = episode.demand_series[day]

    # 1. Deliver pending orders
    delivered = sum(
        o.quantity for o in episode.order_processor.order_queue
        if o.arrival_day == day
    )
    episode.inventory += delivered
    episode.order_processor.order_queue = [
        o for o in episode.order_processor.order_queue if o.arrival_day > day
    ]

    # 2. Daily spoilage (0.143% per day)
    spoilage = episode.inventory * WRITE_OFF_RATE
    episode.inventory = max(0.0, episode.inventory - spoilage)
    episode.performance_tracker.write_offs += spoilage

    # 3. Fulfill demand
    units_sold = min(demand, episode.inventory)
    episode.inventory = max(0.0, episode.inventory - demand)
    lost = max(0.0, demand - units_sold)
    if lost > 0:
        episode.stockouts += 1
    episode.lost_sales += lost
    episode.total_demand += demand
    episode.total_fulfilled += units_sold

    # 4. Reorder if inventory at or below ROP
    rop = max(0.0, action.reorder_point)
    qty = 0
    hist = episode.demand_series[max(0, day - 30):day]
    mean_demand = float(np.mean(hist)) if hist else 0.0
    pipeline = sum(o.quantity for o in episode.order_processor.order_queue)
    inv_position = episode.inventory + pipeline
    if day < SIM_DAYS - LEAD_TIME and inv_position <= rop:
        qty = max(0.0, rop - inv_position + mean_demand * LEAD_TIME)
        if qty > 0:
            episode.order_processor.place_order(day, int(qty))

    # 5. Track performance
    episode.performance_tracker.daily_performance(
        demand_quantity=demand,
        fulfilled_demand=int(units_sold),
        daily_writeoff=0,
    )

    episode.day += 1
    done = episode.day >= SIM_DAYS

    fill_rate = (
        episode.total_fulfilled / episode.total_demand
        if episode.total_demand > 0 else 0.0
    )

    pnl = compute_daily_pnl(
        units_sold=units_sold,
        lost=lost,
        inventory_after=episode.inventory,
        ordered_qty=qty,
        spoilage=spoilage,
        mean_demand=mean_demand,
    )
    reward = pnl["daily_reward"]

    return StepResult(
        observation=episode.get_obs(),
        reward=reward,
        done=done,
        info={
            "fill_rate": fill_rate,
            "stockouts": episode.stockouts,
            "lost_sales": episode.lost_sales,
            "inventory_in": delivered,
            "units_sold": units_sold,
            "daily_profit": pnl["daily_profit"],
            "daily_reward": pnl["daily_reward"],
            "reasoning_logged": action.reasoning[:200] if action.reasoning else "",
        },
    )


@app.get("/state", response_model=StateResponse)
def state():
    if not episode.initialized:
        raise HTTPException(status_code=400, detail="Call /reset first")
    fill_rate = (
        episode.total_fulfilled / episode.total_demand
        if episode.total_demand > 0 else 0.0
    )
    return StateResponse(
        day=episode.day,
        fill_rate=fill_rate,
        done=episode.day >= SIM_DAYS,
        total_demand=episode.total_demand,
        total_fulfilled=episode.total_fulfilled,
        stockouts=episode.stockouts,
        lost_sales=episode.lost_sales,
    )

"""
OpenEnv-compatible Pydantic models for the Inventory Reasoning Environment.

Defines the Action, Observation, and nested types following the OpenEnv SDK pattern
(openenv-core >= 0.2.1).
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation


class PendingOrderModel(BaseModel):
    arrival_day: int
    quantity: int


class InventoryAction(Action):
    """Agent sets a reorder point each simulation day."""

    reorder_point: float = Field(..., description="Inventory level that triggers a replenishment order")
    reasoning: str = Field(default="", description="Optional reasoning for the decision")


class InventoryObservation(Observation):
    """Full observation returned after each simulation step."""

    day: int = Field(default=0, description="Current simulation day")
    current_inventory: float = Field(default=0.0, description="Inventory on hand")
    demand_last_5: List[float] = Field(default_factory=list)
    demand_mean_30d: float = Field(default=0.0)
    demand_std_30d: float = Field(default=0.0)
    fill_rate_so_far: float = Field(default=0.0)
    recent_stockouts: int = Field(default=0)
    recent_lost_sales: float = Field(default=0.0)
    days_remaining: int = Field(default=0)
    pending_orders: List[PendingOrderModel] = Field(default_factory=list)
    demand_last_year_7d: List[float] = Field(default_factory=list)

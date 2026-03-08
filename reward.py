from config import (
    FIXED_ORDER_COST,
    SELLING_PRICE,
    UNIT_COST,
)

# Holding cost rate (fraction of unit cost per day)
HOLDING_RATE = 0.02


# ── Core P&L computation ───────────────────────────────────────────────────────

def compute_daily_pnl(
    units_sold: float,
    lost: float,
    inventory_after: float,
    ordered_qty: float,
    spoilage: float,
    mean_demand: float,
) -> dict[str, float]:
    """
    Compute the full P&L breakdown for one simulation day given actuals.

    Parameters
    ----------
    units_sold      : units fulfilled from inventory
    lost            : units of unmet demand (stockout)
    inventory_after : inventory level AFTER demand fulfillment
    ordered_qty     : units ordered this day (0 if no order placed)
    spoilage        : units written off due to daily spoilage
    mean_demand     : 30-day mean demand (used to normalise reward)

    Returns
    -------
    dict with keys:
        revenue, holding_cost, stockout_penalty, order_cost, writeoff_cost,
        daily_profit, daily_reward
    """
    revenue          = units_sold  * SELLING_PRICE
    holding_cost     = inventory_after * UNIT_COST * HOLDING_RATE
    stockout_penalty = lost        * (SELLING_PRICE - UNIT_COST)
    order_cost       = (FIXED_ORDER_COST if ordered_qty > 0 else 0.0) + ordered_qty * UNIT_COST
    writeoff_cost    = spoilage    * UNIT_COST

    daily_profit = revenue - holding_cost - stockout_penalty - order_cost - writeoff_cost

    baseline     = mean_demand * (SELLING_PRICE - UNIT_COST)
    daily_reward = daily_profit / baseline if baseline > 0 else 0.0

    return {
        "revenue":          revenue,
        "holding_cost":     holding_cost,
        "stockout_penalty": stockout_penalty,
        "order_cost":       order_cost,
        "writeoff_cost":    writeoff_cost,
        "daily_profit":     daily_profit,
        "daily_reward":     daily_reward,
    }

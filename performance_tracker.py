class PerformanceTracker:

    def __init__(self):
        self.total_demand = 0
        self.total_fulfilled_demand = 0
        self.fill_rate = 0
        self.write_offs = 0
        self.stock_out_count = 0
        self.total_lost_sales = 0

    def daily_performance(self, demand_quantity, fulfilled_demand, daily_writeoff):
        self.total_demand += demand_quantity
        self.total_fulfilled_demand += fulfilled_demand
        self.fill_rate = 1 - (self.total_demand - self.total_fulfilled_demand ) / self.total_demand if self.total_demand > 0 else 0
        self.write_offs += daily_writeoff
        self.total_lost_sales += (demand_quantity - fulfilled_demand) if demand_quantity > fulfilled_demand else 0

        if fulfilled_demand < demand_quantity:
            self.stock_out_counter()

    def stock_out_counter(self) -> int:
        self.stock_out_count += 1
        return self.stock_out_count

    def performance_summary(self):
        return {
            "total_demand": self.total_demand,
            "fulfilled_demand": self.total_fulfilled_demand,
            "fill_rate": self.fill_rate,
            "write_offs": self.write_offs,
            "stock_out_count": self.stock_out_count,
            "total_lost_sales": self.total_lost_sales
        }

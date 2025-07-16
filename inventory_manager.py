from config import WRITE_OFF_RATE, BASE_STOCK, DEFAULT_SERVICE_LEVEL, MC_SIMS

class InventoryManager:
    def __init__(self, order_processor, agent):
        self.inventory = BASE_STOCK
        self.order_processor = order_processor
        self.total_write_off_quantity = 0
        self.agent = agent

    def reorder(self, time_period):
        reorder_point = self.agent.compute_reorder_point(time_period)
        if self.inventory <= reorder_point:
            self.order_processor.place_order(time_period, reorder_point)

    def inventory_update(self, demand_quantity):
        if self.inventory >= demand_quantity:
            self.inventory -= demand_quantity
        else:
            self.inventory = 0

    def apply_writeoff(self, time_period):
            write_off_quantity = int(self.inventory * WRITE_OFF_RATE)
            self.inventory -= write_off_quantity
            self.total_write_off_quantity += write_off_quantity
            return write_off_quantity

    def process_deliveries(self, time_period):
        processed_delivery = self.order_processor.manage_order(time_period)
        self.inventory += processed_delivery

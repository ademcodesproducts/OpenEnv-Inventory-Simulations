from dataclasses import dataclass
from typing import List
from config import LEAD_TIME

@dataclass
class Order:
    arrival_day: int
    quantity: int

class OrderProcessor:
    def __init__(self):
        self.order_queue: List[Order] = [] # self.order_queue stores Order objects

    def place_order(self, time_period: int, quantity: int):
        arrival_day = time_period + LEAD_TIME # time_period = current_day
        self.order_queue.append(Order(arrival_day=arrival_day, quantity=quantity))

    def manage_order(self, time_period: int) -> int:
        arrived_orders = [order for order in self.order_queue if order.arrival_day == time_period]
        # if order.arrival_day < current_day order excluded after each call of process_order
        self.order_queue = [order for order in self.order_queue if order.arrival_day > time_period]
        return sum(order.quantity for order in arrived_orders)




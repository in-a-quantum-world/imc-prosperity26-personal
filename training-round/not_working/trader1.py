from datamodel import OrderDepth, TradingState, Order
from typing import List


class Trader:
    
    POSITION_LIMITS = {
        "EMERALDS": 70,
        "TOMATOES": 70,
    }
    
    EMERALD_FAIR_VALUE = 10000

    def run(self, state: TradingState):
        result = {}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            
            if best_bid is None or best_ask is None:
                continue

            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 70)
            buy_capacity  = limit - position
            sell_capacity = limit + position

            if product == "EMERALDS":
                fair_value = self.EMERALD_FAIR_VALUE
            else:
                # Tomatoes: use integer midpoint as fair value
                fair_value = (best_bid + best_ask) // 2

            orders = []

            if best_bid > fair_value:
                if sell_capacity > 0:
                    orders.append(Order(product, best_bid + 1, -sell_capacity))

            elif best_ask < fair_value:
                if buy_capacity > 0:
                    orders.append(Order(product, best_ask - 1, buy_capacity))

            else:
                if buy_capacity > 0:
                    orders.append(Order(product, best_bid + 1, buy_capacity))
                if sell_capacity > 0:
                    orders.append(Order(product, best_ask - 1, -sell_capacity))

            result[product] = orders

        return result, 0, ""

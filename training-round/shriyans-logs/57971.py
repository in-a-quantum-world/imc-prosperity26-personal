from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

class Trader:
    EMERALDS = "EMERALDS"
    TOMATOES = "TOMATOES"

    def run(self, state: TradingState):
        result = {}

        if self.EMERALDS in state.order_depths:
            result[self.EMERALDS] = self.trade_emeralds(state)

        if self.TOMATOES in state.order_depths:
            result[self.TOMATOES] = self.trade_tomatoes(state)

        return result, 0, ""

    def get_order_depth(self, state: TradingState, product: str):
        order_depth = state.order_depths[product]
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def trade_emeralds(self, state: TradingState) -> List[Order]:
        best_bid, best_ask = self.get_order_depth(state, self.EMERALDS)
        if best_bid is None or best_ask is None:
            return []

        EMERALD_FAIR_VALUE = 10000
        EMERALD_SOFT_LIMIT = 70
        current_position = state.position.get(self.EMERALDS, 0)
        return self.market_make(
            self.EMERALDS, best_bid, best_ask, EMERALD_FAIR_VALUE, current_position, EMERALD_SOFT_LIMIT
        )

    def trade_tomatoes(self, state: TradingState) -> List[Order]:
        best_bid, best_ask = self.get_order_depth(state, self.TOMATOES)
        if best_bid is None or best_ask is None:
            return []

        current_position = state.position.get(self.TOMATOES, 0)
        fair_value = (best_bid + best_ask) // 2
        TOMATO_SOFT_LIMIT = 70
        return self.market_make(
            self.TOMATOES, best_bid, best_ask, fair_value, current_position, TOMATO_SOFT_LIMIT
        )

    def market_make(
            self,
            product: str,
            best_bid: int,
            best_ask: int,
            fair_value: int,
            position: int,
            soft_limit: int,
    ) -> List[Order]:
        orders = []

        buy_capacity = soft_limit - position
        sell_capacity = soft_limit + position

        # Short when price is above fair value (best_bid > fair_value)
        if best_bid > fair_value:
            if sell_capacity > 0:
                orders.append(Order(product, best_bid + 1, -sell_capacity))
        # Buy when price is below fair value (best_ask < fair_value)
        elif best_ask < fair_value:
            if buy_capacity > 0:
                orders.append(Order(product, best_ask - 1, buy_capacity))
        # Fair value equals best_ask: submit bid at best_bid + 1 and ask at best_ask - 1
        else:
            if buy_capacity > 0:
                orders.append(Order(product, best_bid + 1, buy_capacity))
            if sell_capacity > 0:
                orders.append(Order(product, best_ask - 1, -sell_capacity))

        return orders
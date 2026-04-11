from datamodel import OrderDepth, TradingState, Order
from typing import List


LIMIT = 79


class Trader:

    EMERALD_FV = 10000

    def run(self, state: TradingState):
        print(state.toJSON())
        result = {}

        if "EMERALDS" in state.order_depths:
            result["EMERALDS"] = self._emeralds(state)

        if "TOMATOES" in state.order_depths:
            result["TOMATOES"] = self._tomatoes(state)

        return result, 0, ""

    def _best(self, od: OrderDepth):
        bid = max(od.buy_orders.keys())  if od.buy_orders  else None
        ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return bid, ask

    # ── EMERALDS ──────────────────────────────────────────────────────────────
    # Exact same logic as trader1.py but with LIMIT=75 instead of 70.

    def _emeralds(self, state: TradingState) -> List[Order]:
        od       = state.order_depths["EMERALDS"]
        bid, ask = self._best(od)
        if bid is None or ask is None:
            return []

        fv   = self.EMERALD_FV
        pos  = state.position.get("EMERALDS", 0)
        bcap = LIMIT - pos
        scap = LIMIT + pos
        orders: List[Order] = []

        if bid > fv:
            if scap > 0:
                orders.append(Order("EMERALDS", bid + 1, -scap))
        elif ask < fv:
            if bcap > 0:
                orders.append(Order("EMERALDS", ask - 1, bcap))
        else:
            if bcap > 0: orders.append(Order("EMERALDS", bid + 1,  bcap))
            if scap > 0: orders.append(Order("EMERALDS", ask - 1, -scap))

        return orders

    def _tomatoes(self, state: TradingState) -> List[Order]:
        od       = state.order_depths["TOMATOES"]
        bid, ask = self._best(od)
        if bid is None or ask is None:
            return []

        pos  = state.position.get("TOMATOES", 0)
        bcap = LIMIT - pos
        scap = LIMIT + pos
        orders: List[Order] = []

        if bid + 1 < ask - 1:
            if bcap > 0: orders.append(Order("TOMATOES", bid + 1,  bcap))
            if scap > 0: orders.append(Order("TOMATOES", ask - 1, -scap))

        return orders

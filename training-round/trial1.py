from datamodel import OrderDepth, TradingState, Order
from typing import List


class Trader:
    EMERALDS = "EMERALDS"
    TOMATOES = "TOMATOES"

    EMERALD_SOFT_LIMIT = 70
    TOMATO_SOFT_LIMIT  = 70
    EMERALD_SKEW_TICKS = 2   # ticks to shift per unit of pos_ratio for emeralds
    TOMATO_SKEW_TICKS  = 1   # gentler skew for tomatoes

    def run(self, state: TradingState):
        result = {}
        if self.EMERALDS in state.order_depths:
            result[self.EMERALDS] = self.trade_emeralds(state)
        if self.TOMATOES in state.order_depths:
            result[self.TOMATOES] = self.trade_tomatoes(state)
        return result, 0, ""

    def get_best_bid_ask(self, od: OrderDepth):
        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return best_bid, best_ask

    def trade_emeralds(self, state: TradingState) -> List[Order]:
        
        od = state.order_depths[self.EMERALDS]
        best_bid, best_ask = self.get_best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return []

        FAIR_VALUE = 10000
        pos      = state.position.get(self.EMERALDS, 0)
        SL       = self.EMERALD_SOFT_LIMIT
        buy_cap  = SL - pos
        sell_cap = SL + pos
        orders: List[Order] = []

        # Aggressive takes
        if best_bid > FAIR_VALUE and sell_cap > 0:
            vol = min(sell_cap, abs(od.buy_orders.get(best_bid, 0)))
            if vol > 0:
                orders.append(Order(self.EMERALDS, best_bid, -vol))
                sell_cap -= vol

        if best_ask < FAIR_VALUE and buy_cap > 0:
            vol = min(buy_cap, abs(od.sell_orders.get(best_ask, 0)))
            if vol > 0:
                orders.append(Order(self.EMERALDS, best_ask, vol))
                buy_cap -= vol

        # Passive MM with inventory skew
        # pos_ratio in [-1, +1]: positive = long, shift quotes DOWN to attract sells
        pos_ratio = pos / SL if SL != 0 else 0
        skew      = round(pos_ratio * self.EMERALD_SKEW_TICKS)

        passive_bid = best_bid + 1 - skew
        passive_ask = best_ask - 1 - skew

        if passive_bid < passive_ask:
            if buy_cap  > 0: orders.append(Order(self.EMERALDS, passive_bid,  buy_cap))
            if sell_cap > 0: orders.append(Order(self.EMERALDS, passive_ask, -sell_cap))
        else:
            # Quotes crossed after skew; fall back to unskewed
            if buy_cap  > 0: orders.append(Order(self.EMERALDS, best_bid + 1,  buy_cap))
            if sell_cap > 0: orders.append(Order(self.EMERALDS, best_ask - 1, -sell_cap))

        return orders

    def trade_tomatoes(self, state: TradingState) -> List[Order]:
       
        od = state.order_depths[self.TOMATOES]
        best_bid, best_ask = self.get_best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return []

        mid      = (best_bid + best_ask) / 2.0
        pos      = state.position.get(self.TOMATOES, 0)
        SL       = self.TOMATO_SOFT_LIMIT
        buy_cap  = SL - pos
        sell_cap = SL + pos
        orders: List[Order] = []

        # Aggressive takes: only when bid/ask moved clearly away from mid
        half_spread = (best_ask - best_bid) / 2.0

        if best_bid > mid + half_spread * 0.5 and sell_cap > 0:
            vol = min(sell_cap, abs(od.buy_orders.get(best_bid, 0)))
            if vol > 0:
                orders.append(Order(self.TOMATOES, best_bid, -vol))
                sell_cap -= vol

        if best_ask < mid - half_spread * 0.5 and buy_cap > 0:
            vol = min(buy_cap, abs(od.sell_orders.get(best_ask, 0)))
            if vol > 0:
                orders.append(Order(self.TOMATOES, best_ask, vol))
                buy_cap -= vol

        # Passive MM with light inventory skew
        pos_ratio = pos / SL if SL != 0 else 0
        skew      = round(pos_ratio * self.TOMATO_SKEW_TICKS)

        passive_bid = best_bid + 1 - skew
        passive_ask = best_ask - 1 - skew

        # Anchor to live mid so we never buy above mid or sell below mid+1
        passive_bid = min(passive_bid, int(mid))
        passive_ask = max(passive_ask, int(mid) + 1)

        if passive_bid < passive_ask:
            if buy_cap  > 0: orders.append(Order(self.TOMATOES, passive_bid,  buy_cap))
            if sell_cap > 0: orders.append(Order(self.TOMATOES, passive_ask, -sell_cap))
        else:
            if buy_cap  > 0: orders.append(Order(self.TOMATOES, best_bid + 1,  buy_cap))
            if sell_cap > 0: orders.append(Order(self.TOMATOES, best_ask - 1, -sell_cap))

        return orders
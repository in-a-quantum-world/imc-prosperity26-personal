from datamodel import OrderDepth, TradingState, Order
from typing import List

class Trader:
    EMERALDS = "EMERALDS"
    TOMATOES = "TOMATOES"

    EMERALD_LIMIT = 75 
    TOMATO_LIMIT  = 75

    
    EMERALD_SKEW = 2
    TOMATO_SKEW  = 1

    def run(self, state: TradingState):
        result = {}
        if self.EMERALDS in state.order_depths:
            result[self.EMERALDS] = self.trade_emeralds(state)
        if self.TOMATOES in state.order_depths:
            result[self.TOMATOES] = self.trade_tomatoes(state)
        return result, 0, ""

    def best_bid_ask(self, od: OrderDepth):
        bid = max(od.buy_orders.keys())  if od.buy_orders  else None
        ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return bid, ask

    def trade_emeralds(self, state: TradingState) -> List[Order]:
        
        od  = state.order_depths[self.EMERALDS]
        bid, ask = self.best_bid_ask(od)
        if bid is None or ask is None:
            return []

        FV   = 10000
        pos  = state.position.get(self.EMERALDS, 0)
        LIM  = self.EMERALD_LIMIT
        bcap = LIM - pos
        scap = LIM + pos
        orders: List[Order] = []

        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price <= FV or scap <= 0:
                break
            vol = min(scap, abs(od.buy_orders[price]))
            orders.append(Order(self.EMERALDS, price, -vol))
            scap -= vol

        for price in sorted(od.sell_orders.keys()):
            if price >= FV or bcap <= 0:
                break
            vol = min(bcap, abs(od.sell_orders[price]))
            orders.append(Order(self.EMERALDS, price, vol))
            bcap -= vol

        pos_ratio = pos / LIM if LIM != 0 else 0
        skew      = round(pos_ratio * self.EMERALD_SKEW)

        pbid = bid + 1 - skew
        pask = ask - 1 - skew

        if pbid < pask:
            if bcap > 0: orders.append(Order(self.EMERALDS, pbid,  bcap))
            if scap > 0: orders.append(Order(self.EMERALDS, pask, -scap))
        else:
            if bcap > 0: orders.append(Order(self.EMERALDS, bid + 1,  bcap))
            if scap > 0: orders.append(Order(self.EMERALDS, ask - 1, -scap))

        return orders

    def trade_tomatoes(self, state: TradingState) -> List[Order]:

        od  = state.order_depths[self.TOMATOES]
        bid, ask = self.best_bid_ask(od)
        if bid is None or ask is None:
            return []

        mid  = (bid + ask) / 2.0
        pos  = state.position.get(self.TOMATOES, 0)
        LIM  = self.TOMATO_LIMIT
        bcap = LIM - pos
        scap = LIM + pos
        orders: List[Order] = []

        
        if bid > mid and scap > 0:
            vol = min(scap, abs(od.buy_orders.get(bid, 0)))
            if vol > 0:
                orders.append(Order(self.TOMATOES, bid, -vol))
                scap -= vol

        if ask < mid and bcap > 0:
            vol = min(bcap, abs(od.sell_orders.get(ask, 0)))
            if vol > 0:
                orders.append(Order(self.TOMATOES, ask, vol))
                bcap -= vol

        pos_ratio = pos / LIM if LIM != 0 else 0
        skew      = round(pos_ratio * self.TOMATO_SKEW)

        pbid = bid + 1 - skew
        pask = ask - 1 - skew

        # Anchor to live mid so we never buy above or sell below fair value
        pbid = min(pbid, int(mid))
        pask = max(pask, int(mid) + 1)

        if pbid < pask:
            if bcap > 0: orders.append(Order(self.TOMATOES, pbid,  bcap))
            if scap > 0: orders.append(Order(self.TOMATOES, pask, -scap))
        else:
            if bcap > 0: orders.append(Order(self.TOMATOES, bid + 1,  bcap))
            if scap > 0: orders.append(Order(self.TOMATOES, ask - 1, -scap))

        return orders
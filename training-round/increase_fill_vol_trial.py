from datamodel import OrderDepth, TradingState, Order
from typing import List

class Trader:
    EMERALDS = "EMERALDS"
    TOMATOES = "TOMATOES"

    EMERALD_LIMIT = 70
    TOMATO_LIMIT  = 70
    EMERALD_SKEW  = 2
    TOMATO_SKEW   = 1

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

    def make_orders(self, product, od, pos, limit, skew_ticks, fv=None):
        
        bid, ask = self.best_bid_ask(od)
        if bid is None or ask is None:
            return []

        bcap = min(limit - pos, limit)   
        scap = min(limit + pos, limit)  
        orders: List[Order] = []

        if fv is not None:
            for price in sorted(od.buy_orders.keys(), reverse=True):
                if price <= fv or scap <= 0:
                    break
                vol = min(scap, abs(od.buy_orders[price]))
                orders.append(Order(product, price, -vol))
                scap -= vol
            for price in sorted(od.sell_orders.keys()):
                if price >= fv or bcap <= 0:
                    break
                vol = min(bcap, abs(od.sell_orders[price]))
                orders.append(Order(product, price, vol))
                bcap -= vol

        pos_ratio = pos / limit if limit != 0 else 0
        skew      = round(pos_ratio * skew_ticks)

        
        primary_bcap = max(bcap // 2, min(bcap, 10))   
        primary_scap = max(scap // 2, min(scap, 10))

        pbid1 = bid + 1 - skew
        pask1 = ask - 1 - skew

        if pbid1 < pask1:
            if primary_bcap > 0:
                orders.append(Order(product, pbid1, primary_bcap))
                bcap -= primary_bcap
            if primary_scap > 0:
                orders.append(Order(product, pask1, -primary_scap))
                scap -= primary_scap
        else:
        
            if bcap > 0:
                orders.append(Order(product, bid + 1, bcap))
                bcap = 0
            if scap > 0:
                orders.append(Order(product, ask - 1, -scap))
                scap = 0

        pbid2 = bid - skew
        pask2 = ask - skew

        if pbid2 < pask2 and pbid2 < pbid1 and pask2 > pask1:
            if bcap > 0:
                orders.append(Order(product, pbid2, bcap))
            if scap > 0:
                orders.append(Order(product, pask2, -scap))

        return orders

    def trade_emeralds(self, state: TradingState) -> List[Order]:
        od  = state.order_depths[self.EMERALDS]
        pos = state.position.get(self.EMERALDS, 0)
        return self.make_orders(
            self.EMERALDS, od, pos,
            self.EMERALD_LIMIT,
            self.EMERALD_SKEW,
            fv=10000
        )

    def trade_tomatoes(self, state: TradingState) -> List[Order]:
        od  = state.order_depths[self.TOMATOES]
        bid, ask = self.best_bid_ask(od)
        if bid is None or ask is None:
            return []
        fv  = (bid + ask) / 2.0   # live mid — no lagging EMA
        pos = state.position.get(self.TOMATOES, 0)
        return self.make_orders(
            self.TOMATOES, od, pos,
            self.TOMATO_LIMIT,
            self.TOMATO_SKEW,
            fv=fv
        )
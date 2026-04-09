from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

# Deliberately overfit to observed patterns in the tutorial round data.
#
#

# TOMATOES  ------ 
#weak mean-reversion (-0.11 coeff, 57.5% accuracy on 20-bar mean).
#Spread normally 13 ticks. When spread < 13, take aggressively.
# Mean-reversion skew: +15 units on the expected-reversion side.


# EMERALDS  — stationary around FV=10000. Pure passive MM at 9993/10007.
# Special cases: ask=10000 → buy it; bid=10000 → post inside spread.


class Trader:
    EMERALDS = "EMERALDS"
    TOMATOES = "TOMATOES"

    EMERALD_LIMIT = 80
    TOMATO_LIMIT  = 80

    MR_WINDOW    = 20
    MR_THRESHOLD = 3.0
    MR_SKEW      = 15


    def run(self, state: TradingState):
        
        tmid_history = []
        try:
            raw = state.traderData
            if raw and isinstance(raw, str):
                saved = json.loads(raw)
                if isinstance(saved, dict):
                    loaded = saved.get('th', [])
                    if isinstance(loaded, list):
                        tmid_history = [float(x) for x in loaded]
        except Exception:
            tmid_history = []

        result = {}
        if self.EMERALDS in state.order_depths:
            result[self.EMERALDS] = self.trade_emeralds(state)
        if self.TOMATOES in state.order_depths:
            result[self.TOMATOES], tmid_history = self.trade_tomatoes(state, tmid_history)

        keep = tmid_history[-(self.MR_WINDOW + 5):]
        trader_data = json.dumps({'th': keep})
        return result, 0, trader_data

    def _best(self, od: OrderDepth):
        bid = max(od.buy_orders.keys())  if od.buy_orders  else None
        ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return bid, ask


    def trade_emeralds(self, state: TradingState) -> List[Order]:
        od = state.order_depths[self.EMERALDS]
        bid, ask = self._best(od)
        if bid is None or ask is None:
            return []

        FV   = 10000
        pos  = state.position.get(self.EMERALDS, 0)
        LIM  = self.EMERALD_LIMIT
        bcap = min(LIM - pos, LIM)
        scap = min(LIM + pos, LIM)
        orders: List[Order] = []

        # Special case: ask=10000 — someone selling at fair value, take it
        if ask == FV and bcap > 0:
            vol = min(bcap, abs(od.sell_orders.get(ask, 0)))
            if vol > 0:
                orders.append(Order(self.EMERALDS, ask, vol))
                bcap -= vol

        # Special case: bid=10000 — spread narrowed to 8, post inside it
        if bid == FV and bcap > 0:
            orders.append(Order(self.EMERALDS, bid + 1, bcap))
            if scap > 0:
                orders.append(Order(self.EMERALDS, ask - 1, -scap))
            return orders

        # Aggressive: sweep levels strictly beyond fair value
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

        # Passive MM with inventory skew
        pos_ratio = pos / LIM if LIM else 0
        skew = round(pos_ratio * 2)
        pbid = bid + 1 - skew
        pask = ask - 1 - skew
        if pbid < pask:
            if bcap > 0: orders.append(Order(self.EMERALDS, pbid,  bcap))
            if scap > 0: orders.append(Order(self.EMERALDS, pask, -scap))
        else:
            if bcap > 0: orders.append(Order(self.EMERALDS, bid + 1,  bcap))
            if scap > 0: orders.append(Order(self.EMERALDS, ask - 1, -scap))

        return orders

    def trade_tomatoes(self, state: TradingState, tmid_history: list):
        """Returns (orders, updated_tmid_history)."""
        od = state.order_depths[self.TOMATOES]
        bid, ask = self._best(od)
        if bid is None or ask is None:
            return [], tmid_history

        mid = (bid + ask) / 2.0
        tmid_history = tmid_history + [mid]   

        pos  = state.position.get(self.TOMATOES, 0)
        LIM  = self.TOMATO_LIMIT
        bcap = min(LIM - pos, LIM)
        scap = min(LIM + pos, LIM)
        orders: List[Order] = []

        # Narrow spread (<13 ticks): someone quoted inside our normal spread.
        # Take aggressively — their ask is well below normal ~5013.
        if (ask - bid) < 13:
            if bcap > 0:
                vol = min(bcap, abs(od.sell_orders.get(ask, 0)))
                if vol > 0:
                    orders.append(Order(self.TOMATOES, ask, vol))
                    bcap -= vol
            if scap > 0:
                vol = min(scap, abs(od.buy_orders.get(bid, 0)))
                if vol > 0:
                    orders.append(Order(self.TOMATOES, bid, -vol))
                    scap -= vol

        # Mean-reversion signal (20-bar rolling mean, 57.5% accuracy)
        mr_extra_buy  = 0
        mr_extra_sell = 0
        if len(tmid_history) >= self.MR_WINDOW:
            roll_mean = sum(tmid_history[-self.MR_WINDOW:]) / self.MR_WINDOW
            deviation = mid - roll_mean
            if deviation > self.MR_THRESHOLD:
                mr_extra_sell = self.MR_SKEW   # above mean → lean short
            elif deviation < -self.MR_THRESHOLD:
                mr_extra_buy  = self.MR_SKEW   # below mean → lean long

        # Passive MM: always both sides with inventory skew + MR skew
        pos_ratio = pos / LIM if LIM else 0
        skew = round(pos_ratio * 1)
        pbid = bid + 1 - skew
        pask = ask - 1 - skew

        buy_size  = min(bcap, max(0, (bcap // 2) + mr_extra_buy))
        sell_size = min(scap, max(0, (scap // 2) + mr_extra_sell))

        if pbid < pask:
            if buy_size  > 0: orders.append(Order(self.TOMATOES, pbid,  buy_size))
            if sell_size > 0: orders.append(Order(self.TOMATOES, pask, -sell_size))
            # Secondary level fills remaining capacity
            remaining_b = bcap - buy_size
            remaining_s = scap - sell_size
            if remaining_b > 0: orders.append(Order(self.TOMATOES, bid,  remaining_b))
            if remaining_s > 0: orders.append(Order(self.TOMATOES, ask, -remaining_s))
        else:
            if bcap > 0: orders.append(Order(self.TOMATOES, bid + 1,  bcap))
            if scap > 0: orders.append(Order(self.TOMATOES, ask - 1, -scap))

        return orders, tmid_history
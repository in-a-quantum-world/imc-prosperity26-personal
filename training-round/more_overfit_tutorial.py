from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

#this ones deliberately overfit to patterns observed, kinda working backwards


#--------------TOMATOES
#MEAN REVERSION COEFFICIENT OF -0.11 AND 57.5% ACCURACY ON 20 BAR ROLLING MEAN SIGNAK
#long run mean 4991


#tomato spread is 13 ticks. when the spread narrows, the insider has quoted tigher, so buy the cheap ask asap since its well below normal ask 

#using mean reversion signal, when mid>3 above mean--> lean short, mid more than 3 brelow mean --> lean long
#-----------EMERALDS 

#thesea re like the stationary asset, fair value defo 10K
#get inside narrow spread so you have better fill of orders
#passive market making seemingly best(?)

class Trader:
    EMERALDS = "EMERALDS"
    TOMATOES = "TOMATOES"

    EMERALD_LIMIT = 80
    TOMATO_LIMIT  = 80

    MR_WINDOW    = 20    
    MR_THRESHOLD = 3.0    
    MR_SKEW      = 15     

    def __init__(self):
        self._tmid_history: List[float] = []

    def run(self, state: TradingState):
        # Restore mid-price history from traderData
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
                self._tmid_history = saved.get('th', [])
            except Exception:
                self._tmid_history = []

        result = {}
        if self.EMERALDS in state.order_depths:
            result[self.EMERALDS] = self.trade_emeralds(state)
        if self.TOMATOES in state.order_depths:
            result[self.TOMATOES] = self.trade_tomatoes(state)

        
        trader_data = json.dumps({'th': self._tmid_history[-(self.MR_WINDOW + 5):]})
        return result, 0, trader_data

    def _best(self, od: OrderDepth):
        bid = max(od.buy_orders.keys())  if od.buy_orders  else None
        ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return bid, ask

    def trade_emeralds(self, state: TradingState) -> List[Order]:
        od  = state.order_depths[self.EMERALDS]
        bid, ask = self._best(od)
        if bid is None or ask is None:
            return []

        FV   = 10000
        pos  = state.position.get(self.EMERALDS, 0)
        LIM  = self.EMERALD_LIMIT
        bcap = min(LIM - pos, LIM)
        scap = min(LIM + pos, LIM)
        orders: List[Order] = []

        spread = ask - bid

        # === SPECIAL CASE: ask = 10000 (occurs 29×) ===
        # Someone is selling AT fair value. Buy it immediately — guaranteed 7-tick
        # profit when we sell later at our passive ask of 10007.
        if ask == FV and bcap > 0:
            vol = min(bcap, abs(od.sell_orders.get(ask, 0)))
            if vol > 0:
                orders.append(Order(self.EMERALDS, ask, vol))
                bcap -= vol

        # inside this tighter spread — more likely to get filled than 9993.
        if bid == FV and bcap > 0:
            # Best passive inside the 10000/10008 spread
            orders.append(Order(self.EMERALDS, bid + 1, bcap))
            bcap = 0
            if scap > 0:
                orders.append(Order(self.EMERALDS, ask - 1, -scap))
            return orders

        #beyond fair value, be aggressive and just get em 
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

        # Passive both sides — inventory skew
        pos_ratio = pos / LIM if LIM else 0
        skew      = round(pos_ratio * 2)
        pbid      = bid + 1 - skew
        pask      = ask - 1 - skew
        if pbid < pask:
            if bcap > 0: orders.append(Order(self.EMERALDS, pbid,  bcap))
            if scap > 0: orders.append(Order(self.EMERALDS, pask, -scap))
        else:
            if bcap > 0: orders.append(Order(self.EMERALDS, bid + 1,  bcap))
            if scap > 0: orders.append(Order(self.EMERALDS, ask - 1, -scap))

        return orders

    def trade_tomatoes(self, state: TradingState) -> List[Order]:
        od  = state.order_depths[self.TOMATOES]
        bid, ask = self._best(od)
        if bid is None or ask is None:
            return []

        mid  = (bid + ask) / 2.0
        self._tmid_history.append(mid)

        pos  = state.position.get(self.TOMATOES, 0)
        LIM  = self.TOMATO_LIMIT
        bcap = min(LIM - pos, LIM)
        scap = min(LIM + pos, LIM)
        orders: List[Order] = []

        spread = ask - bid

        # Post aggressively at bid+1 / ask-1 to capture flow on both sides.
            if bcap > 0:
                vol = min(bcap, abs(od.sell_orders.get(ask, 0)))
                if vol > 0:
                    orders.append(Order(self.TOMATOES, ask, vol))
                    bcap -= vol
            # Also sell into the elevated bid (their bid is above normal too)
            if scap > 0:
                vol = min(scap, abs(od.buy_orders.get(bid, 0)))
                if vol > 0:
                    orders.append(Order(self.TOMATOES, bid, -vol))
                    scap -= vol

        #mean reverion signal if signal deviates by more than 3 ticks form mean
        # Rolling 20-bar mean. 57.5% directional accuracy.
        mr_extra_buy  = 0
        mr_extra_sell = 0
        if len(self._tmid_history) >= self.MR_WINDOW:
            roll_mean  = sum(self._tmid_history[-self.MR_WINDOW:]) / self.MR_WINDOW
            deviation  = mid - roll_mean
            if deviation > self.MR_THRESHOLD:
                # Price above mean → expect fall → lean short
                mr_extra_sell = self.MR_SKEW
            elif deviation < -self.MR_THRESHOLD:
                # Price below mean → expect rise → lean long
                mr_extra_buy = self.MR_SKEW

        #passive market making is strat for emeralds 
        #tomatoes is more aggressive with mean reversion skew to capture flow on both sides and get more fills
        pos_ratio = pos / LIM if LIM else 0
        skew      = round(pos_ratio * 1)
        pbid      = bid + 1 - skew
        pask      = ask - 1 - skew

        # Apply mean-reversion skew: increase size on the "right" side
        buy_size  = min(bcap, max(0, (bcap // 2) + mr_extra_buy))
        sell_size = min(scap, max(0, (scap // 2) + mr_extra_sell))

        if pbid < pask:
            if buy_size  > 0: orders.append(Order(self.TOMATOES, pbid,  buy_size))
            if sell_size > 0: orders.append(Order(self.TOMATOES, pask, -sell_size))
            # Secondary level with remaining capacity
            remaining_b = bcap - buy_size
            remaining_s = scap - sell_size
            if remaining_b > 0: orders.append(Order(self.TOMATOES, bid,  remaining_b))
            if remaining_s > 0: orders.append(Order(self.TOMATOES, ask, -remaining_s))
        else:
            if bcap > 0: orders.append(Order(self.TOMATOES, bid + 1,  bcap))
            if scap > 0: orders.append(Order(self.TOMATOES, ask - 1, -scap))

        return orders
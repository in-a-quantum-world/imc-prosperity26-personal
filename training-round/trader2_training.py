from datamodel import OrderDepth, TradingState, Order
from typing import List
import json



LIMIT = 70    
SKEW  = 15    

class Trader:

    def run(self, state: TradingState):
        last_tmid = None
        try:
            raw = state.traderData
            if raw and isinstance(raw, str):
                saved = json.loads(raw)
                if isinstance(saved, dict):
                    v = saved.get('m')
                    if v is not None:
                        last_tmid = float(v)
        except Exception:
            pass

        result = {}
        current_tmid = None

        if 'TOMATOES' in state.order_depths:
            result['TOMATOES'], current_tmid = self._tomatoes(state, last_tmid)

        if 'EMERALDS' in state.order_depths:
            result['EMERALDS'] = self._emeralds(state)

        td = json.dumps({'m': current_tmid}) if current_tmid is not None else ''
        return result, 0, td

    def _best(self, od):
        bid = max(od.buy_orders.keys())  if od.buy_orders  else None
        ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return bid, ask

    # ── TOMATOES ─────────────────────────────────────────────────────────────

    def _tomatoes(self, state, last_mid):
        od = state.order_depths['TOMATOES']
        bid, ask = self._best(od)
        if bid is None or ask is None:
            return [], None

        mid = (bid + ask) / 2.0
        pos = state.position.get('TOMATOES', 0)

        # Skew based on lag-1 autocorrelation signal
        # Base size = LIMIT - SKEW so we can add SKEW on one side without
        # ever placing more orders than the position limit allows.
        base = LIMIT - SKEW   # = 55

        if last_mid is not None and mid != last_mid:
            if mid > last_mid:
                # Price rose → expect fall → lean short
                buy_size  = base - SKEW   # 40
                sell_size = base + SKEW   # 70
            else:
                # Price fell → expect rise → lean long
                buy_size  = base + SKEW   # 70
                sell_size = base - SKEW   # 40
        else:
            buy_size  = base   # 55
            sell_size = base   # 55

        # Respect position limits
        buy_size  = max(0, min(buy_size,  LIMIT - pos))
        sell_size = max(0, min(sell_size, LIMIT + pos))

        orders: List[Order] = []
        pbid = bid + 1
        pask = ask - 1
        if pbid < pask:
            if buy_size  > 0: orders.append(Order('TOMATOES', pbid,  buy_size))
            if sell_size > 0: orders.append(Order('TOMATOES', pask, -sell_size))

        return orders, mid

    # ── EMERALDS ─────────────────────────────────────────────────────────────

    def _emeralds(self, state) -> List[Order]:
        od = state.order_depths['EMERALDS']
        bid, ask = self._best(od)
        if bid is None or ask is None:
            return []

        FV   = 10000
        pos  = state.position.get('EMERALDS', 0)
        bcap = min(LIMIT - pos, LIMIT)
        scap = min(LIMIT + pos, LIMIT)
        orders: List[Order] = []

        # ask=10000: buy at fair value, guaranteed profitable exit at 10007
        if ask == FV and bcap > 0:
            vol = min(bcap, abs(od.sell_orders.get(ask, 0)))
            if vol > 0:
                orders.append(Order('EMERALDS', ask, vol))
                bcap -= vol

        # bid=10000: spread narrowed, post inside the 8-tick spread
        if bid == FV:
            if bcap > 0: orders.append(Order('EMERALDS', bid + 1,  bcap))
            if scap > 0: orders.append(Order('EMERALDS', ask - 1, -scap))
            return orders

        # Normal: passive MM at 9993/10007
        if bcap > 0: orders.append(Order('EMERALDS', bid + 1,  bcap))
        if scap > 0: orders.append(Order('EMERALDS', ask - 1, -scap))
        return orders
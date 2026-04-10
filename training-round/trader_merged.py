from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

# ── Constants ────────────────────────────────────────────────────────────────

LIMIT        = 95      # real limit is 100, keep small buffer
SKEW_FACTOR  = 2       # inventory skew ticks per unit of pos_ratio
EMA_ALPHA    = 0.05    # slow EMA for tomato fair value (20-bar equivalent)
AGGR_THRESH  = 1       # ticks beyond fair value to trigger aggressive take

# ── Trader ───────────────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState):
        # ── Restore persisted state ───────────────────────────────────────────
        tomato_ema = None
        try:
            saved = json.loads(state.traderData)
            v = saved.get("ema")
            if v is not None:
                tomato_ema = float(v)
        except Exception:
            pass

        result = {}

        if "EMERALDS" in state.order_depths:
            result["EMERALDS"] = self._emeralds(state)

        if "TOMATOES" in state.order_depths:
            result["TOMATOES"], tomato_ema = self._tomatoes(state, tomato_ema)

        td = json.dumps({"ema": tomato_ema}) if tomato_ema is not None else ""
        return result, 0, td

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _best(self, od: OrderDepth):
        bid = max(od.buy_orders.keys())  if od.buy_orders  else None
        ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return bid, ask

    def _skew(self, pos: int) -> int:
        """Inventory skew: shift quotes down when long, up when short."""
        pos_ratio = pos / LIMIT if LIMIT else 0
        return round(pos_ratio * SKEW_FACTOR)

    # ── EMERALDS ──────────────────────────────────────────────────────────────

    def _emeralds(self, state: TradingState) -> List[Order]:
        od       = state.order_depths["EMERALDS"]
        bid, ask = self._best(od)
        if bid is None or ask is None:
            return []

        FV   = 10000
        pos  = state.position.get("EMERALDS", 0)
        bcap = LIMIT - pos
        scap = LIMIT + pos
        orders: List[Order] = []

        # ── Aggressive takes ─────────────────────────────────────────────────
        # Sweep all bids above fair value (sell into them)
        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price <= FV or scap <= 0:
                break
            vol = min(scap, abs(od.buy_orders[price]))
            if vol > 0:
                orders.append(Order("EMERALDS", price, -vol))
                scap -= vol

        # Sweep all asks below fair value (buy from them)
        for price in sorted(od.sell_orders.keys()):
            if price >= FV or bcap <= 0:
                break
            vol = min(bcap, abs(od.sell_orders[price]))
            if vol > 0:
                orders.append(Order("EMERALDS", price, vol))
                bcap -= vol

        # ── Special case: ask == FV (someone selling at fair value) ──────────
        if ask == FV and bcap > 0:
            vol = min(bcap, abs(od.sell_orders.get(ask, 0)))
            if vol > 0:
                orders.append(Order("EMERALDS", ask, vol))
                bcap -= vol

        # ── Special case: bid == FV (spread has narrowed) ────────────────────
        if bid == FV:
            if bcap > 0: orders.append(Order("EMERALDS", bid + 1,  bcap))
            if scap > 0: orders.append(Order("EMERALDS", ask - 1, -scap))
            return orders

        # ── Passive MM with inventory skew ────────────────────────────────────
        skew = self._skew(pos)
        pbid = bid + 1 - skew
        pask = ask - 1 - skew

        if pbid < pask:
            if bcap > 0: orders.append(Order("EMERALDS", pbid,  bcap))
            if scap > 0: orders.append(Order("EMERALDS", pask, -scap))
        elif bid + 1 < ask - 1:
            if bcap > 0: orders.append(Order("EMERALDS", bid + 1,  bcap))
            if scap > 0: orders.append(Order("EMERALDS", ask - 1, -scap))

        return orders

    # ── TOMATOES ──────────────────────────────────────────────────────────────

    def _tomatoes(self, state: TradingState, ema: float):
        od       = state.order_depths["TOMATOES"]
        bid, ask = self._best(od)
        if bid is None or ask is None:
            return [], ema

        mid = (bid + ask) / 2.0

        # Update EMA fair value
        if ema is None:
            ema = mid
        else:
            ema = EMA_ALPHA * mid + (1 - EMA_ALPHA) * ema

        fv   = ema
        pos  = state.position.get("TOMATOES", 0)
        bcap = LIMIT - pos
        scap = LIMIT + pos
        orders: List[Order] = []

        # ── Aggressive takes ─────────────────────────────────────────────────
        # Sell into bids significantly above fair value
        if bid > fv + AGGR_THRESH and scap > 0:
            vol = min(scap, abs(od.buy_orders.get(bid, 0)))
            if vol > 0:
                orders.append(Order("TOMATOES", bid, -vol))
                scap -= vol

        # Buy from asks significantly below fair value
        if ask < fv - AGGR_THRESH and bcap > 0:
            vol = min(bcap, abs(od.sell_orders.get(ask, 0)))
            if vol > 0:
                orders.append(Order("TOMATOES", ask, vol))
                bcap -= vol

        # ── Passive MM with inventory skew ────────────────────────────────────
        skew = self._skew(pos)
        pbid = bid + 1 - skew
        pask = ask - 1 - skew

        # Anchor passive quotes to fair value — never buy above or sell below EMA
        pbid = min(pbid, int(fv))
        pask = max(pask, int(fv) + 1)

        if pbid < pask:
            if bcap > 0: orders.append(Order("TOMATOES", pbid,  bcap))
            if scap > 0: orders.append(Order("TOMATOES", pask, -scap))
        elif bid + 1 < ask - 1:
            if bcap > 0: orders.append(Order("TOMATOES", bid + 1,  bcap))
            if scap > 0: orders.append(Order("TOMATOES", ask - 1, -scap))

        return orders, ema

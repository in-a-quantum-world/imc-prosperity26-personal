"""
ABLATION #5 — HP + OTM tickets (VEV_6000, VEV_6500)
====================================================
HP baseline + "free" deep OTM tickets at VEV_6000 and VEV_6500.

Hypothesis: asks at <= 0 are free lottery tickets. Holding deep OTM
costs nothing if we get them at 0 and we profit if VF rallies hard.

>> Potentially silently bleeding PnL. Worth verifying in isolation. <<

Two concerns with the current implementation to note:
  1. `orders.append(Order(sym, 0, q))` posts our BUY at price 0 rather
     than at the observed best_ask. If best_ask is negative (bot is
     paying us to take inventory), we're leaving money on the table by
     not matching the negative price.
  2. The SELL side posts at price 1, but if best_bid is higher we're
     undercutting ourselves. Also if best_bid is exactly 1 and we're
     already at pos > 0, we may be marked-to-market at 0 while paying
     spread to exit.

Decision rule:
  Δ ≥ +300  → keep it (and fix the pricing bugs noted above)
  |Δ| < 300 → probably keep (low cost, minor lottery upside)
  Δ ≤ -300  → drop, or drop AND investigate the pricing bugs

Note: OTM tickets may matter more in the final backtest day than the
first (if VF drifts far enough to bring them into the money). Look at
the PnL curve shape, not just the total.
"""

from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState,
)
import math
import json

# ═══════════════════════════════════════════════════════════════
# HP CONFIG
# ═══════════════════════════════════════════════════════════════
HP_FAIR = 10_000
HP_SPREAD_HALF = 4
HP_ORDER_SIZE = 15
HP_MAX_POS = 200
HP_SKEW_PER_UNIT = 0.05
HP_TAKE_EDGE = 1


def get_best(depth: OrderDepth):
    bb = max(depth.buy_orders.keys()) if depth.buy_orders else None
    ba = min(depth.sell_orders.keys()) if depth.sell_orders else None
    return bb, ba

def get_mid(depth: OrderDepth):
    bb, ba = get_best(depth)
    return (bb + ba) / 2.0 if bb is not None and ba is not None else None


class Trader:
    def run(self, state: TradingState):
        data = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except:
                data = {}

        result = {}
        positions = state.position or {}

        # HP
        if "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = self._trade_hp(
                state.order_depths["HYDROGEL_PACK"],
                positions.get("HYDROGEL_PACK", 0),
            )

        # Deep OTM free tickets
        for K in [6000, 6500]:
            sym = f"VEV_{K}"
            if sym in state.order_depths:
                pos = positions.get(sym, 0)
                ords = self._collect_otm(state.order_depths[sym], pos, sym)
                if ords:
                    result[sym] = ords

        return result, 0, json.dumps(data)

    # ─────────────────────────────────────────────────────────
    # HP market making (byte-identical across all ablation files)
    # ─────────────────────────────────────────────────────────
    def _trade_hp(self, depth, pos):
        orders = []
        sym = "HYDROGEL_PACK"
        mid = get_mid(depth)
        fair = HP_FAIR
        if mid is not None:
            fair = 0.7 * HP_FAIR + 0.3 * mid
        skew = -pos * HP_SKEW_PER_UNIT
        adj_fair = fair + skew

        if depth.sell_orders:
            for ap in sorted(depth.sell_orders.keys()):
                if ap < adj_fair - HP_TAKE_EDGE:
                    q = min(-depth.sell_orders[ap], HP_MAX_POS - pos)
                    if q > 0:
                        orders.append(Order(sym, ap, q))
                        pos += q
                else:
                    break
        if depth.buy_orders:
            for bp in sorted(depth.buy_orders.keys(), reverse=True):
                if bp > adj_fair + HP_TAKE_EDGE:
                    q = min(depth.buy_orders[bp], HP_MAX_POS + pos)
                    if q > 0:
                        orders.append(Order(sym, bp, -q))
                        pos -= q
                else:
                    break

        bp = math.floor(adj_fair - HP_SPREAD_HALF)
        ap = math.ceil(adj_fair + HP_SPREAD_HALF)
        bs = min(HP_ORDER_SIZE, HP_MAX_POS - pos)
        as_ = min(HP_ORDER_SIZE, HP_MAX_POS + pos)
        if bs > 0:
            orders.append(Order(sym, bp, bs))
        if as_ > 0:
            orders.append(Order(sym, ap, -as_))
        return orders

    # ─────────────────────────────────────────────────────────
    # Free OTM tickets (unchanged from combined — keeping quirks so
    # the measurement reflects what's actually been running in combined)
    # ─────────────────────────────────────────────────────────
    def _collect_otm(self, depth, pos, sym):
        orders = []
        if depth.sell_orders and pos < 50:
            best_ask = min(depth.sell_orders.keys())
            if best_ask <= 0:
                q = min(-depth.sell_orders[best_ask], 50 - pos, 25)
                if q > 0:
                    orders.append(Order(sym, 0, q))
        if pos > 0 and depth.buy_orders:
            best_bid = max(depth.buy_orders.keys())
            if best_bid >= 1:
                q = min(depth.buy_orders[best_bid], pos, 25)
                if q > 0:
                    orders.append(Order(sym, 1, -q))
        return orders

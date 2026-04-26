"""
ABLATION #1 — HP only
=====================
Strips everything except HP market making.
Expected PnL: ~14K/day (matches the standalone HP baseline you mentioned).
This is the BASELINE — every other ablation's delta is measured against this.

If this score is NOT ~14K, something in HP itself has regressed and the other
ablations are worthless until fixed.
"""

from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState,
)
import math
import json

# ═══════════════════════════════════════════════════════════════
# HP CONFIG (unchanged from combined)
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

        if "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = self._trade_hp(
                state.order_depths["HYDROGEL_PACK"],
                positions.get("HYDROGEL_PACK", 0),
            )

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

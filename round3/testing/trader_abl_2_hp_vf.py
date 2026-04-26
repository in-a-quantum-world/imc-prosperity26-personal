"""
ABLATION #2 — HP + VF asymmetric
=================================
HP baseline + VF asymmetric market making (aggressive bid, passive ask).

Hypothesis: simulation predicted +2-6K/day contribution from VF.
Measurement: (this PnL) - (HP-only PnL) = true VF contribution.

Decision rule:
  Δ ≥ +1.5K  → VF is real alpha, keep it
  |Δ| < 1.5K → inside noise, probably drop (complexity without benefit)
  Δ ≤ -1.5K  → VF is bleeding, definitely drop
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

# ═══════════════════════════════════════════════════════════════
# VF CONFIG
# ═══════════════════════════════════════════════════════════════
VF_MAX_POS = 200
VF_TARGET_LONG = 100
VF_BID_SIZE = 20
VF_ASK_SIZE_PASSIVE = 10
VF_ASK_SIZE_UNLOAD = 25
VF_UNLOAD_THRESHOLD = 170
VF_STOP_BID_BELOW = -50
VF_COVER_SHORT_BELOW = -100


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

        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            result["VELVETFRUIT_EXTRACT"] = self._trade_vf_asymmetric(
                state.order_depths["VELVETFRUIT_EXTRACT"],
                positions.get("VELVETFRUIT_EXTRACT", 0),
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

    # ─────────────────────────────────────────────────────────
    # VF asymmetric (aggressive bid, passive ask)
    # ─────────────────────────────────────────────────────────
    def _trade_vf_asymmetric(self, depth, pos):
        orders = []
        sym = "VELVETFRUIT_EXTRACT"
        bb, ba = get_best(depth)
        if bb is None or ba is None:
            return orders
        spread = ba - bb

        # Emergency short cover
        if pos <= VF_COVER_SHORT_BELOW:
            cover = -pos
            for ap in sorted(depth.sell_orders.keys()):
                t = min(-depth.sell_orders[ap], cover)
                if t > 0:
                    orders.append(Order(sym, ap, t))
                    cover -= t
                if cover <= 0:
                    break
            return orders

        # Unloading
        if pos >= VF_UNLOAD_THRESHOLD:
            our_ask = ba - 1 if spread >= 2 else ba
            asz = min(VF_ASK_SIZE_UNLOAD, VF_MAX_POS + pos)
            if asz > 0:
                orders.append(Order(sym, our_ask, -asz))
            bsz = min(5, VF_MAX_POS - pos)
            if bsz > 0:
                orders.append(Order(sym, bb, bsz))
            return orders

        # Normal: aggressive bid, passive ask
        our_bid = bb + 1 if spread >= 3 else bb
        if pos < VF_MAX_POS and pos > VF_STOP_BID_BELOW:
            bsz = min(VF_BID_SIZE, VF_MAX_POS - pos)
            if bsz > 0:
                orders.append(Order(sym, our_bid, bsz))

        if pos > VF_TARGET_LONG:
            our_ask = ba - 1 if spread >= 2 else ba
            asz = min(VF_ASK_SIZE_UNLOAD, VF_MAX_POS + pos)
        else:
            our_ask = ba
            asz = min(VF_ASK_SIZE_PASSIVE, VF_MAX_POS + pos)
        if asz > 0 and our_ask > our_bid:
            orders.append(Order(sym, our_ask, -asz))
        return orders

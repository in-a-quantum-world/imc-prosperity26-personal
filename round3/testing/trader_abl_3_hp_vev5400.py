"""
ABLATION #3 — HP + VEV_5400 structural long
============================================
HP baseline + structural long on VEV_5400 only.

Hypothesis: VEV_5400 prices at IV=0.0120 vs 0.0127 for other strikes.
If the "true" underlying vol is ~0.0127, VEV_5400 is cheap by ~2-3 ticks.
Expected contribution: +0.2K to +0.6K.

Note: VF is read (we need vf_mid for BS fair value) but NOT traded.
Day tracking is kept for TTE calculation.

Decision rule:
  Δ ≥ +500  → keep it
  |Δ| < 500 → drop (too small to justify complexity/variance)
  Δ ≤ -500  → our "structural cheapness" read is wrong — drop

NB: The 14K HP baseline was run without VF being quoted. Since this
variant also doesn't trade VF, HP's fills should be unchanged, so the
PnL delta here really does isolate VEV_5400.
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
# VEV_5400 CONFIG
# ═══════════════════════════════════════════════════════════════
VEV_5400_MAX_POS = 150
VEV_5400_TAKE_EDGE = 1.0
VEV_5400_TAKE_SIZE = 10

# ═══════════════════════════════════════════════════════════════
# BS PARAMS
# ═══════════════════════════════════════════════════════════════
VEV_INITIAL_TTE = 8              # CHANGE TO 5 FOR LIVE
VEV_IV_UNIFORM = 0.0127


# ─── Black-Scholes utilities ───
def norm_cdf(x: float) -> float:
    if x < -10: return 0.0
    if x > 10: return 1.0
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-9 or sigma <= 0:
        return max(S - K, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * norm_cdf(d1) - K * norm_cdf(d2)

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

        # Day/TTE tracking
        day = data.get("day_number", 0)
        prev_ts = data.get("prev_timestamp", 0)
        if state.timestamp < prev_ts and prev_ts > 0:
            day += 1
            data["day_number"] = day
        data["prev_timestamp"] = state.timestamp
        tte = max(VEV_INITIAL_TTE - day - state.timestamp / 1_000_000, 0.001)

        # HP
        if "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = self._trade_hp(
                state.order_depths["HYDROGEL_PACK"],
                positions.get("HYDROGEL_PACK", 0),
            )

        # Read VF mid for BS fair value, do NOT trade VF
        vf_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            vf_mid = get_mid(state.order_depths["VELVETFRUIT_EXTRACT"])

        # VEV_5400 structural long
        if vf_mid is not None and "VEV_5400" in state.order_depths:
            fair_5400 = bs_call(vf_mid, 5400, tte, VEV_IV_UNIFORM)
            result["VEV_5400"] = self._trade_vev_5400(
                state.order_depths["VEV_5400"],
                positions.get("VEV_5400", 0),
                fair_5400,
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
    # VEV_5400 structural lean
    # ─────────────────────────────────────────────────────────
    def _trade_vev_5400(self, depth, pos, fair):
        orders = []
        sym = "VEV_5400"

        # BUY: take asks at/below BS-uniform fair
        if depth.sell_orders and pos < VEV_5400_MAX_POS:
            for ap in sorted(depth.sell_orders.keys()):
                if ap <= fair - VEV_5400_TAKE_EDGE:
                    q = min(
                        -depth.sell_orders[ap],
                        VEV_5400_TAKE_SIZE,
                        VEV_5400_MAX_POS - pos,
                    )
                    if q > 0:
                        orders.append(Order(sym, ap, q))
                        pos += q
                else:
                    break

        # SELL: only if near limit AND market pays well above fair
        if pos > VEV_5400_MAX_POS * 0.8 and depth.buy_orders:
            for bp in sorted(depth.buy_orders.keys(), reverse=True):
                if bp >= fair + 1.0:
                    q = min(
                        depth.buy_orders[bp],
                        pos - int(VEV_5400_MAX_POS * 0.6),
                    )
                    if q > 0:
                        orders.append(Order(sym, bp, -q))
                        pos -= q
                else:
                    break

        return orders

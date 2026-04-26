"""
ABLATION #4 — HP + VEV_5300 unhedged long gamma
================================================
HP baseline + long gamma on VEV_5300 only, unhedged.

Hypothesis: realized vol (~0.017) > implied (0.0127), so long gamma is +EV.
BUT: unhedged means any directional drift in VF hits the options position.

>> My strongest prior for what's bleeding PnL in the combined version. <<
Realized > implied is a theoretical edge that requires delta hedging to
isolate. Without hedging, on a day when VF drifts DOWN, VEV_5300 calls
lose delta and there's no offsetting short VF to cover. On a day when VF
drifts up, it "works" — but you're really just long calls, not long gamma.

Decision rule:
  Δ ≥ +500  → keep it, the unhedged exposure happened to be favorable
  |Δ| < 500 → drop anyway, it's pure directional variance dressed up
  Δ ≤ -500  → this IS the leak, drop definitively

Given the combined variant lost ~3K vs HP-only, and this is the most
fragile component, there's a reasonable chance this alone accounts for
most of the drag.
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
# GAMMA CONFIG
# ═══════════════════════════════════════════════════════════════
GAMMA_STRIKE = 5300
GAMMA_POS_TARGET = 30
GAMMA_TAKE_EDGE = 1.0

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

        # VEV_5300 unhedged long gamma
        if vf_mid is not None and "VEV_5300" in state.order_depths:
            fair_5300 = bs_call(vf_mid, 5300, tte, VEV_IV_UNIFORM)
            result["VEV_5300"] = self._trade_vev_5300_gamma(
                state.order_depths["VEV_5300"],
                positions.get("VEV_5300", 0),
                fair_5300,
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
    # VEV_5300 long gamma (small, unhedged)
    # ─────────────────────────────────────────────────────────
    def _trade_vev_5300_gamma(self, depth, pos, fair):
        orders = []
        sym = "VEV_5300"

        # Buy when clearly cheap relative to BS fair
        if depth.sell_orders and pos < GAMMA_POS_TARGET:
            for ap in sorted(depth.sell_orders.keys()):
                if ap <= fair - GAMMA_TAKE_EDGE:
                    q = min(
                        -depth.sell_orders[ap],
                        5,
                        GAMMA_POS_TARGET - pos,
                    )
                    if q > 0:
                        orders.append(Order(sym, ap, q))
                        pos += q
                else:
                    break

        # Sell when clearly rich
        if depth.buy_orders and pos > 0:
            for bp in sorted(depth.buy_orders.keys(), reverse=True):
                if bp >= fair + GAMMA_TAKE_EDGE:
                    q = min(depth.buy_orders[bp], 5, pos)
                    if q > 0:
                        orders.append(Order(sym, bp, -q))
                        pos -= q
                else:
                    break

        return orders

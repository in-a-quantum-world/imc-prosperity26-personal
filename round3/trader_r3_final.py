"""
IMC Prosperity 4 — Round 3 FINAL
==================================
Based on backtester results:
  - Hydrogel MM: +14K ✅  → CORE strategy, max out
  - Options strategies: all negative → ULTRA-CONSERVATIVE only
  - VF scalp: ~0 → skip

Strategy:
  1. HYDROGEL_PACK aggressive market making (proven +14K)
  2. Options: take-only on EXTREME mispricing (>3 ticks from BS fair)
     - No passive quoting on options (avoids adverse selection)
     - No delta hedging (costs 25K/day in spread)
     - Tiny position limits to cap downside
  3. Free OTM collection (buy VEV_6000/6500 at 0, sell at 1)
"""

from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState,
)
import json
import math
from typing import Any

# ─────────────────────────────────────────────────────────────
# HYDROGEL CONFIG (proven profitable)
# ─────────────────────────────────────────────────────────────
HP_FAIR = 10_000
HP_SPREAD_HALF = 4
HP_ORDER_SIZE = 15
HP_MAX_POS = 200
HP_SKEW_PER_UNIT = 0.05
HP_TAKE_EDGE = 1

# ─────────────────────────────────────────────────────────────
# OPTIONS CONFIG (ultra-conservative take-only)
# ─────────────────────────────────────────────────────────────
VEV_BASE_TTE = 7
VEV_IV = 0.0135               # calibrated from data
VEV_TAKE_THRESH = 3.0          # only take if mispricing > 3 ticks
VEV_TAKE_SIZE = 5              # tiny size to limit exposure
VEV_MAX_POS = 30               # hard cap per strike (very conservative)
VEV_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
VEV_DEEP_ITM = [4000, 4500]
VEV_DEEP_OTM = [6000, 6500]

# ─────────────────────────────────────────────────────────────
# BLACK-SCHOLES (no scipy)
# ─────────────────────────────────────────────────────────────
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

def get_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        data = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}

        result: dict[Symbol, list[Order]] = {}
        conversions = 0
        positions = state.position or {}

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. HYDROGEL_PACK MARKET MAKING (core PnL driver)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = self._trade_hydrogel(
                state.order_depths["HYDROGEL_PACK"],
                positions.get("HYDROGEL_PACK", 0),
            )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 2. OPTIONS: TAKE-ONLY ON EXTREME MISPRICING
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        vf_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            vf_mid = get_mid(state.order_depths["VELVETFRUIT_EXTRACT"])

        if vf_mid is not None:
            day = data.get("day_number", 0)
            tte = max(VEV_BASE_TTE - day - state.timestamp / 1_000_000, 0.001)

            for K in VEV_STRIKES:
                sym = f"VEV_{K}"
                if sym not in state.order_depths:
                    continue
                pos = positions.get(sym, 0)
                fair = bs_call(vf_mid, K, tte, VEV_IV)
                orders = self._take_mispriced_option(
                    state.order_depths[sym], fair, pos, sym
                )
                if orders:
                    result[sym] = orders

            # Deep ITM: take-only at intrinsic
            for K in VEV_DEEP_ITM:
                sym = f"VEV_{K}"
                if sym not in state.order_depths:
                    continue
                pos = positions.get(sym, 0)
                fair = max(vf_mid - K, 0)
                orders = self._take_mispriced_option(
                    state.order_depths[sym], fair, pos, sym,
                    thresh=5.0  # wider threshold for deep ITM (big spread)
                )
                if orders:
                    result[sym] = orders

            # Deep OTM: free lottery tickets
            for K in VEV_DEEP_OTM:
                sym = f"VEV_{K}"
                if sym not in state.order_depths:
                    continue
                pos = positions.get(sym, 0)
                orders = self._collect_free_otm(
                    state.order_depths[sym], pos, sym
                )
                if orders:
                    result[sym] = orders

        # ── Day detection ──
        prev_ts = data.get("prev_timestamp", 0)
        if state.timestamp < prev_ts:
            data["day_number"] = data.get("day_number", 0) + 1
        data["prev_timestamp"] = state.timestamp

        return result, conversions, json.dumps(data)

    # ─────────────────────────────────────────────────────────
    # HYDROGEL MARKET MAKING
    # ─────────────────────────────────────────────────────────
    def _trade_hydrogel(self, depth: OrderDepth, pos: int) -> list[Order]:
        orders: list[Order] = []
        sym = "HYDROGEL_PACK"

        mid = get_mid(depth)
        fair = HP_FAIR
        if mid is not None:
            fair = 0.7 * HP_FAIR + 0.3 * mid

        skew = -pos * HP_SKEW_PER_UNIT
        adj_fair = fair + skew

        # ── Aggressive takes ──
        if depth.sell_orders:
            for ask_p in sorted(depth.sell_orders.keys()):
                if ask_p < adj_fair - HP_TAKE_EDGE:
                    qty = min(-depth.sell_orders[ask_p], HP_MAX_POS - pos)
                    if qty > 0:
                        orders.append(Order(sym, ask_p, qty))
                        pos += qty
                else:
                    break

        if depth.buy_orders:
            for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_p > adj_fair + HP_TAKE_EDGE:
                    qty = min(depth.buy_orders[bid_p], HP_MAX_POS + pos)
                    if qty > 0:
                        orders.append(Order(sym, bid_p, -qty))
                        pos -= qty
                else:
                    break

        # ── Passive quotes ──
        bid_price = math.floor(adj_fair - HP_SPREAD_HALF)
        ask_price = math.ceil(adj_fair + HP_SPREAD_HALF)

        bid_size = min(HP_ORDER_SIZE, HP_MAX_POS - pos)
        ask_size = min(HP_ORDER_SIZE, HP_MAX_POS + pos)

        if bid_size > 0:
            orders.append(Order(sym, bid_price, bid_size))
        if ask_size > 0:
            orders.append(Order(sym, ask_price, -ask_size))

        return orders

    # ─────────────────────────────────────────────────────────
    # OPTIONS: TAKE-ONLY (no passive quoting)
    # ─────────────────────────────────────────────────────────
    def _take_mispriced_option(
        self, depth: OrderDepth, fair: float, pos: int, sym: str,
        thresh: float = VEV_TAKE_THRESH,
    ) -> list[Order]:
        """
        ONLY take liquidity when option is clearly mispriced.
        No passive quotes = no adverse selection risk.
        """
        orders: list[Order] = []

        # Buy if ask is significantly below fair
        if depth.sell_orders and pos < VEV_MAX_POS:
            for ask_p in sorted(depth.sell_orders.keys()):
                if ask_p < fair - thresh:
                    qty = min(
                        -depth.sell_orders[ask_p],
                        VEV_TAKE_SIZE,
                        VEV_MAX_POS - pos,
                    )
                    if qty > 0:
                        orders.append(Order(sym, ask_p, qty))
                        pos += qty
                else:
                    break

        # Sell if bid is significantly above fair
        if depth.buy_orders and pos > -VEV_MAX_POS:
            for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_p > fair + thresh:
                    qty = min(
                        depth.buy_orders[bid_p],
                        VEV_TAKE_SIZE,
                        VEV_MAX_POS + pos,
                    )
                    if qty > 0:
                        orders.append(Order(sym, bid_p, -qty))
                        pos -= qty
                else:
                    break

        return orders

    # ─────────────────────────────────────────────────────────
    # DEEP OTM: FREE LOTTERY TICKETS
    # ─────────────────────────────────────────────────────────
    def _collect_free_otm(
        self, depth: OrderDepth, pos: int, sym: str,
    ) -> list[Order]:
        orders: list[Order] = []

        # Buy at 0
        if depth.sell_orders and pos < 50:
            best_ask = min(depth.sell_orders.keys())
            if best_ask <= 0:
                qty = min(-depth.sell_orders[best_ask], 50 - pos, 25)
                if qty > 0:
                    orders.append(Order(sym, 0, qty))

        # Sell at 1+
        if pos > 0 and depth.buy_orders:
            best_bid = max(depth.buy_orders.keys())
            if best_bid >= 1:
                qty = min(depth.buy_orders[best_bid], pos, 25)
                if qty > 0:
                    orders.append(Order(sym, 1, -qty))

        return orders

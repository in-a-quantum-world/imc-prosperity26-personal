"""
IMC Prosperity 4 — Round 3 V2
================================
HP market making: proven +10K/day
VF market making: penny the bot spread + OBI signal (NEW — estimated +10K/day)  
Options: take-only on extreme mispricing (conservative)

Key findings:
  - VF order book imbalance (bid_vol - ask_vol) predicts next return: r=0.28
  - VF bot spread is 5 ticks. Penny at bid+1 / ask-1 captures the inside
  - HP bot spread is 16 ticks. Penny at ~8-tick spread around 10k
  - Options: correct TTE (day0=8, day1=7, day2=6, live_R3=5)
  - VEV IV stable at ~0.0126 with VEV_5400 at 0.0120

CRITICAL FOR LIVE SUBMISSION:
  Change VEV_INITIAL_TTE from 8 to 5 before submitting to competition.
"""

from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState,
)
import json
import math
from typing import Any

# ─────────────────────────────────────────────────────────────
# HYDROGEL CONFIG
# ─────────────────────────────────────────────────────────────
HP_FAIR = 10_000
HP_SPREAD_HALF = 4
HP_ORDER_SIZE = 15
HP_MAX_POS = 200
HP_SKEW_PER_UNIT = 0.05
HP_TAKE_EDGE = 1

# ─────────────────────────────────────────────────────────────
# VELVETFRUIT CONFIG (NEW)
# ─────────────────────────────────────────────────────────────
VF_MAX_POS = 200
VF_PENNY_SIZE = 15            # size for penny quotes
VF_OBI_LEAN = 1.0             # how much to lean quotes based on OBI
VF_INV_SKEW = 0.03            # inventory skew per unit
VF_TAKE_EDGE = 1              # take if clearly mispriced

# ─────────────────────────────────────────────────────────────
# OPTIONS CONFIG
# ─────────────────────────────────────────────────────────────
VEV_INITIAL_TTE = 8           # CHANGE TO 5 FOR LIVE SUBMISSION
VEV_IV = {
    4000: 0.0126, 4500: 0.0126,
    5000: 0.0126, 5100: 0.0125,
    5200: 0.0127, 5300: 0.0128,
    5400: 0.0120, 5500: 0.0130,
    6000: 0.0130, 6500: 0.0130,
}
VEV_IV_DEFAULT = 0.0126
VEV_TAKE_THRESH = 2.0
VEV_TAKE_SIZE = 10
VEV_MAX_POS = 100             # conservative — avoid large unhedged exposure
VEV_MM_SIZE = 10
VEV_MM_EDGE = 1.0

# ─────────────────────────────────────────────────────────────
# BLACK-SCHOLES
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

def get_best(depth: OrderDepth):
    bb = max(depth.buy_orders.keys()) if depth.buy_orders else None
    ba = min(depth.sell_orders.keys()) if depth.sell_orders else None
    return bb, ba

def get_obi(depth: OrderDepth) -> float:
    """Order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)"""
    bid_vol = sum(depth.buy_orders.values()) if depth.buy_orders else 0
    ask_vol = sum(-v for v in depth.sell_orders.values()) if depth.sell_orders else 0
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


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

        # ── Day tracking ──
        day = data.get("day_number", 0)
        prev_ts = data.get("prev_timestamp", 0)
        if state.timestamp < prev_ts and prev_ts > 0:
            day += 1
            data["day_number"] = day
        data["prev_timestamp"] = state.timestamp

        tte = max(VEV_INITIAL_TTE - day - state.timestamp / 1_000_000, 0.001)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. HYDROGEL_PACK MARKET MAKING
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = self._trade_hydrogel(
                state.order_depths["HYDROGEL_PACK"],
                positions.get("HYDROGEL_PACK", 0),
            )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 2. VELVETFRUIT_EXTRACT PENNY + OBI MARKET MAKING
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        vf_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            vf_depth = state.order_depths["VELVETFRUIT_EXTRACT"]
            vf_mid = get_mid(vf_depth)
            vf_pos = positions.get("VELVETFRUIT_EXTRACT", 0)
            result["VELVETFRUIT_EXTRACT"] = self._trade_vf(
                vf_depth, vf_pos, vf_mid
            )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 3. OPTIONS (correct TTE, market make + take)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if vf_mid is not None:
            for K in [5000, 5100, 5200, 5300, 5400, 5500]:
                sym = f"VEV_{K}"
                if sym not in state.order_depths:
                    continue
                sigma = VEV_IV.get(K, VEV_IV_DEFAULT)
                fair = bs_call(vf_mid, K, tte, sigma)
                pos = positions.get(sym, 0)
                result[sym] = self._trade_vev(
                    state.order_depths[sym], fair, pos, sym
                )

            # Deep ITM
            for K in [4000, 4500]:
                sym = f"VEV_{K}"
                if sym not in state.order_depths:
                    continue
                sigma = VEV_IV.get(K, VEV_IV_DEFAULT)
                fair = bs_call(vf_mid, K, tte, sigma)
                pos = positions.get(sym, 0)
                result[sym] = self._trade_vev(
                    state.order_depths[sym], fair, pos, sym,
                    edge=3.0, take_thresh=4.0,
                )

            # Deep OTM: free tickets
            for K in [6000, 6500]:
                sym = f"VEV_{K}"
                if sym not in state.order_depths:
                    continue
                pos = positions.get(sym, 0)
                result[sym] = self._trade_otm(state.order_depths[sym], pos, sym)

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

        # Aggressive takes
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

        # Passive quotes
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
    # VF PENNY + OBI MARKET MAKING
    # ─────────────────────────────────────────────────────────
    def _trade_vf(
        self, depth: OrderDepth, pos: int, mid: float | None
    ) -> list[Order]:
        orders: list[Order] = []
        sym = "VELVETFRUIT_EXTRACT"

        if mid is None:
            return orders

        best_bid, best_ask = get_best(depth)
        if best_bid is None or best_ask is None:
            return orders

        spread = best_ask - best_bid

        # ── Order book imbalance signal ──
        obi = get_obi(depth)
        # obi > 0 → more bid volume → expect price UP → lean to buy
        obi_lean = obi * VF_OBI_LEAN

        # ── Inventory skew ──
        inv_skew = -pos * VF_INV_SKEW

        # ── Aggressive takes when mispriced ──
        # Use OBI-adjusted fair: if obi positive, fair is slightly higher
        adj_fair = mid + obi_lean + inv_skew

        if depth.sell_orders:
            for ask_p in sorted(depth.sell_orders.keys()):
                if ask_p < adj_fair - VF_TAKE_EDGE and pos < VF_MAX_POS:
                    qty = min(-depth.sell_orders[ask_p], 10, VF_MAX_POS - pos)
                    if qty > 0:
                        orders.append(Order(sym, ask_p, qty))
                        pos += qty
                else:
                    break

        if depth.buy_orders:
            for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_p > adj_fair + VF_TAKE_EDGE and pos > -VF_MAX_POS:
                    qty = min(depth.buy_orders[bid_p], 10, VF_MAX_POS + pos)
                    if qty > 0:
                        orders.append(Order(sym, bid_p, -qty))
                        pos -= qty
                else:
                    break

        # ── Penny the spread: quote inside bot's L1 ──
        if spread >= 3:
            our_bid = best_bid + 1
            our_ask = best_ask - 1
        else:
            our_bid = best_bid
            our_ask = best_ask

        # Apply OBI lean and inventory skew
        our_bid = int(our_bid + obi_lean + inv_skew)
        our_ask = int(our_ask + obi_lean + inv_skew)

        # Ensure valid spread
        if our_bid >= our_ask:
            our_bid = our_ask - 1

        bid_size = min(VF_PENNY_SIZE, VF_MAX_POS - pos)
        ask_size = min(VF_PENNY_SIZE, VF_MAX_POS + pos)

        if bid_size > 0:
            orders.append(Order(sym, our_bid, bid_size))
        if ask_size > 0:
            orders.append(Order(sym, our_ask, -ask_size))

        return orders

    # ─────────────────────────────────────────────────────────
    # VEV OPTIONS MARKET MAKING
    # ─────────────────────────────────────────────────────────
    def _trade_vev(
        self, depth: OrderDepth, fair: float, pos: int, sym: str,
        edge: float = VEV_MM_EDGE, take_thresh: float = VEV_TAKE_THRESH,
    ) -> list[Order]:
        orders: list[Order] = []

        # Aggressive takes
        if depth.sell_orders:
            for ask_p in sorted(depth.sell_orders.keys()):
                if ask_p < fair - take_thresh and pos < VEV_MAX_POS:
                    qty = min(-depth.sell_orders[ask_p], VEV_TAKE_SIZE, VEV_MAX_POS - pos)
                    if qty > 0:
                        orders.append(Order(sym, ask_p, qty))
                        pos += qty
                else:
                    break

        if depth.buy_orders:
            for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_p > fair + take_thresh and pos > -VEV_MAX_POS:
                    qty = min(depth.buy_orders[bid_p], VEV_TAKE_SIZE, VEV_MAX_POS + pos)
                    if qty > 0:
                        orders.append(Order(sym, bid_p, -qty))
                        pos -= qty
                else:
                    break

        # Passive market making
        inv_skew = -pos * 0.02
        bid_p = math.floor(fair - edge + inv_skew)
        ask_p = math.ceil(fair + edge + inv_skew)
        if bid_p >= ask_p:
            bid_p = ask_p - 1
        bid_p = max(bid_p, 0)
        ask_p = max(ask_p, 1)

        bid_sz = min(VEV_MM_SIZE, VEV_MAX_POS - pos)
        ask_sz = min(VEV_MM_SIZE, VEV_MAX_POS + pos)

        if bid_sz > 0:
            orders.append(Order(sym, bid_p, bid_sz))
        if ask_sz > 0:
            orders.append(Order(sym, ask_p, -ask_sz))

        return orders

    # ─────────────────────────────────────────────────────────
    # DEEP OTM
    # ─────────────────────────────────────────────────────────
    def _trade_otm(self, depth: OrderDepth, pos: int, sym: str) -> list[Order]:
        orders: list[Order] = []
        if depth.sell_orders and pos < 100:
            if min(depth.sell_orders.keys()) <= 0:
                qty = min(-depth.sell_orders[min(depth.sell_orders.keys())], 100 - pos, 30)
                if qty > 0:
                    orders.append(Order(sym, 0, qty))
        if pos > 0 and depth.buy_orders:
            best_bid = max(depth.buy_orders.keys())
            if best_bid >= 1:
                qty = min(depth.buy_orders[best_bid], pos, 30)
                if qty > 0:
                    orders.append(Order(sym, 1, -qty))
        return orders

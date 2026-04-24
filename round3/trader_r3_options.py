"""
IMC Prosperity 4 — Round 3
Strategy 2: VEV Option Market Making + Delta Hedge (isolated)
BS fair value with σ=0.0135, fractional TTE, flat smile
Delta hedge via VELVETFRUIT_EXTRACT
"""

from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState,
)
import json
import math
from typing import Any

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
VEV_BASE_TTE = 7             # TTE in days at start of day 1
VEV_IV_DEFAULT = 0.0135      # implied vol per √day
VEV_IV_OVERRIDES = {         # per-strike adjustments from data
    5400: 0.0125,
}

VEV_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
VEV_DEEP_ITM = [4000, 4500]
VEV_DEEP_OTM = [6000, 6500]

VEV_EDGE = 1.0               # passive quote offset from fair
VEV_ORDER_SIZE = 15
VEV_MAX_POS = 300            # per-voucher position limit
VEV_TAKE_THRESH = 2.0        # take liquidity if mispricing > this
VEV_TAKE_SIZE = 20

VF_MAX_POS = 200             # VF position limit (for delta hedging)

# ─────────────────────────────────────────────────────────────
# BLACK-SCHOLES
# ─────────────────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    if x < -10: return 0.0
    if x > 10: return 1.0
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-9 or sigma <= 0:
        return max(S - K, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * norm_cdf(d1) - K * norm_cdf(d2)

def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-9:
        return 1.0 if S > K else 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    return norm_cdf(d1)

def get_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

def get_best_bid_ask(depth: OrderDepth):
    bb = max(depth.buy_orders.keys()) if depth.buy_orders else None
    ba = min(depth.sell_orders.keys()) if depth.sell_orders else None
    return bb, ba


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # ── Load state ──
        data = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}

        result: dict[Symbol, list[Order]] = {}
        conversions = 0
        positions = state.position or {}

        # ── VF mid price (needed for everything) ──
        vf_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            vf_mid = get_mid(state.order_depths["VELVETFRUIT_EXTRACT"])

        if vf_mid is None:
            return result, conversions, json.dumps(data)

        # ── Fractional TTE ──
        day = data.get("day_number", 0)
        tte = max(VEV_BASE_TTE - day - state.timestamp / 1_000_000, 0.001)

        # ── Trade each VEV strike ──
        total_option_delta = 0.0

        for K in VEV_STRIKES:
            symbol = f"VEV_{K}"
            if symbol not in state.order_depths:
                continue

            sigma = VEV_IV_OVERRIDES.get(K, VEV_IV_DEFAULT)
            fair = bs_call_price(vf_mid, K, tte, sigma)
            delta = bs_delta(vf_mid, K, tte, sigma)
            pos = positions.get(symbol, 0)

            orders = self._trade_vev(state.order_depths[symbol], fair, pos, symbol)
            result[symbol] = orders
            total_option_delta += pos * delta

        # ── Deep ITM: trade as synthetic forward (delta=1) ──
        for K in VEV_DEEP_ITM:
            symbol = f"VEV_{K}"
            if symbol not in state.order_depths:
                continue
            pos = positions.get(symbol, 0)
            fair = max(vf_mid - K, 0)
            result[symbol] = self._trade_deep_itm(
                state.order_depths[symbol], fair, pos, symbol
            )
            total_option_delta += pos * 1.0

        # ── Deep OTM: buy at 0, sell at 1 ──
        for K in VEV_DEEP_OTM:
            symbol = f"VEV_{K}"
            if symbol not in state.order_depths:
                continue
            pos = positions.get(symbol, 0)
            result[symbol] = self._trade_deep_otm(
                state.order_depths[symbol], pos, symbol
            )
            # delta ≈ 0, negligible for hedging

        # ── Delta hedge on VF ──
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            vf_pos = positions.get("VELVETFRUIT_EXTRACT", 0)
            target_vf = -round(total_option_delta)
            # Clamp target to VF position limit
            target_vf = max(-VF_MAX_POS, min(VF_MAX_POS, target_vf))

            hedge_orders = self._delta_hedge_vf(
                state.order_depths["VELVETFRUIT_EXTRACT"],
                vf_pos, target_vf, vf_mid,
            )
            result["VELVETFRUIT_EXTRACT"] = hedge_orders

        # ── Persist state (day detection) ──
        prev_ts = data.get("prev_timestamp", 0)
        if state.timestamp < prev_ts:
            data["day_number"] = day + 1
        data["prev_timestamp"] = state.timestamp

        return result, conversions, json.dumps(data)

    def _trade_vev(
        self, depth: OrderDepth, fair: float, pos: int, sym: str
    ) -> list[Order]:
        orders: list[Order] = []

        # ── Aggressive takes ──
        if depth.sell_orders:
            for ask_p in sorted(depth.sell_orders.keys()):
                if ask_p < fair - VEV_TAKE_THRESH:
                    qty = min(-depth.sell_orders[ask_p], VEV_TAKE_SIZE, VEV_MAX_POS - pos)
                    if qty > 0:
                        orders.append(Order(sym, ask_p, qty))
                        pos += qty
                else:
                    break

        if depth.buy_orders:
            for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_p > fair + VEV_TAKE_THRESH:
                    qty = min(depth.buy_orders[bid_p], VEV_TAKE_SIZE, VEV_MAX_POS + pos)
                    if qty > 0:
                        orders.append(Order(sym, bid_p, -qty))
                        pos -= qty
                else:
                    break

        # ── Passive quotes ──
        inv_skew = -pos * 0.02  # subtle skew to shed inventory
        bid_price = math.floor(fair - VEV_EDGE + inv_skew)
        ask_price = math.ceil(fair + VEV_EDGE + inv_skew)
        bid_price = min(bid_price, ask_price - 1)

        bid_size = min(VEV_ORDER_SIZE, VEV_MAX_POS - pos)
        ask_size = min(VEV_ORDER_SIZE, VEV_MAX_POS + pos)

        if bid_size > 0 and bid_price > 0:
            orders.append(Order(sym, bid_price, bid_size))
        if ask_size > 0 and ask_price > 0:
            orders.append(Order(sym, ask_price, -ask_size))

        return orders

    def _trade_deep_itm(
        self, depth: OrderDepth, fair: float, pos: int, sym: str
    ) -> list[Order]:
        orders: list[Order] = []
        edge = 3.0

        if depth.sell_orders:
            for ask_p in sorted(depth.sell_orders.keys()):
                if ask_p < fair - edge:
                    qty = min(-depth.sell_orders[ask_p], VEV_MAX_POS - pos, 15)
                    if qty > 0:
                        orders.append(Order(sym, ask_p, qty))
                        pos += qty

        if depth.buy_orders:
            for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_p > fair + edge:
                    qty = min(depth.buy_orders[bid_p], VEV_MAX_POS + pos, 15)
                    if qty > 0:
                        orders.append(Order(sym, bid_p, -qty))
                        pos -= qty

        bid_p = math.floor(fair - 1)
        ask_p = math.ceil(fair + 1)
        bid_sz = min(10, VEV_MAX_POS - pos)
        ask_sz = min(10, VEV_MAX_POS + pos)
        if bid_sz > 0 and bid_p > 0:
            orders.append(Order(sym, bid_p, bid_sz))
        if ask_sz > 0 and ask_p > 0:
            orders.append(Order(sym, ask_p, -ask_sz))
        return orders

    def _trade_deep_otm(
        self, depth: OrderDepth, pos: int, sym: str
    ) -> list[Order]:
        orders: list[Order] = []
        if depth.sell_orders:
            best_ask = min(depth.sell_orders.keys())
            if best_ask <= 0:
                qty = min(-depth.sell_orders[best_ask], VEV_MAX_POS - pos, 25)
                if qty > 0:
                    orders.append(Order(sym, 0, qty))

        if pos > 0 and depth.buy_orders:
            best_bid = max(depth.buy_orders.keys())
            if best_bid >= 1:
                qty = min(depth.buy_orders[best_bid], pos, 25)
                if qty > 0:
                    orders.append(Order(sym, 1, -qty))
        return orders

    def _delta_hedge_vf(
        self, depth: OrderDepth, vf_pos: int, target_vf: int, vf_mid: float,
    ) -> list[Order]:
        orders: list[Order] = []
        gap = target_vf - vf_pos

        if gap > 0 and depth.sell_orders:
            # Buy VF
            buy_qty = min(gap, VF_MAX_POS - vf_pos)
            if buy_qty > 0:
                # Walk the ask book
                for ask_p in sorted(depth.sell_orders.keys()):
                    can_take = min(-depth.sell_orders[ask_p], buy_qty)
                    if can_take > 0:
                        orders.append(Order("VELVETFRUIT_EXTRACT", ask_p, can_take))
                        buy_qty -= can_take
                    if buy_qty <= 0:
                        break
                # If still need more, post a passive bid
                if buy_qty > 0:
                    orders.append(Order("VELVETFRUIT_EXTRACT", math.floor(vf_mid), buy_qty))

        elif gap < 0 and depth.buy_orders:
            # Sell VF
            sell_qty = min(-gap, VF_MAX_POS + vf_pos)
            if sell_qty > 0:
                for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
                    can_take = min(depth.buy_orders[bid_p], sell_qty)
                    if can_take > 0:
                        orders.append(Order("VELVETFRUIT_EXTRACT", bid_p, -can_take))
                        sell_qty -= can_take
                    if sell_qty <= 0:
                        break
                if sell_qty > 0:
                    orders.append(Order("VELVETFRUIT_EXTRACT", math.ceil(vf_mid), -sell_qty))

        return orders

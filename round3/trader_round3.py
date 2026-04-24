"""
IMC Prosperity 4 — Round 3 Trader
===================================
Strategies (each can be toggled via ENABLE_* flags):
  1. HYDROGEL_PACK market making  — mean-reversion around 10,000
  2. VEV option market making     — BS fair value + delta hedge with VF
  3. VELVETFRUIT_EXTRACT scalping  — fade AC(1) mean-reversion signal

Key calibration from historical data:
  - BS implied vol:  σ ≈ 0.0135  (per √day, NOT the 0.0215 tick-level realized vol)
  - TTE is fractional: T = base_tte - day - timestamp/1_000_000
  - Vol smile is flat across strikes 5000–5500
  - VEV_5400 IV slightly lower (~0.0125)
"""

from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState,
)
import json
import math
from typing import Any

# ─────────────────────────────────────────────────────────────
# STRATEGY TOGGLES
# ─────────────────────────────────────────────────────────────
ENABLE_HYDROGEL = True
ENABLE_VEV_OPTIONS = True
ENABLE_VF_SCALP = True

# ─────────────────────────────────────────────────────────────
# HYDROGEL_PACK CONFIG
# ─────────────────────────────────────────────────────────────
HP_FAIR = 10_000
HP_SPREAD_HALF = 4          # quote at fair ± this
HP_ORDER_SIZE = 10
HP_MAX_POS = 50             # position limit (adjust per round rules)
HP_SKEW_PER_LOT = 1         # skew mid by this much per unit of inventory

# ─────────────────────────────────────────────────────────────
# VEV OPTIONS CONFIG
# ─────────────────────────────────────────────────────────────
VEV_BASE_TTE = 7            # TTE in days at start of day 1
VEV_IV_DEFAULT = 0.0135     # flat implied vol per √day
VEV_IV_OVERRIDES = {        # per-strike IV overrides (from data)
    5400: 0.0125,
}
VEV_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
VEV_DEEP_ITM = [4000, 4500]
VEV_DEEP_OTM = [6000, 6500]
VEV_EDGE = 1.0              # post at BS_fair ± this
VEV_ORDER_SIZE = 10
VEV_MAX_POS = 50            # per-option position limit (adjust per round)
VEV_TAKE_THRESH = 2.0       # take liquidity if mispricing exceeds this
VEV_TAKE_SIZE = 15

# Delta hedging
DELTA_HEDGE_ENABLED = True
VF_MAX_POS = 80             # VF position limit for hedging

# ─────────────────────────────────────────────────────────────
# VF SCALP CONFIG
# ─────────────────────────────────────────────────────────────
VF_SCALP_SIZE = 5
VF_SCALP_EDGE = 1           # only scalp if we expect > this much reversion

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

_SQRT2PI = math.sqrt(2.0 * math.pi)

def norm_cdf(x: float) -> float:
    """Standard normal CDF via Abramowitz & Stegun approximation."""
    if x < -10:
        return 0.0
    if x > 10:
        return 1.0
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI

def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes European call price (r=0)."""
    if T <= 1e-9 or sigma <= 0:
        return max(S - K, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * norm_cdf(d1) - K * norm_cdf(d2)

def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes call delta (r=0)."""
    if T <= 1e-9:
        return 1.0 if S > K else 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    return norm_cdf(d1)

def bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes call gamma (r=0)."""
    if T <= 1e-9 or sigma <= 0 or S <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    return norm_pdf(d1) / (S * sigma * sqrt_T)

def get_mid(order_depth: OrderDepth) -> float | None:
    """Best-bid/ask mid, or None if one side is empty."""
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return (best_bid + best_ask) / 2.0

def get_best_bid_ask(order_depth: OrderDepth):
    """Return (best_bid, best_ask) or (None, None)."""
    bb = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
    ba = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
    return bb, ba


class Trader:
    """
    Round 3 combined strategy trader.
    State persisted across ticks via traderData JSON.
    """

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # ── Load persisted state ──
        data = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}

        prev_vf_mid = data.get("prev_vf_mid", None)

        result: dict[Symbol, list[Order]] = {}
        conversions = 0

        # ── Current positions ──
        positions = state.position if state.position else {}

        # ── Get VF mid (needed for options) ──
        vf_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            vf_mid = get_mid(state.order_depths["VELVETFRUIT_EXTRACT"])

        # ── Compute fractional TTE ──
        # TTE = base - day - timestamp/1_000_000
        # Day numbering: the data says "starting from day 1" with TTE=7
        # If we're on day 0 of the round, that could be day 0 or day 1
        # We need to adjust based on actual round structure.
        # Conservative: assume day in state.timestamp corresponds to the
        # competition day, and TTE starts at VEV_BASE_TTE on the first day.
        day = data.get("day_number", 0)  # will be updated below
        tte = max(VEV_BASE_TTE - day - state.timestamp / 1_000_000, 0.001)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STRATEGY 1: HYDROGEL_PACK MARKET MAKING
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if ENABLE_HYDROGEL and "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = self._trade_hydrogel(
                state.order_depths["HYDROGEL_PACK"],
                positions.get("HYDROGEL_PACK", 0),
            )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STRATEGY 2: VEV OPTION MARKET MAKING + DELTA HEDGE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if ENABLE_VEV_OPTIONS and vf_mid is not None:
            # --- Trade each VEV strike ---
            total_target_vf_delta = 0.0

            for K in VEV_STRIKES:
                symbol = f"VEV_{K}"
                if symbol not in state.order_depths:
                    continue

                sigma = VEV_IV_OVERRIDES.get(K, VEV_IV_DEFAULT)
                fair = bs_call_price(vf_mid, K, tte, sigma)
                delta = bs_delta(vf_mid, K, tte, sigma)
                pos = positions.get(symbol, 0)

                orders = self._trade_vev(
                    state.order_depths[symbol], fair, pos, symbol
                )
                result[symbol] = orders

                # Accumulate target delta from option positions
                total_target_vf_delta += pos * delta

            # --- Deep ITM: trade as synthetic forward ---
            for K in VEV_DEEP_ITM:
                symbol = f"VEV_{K}"
                if symbol not in state.order_depths:
                    continue
                pos = positions.get(symbol, 0)
                fair = max(vf_mid - K, 0)
                orders = self._trade_deep_itm(
                    state.order_depths[symbol], fair, pos, symbol
                )
                result[symbol] = orders
                total_target_vf_delta += pos * 1.0  # delta ≈ 1

            # --- Deep OTM: try to sell any we hold at 1, buy at 0 ---
            for K in VEV_DEEP_OTM:
                symbol = f"VEV_{K}"
                if symbol not in state.order_depths:
                    continue
                pos = positions.get(symbol, 0)
                result[symbol] = self._trade_deep_otm(
                    state.order_depths[symbol], pos, symbol
                )

            # --- Delta hedge on VF ---
            if DELTA_HEDGE_ENABLED and "VELVETFRUIT_EXTRACT" in state.order_depths:
                vf_pos = positions.get("VELVETFRUIT_EXTRACT", 0)
                # We want vf_pos to OFFSET the option delta
                # If we're long options (positive delta), sell VF
                target_vf = -round(total_target_vf_delta)
                hedge_orders = self._delta_hedge_vf(
                    state.order_depths["VELVETFRUIT_EXTRACT"],
                    vf_pos, target_vf, vf_mid,
                )
                # Merge with any VF scalp orders
                if "VELVETFRUIT_EXTRACT" in result:
                    result["VELVETFRUIT_EXTRACT"].extend(hedge_orders)
                else:
                    result["VELVETFRUIT_EXTRACT"] = hedge_orders

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STRATEGY 3: VF MEAN-REVERSION SCALP
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if (
            ENABLE_VF_SCALP
            and not DELTA_HEDGE_ENABLED  # don't scalp if hedging
            and "VELVETFRUIT_EXTRACT" in state.order_depths
            and prev_vf_mid is not None
            and vf_mid is not None
        ):
            vf_pos = positions.get("VELVETFRUIT_EXTRACT", 0)
            scalp_orders = self._scalp_vf(
                state.order_depths["VELVETFRUIT_EXTRACT"],
                vf_mid, prev_vf_mid, vf_pos,
            )
            if "VELVETFRUIT_EXTRACT" in result:
                result["VELVETFRUIT_EXTRACT"].extend(scalp_orders)
            else:
                result["VELVETFRUIT_EXTRACT"] = scalp_orders

        # ── Persist state ──
        data["prev_vf_mid"] = vf_mid
        # Detect new day: if timestamp < previous timestamp, day incremented
        prev_ts = data.get("prev_timestamp", 0)
        if state.timestamp < prev_ts:
            data["day_number"] = day + 1
        data["prev_timestamp"] = state.timestamp

        trader_data = json.dumps(data)
        return result, conversions, trader_data

    # ─────────────────────────────────────────────────────────
    # STRATEGY 1: HYDROGEL_PACK market making
    # ─────────────────────────────────────────────────────────
    def _trade_hydrogel(
        self, depth: OrderDepth, pos: int
    ) -> list[Order]:
        orders: list[Order] = []
        sym = "HYDROGEL_PACK"

        # Adaptive fair: anchor to 10k, but use mid if available
        mid = get_mid(depth)
        fair = HP_FAIR
        if mid is not None:
            # Blend: 70% anchor + 30% market mid
            fair = 0.7 * HP_FAIR + 0.3 * mid

        # Inventory skew
        skew = -pos * HP_SKEW_PER_LOT
        adj_fair = fair + skew

        bid_price = math.floor(adj_fair - HP_SPREAD_HALF)
        ask_price = math.ceil(adj_fair + HP_SPREAD_HALF)

        bid_size = min(HP_ORDER_SIZE, HP_MAX_POS - pos)
        ask_size = min(HP_ORDER_SIZE, HP_MAX_POS + pos)

        # ── Aggressive takes: if price is far from fair, cross spread ──
        best_bid, best_ask = get_best_bid_ask(depth)

        if best_ask is not None and best_ask < fair - 1:
            take_size = min(-depth.sell_orders.get(best_ask, 0), HP_MAX_POS - pos)
            if take_size > 0:
                orders.append(Order(sym, best_ask, take_size))
                pos += take_size

        if best_bid is not None and best_bid > fair + 1:
            take_size = min(depth.buy_orders.get(best_bid, 0), HP_MAX_POS + pos)
            if take_size > 0:
                orders.append(Order(sym, best_bid, -take_size))
                pos -= take_size

        # ── Passive quotes ──
        bid_size = min(HP_ORDER_SIZE, HP_MAX_POS - pos)
        ask_size = min(HP_ORDER_SIZE, HP_MAX_POS + pos)

        if bid_size > 0:
            orders.append(Order(sym, bid_price, bid_size))
        if ask_size > 0:
            orders.append(Order(sym, ask_price, -ask_size))

        return orders

    # ─────────────────────────────────────────────────────────
    # STRATEGY 2a: VEV option market making
    # ─────────────────────────────────────────────────────────
    def _trade_vev(
        self, depth: OrderDepth, fair: float, pos: int, sym: str
    ) -> list[Order]:
        orders: list[Order] = []
        best_bid, best_ask = get_best_bid_ask(depth)

        # ── Aggressive takes on mispricing ──
        if best_ask is not None and best_ask < fair - VEV_TAKE_THRESH:
            take_qty = min(
                -depth.sell_orders.get(best_ask, 0),
                VEV_TAKE_SIZE,
                VEV_MAX_POS - pos,
            )
            if take_qty > 0:
                orders.append(Order(sym, best_ask, take_qty))

        if best_bid is not None and best_bid > fair + VEV_TAKE_THRESH:
            take_qty = min(
                depth.buy_orders.get(best_bid, 0),
                VEV_TAKE_SIZE,
                VEV_MAX_POS + pos,
            )
            if take_qty > 0:
                orders.append(Order(sym, best_bid, -take_qty))

        # ── Passive market making ──
        # Inventory skew: shift quotes to shed inventory
        inv_skew = -pos * 0.05  # subtle skew

        bid_price = math.floor(fair - VEV_EDGE + inv_skew)
        ask_price = math.ceil(fair + VEV_EDGE + inv_skew)

        # Ensure we don't cross ourselves
        bid_price = min(bid_price, ask_price - 1)

        bid_size = min(VEV_ORDER_SIZE, VEV_MAX_POS - pos)
        ask_size = min(VEV_ORDER_SIZE, VEV_MAX_POS + pos)

        if bid_size > 0 and bid_price > 0:
            orders.append(Order(sym, bid_price, bid_size))
        if ask_size > 0 and ask_price > 0:
            orders.append(Order(sym, ask_price, -ask_size))

        return orders

    # ─────────────────────────────────────────────────────────
    # STRATEGY 2b: Deep ITM options (synthetic forward)
    # ─────────────────────────────────────────────────────────
    def _trade_deep_itm(
        self, depth: OrderDepth, fair: float, pos: int, sym: str
    ) -> list[Order]:
        orders: list[Order] = []
        best_bid, best_ask = get_best_bid_ask(depth)

        # These trade at intrinsic. Try to capture spread.
        edge = 2.0

        if best_ask is not None and best_ask < fair - edge:
            take_qty = min(
                -depth.sell_orders.get(best_ask, 0),
                VEV_MAX_POS - pos,
                10,
            )
            if take_qty > 0:
                orders.append(Order(sym, best_ask, take_qty))

        if best_bid is not None and best_bid > fair + edge:
            take_qty = min(
                depth.buy_orders.get(best_bid, 0),
                VEV_MAX_POS + pos,
                10,
            )
            if take_qty > 0:
                orders.append(Order(sym, best_bid, -take_qty))

        # Passive quotes around fair
        bid_p = math.floor(fair - 1)
        ask_p = math.ceil(fair + 1)
        bid_sz = min(5, VEV_MAX_POS - pos)
        ask_sz = min(5, VEV_MAX_POS + pos)

        if bid_sz > 0 and bid_p > 0:
            orders.append(Order(sym, bid_p, bid_sz))
        if ask_sz > 0 and ask_p > 0:
            orders.append(Order(sym, ask_p, -ask_sz))

        return orders

    # ─────────────────────────────────────────────────────────
    # STRATEGY 2c: Deep OTM options (nearly worthless)
    # ─────────────────────────────────────────────────────────
    def _trade_deep_otm(
        self, depth: OrderDepth, pos: int, sym: str
    ) -> list[Order]:
        orders: list[Order] = []
        best_bid, best_ask = get_best_bid_ask(depth)

        # Buy at 0 if possible (free lottery ticket)
        if best_ask is not None and best_ask <= 0:
            qty = min(-depth.sell_orders.get(best_ask, 0), VEV_MAX_POS - pos, 20)
            if qty > 0:
                orders.append(Order(sym, 0, qty))

        # Sell at 1 if we hold any
        if pos > 0 and best_bid is not None and best_bid >= 1:
            qty = min(depth.buy_orders.get(best_bid, 0), pos, 20)
            if qty > 0:
                orders.append(Order(sym, 1, -qty))

        return orders

    # ─────────────────────────────────────────────────────────
    # STRATEGY 2d: Delta hedging on VF
    # ─────────────────────────────────────────────────────────
    def _delta_hedge_vf(
        self, depth: OrderDepth, vf_pos: int, target_vf: int, vf_mid: float,
    ) -> list[Order]:
        orders: list[Order] = []
        delta_gap = target_vf - vf_pos

        # Clamp to position limits
        if delta_gap > 0:
            # Need to buy VF
            buy_qty = min(delta_gap, VF_MAX_POS - vf_pos)
            if buy_qty > 0:
                # Use a slightly aggressive price to ensure fill
                best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
                price = best_ask if best_ask is not None else math.ceil(vf_mid + 2)
                orders.append(Order("VELVETFRUIT_EXTRACT", price, buy_qty))

        elif delta_gap < 0:
            # Need to sell VF
            sell_qty = min(-delta_gap, VF_MAX_POS + vf_pos)
            if sell_qty > 0:
                best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
                price = best_bid if best_bid is not None else math.floor(vf_mid - 2)
                orders.append(Order("VELVETFRUIT_EXTRACT", price, -sell_qty))

        return orders

    # ─────────────────────────────────────────────────────────
    # STRATEGY 3: VF mean-reversion scalp
    # ─────────────────────────────────────────────────────────
    def _scalp_vf(
        self, depth: OrderDepth, vf_mid: float, prev_mid: float, pos: int,
    ) -> list[Order]:
        """
        AC(1) ≈ -0.155: after an up-tick, price tends to revert.
        Fade the last move with a small passive order.
        """
        orders: list[Order] = []
        move = vf_mid - prev_mid

        # Expected reversion ≈ -0.155 * move
        expected_revert = -0.155 * move

        if abs(expected_revert) < VF_SCALP_EDGE:
            return orders

        if expected_revert > 0 and pos < VF_MAX_POS:
            # Price just dropped, expect bounce — buy
            buy_price = math.floor(vf_mid)
            qty = min(VF_SCALP_SIZE, VF_MAX_POS - pos)
            if qty > 0:
                orders.append(Order("VELVETFRUIT_EXTRACT", buy_price, qty))

        elif expected_revert < 0 and pos > -VF_MAX_POS:
            # Price just rose, expect pullback — sell
            sell_price = math.ceil(vf_mid)
            qty = min(VF_SCALP_SIZE, VF_MAX_POS + pos)
            if qty > 0:
                orders.append(Order("VELVETFRUIT_EXTRACT", sell_price, -qty))

        return orders

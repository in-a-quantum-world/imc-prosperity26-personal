"""
IMC Prosperity 4 — Round 3
Strategy 2 REVISED: IV Scalping + Unhedged (Frankfurt Hedgehogs style)

Key findings from data analysis:
  - Vol smile exists (parabola a≈0.1, b≈0, c≈0.0135)
  - IV deviations from smile mean-revert at different speeds
  - Fast-reverting: K=5300 (111 ticks), K=5100 (154 ticks) → SCALP THESE
  - Slow/structural: K=5400, K=5500 → don't scalp, but can lean directionally
  - Delta hedging costs ~25K/day in spread → DO NOT HEDGE
  - VF has AC(1)=-0.15 mean reversion → unhedged option positions benefit
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
VEV_BASE_TTE = 7

# Smile parabola initial params (a*m² + b*m + c)
# Will be re-fitted live each tick
SMILE_A_INIT = 0.10
SMILE_B_INIT = 0.0
SMILE_C_INIT = 0.0135

# Strikes to actively scalp (fast-reverting IV deviations)
SCALP_STRIKES = [5100, 5200, 5300]
# Strikes with structural mispricing (slow reversion, lean but don't scalp)
LEAN_STRIKES = [5400]  # persistently cheap → buy bias
# Deep ITM / OTM
DEEP_ITM = [4000, 4500]
DEEP_OTM = [6000, 6500]
# All strikes used for smile fitting
FIT_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]

# Scalp parameters
SCALP_IV_THRESH = 0.00015    # IV deviation threshold to trigger trade
SCALP_SIZE = 15              # order size for scalp trades
SCALP_MAX_POS = 200          # max position per strike (limit is 300)

# Lean parameters
LEAN_SIZE = 5                # smaller size for structural trades
LEAN_MAX_POS = 100

# Deep OTM: buy at 0 (free lottery tickets)
OTM_MAX_POS = 50

# Position limit
VEV_POS_LIMIT = 300


# ─────────────────────────────────────────────────────────────
# BLACK-SCHOLES
# ─────────────────────────────────────────────────────────────
def norm_cdf(x: float) -> float:
    if x < -10: return 0.0
    if x > 10: return 1.0
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-9 or sigma <= 0:
        return max(S - K, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * norm_cdf(d1) - K * norm_cdf(d2)

def bs_iv(price: float, S: float, K: float, T: float) -> float:
    """Implied vol via bisection (no scipy dependency)."""
    intrinsic = max(S - K, 0)
    if price <= intrinsic + 0.01 or price >= S - 0.01 or T <= 1e-9:
        return float('nan')
    lo, hi = 0.001, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_call(S, K, T, mid) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-9 or sigma <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    return S * norm_pdf(d1) * sqrt_T

def get_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


def fit_smile(ivs: dict[int, float], S: float) -> tuple[float, float, float] | None:
    """
    Fit parabola IV = a*m² + b*m + c to observed IVs.
    ivs: {strike: implied_vol}
    Returns (a, b, c) or None if insufficient data.
    """
    if len(ivs) < 3:
        return None
    
    strikes_list = sorted(ivs.keys())
    m_vals = [math.log(K / S) for K in strikes_list]
    iv_vals = [ivs[K] for K in strikes_list]
    
    # Manual least-squares for parabola (no numpy needed)
    n = len(m_vals)
    # Build normal equations for y = a*x² + b*x + c
    S0 = n
    S1 = sum(m_vals)
    S2 = sum(m * m for m in m_vals)
    S3 = sum(m * m * m for m in m_vals)
    S4 = sum(m * m * m * m for m in m_vals)
    T0 = sum(iv_vals)
    T1 = sum(m * iv for m, iv in zip(m_vals, iv_vals))
    T2 = sum(m * m * iv for m, iv in zip(m_vals, iv_vals))
    
    # Solve 3x3 system using Cramer's rule
    # [S4 S3 S2] [a]   [T2]
    # [S3 S2 S1] [b] = [T1]
    # [S2 S1 S0] [c]   [T0]
    det = S4*(S2*S0 - S1*S1) - S3*(S3*S0 - S1*S2) + S2*(S3*S1 - S2*S2)
    if abs(det) < 1e-20:
        return None
    
    a = (T2*(S2*S0 - S1*S1) - S3*(T1*S0 - S1*T0) + S2*(T1*S1 - S2*T0)) / det
    b = (S4*(T1*S0 - S1*T0) - T2*(S3*S0 - S1*S2) + S2*(S3*T0 - T1*S2)) / det
    c = (S4*(S2*T0 - T1*S1) - S3*(S3*T0 - T1*S2) + T2*(S3*S1 - S2*S2)) / det
    
    return (a, b, c)


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

        # ── VF mid ──
        vf_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            vf_mid = get_mid(state.order_depths["VELVETFRUIT_EXTRACT"])
        
        if vf_mid is None:
            return result, conversions, json.dumps(data)

        # ── TTE ──
        day = data.get("day_number", 0)
        tte = max(VEV_BASE_TTE - day - state.timestamp / 1_000_000, 0.001)

        # ── Step 1: Compute IV for each strike ──
        live_ivs: dict[int, float] = {}
        live_mids: dict[int, float] = {}
        
        for K in FIT_STRIKES:
            sym = f"VEV_{K}"
            if sym not in state.order_depths:
                continue
            mid = get_mid(state.order_depths[sym])
            if mid is None or mid <= 0:
                continue
            iv = bs_iv(mid, vf_mid, K, tte)
            if math.isnan(iv):
                continue
            live_ivs[K] = iv
            live_mids[K] = mid

        # ── Step 2: Fit volatility smile ──
        smile = fit_smile(live_ivs, vf_mid)
        if smile is None:
            # Fall back to stored params
            smile = (
                data.get("smile_a", SMILE_A_INIT),
                data.get("smile_b", SMILE_B_INIT),
                data.get("smile_c", SMILE_C_INIT),
            )
        
        a, b, c = smile
        data["smile_a"] = a
        data["smile_b"] = b
        data["smile_c"] = c

        # ── Step 3: Compute deviations and fair prices ──
        for K in FIT_STRIKES:
            if K not in live_ivs:
                continue
            
            sym = f"VEV_{K}"
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            
            m = math.log(K / vf_mid)
            fitted_iv = a * m * m + b * m + c
            actual_iv = live_ivs[K]
            iv_dev = actual_iv - fitted_iv
            
            fair_price = bs_call(vf_mid, K, tte, fitted_iv)
            pos = positions.get(sym, 0)
            
            orders: list[Order] = []
            
            if K in SCALP_STRIKES:
                orders = self._iv_scalp(
                    depth, fair_price, iv_dev, pos, sym, K,
                    SCALP_IV_THRESH, SCALP_SIZE, SCALP_MAX_POS,
                )
            elif K in LEAN_STRIKES:
                # VEV_5400 is structurally cheap → buy bias
                orders = self._lean_buy(
                    depth, fair_price, pos, sym,
                    LEAN_SIZE, LEAN_MAX_POS,
                )
            
            if orders:
                result[sym] = orders

        # ── Step 4: Deep ITM — trade at intrinsic ──
        for K in DEEP_ITM:
            sym = f"VEV_{K}"
            if sym not in state.order_depths:
                continue
            pos = positions.get(sym, 0)
            fair = max(vf_mid - K, 0)
            orders = self._trade_intrinsic(state.order_depths[sym], fair, pos, sym)
            if orders:
                result[sym] = orders

        # ── Step 5: Deep OTM — collect free options ──
        for K in DEEP_OTM:
            sym = f"VEV_{K}"
            if sym not in state.order_depths:
                continue
            pos = positions.get(sym, 0)
            orders = self._collect_free_otm(state.order_depths[sym], pos, sym)
            if orders:
                result[sym] = orders

        # ── NO DELTA HEDGE — deliberately unhedged ──
        # VF mean-reversion makes unhedged profitable per Frankfurt Hedgehogs

        # ── Day detection ──
        prev_ts = data.get("prev_timestamp", 0)
        if state.timestamp < prev_ts:
            data["day_number"] = day + 1
        data["prev_timestamp"] = state.timestamp

        return result, conversions, json.dumps(data)

    def _iv_scalp(
        self, depth: OrderDepth, fair: float, iv_dev: float,
        pos: int, sym: str, K: int,
        thresh: float, size: int, max_pos: int,
    ) -> list[Order]:
        """
        Core IV scalping logic:
        - If IV is BELOW smile (iv_dev < -thresh): option is cheap → BUY
        - If IV is ABOVE smile (iv_dev > +thresh): option is rich → SELL
        """
        orders: list[Order] = []
        
        # ── Aggressive: take liquidity on mispricing ──
        if iv_dev < -thresh:
            # Option is cheap → buy
            if depth.sell_orders and pos < max_pos:
                for ask_p in sorted(depth.sell_orders.keys()):
                    if ask_p <= fair + 0.5:  # buy at or below fair
                        qty = min(-depth.sell_orders[ask_p], size, max_pos - pos)
                        if qty > 0:
                            orders.append(Order(sym, ask_p, qty))
                            pos += qty
                    else:
                        break

        elif iv_dev > thresh:
            # Option is rich → sell
            if depth.buy_orders and pos > -max_pos:
                for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
                    if bid_p >= fair - 0.5:  # sell at or above fair
                        qty = min(depth.buy_orders[bid_p], size, max_pos + pos)
                        if qty > 0:
                            orders.append(Order(sym, bid_p, -qty))
                            pos -= qty
                    else:
                        break

        # ── Passive: quote around fair to capture spread ──
        # Shift quotes based on IV deviation to lean into the scalp
        lean = 0.0
        if iv_dev < -thresh * 0.5:
            lean = 0.5   # lean bid up (want to buy)
        elif iv_dev > thresh * 0.5:
            lean = -0.5  # lean ask down (want to sell)
        
        # Inventory skew
        inv_skew = -pos * 0.02
        
        bid_p = math.floor(fair - 0.5 + lean + inv_skew)
        ask_p = math.ceil(fair + 0.5 + lean + inv_skew)
        bid_p = min(bid_p, ask_p - 1)
        
        bid_sz = min(size, max_pos - pos)
        ask_sz = min(size, max_pos + pos)
        
        if bid_sz > 0 and bid_p > 0:
            orders.append(Order(sym, bid_p, bid_sz))
        if ask_sz > 0 and ask_p > 0:
            orders.append(Order(sym, ask_p, -ask_sz))

        return orders

    def _lean_buy(
        self, depth: OrderDepth, fair: float,
        pos: int, sym: str, size: int, max_pos: int,
    ) -> list[Order]:
        """For structurally cheap options (VEV_5400): buy bias."""
        orders: list[Order] = []
        
        # Buy if at or below fair
        if depth.sell_orders and pos < max_pos:
            for ask_p in sorted(depth.sell_orders.keys()):
                if ask_p <= fair:
                    qty = min(-depth.sell_orders[ask_p], size, max_pos - pos)
                    if qty > 0:
                        orders.append(Order(sym, ask_p, qty))
                        pos += qty
                else:
                    break
        
        # Passive: lean bid
        bid_p = math.floor(fair - 0.5)
        bid_sz = min(size, max_pos - pos)
        if bid_sz > 0 and bid_p > 0:
            orders.append(Order(sym, bid_p, bid_sz))
        
        # Light ask to manage inventory
        if pos > max_pos * 0.5:
            ask_p = math.ceil(fair + 1.5)
            ask_sz = min(size // 2, pos)
            if ask_sz > 0:
                orders.append(Order(sym, ask_p, -ask_sz))
        
        return orders

    def _trade_intrinsic(
        self, depth: OrderDepth, fair: float, pos: int, sym: str,
    ) -> list[Order]:
        """Deep ITM: buy below intrinsic, sell above."""
        orders: list[Order] = []
        edge = 3.0
        
        if depth.sell_orders:
            for ask_p in sorted(depth.sell_orders.keys()):
                if ask_p < fair - edge:
                    qty = min(-depth.sell_orders[ask_p], VEV_POS_LIMIT - pos, 10)
                    if qty > 0:
                        orders.append(Order(sym, ask_p, qty))
                        pos += qty
        
        if depth.buy_orders:
            for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_p > fair + edge:
                    qty = min(depth.buy_orders[bid_p], VEV_POS_LIMIT + pos, 10)
                    if qty > 0:
                        orders.append(Order(sym, bid_p, -qty))
                        pos -= qty
        
        return orders

    def _collect_free_otm(
        self, depth: OrderDepth, pos: int, sym: str,
    ) -> list[Order]:
        """Deep OTM: buy at 0, sell at 1."""
        orders: list[Order] = []
        
        if depth.sell_orders:
            best_ask = min(depth.sell_orders.keys())
            if best_ask <= 0 and pos < OTM_MAX_POS:
                qty = min(-depth.sell_orders[best_ask], OTM_MAX_POS - pos, 25)
                if qty > 0:
                    orders.append(Order(sym, 0, qty))
        
        if pos > 0 and depth.buy_orders:
            best_bid = max(depth.buy_orders.keys())
            if best_bid >= 1:
                qty = min(depth.buy_orders[best_bid], pos, 25)
                if qty > 0:
                    orders.append(Order(sym, 1, -qty))
        
        return orders

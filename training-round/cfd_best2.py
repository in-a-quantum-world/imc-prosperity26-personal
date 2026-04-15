"""
IMC Prosperity 4 — Tutorial Round: LES-Inspired Trader
═══════════════════════════════════════════════════════
Treats price as an Itô process:  dP = μ dt + σ dW

Uses Large Eddy Simulation (LES) concepts from CFD to decompose
price into resolved (drift) and sub-grid (noise) components:
    P(t) = P̄(t) + P'(t)

Regime classification via Reynolds number analog:
    Re = |drift| × L / σ²_subgrid
    
    Re > threshold  →  "Laminar"  (drift-dominated, predictable)
      • FV from Taylor series extrapolation of filtered price
      • Tighter spreads (trend is reliable)
      • Asymmetric quotes leaning into drift direction
      • Lighter inventory skew (trend provides natural exit)
    
    Re ≤ threshold  →  "Turbulent" (noise-dominated, unpredictable)
      • FV = fast EMA (responsive to local conditions)
      • Wider spreads (more edge per fill for protection)
      • Symmetric quotes
      • Heavier inventory skew (mean-revert aggressively)

EMERALDS: FV = 10,000 exactly. Pure penny market making.
TOMATOES: LES regime-adaptive market making.
"""

from datamodel import OrderDepth, TradingState, Order
import json
import math


class Trader:
    LIMITS = {"EMERALDS": 80, "TOMATOES": 80}
    DAY_LENGTH = 999900

    # ── EMERALDS ─────────────────────────────────────────────────
    EM_FV = 10000
    EM_MAKE_WIDTH = 7
    EM_INV_SKEW = 0.15

    # ── TOMATOES: LES parameters ─────────────────────────────────
    # Itô process filter scales
    TOM_SLOW_ALPHA = 0.05     # LES resolved-scale filter (P̄)
    TOM_FAST_ALPHA = 0.12     # sub-grid responsive filter

    # Regime classification
    RE_THRESHOLD = 0.7        # Reynolds number boundary
    LES_WINDOW = 50           # characteristic length scale L
    DRIFT_EXTRAP = 8          # Taylor series extrapolation horizon

    # Laminar regime parameters
    LAM_MAKE_WIDTH = 4        # tighter spread (predictable flow)
    LAM_INV_SKEW = 0.12       # lighter skew (drift provides exit)
    LAM_DRIFT_LEAN = 0.3      # asymmetric lean into drift

    # Turbulent regime parameters
    TURB_MAKE_WIDTH = 5       # wider spread (noise protection)
    TURB_INV_SKEW = 0.28      # heavier skew (mean-revert hard)

    # ─────────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        result = {}
        data = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}

        for product in ["EMERALDS", "TOMATOES"]:
            if product not in state.order_depths:
                continue
            od = state.order_depths[product]
            pos = state.position.get(product, 0)
            limit = self.LIMITS[product]
            ts = state.timestamp

            if product == "EMERALDS":
                orders, data = self._trade_emeralds(od, pos, limit, data, ts)
            else:
                orders, data = self._trade_tomatoes(od, pos, limit, data, ts)
            result[product] = orders

        return result, 0, json.dumps(data)

    # ═════════════════════════════════════════════════════════════
    #  EMERALDS — FV = 10,000 penny MM (identical to v2)
    # ═════════════════════════════════════════════════════════════
    def _trade_emeralds(self, od, pos, limit, data, ts):
        orders = []
        fv = self.EM_FV
        pos_start = pos

        # TAKE
        pos, take_buy = self._sweep_sells(orders, od, "EMERALDS", pos, limit, fv)
        pos, take_sell = self._sweep_buys(orders, od, "EMERALDS", pos, limit, fv)

        # MAKE
        eod = self._eod_mult(ts)
        skew = -pos * self.EM_INV_SKEW * eod
        adj_fv = fv + skew

        bid_px = int(math.floor(adj_fv)) - self.EM_MAKE_WIDTH
        ask_px = int(math.ceil(adj_fv)) + self.EM_MAKE_WIDTH
        bid_px, ask_px = self._penny(od, bid_px, ask_px)
        bid_px = min(bid_px, fv - 1)
        ask_px = max(ask_px, fv + 1)

        make_buy = max(0, (limit - pos_start) - take_buy)
        make_sell = max(0, (limit + pos_start) - take_sell)
        if make_buy > 0:
            orders.append(Order("EMERALDS", bid_px, make_buy))
        if make_sell > 0:
            orders.append(Order("EMERALDS", ask_px, -make_sell))

        return orders, data

    # ═════════════════════════════════════════════════════════════
    #  TOMATOES — LES regime-adaptive market making
    # ═════════════════════════════════════════════════════════════
    def _trade_tomatoes(self, od, pos, limit, data, ts):
        orders = []
        if not od.buy_orders or not od.sell_orders:
            return orders, data

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid = (best_bid + best_ask) / 2
        pos_start = pos

        # ── Update Itô process filters ──────────────────────────
        # Resolved scale P̄ (slow LES filter)
        prev_slow = data.get("les_slow", mid)
        p_bar = self.TOM_SLOW_ALPHA * mid + (1 - self.TOM_SLOW_ALPHA) * prev_slow
        data["les_slow"] = p_bar

        # Fast EMA for responsive FV
        prev_fast = data.get("les_fast", mid)
        ema_fast = self.TOM_FAST_ALPHA * mid + (1 - self.TOM_FAST_ALPHA) * prev_fast
        data["les_fast"] = ema_fast

        # ── LES decomposition: P' = P - P̄ ──────────────────────
        # Track sub-grid variance (rolling)
        p_prime = mid - p_bar
        sg_history = data.get("les_sg", [])
        sg_history.append(p_prime)
        if len(sg_history) > self.LES_WINDOW:
            sg_history = sg_history[-self.LES_WINDOW:]
        data["les_sg"] = sg_history

        # Track drift history for acceleration
        drift_history = data.get("les_drift", [])
        prev_p_bar = data.get("les_prev_pbar", p_bar)
        instant_drift = p_bar - prev_p_bar
        drift_history.append(instant_drift)
        if len(drift_history) > self.LES_WINDOW:
            drift_history = drift_history[-self.LES_WINDOW:]
        data["les_drift"] = drift_history
        data["les_prev_pbar"] = p_bar

        # ── Regime classification ────────────────────────────────
        if len(sg_history) >= 10 and len(drift_history) >= 10:
            # Drift rate (resolved-scale velocity)
            drift = sum(drift_history[-self.LES_WINDOW:]) / len(drift_history[-self.LES_WINDOW:])

            # Sub-grid turbulent intensity
            sg_mean = sum(sg_history) / len(sg_history)
            sg_var = sum((x - sg_mean)**2 for x in sg_history) / len(sg_history)
            sigma_sg = math.sqrt(sg_var) if sg_var > 0 else 0.01

            # Reynolds number analog: Re = |μ| × L / ν
            nu = sigma_sg**2 if sigma_sg > 0 else 0.01
            Re = abs(drift) * self.LES_WINDOW / nu

            is_laminar = Re > self.RE_THRESHOLD
        else:
            # Not enough data yet — default to turbulent (conservative)
            is_laminar = False
            drift = 0.0
            Re = 0.0

        # ── Compute fair value based on regime ───────────────────
        if is_laminar:
            # LAMINAR: Taylor series extrapolation of drift
            # P(t+Δ) ≈ P̄(t) + μΔ
            # Use drift acceleration if we have enough history
            accel = 0.0
            if len(drift_history) >= 20:
                recent = drift_history[-10:]
                earlier = drift_history[-20:-10]
                accel = (sum(recent)/len(recent) - sum(earlier)/len(earlier)) / 10

            fv = p_bar + drift * self.DRIFT_EXTRAP + 0.5 * accel * self.DRIFT_EXTRAP**2

            make_width = self.LAM_MAKE_WIDTH
            inv_skew = self.LAM_INV_SKEW

            # Asymmetric lean: shift quotes in drift direction
            # If drifting UP: make ask tighter (get filled on the winning side)
            drift_shift = drift * self.LAM_DRIFT_LEAN * self.LES_WINDOW
        else:
            # TURBULENT: mean-revert to fast EMA
            fv = ema_fast
            make_width = self.TURB_MAKE_WIDTH
            inv_skew = self.TURB_INV_SKEW
            drift_shift = 0.0

        # ── TAKE: sweep mispriced orders ─────────────────────────
        pos, take_buy = self._sweep_sells(orders, od, "TOMATOES", pos, limit, fv - 1)
        pos, take_sell = self._sweep_buys(orders, od, "TOMATOES", pos, limit, fv + 1)

        # ── MAKE: passive quotes ─────────────────────────────────
        eod = self._eod_mult(ts)
        skew = -pos * inv_skew * eod
        adj_fv = fv + skew + drift_shift

        bid_px = int(math.floor(adj_fv - make_width))
        ask_px = int(math.ceil(adj_fv + make_width))

        bid_px, ask_px = self._penny(od, bid_px, ask_px)

        # Clamp: don't cross adj_fv
        adj_floor = int(math.floor(adj_fv))
        adj_ceil = int(math.ceil(adj_fv))
        bid_px = min(bid_px, adj_floor - 1)
        ask_px = max(ask_px, adj_ceil + 1)

        if bid_px >= ask_px:
            bid_px = adj_floor - 1
            ask_px = adj_ceil + 1

        make_buy = max(0, (limit - pos_start) - take_buy)
        make_sell = max(0, (limit + pos_start) - take_sell)

        if make_buy > 0:
            orders.append(Order("TOMATOES", bid_px, make_buy))
        if make_sell > 0:
            orders.append(Order("TOMATOES", ask_px, -make_sell))

        return orders, data

    # ═════════════════════════════════════════════════════════════
    #  Helpers
    # ═════════════════════════════════════════════════════════════
    @staticmethod
    def _sweep_sells(orders, od, symbol, pos, limit, threshold):
        total = 0
        if not od.sell_orders:
            return pos, total
        for price in sorted(od.sell_orders):
            if price > threshold:
                break
            room = limit - pos
            if room <= 0:
                break
            vol = -od.sell_orders[price]
            qty = min(vol, room)
            if qty > 0:
                orders.append(Order(symbol, price, qty))
                pos += qty
                total += qty
        return pos, total

    @staticmethod
    def _sweep_buys(orders, od, symbol, pos, limit, threshold):
        total = 0
        if not od.buy_orders:
            return pos, total
        for price in sorted(od.buy_orders, reverse=True):
            if price < threshold:
                break
            room = limit + pos
            if room <= 0:
                break
            vol = od.buy_orders[price]
            qty = min(vol, room)
            if qty > 0:
                orders.append(Order(symbol, price, -qty))
                pos -= qty
                total += qty
        return pos, total

    @staticmethod
    def _penny(od, bid_px, ask_px):
        if od.buy_orders:
            npc_best_bid = max(od.buy_orders)
            bid_px = max(bid_px, npc_best_bid + 1)
        if od.sell_orders:
            npc_best_ask = min(od.sell_orders)
            ask_px = min(ask_px, npc_best_ask - 1)
        return bid_px, ask_px

    def _eod_mult(self, ts):
        progress = ts / self.DAY_LENGTH
        if progress > 0.90:
            return 1.0 + 2.0 * (progress - 0.90) / 0.10
        return 1.0
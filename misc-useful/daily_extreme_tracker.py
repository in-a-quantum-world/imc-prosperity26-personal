"""
Daily Extreme Insider Tracker
──────────────────────────────
Based on the Frankfurt Hedgehogs' P3 approach:

  "Track the daily running minimum and maximum.  When a trade occurred at
   a daily extreme — and in the expected direction relative to the mid price
   — we flagged it as a signal and positioned accordingly.  False positives
   were managed by monitoring for corresponding new extrema that contradicted
   earlier signals."

Algorithm (per product, per tick)
───────────────────────────────────
1. Maintain running daily min/max of ALL trade prices seen so far today.
2. Each tick, scan market_trades for each product:
     • If any trade price <= current daily min  →  new daily min.
       If that new min is BELOW the current mid price  →  LONG signal.
       (Someone just bought at the cheapest price of the day, below fair value.)
     • If any trade price >= current daily max  →  new daily max.
       If that new max is ABOVE the current mid price  →  SHORT signal.
       (Someone just sold at the most expensive price of the day, above fair value.)
3. False-positive cancellation:
     • If we hold a LONG signal and a new daily MIN is set at a price
       LOWER than the one that triggered the signal  →  the original low
       has been broken; cancel the signal and flatten position.
     • Mirror logic for SHORT.

HOW TO USE
──────────
    from daily_extreme_tracker import DailyExtremeTracker, LONG, SHORT, NEUTRAL

    class Trader:
        def run(self, state):
            tracker = DailyExtremeTracker.from_trader_data(state.traderData)
            tracker.update(state)
            orders = {}
            for product in state.order_depths:
                signal, confidence = tracker.get_signal(product)
                if signal != NEUTRAL:
                    orders[product] = tracker.follow_orders(
                        product, state, pos_limit=20
                    )
            return orders, 0, tracker.to_trader_data()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

# ── Direction constants ────────────────────────────────────────────────────────
LONG    =  1
NEUTRAL =  0
SHORT   = -1

# ── Config ─────────────────────────────────────────────────────────────────────

# A trade is "at the daily extreme" if within this many price units of it.
# Set 0.0 for exact match; 0.5 allows for half-tick rounding in CSV data.
PRICE_TOLERANCE: float = 0.5

# How many ticks the signal stays active after being emitted.
# If no contradicting extreme arrives, signal expires after this many ticks.
# Set to a large number (e.g. 999_999) to hold all day.
SIGNAL_TTL_TICKS: int = 999_999

# Minimum gap between the trade price and the mid price for the signal to fire.
# Guards against trades inside the spread being counted as meaningful extrema.
# 0.0 = fire on any trade at extreme; 0.5 = must be at least 0.5 below mid (LONG)
MIN_DISTANCE_FROM_MID: float = 0.0


# ── Per-product state ──────────────────────────────────────────────────────────

@dataclass
class ProductExtremeState:
    product: str

    # Daily running extrema (reset when day changes)
    current_day: Optional[int] = None
    daily_min: float = float('inf')
    daily_max: float = float('-inf')

    # Active signal
    signal_direction: int = NEUTRAL
    signal_price: float = 0.0       # the extreme price that triggered the signal
    signal_ts: int = 0
    signal_ticks_alive: int = 0

    # Activity log for this tick (cleared each tick)
    log: list = field(default_factory=list)

    # ── Day reset ─────────────────────────────────────────────────────────────

    def _maybe_reset_day(self, day: int) -> None:
        """Reset running extrema at the start of a new day."""
        if day != self.current_day:
            self.current_day = day
            self.daily_min   = float('inf')
            self.daily_max   = float('-inf')
            # Do NOT reset signal — it may have been set at end of prior day
            self.log.append(f"  [Day reset → day {day}]")

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self, day: int, ts: int, mid: float, trades: list) -> None:
        """
        Feed one tick.  `trades` is the list of Trade objects for this product
        at this timestamp (from state.market_trades + state.own_trades).
        """
        self.log = []
        self._maybe_reset_day(day)

        # Age the signal
        if self.signal_direction != NEUTRAL:
            self.signal_ticks_alive += 1
            if self.signal_ticks_alive > SIGNAL_TTL_TICKS:
                self.log.append(f"  Signal expired after {SIGNAL_TTL_TICKS} ticks")
                self.signal_direction = NEUTRAL

        for trade in trades:
            price = trade.price

            # ── Check for new daily MIN ───────────────────────────────────────
            if price <= self.daily_min + PRICE_TOLERANCE:
                new_min = min(self.daily_min, price)

                # Is this genuinely below current fair value?
                below_mid = mid - price >= MIN_DISTANCE_FROM_MID

                if price < self.daily_min - PRICE_TOLERANCE:
                    # Strictly new low
                    self.daily_min = new_min

                    if self.signal_direction == LONG and price < self.signal_price - PRICE_TOLERANCE:
                        # New low BELOW our LONG trigger → false positive, cancel
                        self.log.append(
                            f"  [ts={ts}] LONG signal CANCELLED — new low {price:.1f} "
                            f"< trigger {self.signal_price:.1f}"
                        )
                        self.signal_direction = NEUTRAL
                    elif below_mid:
                        # New genuine low below mid → LONG signal
                        self.signal_direction   = LONG
                        self.signal_price       = price
                        self.signal_ts          = ts
                        self.signal_ticks_alive = 0
                        self.log.append(
                            f"  [ts={ts}] LONG signal @ new daily min {price:.1f} "
                            f"(mid={mid:.1f}, gap={mid-price:.1f})"
                        )
                elif price <= self.daily_min + PRICE_TOLERANCE:
                    # At (not below) daily min — still a confirming trade
                    self.daily_min = new_min
                    if below_mid and self.signal_direction != LONG:
                        self.signal_direction   = LONG
                        self.signal_price       = price
                        self.signal_ts          = ts
                        self.signal_ticks_alive = 0
                        self.log.append(
                            f"  [ts={ts}] LONG signal @ daily min {price:.1f} "
                            f"(mid={mid:.1f}, gap={mid-price:.1f})"
                        )

            # ── Check for new daily MAX ───────────────────────────────────────
            if price >= self.daily_max - PRICE_TOLERANCE:
                new_max = max(self.daily_max, price)

                above_mid = price - mid >= MIN_DISTANCE_FROM_MID

                if price > self.daily_max + PRICE_TOLERANCE:
                    # Strictly new high
                    self.daily_max = new_max

                    if self.signal_direction == SHORT and price > self.signal_price + PRICE_TOLERANCE:
                        # New high ABOVE our SHORT trigger → false positive, cancel
                        self.log.append(
                            f"  [ts={ts}] SHORT signal CANCELLED — new high {price:.1f} "
                            f"> trigger {self.signal_price:.1f}"
                        )
                        self.signal_direction = NEUTRAL
                    elif above_mid:
                        self.signal_direction   = SHORT
                        self.signal_price       = price
                        self.signal_ts          = ts
                        self.signal_ticks_alive = 0
                        self.log.append(
                            f"  [ts={ts}] SHORT signal @ new daily max {price:.1f} "
                            f"(mid={mid:.1f}, gap={price-mid:.1f})"
                        )
                elif price >= self.daily_max - PRICE_TOLERANCE:
                    self.daily_max = new_max
                    if above_mid and self.signal_direction != SHORT:
                        self.signal_direction   = SHORT
                        self.signal_price       = price
                        self.signal_ts          = ts
                        self.signal_ticks_alive = 0
                        self.log.append(
                            f"  [ts={ts}] SHORT signal @ daily max {price:.1f} "
                            f"(mid={mid:.1f}, gap={price-mid:.1f})"
                        )

    # ── Signal query ──────────────────────────────────────────────────────────

    def get_signal(self) -> tuple[int, float]:
        """
        Returns (direction, confidence).
        Confidence decays linearly from 1.0 to 0.5 over SIGNAL_TTL_TICKS.
        With TTL = 999_999 (hold all day) confidence is effectively always 1.0.
        """
        if self.signal_direction == NEUTRAL:
            return NEUTRAL, 0.0
        ttl_frac = 1.0 - self.signal_ticks_alive / max(SIGNAL_TTL_TICKS, 1)
        conf = 0.5 + 0.5 * ttl_frac
        return self.signal_direction, max(conf, 0.5)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "product":            self.product,
            "current_day":        self.current_day,
            "daily_min":          self.daily_min,
            "daily_max":          self.daily_max,
            "signal_direction":   self.signal_direction,
            "signal_price":       self.signal_price,
            "signal_ts":          self.signal_ts,
            "signal_ticks_alive": self.signal_ticks_alive,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProductExtremeState":
        obj = cls(product=d["product"])
        obj.current_day        = d.get("current_day")
        obj.daily_min          = d.get("daily_min",  float('inf'))
        obj.daily_max          = d.get("daily_max",  float('-inf'))
        obj.signal_direction   = d.get("signal_direction",   NEUTRAL)
        obj.signal_price       = d.get("signal_price",       0.0)
        obj.signal_ts          = d.get("signal_ts",          0)
        obj.signal_ticks_alive = d.get("signal_ticks_alive", 0)
        return obj


# ── Main tracker ───────────────────────────────────────────────────────────────

class DailyExtremeTracker:
    """
    Tracks per-product signals based on trades at daily running extrema.
    Compatible with InsiderTracker / AnonymousInsiderTracker interface.
    """

    def __init__(self, states: dict[str, ProductExtremeState] | None = None):
        self._states: dict[str, ProductExtremeState] = states or {}
        self._current_ts: int = 0
        self._current_day: int = 0

    # ── Persistence ───────────────────────────────────────────────────────────

    @classmethod
    def from_trader_data(cls, trader_data_str: str) -> "DailyExtremeTracker":
        try:
            if trader_data_str:
                blob = json.loads(trader_data_str)
                raw  = blob.get("daily_extreme_tracker", {})
                states = {prod: ProductExtremeState.from_dict(val)
                          for prod, val in raw.items()}
                return cls(states)
        except Exception:
            pass
        return cls()

    def to_trader_data(self, existing: str = "") -> str:
        try:
            blob = json.loads(existing) if existing else {}
        except Exception:
            blob = {}
        blob["daily_extreme_tracker"] = {
            prod: s.to_dict() for prod, s in self._states.items()
        }
        return json.dumps(blob)

    # ── Core update ───────────────────────────────────────────────────────────

    def update(self, state, day: int = 0) -> None:
        """
        Call once per tick.  Pass `day` explicitly (Prosperity day number:
        -2, -1, 0, 1, …) since TradingState doesn't expose it directly.
        If you don't know the day, derive it from traderData or pass 0.
        """
        self._current_ts  = state.timestamp
        self._current_day = day

        for product, order_depth in state.order_depths.items():
            # Compute mid from best bid / best ask
            bids = order_depth.buy_orders
            asks = order_depth.sell_orders
            if not bids or not asks:
                continue
            mid = (max(bids.keys()) + min(asks.keys())) / 2.0

            trades = (
                list(state.market_trades.get(product, []))
                + list(state.own_trades.get(product, []))
            )

            if product not in self._states:
                self._states[product] = ProductExtremeState(product=product)

            self._states[product].update(day, state.timestamp, mid, trades)

    # ── Signal queries ─────────────────────────────────────────────────────────

    def get_signal(self, product: str) -> tuple[int, float]:
        if product not in self._states:
            return NEUTRAL, 0.0
        return self._states[product].get_signal()

    def get_all_active_signals(self) -> dict[str, tuple[int, float]]:
        result = {}
        for prod, s in self._states.items():
            d, c = s.get_signal()
            if d != NEUTRAL:
                result[prod] = (d, c)
        return result

    # ── Order generation ───────────────────────────────────────────────────────

    def follow_orders(
        self,
        product: str,
        state,
        pos_limit: int,
        aggression: float = 1.0,
    ) -> list:
        """Generate aggressive crossing orders to follow the signal."""
        from datamodel import Order

        direction, confidence = self.get_signal(product)
        if direction == NEUTRAL:
            return []

        order_depth = state.order_depths.get(product)
        if not order_depth:
            return []

        current_pos = state.position.get(product, 0)
        target_pos  = int(direction * pos_limit * aggression)
        remaining   = target_pos - current_pos
        orders      = []

        if remaining > 0:
            for ask_price in sorted(order_depth.sell_orders):
                if remaining <= 0:
                    break
                fill = min(remaining, abs(order_depth.sell_orders[ask_price]))
                orders.append(Order(product, ask_price, fill))
                remaining -= fill
        elif remaining < 0:
            for bid_price in sorted(order_depth.buy_orders, reverse=True):
                if remaining >= 0:
                    break
                fill = min(abs(remaining), order_depth.buy_orders[bid_price])
                orders.append(Order(product, bid_price, -fill))
                remaining += fill

        return orders

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def log_summary(self) -> str:
        lines = [f"[DailyExtremeTracker ts={self._current_ts} day={self._current_day}]"]
        for prod, s in self._states.items():
            for entry in s.log:
                lines.append(f"  {prod}: {entry}")
            d, c = s.get_signal()
            if d != NEUTRAL:
                lines.append(
                    f"  {prod}: active {'LONG' if d==LONG else 'SHORT'} "
                    f"signal (conf={c:.2f}) triggered @ {s.signal_price:.1f}, "
                    f"daily range [{s.daily_min:.1f}, {s.daily_max:.1f}]"
                )
        if len(lines) == 1:
            lines.append("  No active signals.")
        return "\n".join(lines)

    def dump_state(self) -> dict:
        return {
            prod: {
                "signal":      s.get_signal(),
                "daily_min":   s.daily_min,
                "daily_max":   s.daily_max,
                "signal_price":s.signal_price,
                "ticks_alive": s.signal_ticks_alive,
            }
            for prod, s in self._states.items()
        }

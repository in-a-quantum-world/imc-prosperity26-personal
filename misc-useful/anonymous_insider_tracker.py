"""
Anonymous Insider Tracker
──────────────────────────
Detects an insider bot WITHOUT knowing their name, purely from order-book
behaviour.

Core insight (observed in P3)
──────────────────────────────
An insider who knows future price direction places *passive* (maker) orders
at price levels that are genuinely local extrema:

  • Large BID at the rolling local MINIMUM  →  they know the price won't go
    lower, so they harvest cheap inventory there.  Signal for us: GO LONG.

  • Large ASK at the rolling local MAXIMUM  →  they know the price won't go
    higher, so they dump inventory there.  Signal for us: GO SHORT.

A random market-maker's volume is spread across many levels; the insider
concentrates unusual size exactly at the extremum.  We detect that anomaly.

Detection algorithm (per product, per tick)
────────────────────────────────────────────
1. Maintain a rolling window of mid-prices (ROLLING_WINDOW ticks).
2. Find the rolling min and rolling max of that window.
3. For the current best bid:
     – Is it within PRICE_TOLERANCE of the rolling min?
     – Is its volume > AVG_VOLUME_MULTIPLIER × mean book volume on bid side?
   → If both: emit LONG signal.
4. For the current best ask:
     – Is it within PRICE_TOLERANCE of the rolling max?
     – Is its volume > AVG_VOLUME_MULTIPLIER × mean book volume on ask side?
   → If both: emit SHORT signal.
5. Track historical signal accuracy (did subsequent price moves confirm?).
   Confidence = min(raw_confidence, hit_rate) so a low-accuracy signal is
   down-weighted automatically.

HOW TO USE
──────────
1.  Instantiate AnonymousInsiderTracker (one shared instance across all
    products is fine — it keeps per-product state internally).

2.  Each tick inside Trader.run():
      a.  Restore from traderData:
            tracker = AnonymousInsiderTracker.from_trader_data(state.traderData)
      b.  Feed the current state:
            tracker.update(state)
      c.  Query:
            signal, confidence = tracker.get_signal("KELP")
      d.  Optionally generate orders:
            orders = tracker.follow_orders("KELP", state, pos_limit=20)
      e.  Persist:
            trader_data = tracker.to_trader_data()

MINIMAL EXAMPLE
───────────────
    from anonymous_insider_tracker import AnonymousInsiderTracker, LONG, SHORT, NEUTRAL

    class Trader:
        def run(self, state):
            tracker = AnonymousInsiderTracker.from_trader_data(state.traderData)
            tracker.update(state)
            orders = {}
            for product in state.order_depths:
                signal, confidence = tracker.get_signal(product)
                if signal != NEUTRAL and confidence > 0.4:
                    orders[product] = tracker.follow_orders(product, state, pos_limit=20)
            return orders, 0, tracker.to_trader_data()
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ── Direction constants ────────────────────────────────────────────────────────
LONG    =  1
NEUTRAL =  0
SHORT   = -1

# ── Tunable parameters ─────────────────────────────────────────────────────────

# How many ticks of mid-price history to keep for finding the rolling extremum.
# 50 ticks x 100 ts/tick = 5 000 timestamp units (~5 seconds of sim).
ROLLING_WINDOW: int = 50

# A bid/ask qualifies as "at the extremum" if it's within this many price units.
PRICE_TOLERANCE: float = 1.0

# ABSOLUTE volume threshold — order must be at least this size to be flagged.
# Calibrated from Round 1 data: order books have 1–2 levels per tick; there is
# a clean bimodal split between "small" bots (vol 10–15) and a distinct actor
# placing larger orders (vol 17–30).  Setting 17 captures that actor for ASH;
# use 15 for INTARIAN_PEPPER_ROOT.  Override per-product via PRODUCT_VOL_OVERRIDE.
# NOTE: do NOT use the relative-multiplier approach — with only 1-2 levels in
# the book the average is too noisy and the ratio is always ~1.0.
ABSOLUTE_VOL_THRESHOLD: int = 17

# Per-product overrides: {product_name: min_volume_to_flag}
PRODUCT_VOL_OVERRIDE: dict[str, int] = {
    "INTARIAN_PEPPER_ROOT": 15,
}

# How many ticks after emitting a signal we wait to check whether it was right.
# At 100 ts/tick, 10 ticks = 1 000 ts.
CONFIRMATION_LAG: int = 10

# Minimum number of confirmed signals before the hit-rate starts affecting
# confidence (avoids dividing by tiny samples at the start).
MIN_CONFIRMATIONS: int = 5

# Signal stays "hot" for this many ticks after it was first emitted.
SIGNAL_TTL: int = 8


# ── Per-product state ──────────────────────────────────────────────────────────

@dataclass
class ProductAnonState:
    product: str

    # Rolling price history: list of (timestamp, mid_price)
    price_history: deque = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))

    # Pending confirmations: list of (emit_ts, signal_direction, emit_price)
    pending: list = field(default_factory=list)

    # Hit-rate tracking
    confirmed_correct: int = 0
    confirmed_total: int = 0

    # Most recent signal
    last_signal_direction: int = NEUTRAL
    last_signal_ts: int = 0
    last_signal_confidence: float = 0.0

    # ── Helpers ──────────────────────────────────────────────────────────────

    def rolling_min(self) -> Optional[float]:
        if not self.price_history:
            return None
        return min(p for _, p in self.price_history)

    def rolling_max(self) -> Optional[float]:
        if not self.price_history:
            return None
        return max(p for _, p in self.price_history)

    def hit_rate(self) -> float:
        """Fraction of past signals that were confirmed correct."""
        if self.confirmed_total < MIN_CONFIRMATIONS:
            return 1.0   # no penalty until we have enough data
        return self.confirmed_correct / self.confirmed_total

    # ── Main update ──────────────────────────────────────────────────────────

    def update(self, ts: int, mid_price: float, order_depth) -> tuple[int, float]:
        """
        Feed one tick of data.  Returns (signal_direction, raw_confidence).
        raw_confidence is in [0, 1]; multiply by hit_rate() for final confidence.
        """
        # 1. Update price history
        self.price_history.append((ts, mid_price))

        # 2. Resolve any pending confirmations
        if len(self.price_history) >= 2:
            self._resolve_pending(ts, mid_price)

        # 3. Need at least half a window before emitting signals
        if len(self.price_history) < ROLLING_WINDOW // 2:
            return NEUTRAL, 0.0

        roll_min = self.rolling_min()
        roll_max = self.rolling_max()
        if roll_min is None or roll_max is None:
            return NEUTRAL, 0.0
        roll_range = roll_max - roll_min
        if roll_range < 1e-6:   # flat price — nothing to detect
            return NEUTRAL, 0.0

        # 4. Analyse order book
        signal, raw_conf = self._check_book(ts, mid_price, order_depth,
                                            roll_min, roll_max, roll_range)

        if signal != NEUTRAL:
            self.last_signal_direction  = signal
            self.last_signal_ts         = ts
            self.last_signal_confidence = raw_conf
            # Queue this signal for confirmation CONFIRMATION_LAG ticks later
            self.pending.append((ts, signal, mid_price))

        return signal, raw_conf

    def _check_book(
        self,
        ts: int,
        mid: float,
        order_depth,
        roll_min: float,
        roll_max: float,
        roll_range: float,
    ) -> tuple[int, float]:
        """
        Inspect the order book for anomalous volume at extremum prices.
        Returns (direction, raw_confidence).
        """
        bids = order_depth.buy_orders   # {price: volume}  volume > 0
        asks = order_depth.sell_orders  # {price: volume}  volume < 0  (Prosperity convention)

        if not bids or not asks:
            return NEUTRAL, 0.0

        # ── Bid side: check for insider accumulation at the rolling low ──────
        bid_signal, bid_conf = self._check_side(
            levels=bids,
            extremum=roll_min,
            volumes_are_positive=True,
            direction=LONG,
        )

        # ── Ask side: check for insider distribution at the rolling high ─────
        ask_signal, ask_conf = self._check_side(
            levels=asks,
            extremum=roll_max,
            volumes_are_positive=False,
            direction=SHORT,
        )

        # ── Pick the stronger signal ─────────────────────────────────────────
        if bid_conf >= ask_conf and bid_conf > 0:
            # Normalise confidence by how close to the extremum we are.
            # Maximum confidence when mid ≈ roll_min (price IS at the low).
            proximity = 1.0 - min(abs(mid - roll_min) / roll_range, 1.0)
            return LONG, bid_conf * (0.5 + 0.5 * proximity)

        if ask_conf > bid_conf and ask_conf > 0:
            proximity = 1.0 - min(abs(mid - roll_max) / roll_range, 1.0)
            return SHORT, ask_conf * (0.5 + 0.5 * proximity)

        return NEUTRAL, 0.0

    def _check_side(
        self,
        levels: dict,
        extremum: float,
        volumes_are_positive: bool,
        direction: int,
    ) -> tuple[int, float]:
        """
        For one side of the book, check whether any level near `extremum`
        has a volume >= ABSOLUTE_VOL_THRESHOLD (or the per-product override).

        Why absolute rather than relative:
          The Round 1 book only has 1-2 levels per tick, so the "average" is
          just the single best-level volume and the ratio is always 1.0.
          An absolute cutoff cleanly separates the two bot populations observed:
            small orders (vol 10-15) = normal market makers
            large orders (vol 17+)   = the predictive actor
        """
        threshold = PRODUCT_VOL_OVERRIDE.get(self.product, ABSOLUTE_VOL_THRESHOLD)

        for price, raw_vol in levels.items():
            vol = abs(raw_vol)
            if abs(price - extremum) <= PRICE_TOLERANCE and vol >= threshold:
                # Confidence scales with how much the volume exceeds the threshold.
                # vol == threshold -> conf = 0.33; vol == threshold*2 -> conf = 0.67; capped at 1.0
                raw_conf = min(vol / (threshold * 1.5), 1.0)
                return direction, raw_conf

        return NEUTRAL, 0.0

    def _resolve_pending(self, current_ts: int, current_price: float) -> None:
        """
        Check pending signals: if CONFIRMATION_LAG ticks have elapsed,
        see whether the price moved in the predicted direction.
        """
        still_pending = []
        for (emit_ts, signal_dir, emit_price) in self.pending:
            ticks_elapsed = (current_ts - emit_ts) // 100  # 100 ts per tick
            if ticks_elapsed >= CONFIRMATION_LAG:
                price_move = current_price - emit_price
                if signal_dir == LONG and price_move > 0:
                    self.confirmed_correct += 1
                elif signal_dir == SHORT and price_move < 0:
                    self.confirmed_correct += 1
                self.confirmed_total += 1
            else:
                still_pending.append((emit_ts, signal_dir, emit_price))
        self.pending = still_pending

    def current_signal(self, current_ts: int) -> tuple[int, float]:
        """
        Return the active signal if it's still within SIGNAL_TTL ticks,
        adjusted by the historical hit rate.
        """
        if self.last_signal_direction == NEUTRAL:
            return NEUTRAL, 0.0
        ticks_since = (current_ts - self.last_signal_ts) // 100
        if ticks_since > SIGNAL_TTL:
            return NEUTRAL, 0.0
        # Decay linearly over TTL
        ttl_factor = 1.0 - ticks_since / SIGNAL_TTL
        final_conf  = self.last_signal_confidence * ttl_factor * self.hit_rate()
        if final_conf < 0.05:
            return NEUTRAL, 0.0
        return self.last_signal_direction, final_conf

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "product":               self.product,
            "price_history":         list(self.price_history),
            "pending":               self.pending,
            "confirmed_correct":     self.confirmed_correct,
            "confirmed_total":       self.confirmed_total,
            "last_signal_direction": self.last_signal_direction,
            "last_signal_ts":        self.last_signal_ts,
            "last_signal_confidence":self.last_signal_confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProductAnonState":
        obj = cls(product=d["product"])
        obj.price_history         = deque(
            [tuple(x) for x in d.get("price_history", [])],
            maxlen=ROLLING_WINDOW,
        )
        obj.pending               = [tuple(x) for x in d.get("pending", [])]
        obj.confirmed_correct     = d.get("confirmed_correct", 0)
        obj.confirmed_total       = d.get("confirmed_total", 0)
        obj.last_signal_direction = d.get("last_signal_direction", NEUTRAL)
        obj.last_signal_ts        = d.get("last_signal_ts", 0)
        obj.last_signal_confidence= d.get("last_signal_confidence", 0.0)
        return obj


# ── Main tracker ───────────────────────────────────────────────────────────────

class AnonymousInsiderTracker:
    """
    Drop-in companion to InsiderTracker — detects an unnamed insider bot
    from order-book volume anomalies at rolling price extrema.
    """

    def __init__(self, states: dict[str, ProductAnonState] | None = None):
        self._states: dict[str, ProductAnonState] = states or {}
        self._current_ts: int = 0

    # ── Persistence ───────────────────────────────────────────────────────────

    @classmethod
    def from_trader_data(cls, trader_data_str: str) -> "AnonymousInsiderTracker":
        try:
            if trader_data_str:
                blob = json.loads(trader_data_str)
                raw  = blob.get("anon_insider_tracker", {})
                states = {prod: ProductAnonState.from_dict(val)
                          for prod, val in raw.items()}
                return cls(states)
        except Exception:
            pass
        return cls()

    def to_trader_data(self, existing_trader_data_str: str = "") -> str:
        try:
            blob = json.loads(existing_trader_data_str) if existing_trader_data_str else {}
        except Exception:
            blob = {}
        blob["anon_insider_tracker"] = {
            prod: s.to_dict() for prod, s in self._states.items()
        }
        return json.dumps(blob)

    # ── Core update ───────────────────────────────────────────────────────────

    def update(self, state) -> None:
        """Call once per tick at the top of Trader.run()."""
        self._current_ts = state.timestamp

        for product, order_depth in state.order_depths.items():
            # Compute mid-price from the book
            bids = order_depth.buy_orders
            asks = order_depth.sell_orders
            if not bids or not asks:
                continue
            best_bid = max(bids.keys())
            best_ask = min(asks.keys())
            mid = (best_bid + best_ask) / 2.0

            if product not in self._states:
                self._states[product] = ProductAnonState(product=product)

            self._states[product].update(state.timestamp, mid, order_depth)

    # ── Signal queries ─────────────────────────────────────────────────────────

    def get_signal(self, product: str) -> tuple[int, float]:
        """
        Returns (direction, confidence) for a product.
        confidence is 0 if no active signal.
        """
        if product not in self._states:
            return NEUTRAL, 0.0
        return self._states[product].current_signal(self._current_ts)

    def get_all_active_signals(self) -> dict[str, tuple[int, float]]:
        return {
            prod: (d, c)
            for prod, s in self._states.items()
            if (d := s.current_signal(self._current_ts)[0]) != NEUTRAL
            for c in [s.current_signal(self._current_ts)[1]]
        }

    def hit_rate(self, product: str) -> float:
        """Returns the historical signal accuracy for a product."""
        if product not in self._states:
            return 0.0
        return self._states[product].hit_rate()

    # ── Order generation ───────────────────────────────────────────────────────

    def follow_orders(
        self,
        product: str,
        state,
        pos_limit: int,
        min_confidence: float = 0.3,
    ) -> list:
        """
        Generate aggressive crossing orders to follow the anonymous insider.
        Only trades when confidence >= min_confidence.
        """
        from datamodel import Order

        direction, confidence = self.get_signal(product)
        if direction == NEUTRAL or confidence < min_confidence:
            return []

        order_depth = state.order_depths.get(product)
        if order_depth is None:
            return []

        current_position = state.position.get(product, 0)
        target_position  = int(direction * pos_limit * confidence)
        remaining        = target_position - current_position
        orders           = []

        if remaining > 0:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if remaining <= 0:
                    break
                fill = min(remaining, abs(order_depth.sell_orders[ask_price]))
                orders.append(Order(product, ask_price, fill))
                remaining -= fill

        elif remaining < 0:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if remaining >= 0:
                    break
                fill = min(abs(remaining), order_depth.buy_orders[bid_price])
                orders.append(Order(product, bid_price, -fill))
                remaining += fill

        return orders

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def log_summary(self) -> str:
        lines = [f"[AnonInsiderTracker ts={self._current_ts}]"]
        for prod, s in self._states.items():
            d, c = s.current_signal(self._current_ts)
            if d != NEUTRAL:
                dir_name = "LONG" if d == LONG else "SHORT"
                hr = s.hit_rate()
                lines.append(
                    f"  {prod}: {dir_name} conf={c:.2f} "
                    f"hit_rate={hr:.0%} ({s.confirmed_correct}/{s.confirmed_total})"
                )
        if len(lines) == 1:
            lines.append("  No active signals.")
        return "\n".join(lines)

    def dump_state(self) -> dict:
        return {
            prod: {
                "signal":     s.current_signal(self._current_ts),
                "hit_rate":   s.hit_rate(),
                "roll_min":   s.rolling_min(),
                "roll_max":   s.rolling_max(),
                "history_len":len(s.price_history),
                "pending":    len(s.pending),
            }
            for prod, s in self._states.items()
        }


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Simulates an insider who places a suspiciously large bid right at the
    rolling local minimum.  Expects the tracker to emit a LONG signal.
    Run: python anonymous_insider_tracker.py
    """
    import pprint

    class MockOrderDepth:
        def __init__(self, bids: dict, asks: dict):
            self.buy_orders  = bids   # {price: volume}  positive
            self.sell_orders = asks   # {price: volume}  negative (Prosperity conv)

    class MockState:
        def __init__(self, ts, depths):
            self.timestamp    = ts
            self.order_depths = depths
            self.position     = {}

    print("=== AnonymousInsiderTracker Sanity Test ===\n")

    tracker = AnonymousInsiderTracker()

    # Fill rolling window with prices oscillating 998–1002
    base_prices = [1000 + (i % 5) - 2 for i in range(ROLLING_WINDOW)]
    for i, bp in enumerate(base_prices):
        ts = 100 * (i + 1)
        depth = MockOrderDepth(
            bids={bp - 1: 5, bp - 2: 3},     # normal volume
            asks={bp + 1: -5, bp + 2: -3},
        )
        state = MockState(ts=ts, depths={"KELP": depth})
        tracker.update(state)

    roll_min = tracker._states["KELP"].rolling_min()
    print(f"Rolling min after warm-up: {roll_min}")
    print(f"Signal (should be NEUTRAL): {tracker.get_signal('KELP')}\n")

    # Now inject a tick with a VERY large bid at the rolling min
    insider_ts = 100 * (ROLLING_WINDOW + 1)
    insider_depth = MockOrderDepth(
        bids={
            roll_min: 50,          # anomalously large order at the rolling low
            roll_min - 1: 3,
            roll_min - 2: 2,
        },
        asks={roll_min + 2: -5, roll_min + 3: -3},
    )
    insider_state = MockState(ts=insider_ts, depths={"KELP": insider_depth})
    tracker.update(insider_state)

    signal, conf = tracker.get_signal("KELP")
    dir_name = "LONG" if signal == LONG else "SHORT" if signal == SHORT else "NEUTRAL"
    print(f"After large bid at rolling min ({roll_min}):")
    print(f"  Signal    : {dir_name}")
    print(f"  Confidence: {conf:.2f}  (should be > 0)")
    print()
    print(tracker.log_summary())
    print()
    print("=== State dump ===")
    pprint.pprint(tracker.dump_state())

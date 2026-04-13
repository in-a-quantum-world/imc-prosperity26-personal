"""
HOW TO USE
──────────
1.  Import InsiderTracker at the top of your trader.py.
2.  On each call to Trader.run():
      a.  Restore tracker state from traderData.
      b.  Call tracker.update(state) to process the current tick.
      c.  Call tracker.get_signal(product) for a direction + confidence.
      d.  Call tracker.follow_orders(product, state, pos_limit) to get orders.
      e.  Save tracker state back into traderData before returning.

MINIMAL EXAMPLE
───────────────
    from insider_tracker import InsiderTracker, LONG, SHORT, NEUTRAL

    class Trader:
        def run(self, state: TradingState):
            tracker = InsiderTracker.from_trader_data(state.traderData)
            tracker.update(state)

            orders: dict[str, list[Order]] = {}
            for product in state.order_depths:
                signal, confidence = tracker.get_signal(product)
                if signal != NEUTRAL:
                    new_orders = tracker.follow_orders(
                        product, state, pos_limit=20, aggression=confidence
                    )
                    if new_orders:
                        orders[product] = new_orders

            trader_data = tracker.to_trader_data()
            return orders, 0, trader_data
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

# ── Direction constants ────────────────────────────────────────────────────────
LONG    =  1
NEUTRAL =  0
SHORT   = -1

# ── Insider configuration ──────────────────────────────────────────────────────
# Update this list before each round based on known/suspected bots.
# In P3: 'Olivia' traded KELP and SQUID_INK.
# In P2 / older rounds other names spotted: 'Vladimir', 'Caesar', 'Charlie', 'Ruby'.
# Set FOLLOW_WINDOW to how many timestamps after their last trade you trust the signal.

KNOWN_INSIDERS: list[str] = [
    "Olivia",       # P3 confirmed insider — KELP, SQUID_INK
    "Vladimir",     # spotted in earlier rounds
    "Caesar",       # spotted in earlier rounds
    "Charlie",      # spotted in earlier rounds
]

# How long (in timestamp units) the insider signal stays "hot".
# Prosperity timestamps increment by 100 each tick; 500 = 5 ticks.
FOLLOW_WINDOW: int = 500

# Confidence decay: each tick after the signal the confidence drops by this amount.
# At 1.0 immediately after the trade, drops to 0.0 at FOLLOW_WINDOW ticks later.
CONFIDENCE_DECAY: float = 1.0 / (FOLLOW_WINDOW / 100)


# ── Per-product insider state ──────────────────────────────────────────────────

@dataclass
class ProductInsiderState:
    """Tracks the last observed buy and sell timestamp for one insider on one product."""
    insider_id: str
    product: str
    last_bought_ts: Optional[int] = None   # most recent ts where insider was buyer
    last_sold_ts: Optional[int] = None     # most recent ts where insider was seller

    def update(self, trades: list) -> bool:
        """
        Scan a list of Trade objects, update timestamps.
        Returns True if new insider activity was detected this tick.
        """
        found = False
        for trade in trades:
            if trade.buyer == self.insider_id:
                if self.last_bought_ts is None or trade.timestamp >= self.last_bought_ts:
                    self.last_bought_ts = trade.timestamp
                    found = True
            if trade.seller == self.insider_id:
                if self.last_sold_ts is None or trade.timestamp >= self.last_sold_ts:
                    self.last_sold_ts = trade.timestamp
                    found = True
        return found

    def direction(self) -> int:
        """
        Return LONG / SHORT / NEUTRAL based on which action happened more recently.
        Logic mirrors Frankfurt Hedgehogs' check_for_informed():
          - only bought  → LONG
          - only sold    → SHORT
          - both:
              bought later → LONG  (accumulating)
              sold   later → SHORT (distributing)
              same ts      → NEUTRAL (noise)
          - neither      → NEUTRAL
        """
        b, s = self.last_bought_ts, self.last_sold_ts
        if b is None and s is None:
            return NEUTRAL
        if b is None:
            return SHORT
        if s is None:
            return LONG
        if b > s:
            return LONG
        if s > b:
            return SHORT
        return NEUTRAL

    def confidence(self, current_ts: int) -> float:
        """
        Returns a float [0.0, 1.0] representing how fresh the signal is.
        1.0 = insider traded this tick. 0.0 = signal expired.
        Decays linearly to 0 over FOLLOW_WINDOW timestamps.
        """
        direction = self.direction()
        if direction == NEUTRAL:
            return 0.0

        ref_ts = max(
            t for t in [self.last_bought_ts, self.last_sold_ts] if t is not None
        )
        age = current_ts - ref_ts
        if age > FOLLOW_WINDOW:
            return 0.0
        return max(0.0, 1.0 - (age / FOLLOW_WINDOW))

    def to_dict(self) -> dict:
        return {
            "insider_id":    self.insider_id,
            "product":       self.product,
            "last_bought_ts": self.last_bought_ts,
            "last_sold_ts":  self.last_sold_ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProductInsiderState":
        return cls(
            insider_id=d["insider_id"],
            product=d["product"],
            last_bought_ts=d.get("last_bought_ts"),
            last_sold_ts=d.get("last_sold_ts"),
        )


# ── Main tracker ───────────────────────────────────────────────────────────────

class InsiderTracker:
    """
    Tracks all configured insider bots across all products.

    State key: (product, insider_id) → ProductInsiderState
    Persisted as JSON in traderData under key "insider_tracker".
    """

    def __init__(self, states: dict[tuple[str, str], ProductInsiderState] | None = None):
        # {(product, insider_id): ProductInsiderState}
        self._states: dict[tuple[str, str], ProductInsiderState] = states or {}
        self._current_ts: int = 0
        self._activity_log: list[str] = []  # human-readable log of this tick's events

    # ── Persistence ───────────────────────────────────────────────────────────

    @classmethod
    def from_trader_data(cls, trader_data_str: str) -> "InsiderTracker":
        """Deserialise from the traderData JSON string."""
        try:
            if trader_data_str:
                blob = json.loads(trader_data_str)
                raw = blob.get("insider_tracker", {})
                states = {}
                for key_str, val in raw.items():
                    product, insider_id = key_str.split("|", 1)
                    s = ProductInsiderState.from_dict(val)
                    states[(product, insider_id)] = s
                return cls(states)
        except Exception:
            pass
        return cls()

    def to_trader_data(self, existing_trader_data_str: str = "") -> str:
        """
        Serialise tracker state, merging into an existing traderData JSON blob.
        Call this at the end of Trader.run() and return the result as trader_data.
        """
        try:
            blob = json.loads(existing_trader_data_str) if existing_trader_data_str else {}
        except Exception:
            blob = {}

        raw = {}
        for (product, insider_id), s in self._states.items():
            raw[f"{product}|{insider_id}"] = s.to_dict()

        blob["insider_tracker"] = raw
        return json.dumps(blob)

    # ── Core update ───────────────────────────────────────────────────────────

    def update(self, state) -> None:
        """
        Call once per tick at the top of Trader.run().
        Scans market_trades and own_trades for all products and all known insiders.
        Updates internal state and logs any new activity.
        """
        self._current_ts = state.timestamp
        self._activity_log = []

        all_products = set(state.market_trades.keys()) | set(state.own_trades.keys())

        for product in all_products:
            trades = (
                state.market_trades.get(product, [])
                + state.own_trades.get(product, [])
            )
            if not trades:
                continue

            for insider_id in KNOWN_INSIDERS:
                key = (product, insider_id)
                if key not in self._states:
                    self._states[key] = ProductInsiderState(
                        insider_id=insider_id, product=product
                    )

                s = self._states[key]
                found = s.update(trades)

                if found:
                    dir_name = {LONG: "LONG", SHORT: "SHORT", NEUTRAL: "NEUTRAL"}[s.direction()]
                    conf = s.confidence(self._current_ts)
                    self._activity_log.append(
                        f"[ts={state.timestamp}] {insider_id} on {product}: "
                        f"{dir_name} (conf={conf:.2f}), "
                        f"last_buy={s.last_bought_ts}, last_sell={s.last_sold_ts}"
                    )

    # ── Signal queries ────────────────────────────────────────────────────────

    def get_signal(self, product: str) -> tuple[int, float]:
        """
        Returns (direction, confidence) for a product by aggregating all insiders.
        Direction is the consensus among active (non-expired) insider signals.
        Confidence is the max confidence among all insiders on this product.

        direction  ∈ {LONG=1, NEUTRAL=0, SHORT=-1}
        confidence ∈ [0.0, 1.0]
        """
        long_conf  = 0.0
        short_conf = 0.0

        for insider_id in KNOWN_INSIDERS:
            key = (product, insider_id)
            if key not in self._states:
                continue
            s = self._states[key]
            d = s.direction()
            c = s.confidence(self._current_ts)
            if c == 0.0:
                continue
            if d == LONG:
                long_conf  = max(long_conf, c)
            elif d == SHORT:
                short_conf = max(short_conf, c)

        if long_conf == 0.0 and short_conf == 0.0:
            return NEUTRAL, 0.0
        if long_conf >= short_conf:
            return LONG, long_conf
        return SHORT, short_conf

    def get_all_active_signals(self) -> dict[str, tuple[int, float]]:
        """Returns {product: (direction, confidence)} for all products with active signals."""
        products = {product for (product, _) in self._states}
        result = {}
        for product in products:
            direction, confidence = self.get_signal(product)
            if direction != NEUTRAL:
                result[product] = (direction, confidence)
        return result

    def is_active(self, product: str) -> bool:
        """True if any insider has a non-expired signal on this product."""
        _, conf = self.get_signal(product)
        return conf > 0.0

    # ── Order generation ──────────────────────────────────────────────────────

    def follow_orders(
        self,
        product: str,
        state,
        pos_limit: int,
        aggression: float = 1.0,
    ) -> list:
        """
        Generate Order objects to follow the insider's implied position.

        Parameters
        ----------
        product    : the symbol to trade
        state      : current TradingState
        pos_limit  : maximum absolute position allowed for this product
        aggression : float [0, 1] scaling the target position.
                     Pass the confidence value from get_signal() to size
                     proportionally — full size only when confidence is 1.0.

        Returns a list of Order objects (empty if no signal or no liquidity).

        The orders are MARKET-CROSSING (aggressive) — they cross the spread
        to guarantee a fill, because the insider signal is time-sensitive.
        Use only when confidence > 0 and within FOLLOW_WINDOW.
        """
        from datamodel import Order  # imported locally to avoid circular imports

        direction, confidence = self.get_signal(product)
        if direction == NEUTRAL or confidence == 0.0:
            return []

        order_depth = state.order_depths.get(product)
        if order_depth is None:
            return []

        current_position = state.position.get(product, 0)

        # Scale target position by aggression (confidence)
        target_position = int(direction * pos_limit * aggression)
        remaining = target_position - current_position

        orders = []

        if remaining > 0:
            # Need to BUY — cross the ask (take the cheapest offer)
            asks = sorted(order_depth.sell_orders.keys())  # ascending
            volume_to_fill = remaining
            for ask_price in asks:
                if volume_to_fill <= 0:
                    break
                available = abs(order_depth.sell_orders[ask_price])
                fill = min(volume_to_fill, available)
                orders.append(Order(product, ask_price, fill))
                volume_to_fill -= fill

        elif remaining < 0:
            # Need to SELL — cross the bid (hit the highest bid)
            bids = sorted(order_depth.buy_orders.keys(), reverse=True)  # descending
            volume_to_fill = abs(remaining)
            for bid_price in bids:
                if volume_to_fill <= 0:
                    break
                available = order_depth.buy_orders[bid_price]
                fill = min(volume_to_fill, available)
                orders.append(Order(product, bid_price, -fill))
                volume_to_fill -= fill

        return orders

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def log_summary(self) -> str:
        """Return a human-readable summary of all active signals this tick."""
        if not self._activity_log:
            return f"[InsiderTracker ts={self._current_ts}] No insider activity detected."
        return "\n".join(self._activity_log)

    def dump_state(self) -> dict:
        """Return the full internal state as a dict for debugging."""
        return {
            "current_ts": self._current_ts,
            "states": {
                f"{product}|{insider_id}": {
                    "direction": s.direction(),
                    "confidence": s.confidence(self._current_ts),
                    "last_bought_ts": s.last_bought_ts,
                    "last_sold_ts": s.last_sold_ts,
                }
                for (product, insider_id), s in self._states.items()
            },
        }


# ── Standalone test (no Prosperity datamodel needed) ──────────────────────────

if __name__ == "__main__":
    """
    Quick sanity test using mock objects — no Prosperity datamodel dependency.
    Run: python insider_tracker.py
    """

    class MockTrade:
        def __init__(self, buyer, seller, price, quantity, timestamp, symbol="KELP"):
            self.buyer     = buyer
            self.seller    = seller
            self.price     = price
            self.quantity  = quantity
            self.timestamp = timestamp
            self.symbol    = symbol

    class MockOrderDepth:
        def __init__(self):
            self.buy_orders  = {1999: 10, 1998: 5}
            self.sell_orders = {2001: -10, 2002: -5}

    class MockState:
        def __init__(self, ts, market_trades, traderData=""):
            self.timestamp    = ts
            self.market_trades = market_trades
            self.own_trades   = {}
            self.position     = {"KELP": 0, "SQUID_INK": 0}
            self.order_depths = {"KELP": MockOrderDepth(), "SQUID_INK": MockOrderDepth()}
            self.traderData   = traderData

    print("=== InsiderTracker Sanity Test ===\n")

    # Tick 1: Olivia buys KELP → LONG signal
    state1 = MockState(
        ts=1000,
        market_trades={
            "KELP": [MockTrade("Olivia", "MarketMaker", 2000, 15, 1000, "KELP")],
            "SQUID_INK": [],
        },
    )
    tracker = InsiderTracker()
    tracker.update(state1)
    print(tracker.log_summary())
    direction, conf = tracker.get_signal("KELP")
    print(f"KELP signal: direction={'LONG' if direction==1 else 'SHORT' if direction==-1 else 'NEUTRAL'}, confidence={conf:.2f}")
    print()

    # Persist state
    td = tracker.to_trader_data()

    # Tick 2 (100 ts later): Olivia sells KELP → flips to SHORT
    state2 = MockState(
        ts=1100,
        market_trades={
            "KELP": [MockTrade("MarketMaker", "Olivia", 2010, 15, 1100, "KELP")],
            "SQUID_INK": [],
        },
        traderData=td,
    )
    tracker2 = InsiderTracker.from_trader_data(state2.traderData)
    tracker2.update(state2)
    print(tracker2.log_summary())
    direction, conf = tracker2.get_signal("KELP")
    print(f"KELP signal: direction={'LONG' if direction==1 else 'SHORT' if direction==-1 else 'NEUTRAL'}, confidence={conf:.2f}")
    print()

    # Tick 3 (700 ts after sell): signal should be expired
    state3 = MockState(ts=1800, market_trades={"KELP": [], "SQUID_INK": []}, traderData=tracker2.to_trader_data())
    tracker3 = InsiderTracker.from_trader_data(state3.traderData)
    tracker3.update(state3)
    direction, conf = tracker3.get_signal("KELP")
    print(f"KELP signal after {1800-1100} ts (window={FOLLOW_WINDOW}): "
          f"direction={'LONG' if direction==1 else 'SHORT' if direction==-1 else 'NEUTRAL'}, "
          f"confidence={conf:.2f} (should be 0.0 -> NEUTRAL)")

    print("\n=== All active signals ===")
    print(tracker3.get_all_active_signals())
    print("\n=== State dump ===")
    import pprint
    pprint.pprint(tracker3.dump_state())

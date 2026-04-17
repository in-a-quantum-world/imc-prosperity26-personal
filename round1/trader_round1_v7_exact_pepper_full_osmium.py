
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import jsonpickle


class Trader:
    """
    Version 7:
      - INTARIAN_PEPPER_ROOT:
          exact day-counter directional carry trader
      - ASH_COATED_OSMIUM:
          full-capacity simplified EMA/imbalance/z-score market maker

    Design goals:
      1) Pepper should reliably deliver the 7.4k-ish baseline.
      2) Osmium should be where the extra edge comes from, but through simple,
         robust logic rather than cycle / insider / regime spaghetti.
      3) Layered orders improve fill quality without requiring constant crossing.
    """

    LIMITS = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    PEPPER_SOFT_LIMIT = 80
    OSMIUM_SOFT_LIMIT = 78
    DAY_LEN_EST = 1_000_000

    def run(self, state: TradingState):
        data = self._load_data(state)
        self._handle_day_reset(state, data)

        result = {
            self.PEPPER: [],
            self.OSMIUM: [],
        }

        if self.PEPPER in state.order_depths:
            result[self.PEPPER] = self.trade_pepper(state, state.order_depths[self.PEPPER], data)

        if self.OSMIUM in state.order_depths:
            result[self.OSMIUM] = self.trade_osmium(state, state.order_depths[self.OSMIUM], data)

        traderData = jsonpickle.encode(data)
        return result, 0, traderData

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def _load_data(self, state: TradingState) -> Dict:
        if getattr(state, "traderData", None):
            try:
                data = jsonpickle.decode(state.traderData)
                if isinstance(data, dict):
                    data.setdefault("last_timestamp", None)
                    data.setdefault("day_index", 0)
                    data.setdefault("osmium_short_ema", None)
                    data.setdefault("osmium_long_ema", None)
                    data.setdefault("osmium_last_imbalance", 0.0)
                    data.setdefault("osmium_mid_hist", [])
                    return data
            except Exception:
                pass
        return {
            "last_timestamp": None,
            "day_index": 0,
            "osmium_short_ema": None,
            "osmium_long_ema": None,
            "osmium_last_imbalance": 0.0,
            "osmium_mid_hist": [],
        }

    def _handle_day_reset(self, state: TradingState, data: Dict) -> None:
        last_ts = data.get("last_timestamp")
        if last_ts is not None and state.timestamp < last_ts:
            data["day_index"] = int(data.get("day_index", 0)) + 1
            data["osmium_short_ema"] = None
            data["osmium_long_ema"] = None
            data["osmium_last_imbalance"] = 0.0
            data["osmium_mid_hist"] = []
        data["last_timestamp"] = state.timestamp

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def _sorted_book(self, order_depth: OrderDepth) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        bids = sorted(order_depth.buy_orders.items(), reverse=True)
        asks = sorted(order_depth.sell_orders.items())
        return bids, asks

    def _best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int], int, int]:
        bids, asks = self._sorted_book(order_depth)
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        bid_vol = bids[0][1] if bids else 0
        ask_vol = -asks[0][1] if asks else 0
        return best_bid, best_ask, bid_vol, ask_vol

    def _mid(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask, _, _ = self._best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    def _microprice(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask, bid_vol, ask_vol = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return self._mid(order_depth)
        total = bid_vol + ask_vol
        if total <= 0:
            return (best_bid + best_ask) / 2.0
        return (best_ask * bid_vol + best_bid * ask_vol) / total

    def _spread(self, order_depth: OrderDepth) -> Optional[int]:
        best_bid, best_ask, _, _ = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return None
        return best_ask - best_bid

    def _imbalance(self, order_depth: OrderDepth) -> float:
        _, _, bid_vol, ask_vol = self._best_bid_ask(order_depth)
        denom = bid_vol + ask_vol
        if denom <= 0:
            return 0.0
        return (bid_vol - ask_vol) / denom

    def _update_ema(self, prev: Optional[float], x: float, alpha: float) -> float:
        if prev is None:
            return x
        return alpha * x + (1.0 - alpha) * prev

    def _append_hist(self, xs: List[float], x: float, cap: int = 60) -> None:
        xs.append(x)
        if len(xs) > cap:
            del xs[0]

    def _mean_std(self, xs: List[float]) -> Tuple[float, float]:
        if not xs:
            return 0.0, 1.0
        n = len(xs)
        mu = sum(xs) / n
        var = sum((x - mu) ** 2 for x in xs) / max(1, n)
        std = var ** 0.5
        return mu, max(std, 1e-6)

    def _add_buy(self, orders: List[Order], product: str, price: int, qty: int, pos: int, limit: int) -> int:
        if qty <= 0:
            return pos
        qty = min(qty, limit - pos)
        if qty > 0:
            orders.append(Order(product, int(price), int(qty)))
            pos += qty
        return pos

    def _add_sell(self, orders: List[Order], product: str, price: int, qty: int, pos: int, limit: int) -> int:
        if qty <= 0:
            return pos
        qty = min(qty, limit + pos)
        if qty > 0:
            orders.append(Order(product, int(price), int(-qty)))
            pos -= qty
        return pos

    # ------------------------------------------------------------------
    # PEPPER: exact day-counter directional
    # ------------------------------------------------------------------

    def _pepper_fair_value(self, state: TradingState, data: Dict, order_depth: OrderDepth) -> float:
        day_index = int(data.get("day_index", 0))
        # uploaded files are day -2, -1, 0 -> anchors 10000, 11000, 12000
        anchor = 10000 + 1000 * day_index
        mid = self._mid(order_depth)
        micro = self._microprice(order_depth)
        imb = self._imbalance(order_depth)
        base = anchor + 0.001 * state.timestamp
        if mid is not None and micro is not None:
            base += 0.20 * (micro - mid)
        base += 0.60 * imb
        return base

    def _pepper_target_position(self, timestamp: int) -> int:
        # stay max long basically all day
        if timestamp < 985_000:
            return 80
        return 40

    def trade_pepper(self, state: TradingState, order_depth: OrderDepth, data: Dict) -> List[Order]:
        product = self.PEPPER
        limit = self.LIMITS[product]
        pos = self._position(state, product)

        best_bid, best_ask, _, _ = self._best_bid_ask(order_depth)
        mid = self._mid(order_depth)
        spread = self._spread(order_depth)

        if best_bid is None or best_ask is None or mid is None:
            return []

        fv = self._pepper_fair_value(state, data, order_depth)
        target = self._pepper_target_position(state.timestamp)
        gap = target - pos

        orders: List[Order] = []
        bids, asks = self._sorted_book(order_depth)

        # Aggressively lift offers while under target.
        if gap > 0:
            if state.timestamp < 50_000:
                cross_edge = 4.0
            elif state.timestamp < 150_000:
                cross_edge = 3.5
            else:
                cross_edge = 2.5

            # account for the known early drawdown -> do not go insane at t=0
            if state.timestamp < 5_000:
                cross_edge -= 1.0
            if gap >= 50:
                cross_edge += 0.5

            remaining = min(gap, limit - pos)
            for ask_price, ask_qty_signed in asks:
                ask_qty = -ask_qty_signed
                if remaining <= 0:
                    break
                if ask_price <= fv + cross_edge:
                    take = min(ask_qty, remaining)
                    if take > 0:
                        pos = self._add_buy(orders, product, ask_price, take, pos, limit)
                        remaining -= take
                else:
                    break

        # Layered refill bids
        gap = target - pos
        if spread is None:
            spread = 2

        if gap > 35:
            bid_sizes = [22, 14, 8]
        elif gap > 10:
            bid_sizes = [12, 8, 5]
        else:
            bid_sizes = [5, 3, 2]

        bid_levels = [
            min(best_bid + (1 if spread >= 2 else 0), int(round(fv))),
            int(round(fv - 1)),
            int(round(fv - 2)),
        ]

        for px, qty in zip(bid_levels, bid_sizes):
            if px < best_ask:
                pos = self._add_buy(orders, product, px, qty, pos, limit)

        # Very limited selling: only obviously rich or extremely late
        rich = best_bid - fv
        if pos > target + 10 or rich >= 4 or state.timestamp > 995_000:
            ask_levels = [max(best_bid + 1, int(round(fv + 2))), int(round(fv + 4))]
            ask_sizes = [6, 4] if pos < 75 else [10, 6]
            for px, qty in zip(ask_levels, ask_sizes):
                px = max(px, best_bid + 1)
                pos = self._add_sell(orders, product, px, qty, pos, limit)

        return orders

    # ------------------------------------------------------------------
    # OSMIUM: simplified full-capacity MM
    # ------------------------------------------------------------------

    def trade_osmium(self, state: TradingState, order_depth: OrderDepth, data: Dict) -> List[Order]:
        product = self.OSMIUM
        hard_limit = self.LIMITS[product]
        soft_limit = self.OSMIUM_SOFT_LIMIT
        pos = self._position(state, product)

        best_bid, best_ask, _, _ = self._best_bid_ask(order_depth)
        mid = self._mid(order_depth)
        spread = self._spread(order_depth)
        micro = self._microprice(order_depth)
        imb = self._imbalance(order_depth)

        if None in (best_bid, best_ask, mid, spread):
            return []

        short_ema = self._update_ema(data.get("osmium_short_ema"), mid, 0.10)
        long_ema = self._update_ema(data.get("osmium_long_ema"), mid, 0.03)
        data["osmium_short_ema"] = short_ema
        data["osmium_long_ema"] = long_ema

        self._append_hist(data["osmium_mid_hist"], mid, cap=60)
        hist = data["osmium_mid_hist"]
        mu, sigma = self._mean_std(hist[-30:] if len(hist) >= 10 else hist)
        z = (mid - mu) / sigma if sigma > 1e-9 else 0.0

        last_imb = data.get("osmium_last_imbalance", 0.0)
        imb_change = imb - last_imb
        data["osmium_last_imbalance"] = imb

        # Simplified FV: EMA spread + imbalance + mean reversion.
        fv = 10000.0
        fv += 0.75 * (short_ema - long_ema)
        fv += 0.80 * imb
        fv += 0.15 * imb_change
        fv += 0.08 * (10000.0 - mid)
        fv += -0.55 * z
        if micro is not None:
            fv += 0.20 * (micro - mid)
        fv += -0.05 * pos

        signal = fv - mid
        abs_signal = abs(signal)

        # Size/width chosen to be active but not insane.
        if spread >= 24:
            halfwidth = 4
            base_size = 7
        elif spread >= 18:
            halfwidth = 3
            base_size = 7
        else:
            halfwidth = 2
            base_size = 6

        if abs(z) > 2.2:
            base_size = max(4, base_size - 2)

        orders: List[Order] = []
        bids, asks = self._sorted_book(order_depth)

        # Strong-signal taking only.
        TAKE_THRESHOLD = 3.0
        TAKE_EDGE = 1.5

        if signal > TAKE_THRESHOLD and pos < soft_limit:
            remaining = soft_limit - pos
            for ask_price, ask_qty_signed in asks:
                ask_qty = -ask_qty_signed
                if ask_price <= fv - TAKE_EDGE and mid <= fv:
                    take = min(ask_qty, 20, remaining)
                    if take > 0:
                        pos = self._add_buy(orders, product, ask_price, take, pos, hard_limit)
                        remaining -= take
                else:
                    break

        if signal < -TAKE_THRESHOLD and pos > -soft_limit:
            remaining = soft_limit + pos
            for bid_price, bid_qty in bids:
                if bid_price >= fv + TAKE_EDGE and mid >= fv:
                    take = min(bid_qty, 20, remaining)
                    if take > 0:
                        pos = self._add_sell(orders, product, bid_price, take, pos, hard_limit)
                        remaining -= take
                else:
                    break

        # Layered passive quotes.
        improve_bid = 1 if signal > 0.6 and spread >= 2 else 0
        improve_ask = 1 if signal < -0.6 and spread >= 2 else 0

        bid_levels = [
            min(best_bid + improve_bid, int(round(fv - halfwidth))),
            int(round(fv - halfwidth - 1)),
            int(round(fv - halfwidth - 2)),
        ]
        ask_levels = [
            max(best_ask - improve_ask, int(round(fv + halfwidth))),
            int(round(fv + halfwidth + 1)),
            int(round(fv + halfwidth + 2)),
        ]

        if signal > 1.0:
            bid_sizes = [10, 8, 5]
            ask_sizes = [2, 2, 1]
        elif signal < -1.0:
            bid_sizes = [2, 2, 1]
            ask_sizes = [10, 8, 5]
        else:
            bid_sizes = [7, 5, 3]
            ask_sizes = [7, 5, 3]

        # Inventory skew
        if pos > 55:
            bid_sizes = [max(0, x - 5) for x in bid_sizes]
            ask_sizes = [x + 3 for x in ask_sizes]
        elif pos < -55:
            ask_sizes = [max(0, x - 5) for x in ask_sizes]
            bid_sizes = [x + 3 for x in bid_sizes]

        for px, qty in zip(bid_levels, bid_sizes):
            if qty > 0 and px < best_ask:
                pos = self._add_buy(orders, product, px, qty, pos, hard_limit)

        for px, qty in zip(ask_levels, ask_sizes):
            px = max(px, best_bid + 1)
            if qty > 0:
                pos = self._add_sell(orders, product, px, qty, pos, hard_limit)

        return orders
